"""Test unitari per la pipeline a 4 Step di estrazione e ingestion referti clinici."""
import asyncio
import pymupdf
import pytest

from vitality_be.llm.base import ChatMessage
from vitality_be.llm.embedder import EchoEmbedder
from vitality_be.schemas.extraction import (
    LLMReportExtractionBatch,
    RawExtractedItem,
    ReportExtractionResult,
)
from vitality_be.services.extractor import extract_from_report
from vitality_be.services.multimodal import inspect_pdf_in_ram, process_raw_attachment_bytes
from vitality_be.services.patient_context import PatientMaskingMap, scrub_report_text
from vitality_be.services.reports import process_and_ingest_report


class FakeStructuredReportLLM:
    """Mock LLM con output strutturato Pydantic conforme per i referti."""

    def __init__(self, batch: LLMReportExtractionBatch | None = None):
        self.batch = batch or LLMReportExtractionBatch(
            is_medical_report=True,
            report_type="ANALISI_SANGUE",
            report_date_iso="2026-10-15T08:00:00+02:00",
            clean_summary="Profilo glicemico e lipidico stabile. HbA1c nei limiti terapeutici.",
            items=[
                RawExtractedItem(
                    target_table="diabetes_data",
                    metric_type="HbA1c",
                    metric_value="6.8",
                    timestamp_iso="2026-10-15T08:00:00+02:00",
                    summary_title="Emoglobina Glicata: 6.8 %",
                ),
                RawExtractedItem(
                    target_table="diabetes_data",
                    metric_type="GLICEMIA",
                    metric_value="120",
                    timestamp_iso="2026-10-15T08:00:00+02:00",
                    summary_title="Glicemia: 120 mg/dL",
                ),
            ],
        )

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        response_schema: type,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ):
        return self.batch


def create_synthetic_pdf(text: str) -> bytes:
    """Genera un PDF digitale nativo valido interamente in RAM."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_step1_pdf_inspection_native():
    """Verifica che un PDF con testo venga rilevato come digitale nativo (Step 1)."""
    text = (
        "OSPEDALE MAGGIORE DI MILANO - LABORATORIO ANALISI\n"
        "Paziente: Mario Rossi - Codice Fiscale: RSSMRA70A01H501U\n"
        "Data esame: 15/10/2026\n"
        "Emoglobina Glicata (HbA1c): 6.8 %\n"
        "Glucosio plasmatico: 125 mg/dL\n"
        "Colesterolo Totale: 195 mg/dL\n"
        "Colesterolo LDL: 110 mg/dL\n"
        "Trigliceridi: 150 mg/dL\n"
    )
    pdf_bytes = create_synthetic_pdf(text)
    result = inspect_pdf_in_ram(pdf_bytes)
    assert result.is_native_digital is True
    assert result.page_count == 1
    assert result.alnum_char_count >= 80
    assert "Emoglobina Glicata" in result.extracted_text


def test_step2_pii_scrubbing_report():
    """Verifica la rimozione di CF, ospedale, intestazioni e nomi dal testo del referto (Step 2)."""
    raw_text = (
        "Ospedale San Raffaele - Laboratorio Analisi Cliniche\n"
        "Paziente: Mario Rossi\n"
        "Codice Fiscale: RSSMRA70A01H501U\n"
        "Indirizzo: Via Roma 12, Milano\n"
        "Telefono: 3331234567\n"
        "Medico Curante: Dott. Marco Bianchi\n"
        "Esame: Glicemia 120 mg/dL, HbA1c 6.8 %.\n"
    )
    masking_map = PatientMaskingMap(first_name="Mario", last_name="Rossi")
    scrubbed, updated_map = scrub_report_text(raw_text, masking_map)

    # Verifiche GDPR Art. 9
    assert "RSSMRA70A01H501U" not in scrubbed
    assert "Mario Rossi" not in scrubbed
    assert "3331234567" not in scrubbed
    assert "Via Roma 12" not in scrubbed
    assert "Ospedale San Raffaele" not in scrubbed
    assert "[CF_RIMOSSO]" in scrubbed or "[INTESTAZIONE_ANAGRAFICA_RIMOSSA]" in scrubbed
    assert "Glicemia 120" in scrubbed


def _run_coro_sync(coro):
    """Esegue una coroutine puramente in-memory senza aprire socket di loopback su Windows sandbox."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


def test_step3_extract_from_report_echo():
    """Verifica l'estrazione strutturata Pydantic v2 tramite FakeStructuredReportLLM (Step 3)."""
    llm = FakeStructuredReportLLM()
    # Testo con >= 80 caratteri alfanumerici per attivare PyMuPDF native
    pdf_bytes = create_synthetic_pdf(
        "Referto Laboratorio Analisi Cliniche\n"
        "Esami ematochimici eseguiti in data 15/10/2026.\n"
        "Glicemia a digiuno: 120 mg/dL. Emoglobina Glicata HbA1c: 6.8 %.\n"
        "Quadro metabolico soddisfacente in terapia corrente.\n"
    )
    result: ReportExtractionResult = _run_coro_sync(
        extract_from_report(
            llm,
            media_bytes=pdf_bytes,
            media_mime_type="application/pdf",
        )
    )
    assert result.extraction_method == "pymupdf_native"
    assert result.report_type == "ANALISI_SANGUE"
    assert len(result.proposed_entries) == 2
    assert any(p.data.get("metric_type") == "HbA1c" for p in result.proposed_entries)


def test_step4_reports_pipeline_ingest():
    """Verifica la pipeline end-to-end a 4 step con EchoEmbedder e Mock DB (Step 4)."""
    llm = FakeStructuredReportLLM()
    embedder = EchoEmbedder(dim=1536)

    class MockTable:
        def __init__(self, name):
            self.name = name
            self.last_inserted = None

        def insert(self, data):
            self.last_inserted = data
            return self

        def select(self, *args, **kwargs):
            return self

        def limit(self, *args, **kwargs):
            return self

        def execute(self):
            class Resp:
                data = [{"id": 101, "name": "Mario", "surname": "Rossi"}]
            return Resp()

    class MockUserDb:
        def __init__(self):
            self.tables = {}

        def table(self, name):
            if name not in self.tables:
                self.tables[name] = MockTable(name)
            return self.tables[name]

    user_db = MockUserDb()
    pdf_bytes = create_synthetic_pdf(
        "Laboratorio Analisi Mediche - Referto Esami Sangue Completo\n"
        "Paziente: Mario Rossi - Codice Fiscale: RSSMRA70A01H501U\n"
        "Emoglobina Glicata: 6.8 % - Glicemia: 120 mg/dL.\n"
        "Profilo lipidico e renale nei limiti della norma.\n"
    )
    processed = process_raw_attachment_bytes(
        raw_bytes=pdf_bytes,
        mime_type="application/pdf",
        kind="medical_report",
    )

    response = _run_coro_sync(
        process_and_ingest_report(
            attachment=processed,
            title="Analisi Ottobre",
            user_db=user_db,
            llm=llm,
            embedder=embedder,
        )
    )

    assert response.id == "101"
    assert response.extraction_method == "pymupdf_native"
    assert len(response.extracted) == 2
    assert user_db.table("patient_reports").last_inserted is not None
    saved_row = user_db.table("patient_reports").last_inserted
    assert saved_row["embedding"] is not None
    assert saved_row["embedding"].startswith("[")
    assert saved_row["clean_summary"] != ""
