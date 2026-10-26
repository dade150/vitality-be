import asyncio
from datetime import date, datetime, timezone
import types as pytypes

import pytest


@pytest.fixture(autouse=True)
def _inline_to_thread(monkeypatch):
    async def _run_sync(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _run_sync)

from vitality_be.services.patient_context import (
    TOKEN_DOCTOR_NAME,
    TOKEN_PATIENT_NAME,
    PatientMaskingMap,
    build_dual_rag_message,
    calculate_age_years,
    calculate_bmi,
    fetch_patient_context,
    normalize_metric_type,
    rehydrate_assistant_reply,
    scrub_free_text,
)
from vitality_be.services.retriever import RetrievedChunk


class FakeTableQuery:
    def __init__(self, rows=None, exc: Exception | None = None):
        self._rows = rows if rows is not None else []
        self._exc = exc

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self._exc is not None:
            raise self._exc
        return pytypes.SimpleNamespace(data=self._rows)


class FakePatientDb:
    def __init__(self, tables: dict[str, list[dict]] | None = None, failing_tables: set[str] | None = None):
        self.tables = tables or {}
        self.failing_tables = failing_tables or set()

    def table(self, name: str) -> FakeTableQuery:
        if name in self.failing_tables:
            return FakeTableQuery(exc=RuntimeError(f"Errore DB simulato su {name}"))
        return FakeTableQuery(rows=self.tables.get(name, []))


def test_normalize_metric_type_handles_duplicates():
    assert normalize_metric_type("HbA1c") == "HBA1C"
    assert normalize_metric_type("HBA1C") == "HBA1C"
    assert normalize_metric_type("eGFR") == "EGFR"
    assert normalize_metric_type("EGFR") == "EGFR"
    assert normalize_metric_type("Colesterolo Totale") == "COLESTEROLO_TOTALE"
    assert normalize_metric_type("COLESTEROLO_TOTALE") == "COLESTEROLO_TOTALE"
    assert normalize_metric_type("Trigliceridi") == "TRIGLICERIDI"


def test_calculate_age_and_bmi():
    assert calculate_age_years("1960-05-20", today=date(2026, 5, 19)) == 65
    assert calculate_age_years("1960-05-20", today=date(2026, 5, 20)) == 66
    assert calculate_age_years("invalid-date") is None

    bmi_res = calculate_bmi(82, 172)
    assert bmi_res == (27.7, "sovrappeso")
    assert calculate_bmi(-5, 170) is None


def test_scrub_free_text_and_rehydrate_reply():
    m = PatientMaskingMap(
        first_name="Mario",
        last_name="Rossi",
        primary_doctor="Dott.ssa Verdi",
        doctor_tokens={TOKEN_DOCTOR_NAME: "Dott.ssa Verdi", "{{MEDICO_2}}": "Prof. Neri"},
    )
    dirty = (
        "Sono Mario Rossi, la mia mail è mario.rossi@example.com, "
        "cell 333 1234567, CF RSSMRA60E20H501Z, seguito da Dott.ssa Verdi."
    )
    cleaned = scrub_free_text(dirty, m)
    assert "Mario Rossi" not in cleaned
    assert "Rossi" not in cleaned
    assert "mario.rossi@example.com" not in cleaned
    assert "333 1234567" not in cleaned
    assert "RSSMRA60E20H501Z" not in cleaned
    assert "Dott.ssa Verdi" not in cleaned
    assert TOKEN_PATIENT_NAME in cleaned
    assert TOKEN_DOCTOR_NAME in cleaned

    llm_out = "Ciao {{NOME}}! Ottimo lavoro oggi. Parlane con {{MEDICO}} o con {{MEDICO_2}} alla visita."
    rehydrated = rehydrate_assistant_reply(llm_out, m)
    assert rehydrated == (
        "Ciao Mario! Ottimo lavoro oggi. Parlane con Dott.ssa Verdi o con Prof. Neri alla visita."
    )

    # Caso senza nome o medico salvato nel DB -> pulizia elegante dei segnaposto
    empty_rehydrated = rehydrate_assistant_reply(
        "Buongiorno, {{NOME}}! Ricordati di sentire {{MEDICO}}.",
        PatientMaskingMap(),
    )
    assert "{{NOME}}" not in empty_rehydrated
    assert "{{MEDICO}}" not in empty_rehydrated
    assert empty_rehydrated == "Buongiorno! Ricordati di sentire il tuo medico di riferimento."


@pytest.mark.asyncio
async def test_fetch_patient_context_full_and_zero_pii_leak():
    fixed_now = datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc)
    db = FakePatientDb(
        tables={
            "user": [
                {
                    "name": "Giulia",
                    "surname": "Bianchi",
                    "birth": "1964-04-12",
                }
            ],
            "profile": [
                {
                    "peso": 74,
                    "altezza": 165,
                    "sesso": "femmina",
                    "note_anamnestiche": "Diabete tipo 2 dal 2019, ipertensione lieve. Contatto: giulia@test.it",
                }
            ],
            "diabetes_data": [
                {"metric_type": "GLICEMIA", "metric_value": "132", "measured_at": "2026-10-26T07:30:00Z"},
                {"metric_type": "GLICEMIA", "metric_value": "64", "measured_at": "2026-10-25T19:00:00Z"},
                {"metric_type": "GLICEMIA", "metric_value": "195", "measured_at": "2026-10-25T14:00:00Z"},
                {"metric_type": "PRESSIONE", "metric_value": "128/78", "measured_at": "2026-10-26T08:00:00Z"},
                {"metric_type": "HbA1c", "metric_value": "6.9", "measured_at": "2026-09-10T09:00:00Z"},
                {"metric_type": "EGFR", "metric_value": "78", "measured_at": "2026-09-10T09:00:00Z"},
                {"metric_type": "Colesterolo Totale", "metric_value": "175", "measured_at": "2026-09-10T09:00:00Z"},
            ],
            "therapies": [
                {
                    "id": 1,
                    "therapy_type": "COMPRESSE_ORALI",
                    "drug_name": "Metformina",
                    "dosage": "1000 mg",
                    "frequency": "2 volte al giorno",
                    "scheduled_time": "08:00:00",
                    "notes": "Ai pasti principali",
                    "is_active": True,
                }
            ],
            "therapy_logs": [
                {"therapy_id": 1, "taken_at": "2026-10-26T08:05:00Z", "status": "ASSUNTA"}
            ],
            "meals": [
                {
                    "meal_type": "COLAZIONE",
                    "description": "Yogurt greco bianco e 30g fiocchi d'avena",
                    "carbs_grams": 24,
                    "calories": 210,
                    "is_estimated": False,
                    "meal_at": "2026-10-26T07:45:00Z",
                }
            ],
            "physical_activities": [
                {
                    "activity_type": "CAMMINATA",
                    "duration_minutes": 40,
                    "intensity": "MODERATA",
                    "steps": 4500,
                    "notes": "Passeggiata al parco",
                    "activity_at": "2026-10-25T17:00:00Z",
                }
            ],
            "appointments": [
                {
                    "title": "Controllo diabetologico semestrale",
                    "doctor_name": "Dott. Marco Ferri",
                    "appointment_at": "2026-11-15T10:30:00Z",
                    "notes": "Portare diario glicemico",
                }
            ],
            "patient_reports": [
                {
                    "report_type": "VISITA_DIABETOLOGICA",
                    "source_name": "Ospedale",
                    "content": "Paziente Giulia Bianchi con buon compenso glicemico generale.",
                    "report_at": "2026-05-15T10:00:00Z",
                }
            ],
        }
    )

    res = await fetch_patient_context(db, now=fixed_now)
    xml = res.context_xml

    assert xml.startswith("<contesto_paziente>")
    assert xml.endswith("</contesto_paziente>")
    assert res.sections_count == 8

    # ZERO PII leak: né il nome reale, né il cognome, né la data di nascita esatta, né il medico reale, né l'email
    assert "Giulia" not in xml
    assert "Bianchi" not in xml
    assert "1964-04-12" not in xml
    assert "Dott. Marco Ferri" not in xml
    assert "giulia@test.it" not in xml

    # Presenza dei dati clinici generalizzati e dei token reversibili
    assert "Età: 62 anni" in xml
    assert "IMC 27.2 kg/m² - sovrappeso" in xml
    assert TOKEN_PATIENT_NAME in xml
    assert TOKEN_DOCTOR_NAME in xml
    assert "media 130 mg/dL (min 64, max 195 mg/dL | in range 70-180: 1, ipoglicemie <70: 1, iperglicemie >180: 1)" in xml
    assert "HbA1c (Emoglobina Glicata): 6.9 %" in xml
    assert "eGFR (Filtrato Glomerulare): 78 mL/min/1.73m²" in xml
    assert "Colesterolo Totale: 175 mg/dL" in xml
    assert "Metformina" in xml
    assert "Totale movimento ultimi 7 giorni: 40 minuti" in xml


@pytest.mark.asyncio
async def test_fetch_patient_context_graceful_partial_degradation():
    db = FakePatientDb(
        tables={
            "profile": [{"peso": 70, "altezza": 175, "sesso": "maschio", "note_anamnestiche": None}],
        },
        failing_tables={"user", "diabetes_data", "therapies", "meals"},
    )
    res = await fetch_patient_context(db)
    assert "<contesto_paziente>" in res.context_xml
    assert "IMC 22.9 kg/m² - normopeso" in res.context_xml
    assert res.sections_count == 1


def test_build_dual_rag_message_combines_both_contexts():
    chunk = RetrievedChunk(
        id=1,
        doc_id="ada-1",
        chunk_index=0,
        source_name="ADA 2026",
        document_type="guideline",
        heading_path="Attività fisica",
        content="Si raccomandano 150 minuti a settimana di esercizio aerobico moderato.",
        language="it",
        similarity=0.9,
        fts_rank=0.4,
        rrf_score=0.03,
    )
    msg = build_dual_rag_message(
        "Quanto sto camminando rispetto alle linee guida?",
        patient_context_xml="<contesto_paziente>\n[Attività Fisica Recente]\n- 40 min\n</contesto_paziente>",
        clinical_chunks=[chunk],
    )
    assert msg.index("<contesto_paziente>") < msg.index("<contesto_clinico>") < msg.index("<messaggio_utente>")


@pytest.mark.asyncio
async def test_chat_reply_dual_rag_scrubs_input_and_rehydrates_output():
    from vitality_be.llm.base import ChatMessage
    from vitality_be.schemas.chat import ChatMessageIn
    from vitality_be.services import chat as chat_service

    captured: list[tuple[list[ChatMessage], str | None]] = []

    class CapturingLLM:
        async def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str:
            captured.append((messages, system))
            return "Ciao {{NOME}}! La tua HbA1c è buona. Ne parlerai con {{MEDICO}} alla prossima visita."

    class DualFakeDb(FakePatientDb):
        def rpc(self, fn_name: str, _params: dict):
            if fn_name == "get_patient_context_bundle":
                return FakeTableQuery(
                    rows={
                        "user": [{"name": "Luca", "surname": "Moretti", "birth": "1970-01-10"}],
                        "profile": [{"peso": 80, "altezza": 178, "sesso": "maschio", "note_anamnestiche": None}],
                        "diabetes_data": [
                            {"metric_type": "HBA1C", "metric_value": "6.7", "measured_at": "2026-10-01T08:00:00Z"}
                        ],
                        "therapies": [],
                        "therapy_logs": [],
                        "meals": [],
                        "physical_activities": [],
                        "appointments": [
                            {
                                "title": "Visita diabetologica",
                                "doctor_name": "Dott.ssa Elena Riva",
                                "appointment_at": "2026-11-20T09:00:00Z",
                                "notes": None,
                            }
                        ],
                        "patient_reports": [],
                    }
                )
            return FakeTableQuery(
                rows=[
                    {
                        "id": 1,
                        "doc_id": "sid-1",
                        "chunk_index": 0,
                        "source_name": "Linee Guida SID/AMD 2025",
                        "document_type": "guideline",
                        "heading_path": "Target HbA1c",
                        "content": "Un valore di HbA1c intorno a 6.5-7.0% indica un buon compenso.",
                        "language": "it",
                        "similarity": 0.91,
                        "fts_rank": 0.5,
                        "rrf_score": 0.033,
                    }
                ]
            )

    class SimpleEmbedder:
        async def embed_query(self, _q: str) -> list[float]:
            return [0.1, 0.2]

    db = DualFakeDb()
    out = await chat_service.reply(
        CapturingLLM(),
        [
            ChatMessageIn(
                role="user",
                content="Ciao, sono Luca Moretti (luca@email.it), come va la mia glicata?",
            )
        ],
        user_db=db,
        embedder=SimpleEmbedder(),
    )

    # 1. La risposta finale al FE ha i token re-idratati con i nomi reali
    assert out == "Ciao Luca! La tua HbA1c è buona. Ne parlerai con Dott.ssa Elena Riva alla prossima visita."

    # 2. Ciò che è stato inviato a Vertex AI NON contiene alcun dato identificativo reale
    sent_messages, sent_system = captured[0]
    sent_user_content = sent_messages[-1].content
    assert "Luca" not in sent_user_content
    assert "Moretti" not in sent_user_content
    assert "luca@email.it" not in sent_user_content
    assert "Dott.ssa Elena Riva" not in sent_user_content
    assert "<contesto_paziente>" in sent_user_content
    assert "<contesto_clinico>" in sent_user_content
    assert "HbA1c (Emoglobina Glicata): 6.7 %" in sent_user_content
    assert "Linee Guida SID/AMD 2025" in sent_user_content
    assert sent_system is not None
    assert "<contesto_paziente>" in sent_system  # Istruzioni sull'uso del blocco nel system prompt
    assert "HbA1c (Emoglobina Glicata): 6.7 %" not in sent_system  # Ma MAI i dati reali nel system prompt
