from datetime import datetime

import pytest

from vitality_be.llm.base import ChatMessage
from vitality_be.schemas.chat import ChatMessageIn
from vitality_be.schemas.extraction import (
    LLMExtractionBatch,
    LLMReportExtractionBatch,
    RawExtractedItem,
    canonicalize_metric_type,
    normalize_and_validate_metric_value,
)
from vitality_be.services import chat as chat_service
from vitality_be.services.extractor import (
    ROME_TZ,
    extract_clinical_proposals,
    extract_from_report,
    should_attempt_extraction,
)
from vitality_be.services.patient_context import (
    PatientMaskingMap,
    mask_user_text_with_dynamic_tokens,
)


class FakeStructuredLLM:
    """Mock LLM che cattura tutti i prompt inviati (chat + structured) per verificare zero PII leak."""

    def __init__(
        self,
        *,
        chat_reply: str = "Perfetto {{NOME}}, puoi confermare i dati nella scheda qui sotto.",
        structured_batch: LLMExtractionBatch | LLMReportExtractionBatch | None = None,
        structured_exc: Exception | None = None,
    ):
        self.chat_reply = chat_reply
        self.structured_batch = structured_batch
        self.structured_exc = structured_exc
        self.chat_calls: list[tuple[list[ChatMessage], str | None]] = []
        self.structured_calls: list[dict] = []

    async def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str:
        self.chat_calls.append((messages, system))
        return self.chat_reply

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        response_schema: type,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ):
        self.structured_calls.append(
            {
                "messages": messages,
                "system": system,
                "response_schema": response_schema,
                "media_bytes": media_bytes,
                "media_mime_type": media_mime_type,
            }
        )
        if self.structured_exc is not None:
            raise self.structured_exc
        if self.structured_batch is not None:
            return self.structured_batch
        return response_schema()


def test_canonicalize_and_validate_glycemic_lipid_pressure_metrics():
    assert canonicalize_metric_type("HBA1C") == "HbA1c"
    assert canonicalize_metric_type("COLESTEROLO_TOTALE") == "Colesterolo Totale"
    assert canonicalize_metric_type("colesterolo") == "Colesterolo Totale"
    assert canonicalize_metric_type("hdl") == "HDL"
    assert canonicalize_metric_type("ldl") == "LDL"
    assert canonicalize_metric_type("TRIGLICERIDI") == "Trigliceridi"
    assert canonicalize_metric_type("EGFR") == "eGFR"

    assert normalize_and_validate_metric_value("GLICEMIA", "142 mg/dL") == "142"
    assert normalize_and_validate_metric_value("PRESSIONE", "130 / 80 mmHg") == "130/80"
    assert normalize_and_validate_metric_value("HbA1c", "6,8 %") == "6.8"
    assert normalize_and_validate_metric_value("Colesterolo Totale", "195 mg/dL") == "195"
    assert normalize_and_validate_metric_value("LDL", "98") == "98"
    assert normalize_and_validate_metric_value("HDL", "52") == "52"
    assert normalize_and_validate_metric_value("Trigliceridi", "145") == "145"

    with pytest.raises(ValueError):
        normalize_and_validate_metric_value("GLICEMIA", "valore normale")
    with pytest.raises(ValueError):
        normalize_and_validate_metric_value("GLICEMIA", "5")  # fuori range fisiologico
    with pytest.raises(ValueError):
        normalize_and_validate_metric_value("PRESSIONE", "80/130")  # sistolica < diastolica


def test_should_attempt_extraction_skips_greetings_and_hypotheticals():
    assert not should_attempt_extraction([ChatMessage("user", "Ciao!")])
    assert not should_attempt_extraction([ChatMessage("user", "Grazie mille")])
    assert not should_attempt_extraction(
        [ChatMessage("user", "Cosa succede se ho la glicemia a 200?")]
    )
    assert not should_attempt_extraction(
        [ChatMessage("user", "Qual è il valore normale del colesterolo LDL?")]
    )
    assert should_attempt_extraction(
        [ChatMessage("user", "Stamattina ho misurato la glicemia ed era 138")]
    )


def test_mask_sensitive_user_input_and_dynamic_doctor_tokens():
    initial_map = PatientMaskingMap(first_name="Mario", last_name="Rossi")
    raw_input = (
        "Sono Mario Rossi, abito in Via Garibaldi, 42, il mio CF è RSSMRA60E20H501Z, "
        "tessera sanitaria 80380001200345678901, IBAN IT60X0542811101000000123456, "
        "cell 333 9876543, mail mario@test.it. Ho prenotato la visita oculistica "
        "con il Dott. Roberto Galli per il 30 novembre."
    )
    scrubbed, updated_map = mask_user_text_with_dynamic_tokens(raw_input, initial_map)

    assert "Mario" not in scrubbed
    assert "Rossi" not in scrubbed
    assert "Via Garibaldi, 42" not in scrubbed
    assert "RSSMRA60E20H501Z" not in scrubbed
    assert "80380001200345678901" not in scrubbed
    assert "IT60X0542811101000000123456" not in scrubbed
    assert "333 9876543" not in scrubbed
    assert "mario@test.it" not in scrubbed
    assert "Dott. Roberto Galli" not in scrubbed

    assert "{{NOME}}" in scrubbed
    assert "{{MEDICO}}" in scrubbed
    assert updated_map.primary_doctor == "Dott. Roberto Galli"


def _run_coro_sync(coro):
    """Esegue una coroutine puramente in-memory senza aprire socket di loopback su Windows sandbox."""
    try:
        while True:
            coro.send(None)
    except StopIteration as exc:
        return exc.value


def test_extract_glycemic_lipid_meal_activity_and_anti_hallucination():
    fixed_now = datetime(2026, 10, 28, 13, 30, tzinfo=ROME_TZ)
    batch = LLMExtractionBatch(
        items=[
            # 1. Glicemia reale presente nel testo -> OK
            RawExtractedItem(
                target_table="diabetes_data",
                metric_type="GLICEMIA",
                metric_value="142",
                timestamp_iso="2026-10-28T08:00:00+01:00",
            ),
            # 2. Colesterolo LDL reale presente nel testo -> OK
            RawExtractedItem(
                target_table="diabetes_data",
                metric_type="LDL",
                metric_value="112",
                timestamp_iso="2026-10-28T08:00:00+01:00",
            ),
            # 3. ALLUCINAZIONE LLM (HbA1c 7.4 non detta dall'utente!) -> DEVE ESSERE SCARTATA
            RawExtractedItem(
                target_table="diabetes_data",
                metric_type="HbA1c",
                metric_value="7.4",
                timestamp_iso="2026-10-28T08:00:00+01:00",
            ),
            # 4. Pasto senza grammi espliciti -> is_estimated deve diventare True
            RawExtractedItem(
                target_table="meals",
                meal_type="PRANZO",
                meal_description="Piatto di pasta al pomodoro",
                carbs_grams=65,
                calories=370,
                is_estimated=False,  # Anche se l'LLM mettesse False, il validatore vede che 65 non è nel testo utente!
                timestamp_iso="2026-10-28T13:00:00+01:00",
            ),
            # 5. Attività fisica incompleta (manca la durata in minuti)
            RawExtractedItem(
                target_table="physical_activities",
                activity_type="CAMMINATA",
                duration_minutes=None,
                intensity="MODERATA",
                timestamp_iso="2026-10-28T10:00:00+01:00",
            ),
            # 6. Assunzione farmaco con matching automatico di therapy_id
            RawExtractedItem(
                target_table="therapy_logs",
                drug_name="Metformina",
                therapy_status="ASSUNTA",
                timestamp_iso="2026-10-28T08:05:00+01:00",
            ),
        ]
    )

    llm = FakeStructuredLLM(structured_batch=batch)
    user_msg = (
        "Stamattina alle 8 avevo la glicemia a 142 e il colesterolo LDL a 112, "
        "ho preso la metformina, poi sono andato a camminare e a pranzo ho mangiato "
        "un piatto di pasta al pomodoro."
    )
    proposals = _run_coro_sync(
        extract_clinical_proposals(
            llm,
            [ChatMessage("user", user_msg)],
            active_therapies=[{"id": 42, "drug_name": "Metformina Teva", "therapy_type": "COMPRESSE_ORALI", "dosage": "1000 mg"}],
            now=fixed_now,
        )
    )

    # L'HbA1c allucinata (7.4) è stata bloccata: restano 5 proposte valide
    assert len(proposals) == 5

    glic = proposals[0]
    assert glic.target_table == "diabetes_data"
    assert glic.is_complete is True
    assert glic.data["metric_type"] == "GLICEMIA"
    assert glic.data["metric_value"] == "142"
    assert glic.data["unit"] == "mg/dL"

    ldl = proposals[1]
    assert ldl.target_table == "diabetes_data"
    assert ldl.data["metric_type"] == "LDL"
    assert ldl.data["metric_value"] == "112"

    meal = proposals[2]
    assert meal.target_table == "meals"
    assert meal.data["is_estimated"] is True
    assert meal.data["carbs_grams"] == 65

    act = proposals[3]
    assert act.target_table == "physical_activities"
    assert act.is_complete is False
    assert "duration_minutes" in act.missing_fields
    assert act.data["duration_minutes"] is None

    tlog = proposals[4]
    assert tlog.target_table == "therapy_logs"
    assert tlog.is_complete is True
    assert tlog.data["therapy_id"] == 42


def test_extract_from_report_text_and_photo_reuses_glycemic_lipid_tools_and_scrubs_pii():
    fixed_now = datetime(2026, 10, 28, 9, 0, tzinfo=ROME_TZ)
    report_batch = LLMReportExtractionBatch(
        report_type="ANALISI_SANGUE",
        report_date_iso="2026-10-25T08:30:00+01:00",
        clean_summary=(
            "Paziente Mario Rossi (CF RSSMRA60E20H501Z, tel 3331234567): "
            "profilo glicemico con HbA1c 6.7%, profilo lipidico con Colesterolo Totale 188 mg/dL, "
            "HDL 50 mg/dL, LDL 104 mg/dL, Trigliceridi 135 mg/dL, eGFR 82 mL/min/1.73m²."
        ),
        items=[
            RawExtractedItem(target_table="diabetes_data", metric_type="GLICEMIA", metric_value="116"),
            RawExtractedItem(target_table="diabetes_data", metric_type="HBA1C", metric_value="6.7"),
            RawExtractedItem(target_table="diabetes_data", metric_type="COLESTEROLO_TOTALE", metric_value="188"),
            RawExtractedItem(target_table="diabetes_data", metric_type="HDL", metric_value="50"),
            RawExtractedItem(target_table="diabetes_data", metric_type="LDL", metric_value="104"),
            RawExtractedItem(target_table="diabetes_data", metric_type="TRIGLICERIDI", metric_value="135"),
            RawExtractedItem(target_table="diabetes_data", metric_type="EGFR", metric_value="82"),
            RawExtractedItem(target_table="diabetes_data", metric_type="UACR", metric_value="18"),
        ],
    )

    llm = FakeStructuredLLM(structured_batch=report_batch)
    fake_jpeg = b"\xff\xd8\xff\xe0FAKE_REPORT_IMAGE_BYTES"
    masking_map = PatientMaskingMap(first_name="Mario", last_name="Rossi")

    result = _run_coro_sync(
        extract_from_report(
            llm,
            media_bytes=fake_jpeg,
            media_mime_type="image/jpeg",
            masking_map=masking_map,
            source_report_id=7,
            now=fixed_now,
        )
    )

    # Verifica che i byte della foto del referto siano stati passati al metodo multimodale
    assert llm.structured_calls[0]["media_bytes"] == fake_jpeg
    assert llm.structured_calls[0]["media_mime_type"] == "image/jpeg"

    # Verifica che la sintesi abbia rimosso CF, telefono e cognome
    assert "RSSMRA60E20H501Z" not in result.clean_summary
    assert "3331234567" not in result.clean_summary
    assert "Rossi" not in result.clean_summary
    assert result.report_type == "ANALISI_SANGUE"

    # Verifica che tutti gli 8 valori (profilo glicemico, lipidico/colesterolo e renale) siano pronti per il FE
    extracted_metrics = {
        e.data["metric_type"]: (e.data["metric_value"], e.data["source_report_id"])
        for e in result.proposed_entries
    }
    assert extracted_metrics == {
        "GLICEMIA": ("116", 7),
        "HbA1c": ("6.7", 7),
        "Colesterolo Totale": ("188", 7),
        "HDL": ("50", 7),
        "LDL": ("104", 7),
        "Trigliceridi": ("135", 7),
        "eGFR": ("82", 7),
        "UACR": ("18", 7),
    }


def test_chat_reply_with_proposals_rehydrates_doctor_and_degrades_gracefully(monkeypatch):
    import asyncio

    async def _fake_gather(*coros):
        return [_run_coro_sync(c) for c in coros]

    monkeypatch.setattr(asyncio, "gather", _fake_gather)

    batch = LLMExtractionBatch(
        items=[
            RawExtractedItem(
                target_table="appointments",
                appointment_title="Visita oculistica",
                doctor_name="{{MEDICO}}",
                timestamp_iso="2026-11-30T09:00:00+01:00",
                has_explicit_time=False,
            )
        ]
    )
    llm = FakeStructuredLLM(
        chat_reply="Ho preparato la scheda per la visita con {{MEDICO}}. A che ora hai l'appuntamento?",
        structured_batch=batch,
    )

    reply_text, proposals = _run_coro_sync(
        chat_service.reply_with_proposals(
            llm,
            [
                ChatMessageIn(
                    role="user",
                    content="Ho prenotato la visita oculistica con il Dott. Roberto Galli per il 30 novembre (mio cell 333 1122334).",
                )
            ],
        )
    )

    # 1. Nessun dato sensibile (né telefono né nome reale del medico) è stato inviato a Vertex AI
    sent_chat_msg = llm.chat_calls[0][0][-1].content
    sent_ext_msg = llm.structured_calls[0]["messages"][-1].content
    assert "333 1122334" not in sent_chat_msg
    assert "Dott. Roberto Galli" not in sent_chat_msg
    assert "333 1122334" not in sent_ext_msg
    assert "Dott. Roberto Galli" not in sent_ext_msg

    # 2. Sia la risposta chat sia la Card dell'appuntamento sono state re-idratate per il FE Flutter
    assert "Dott. Roberto Galli" in reply_text
    assert len(proposals) == 1
    assert proposals[0].target_table == "appointments"
    assert proposals[0].data["doctor_name"] == "Dott. Roberto Galli"
    assert proposals[0].is_complete is False
    assert "appointment_time" in proposals[0].missing_fields

    # 3. Degradazione graziosa: se l'estrazione strutturata fallisce, la risposta chat arriva comunque
    failing_llm = FakeStructuredLLM(
        chat_reply="Risposta comunque consegnata.",
        structured_exc=RuntimeError("Errore simulato nell'estrattore"),
    )
    reply_ok, empty_props = _run_coro_sync(
        chat_service.reply_with_proposals(
            failing_llm,
            [ChatMessageIn(role="user", content="Ho misurato la glicemia ed è 125")],
        )
    )
    assert reply_ok == "Risposta comunque consegnata."
    assert empty_props == []
