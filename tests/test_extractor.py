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
            "Referto clinico di Mario Rossi (CF RSSMRA60E20H501Z, tel 3331234567): "
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

    # Verifica che la sintesi abbia rimosso CF, telefono e cognome (A3: clean_summary de-identificata, display re-idratata)
    assert "RSSMRA60E20H501Z" not in result.clean_summary
    assert "3331234567" not in result.clean_summary
    assert "Rossi" not in result.clean_summary
    assert "Mario" not in result.clean_summary
    assert "Mario" in result.clean_summary_display
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


def test_pii_scrubbing_advances_birthdate_relatives_and_hospital_filter():
    from vitality_be.services.patient_context import scrub_free_text, scrub_report_text

    # A6: Non deve censurare termini clinici legittimi
    clinical_text = "Valutazione clinica generale buona. Laboratorio analisi del sangue di routine. Risposta clinica positiva."
    scrubbed_clin = scrub_free_text(clinical_text)
    assert "[STRUTTURA_SANITARIA_RIMOSSA]" not in scrubbed_clin
    assert "clinica generale" in scrubbed_clin
    assert "analisi del sangue" in scrubbed_clin

    # A6: Ma deve censurare i veri ospedali/cliniche
    hosp_text = "Ricoverato presso Ospedale San Raffaele e poi visitato alla Clinica Columbus."
    scrubbed_hosp = scrub_free_text(hosp_text)
    assert "San Raffaele" not in scrubbed_hosp
    assert "Columbus" not in scrubbed_hosp
    assert "[STRUTTURA_SANITARIA_RIMOSSA]" in scrubbed_hosp

    # A5: Date di nascita senza due punti o nel corpo del testo
    birth_text = "Paziente nato il 01/01/1960 a Roma. Nata a Milano il 12/03/1958. Data di nascita 15/05/1975."
    scrubbed_birth = scrub_free_text(birth_text)
    assert "01/01/1960" not in scrubbed_birth
    assert "12/03/1958" not in scrubbed_birth
    assert "15/05/1975" not in scrubbed_birth
    assert "[DATA_NASCITA_RIMOSSA]" in scrubbed_birth

    # A5: Nomi di familiari e terzi rimossi preservando la relazione clinica
    rel_text = "Accompagnato dalla moglie Maria Rossi. Il figlio Luca riferisce ipoglicemie. Madre diabetica."
    scrubbed_rel = scrub_free_text(rel_text)
    assert "Maria Rossi" not in scrubbed_rel
    assert "Luca" not in scrubbed_rel
    assert "moglie [FAMILIARE_RIMOSSO]" in scrubbed_rel
    assert "figlio [FAMILIARE_RIMOSSO]" in scrubbed_rel
    assert "Madre diabetica" in scrubbed_rel

    # A5: Intestazioni tabellari tipiche dei referti
    tab_report = "ROSSI MARIO  M  01/01/1960  RSSMRA60A01H501Z\nGLICEMIA BASE: 110 mg/dL"
    scrubbed_tab, _ = scrub_report_text(tab_report)
    assert "ROSSI MARIO" not in scrubbed_tab
    assert "RSSMRA60A01H501Z" not in scrubbed_tab
    assert "01/01/1960" not in scrubbed_tab
    assert "GLICEMIA BASE: 110 mg/dL" in scrubbed_tab


def test_extract_clinical_proposals_multimodal_meal_and_glucometer_photos():
    fixed_now = datetime(2026, 11, 20, 16, 0, tzinfo=ROME_TZ)
    fake_photo = b"\xff\xd8\xff\xe0" + b"\x00" * 64

    # 1. Foto pasto (yogurt) anche con didascalia breve/neutra
    meal_batch = LLMExtractionBatch(
        items=[
            RawExtractedItem(
                target_table="meals",
                meal_type="SPUNTINO_POMERIGGIO",
                meal_description="Yogurt bianco intero 125 g",
                carbs_grams=6,
                calories=82,
                is_estimated=False,
                timestamp_iso="2026-11-20T16:00:00+01:00",
            )
        ]
    )
    meal_llm = FakeStructuredLLM(structured_batch=meal_batch)
    meal_props = _run_coro_sync(
        extract_clinical_proposals(
            meal_llm,
            [ChatMessage("user", "Ciao")],
            media_bytes=fake_photo,
            media_mime_type="image/jpeg",
            attachment_kind="meal_photo",
            now=fixed_now,
        )
    )
    assert len(meal_llm.structured_calls) == 1
    assert meal_llm.structured_calls[0]["media_bytes"] == fake_photo
    assert meal_llm.structured_calls[0]["media_mime_type"] == "image/jpeg"
    assert '<allegato_multimodale tipo="meal_photo">' in meal_llm.structured_calls[0]["messages"][0].content
    assert len(meal_props) == 1
    assert meal_props[0].target_table == "meals"
    assert meal_props[0].data["description"] == "Yogurt bianco intero 125 g"
    assert meal_props[0].data["carbs_grams"] == 6

    # 2. Foto glucometro senza numero scritto nel testo -> NON deve essere scartato dal guardrail testuale
    gluc_batch = LLMExtractionBatch(
        items=[
            RawExtractedItem(
                target_table="diabetes_data",
                metric_type="GLICEMIA",
                metric_value="128",
                timestamp_iso="2026-11-20T16:00:00+01:00",
            )
        ]
    )
    gluc_llm = FakeStructuredLLM(structured_batch=gluc_batch)
    gluc_props = _run_coro_sync(
        extract_clinical_proposals(
            gluc_llm,
            [ChatMessage("user", "Ecco la lettura del mio glucometro.")],
            media_bytes=fake_photo,
            media_mime_type="image/jpeg",
            attachment_kind="glucometer_photo",
            now=fixed_now,
        )
    )
    assert len(gluc_props) == 1
    assert gluc_props[0].target_table == "diabetes_data"
    assert gluc_props[0].data["metric_type"] == "GLICEMIA"
    assert gluc_props[0].data["metric_value"] == "128"


def test_chat_pipeline_meal_photo_passes_media_to_extractor(monkeypatch):
    fake_photo = b"\xff\xd8\xff\xe0" + b"\x00" * 64
    meal_batch = LLMExtractionBatch(
        items=[
            RawExtractedItem(
                target_table="meals",
                meal_type="SPUNTINO_POMERIGGIO",
                meal_description="Yogurt greco alla fragola 150 g",
                carbs_grams=14,
                calories=130,
                is_estimated=True,
            )
        ]
    )

    class MultimodalFakeLLM(FakeStructuredLLM):
        async def chat(
            self,
            messages: list[ChatMessage],
            *,
            system: str | None = None,
            media_bytes: bytes | None = None,
            media_mime_type: str | None = None,
        ) -> str:
            self.chat_calls.append((messages, system))
            return "Ottimo spuntino! Puoi confermare lo yogurt nella scheda qui sotto."

    async def _seq_gather(*coros):
        return [await c for c in coros]

    monkeypatch.setattr("vitality_be.services.chat.asyncio.gather", _seq_gather)
    llm = MultimodalFakeLLM(structured_batch=meal_batch)

    res = _run_coro_sync(
        chat_service.run_chat_pipeline(
            llm,
            user_message="Ecco la foto del mio pasto. Puoi analizzarlo dal punto di vista nutrizionale?",
            media_bytes=fake_photo,
            media_mime_type="image/jpeg",
            attachment_kind="meal_photo",
        )
    )
    assert len(res.proposed_entries) == 1
    assert res.proposed_entries[0].target_table == "meals"
    assert "Yogurt" in res.proposed_entries[0].data["description"]
    assert any(c["media_bytes"] == fake_photo for c in llm.structured_calls)


def test_therapy_log_generic_phrase_time_disambiguation_and_id_validation():
    fixed_now = datetime(2026, 11, 20, 8, 15, tzinfo=ROME_TZ)

    # 1. Utente dice "Ho preso il farmaco" senza nominarlo e ha 1 sola terapia attiva -> auto-match!
    generic_batch = LLMExtractionBatch(
        items=[
            RawExtractedItem(
                target_table="therapy_logs",
                drug_name=None,
                therapy_id=None,
                therapy_status="ASSUNTA",
                timestamp_iso="2026-11-20T08:15:00+01:00",
            )
        ]
    )
    single_therapies = [
        {"id": 11, "drug_name": "Metformina 500", "therapy_type": "COMPRESSE_ORALI", "dosage": "500 mg", "scheduled_time": "08:00"},
    ]
    llm1 = FakeStructuredLLM(structured_batch=generic_batch)
    props1 = _run_coro_sync(
        extract_clinical_proposals(
            llm1,
            [ChatMessage("user", "Ho preso il farmaco")],
            active_therapies=single_therapies,
            now=fixed_now,
        )
    )
    assert len(props1) == 1
    assert props1[0].target_table == "therapy_logs"
    assert props1[0].is_complete is True
    assert props1[0].data["therapy_id"] == 11
    assert props1[0].data["drug_name_mentioned"] == "Metformina 500"
    assert "orario: 08:00" in llm1.structured_calls[0]["messages"][0].content

    # 2. Utente dice "Ho preso il farmaco" con 2 terapie diverse -> mostra comunque la card con therapy_id mancante (menu a tendina)
    two_therapies = [
        {"id": 11, "drug_name": "Metformina", "therapy_type": "COMPRESSE_ORALI", "dosage": "500 mg", "scheduled_time": "08:00"},
        {"id": 12, "drug_name": "Ramipril", "therapy_type": "COMPRESSE_ORALI", "dosage": "5 mg", "scheduled_time": "08:00"},
    ]
    llm2 = FakeStructuredLLM(structured_batch=generic_batch)
    props2 = _run_coro_sync(
        extract_clinical_proposals(
            llm2,
            [ChatMessage("user", "Ho preso il farmaco")],
            active_therapies=two_therapies,
            now=fixed_now,
        )
    )
    assert len(props2) == 1
    assert props2[0].target_table == "therapy_logs"
    assert props2[0].is_complete is False
    assert "therapy_id" in props2[0].missing_fields
    assert props2[0].data["therapy_id"] is None

    # 3. Stesso farmaco in 2 orari (08:00 e 20:00) + LLM allucina therapy_id=999 -> valida e disambigua per orario!
    same_drug_two_times = [
        {"id": 21, "drug_name": "Metformina", "therapy_type": "COMPRESSE_ORALI", "dosage": "1000 mg", "scheduled_time": "08:00"},
        {"id": 22, "drug_name": "Metformina", "therapy_type": "COMPRESSE_ORALI", "dosage": "1000 mg", "scheduled_time": "20:00"},
    ]
    evening_batch = LLMExtractionBatch(
        items=[
            RawExtractedItem(
                target_table="therapy_logs",
                drug_name="Metformina",
                therapy_id=999,  # ID inesistente allucinato!
                therapy_status="ASSUNTA",
                timestamp_iso="2026-11-20T20:10:00+01:00",
            )
        ]
    )
    llm3 = FakeStructuredLLM(structured_batch=evening_batch)
    props3 = _run_coro_sync(
        extract_clinical_proposals(
            llm3,
            [ChatMessage("user", "Stasera alle 20:10 ho preso la metformina")],
            active_therapies=same_drug_two_times,
            now=datetime(2026, 11, 20, 20, 15, tzinfo=ROME_TZ),
        )
    )
    assert len(props3) == 1
    assert props3[0].is_complete is True
    assert props3[0].data["therapy_id"] == 22

    # Alle 16:00, tra 08:00 (finestra aperta) e 20:00 (finestra chiusa fino alle 19:30), sceglie 08:00 (id=21)
    afternoon_batch = LLMExtractionBatch(
        items=[
            RawExtractedItem(
                target_table="therapy_logs",
                drug_name="Metformina",
                therapy_id=None,
                therapy_status="ASSUNTA",
                timestamp_iso="2026-11-20T16:00:00+01:00",
            )
        ]
    )
    llm4 = FakeStructuredLLM(structured_batch=afternoon_batch)
    props4 = _run_coro_sync(
        extract_clinical_proposals(
            llm4,
            [ChatMessage("user", "Ho preso la metformina")],
            active_therapies=same_drug_two_times,
            now=datetime(2026, 11, 20, 16, 0, tzinfo=ROME_TZ),
        )
    )
    assert len(props4) == 1
    assert props4[0].data["therapy_id"] == 21


