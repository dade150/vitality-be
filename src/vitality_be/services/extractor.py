"""Servizio di Estrazione Strutturata Dati Clinici (Human-in-the-Loop, senza LangChain).

Supporta due modalità con lo stesso motore di validazione Pydantic v2:
1. `extract_clinical_proposals(...)`: analizza i messaggi dell'utente in `/chat` (testo o
   trascrizione vocale) in parallelo alla generazione della risposta e produce una lista
   di `ProposedEntry` per il widget di conferma/modifica sul Front-End Flutter.
2. `extract_from_report(...)`: analizza un referto clinico (da testo o da foto/PDF tramite
   input multimodale Vertex AI) ed estrae sia i valori del profilo glicemico (`GLICEMIA`,
   `HbA1c`), lipidico (`Colesterolo Totale`, `HDL`, `LDL`, `Trigliceridi`), renale (`eGFR`,
   `UACR`) e pressorio (`PRESSIONE`) come `ProposedEntry`, sia una sintesi clinica de-identificata
   per `public.patient_reports`.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone, tzinfo
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from vitality_be.llm.base import ChatMessage, LLMProvider
from vitality_be.llm.prompts import get_extraction_system_prompt
from vitality_be.schemas.extraction import (
    DEFAULT_METRIC_UNITS,
    AppointmentProposal,
    DiabetesDataProposal,
    LLMExtractionBatch,
    LLMReportExtractionBatch,
    MealProposal,
    PhysicalActivityProposal,
    ProfileUpdateProposal,
    ProposedEntry,
    RawExtractedItem,
    ReportExtractionResult,
    TherapyLogProposal,
    TherapyPlanProposal,
    canonicalize_metric_type,
)
from vitality_be.services.patient_context import (
    PatientMaskingMap,
    mask_user_text_with_dynamic_tokens,
    rehydrate_assistant_reply,
    scrub_free_text,
)


def get_rome_tz() -> tzinfo:
    """Restituisce il fuso `Europe/Rome` (con fallback CET UTC+1 su Windows se `tzdata` non è installato)."""
    try:
        return ZoneInfo("Europe/Rome")
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=1), name="CET")


ROME_TZ = get_rome_tz()

_ITALIAN_WEEKDAYS = (
    "Lunedì",
    "Martedì",
    "Mercoledì",
    "Giovedì",
    "Venerdì",
    "Sabato",
    "Domenica",
)

# Messaggi puramente di cortesia che non contengono mai dati da salvare
_TRIVIAL_GREETING_RE = re.compile(
    r"^\s*(?:ciao|buongiorno|buonasera|salve|hey|ehi|grazie(?:\s+mille)?|ok|va\s+bene|perfetto|capito|a\s+presto)[!.,?\s]*$",
    re.IGNORECASE,
)

# Domande ipotetiche pure ("cosa succede se...", "qual è il valore normale...")
_HYPOTHETICAL_QUESTION_RE = re.compile(
    r"^\s*(?:cosa\s+succede\s+se|che\s+succede\s+se|qual\s+[eè]\s+il\s+valore\s+normale|"
    r"quali\s+sono\s+i\s+valori\s+normali|quanti\s+carboidrati\s+ha\b|quante\s+calorie\s+ha\b|"
    r"mi\s+spieghi\s+cos['’]?[eè])\b",
    re.IGNORECASE,
)

_METRIC_TITLES: dict[str, str] = {
    "GLICEMIA": "Glicemia",
    "PRESSIONE": "Pressione arteriosa",
    "HbA1c": "Emoglobina glicata (HbA1c)",
    "eGFR": "Filtrato glomerulare (eGFR)",
    "UACR": "Albuminuria (UACR)",
    "Colesterolo Totale": "Colesterolo Totale",
    "HDL": "Colesterolo HDL",
    "LDL": "Colesterolo LDL",
    "Trigliceridi": "Trigliceridi",
}


def should_attempt_extraction(messages: Sequence[ChatMessage]) -> bool:
    """Ritorna `False` per saluti banali o domande teoriche/ipotetiche pure, risparmiando chiamate LLM."""
    if not messages:
        return False
    last_text = messages[-1].content.strip()
    if not last_text or len(last_text) < 2:
        return False
    if _TRIVIAL_GREETING_RE.match(last_text):
        return False
    if _HYPOTHETICAL_QUESTION_RE.match(last_text) and "?" in last_text:
        return False
    return True


def _format_rome_now(now: datetime | None = None) -> tuple[str, str]:
    ref = (now or datetime.now(ROME_TZ)).astimezone(ROME_TZ)
    iso_str = ref.isoformat(timespec="seconds")
    weekday = _ITALIAN_WEEKDAYS[ref.weekday()]
    human_str = f"{weekday} {ref.strftime('%d/%m/%Y ore %H:%M')} (Europe/Rome, ISO: {iso_str})"
    return iso_str, human_str


def _normalize_iso_timestamp(raw_ts: str | None, *, fallback_iso: str) -> str:
    if not raw_ts or not str(raw_ts).strip():
        return fallback_iso
    cleaned = str(raw_ts).strip()
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ROME_TZ)
        return dt.astimezone(ROME_TZ).isoformat(timespec="seconds")
    except ValueError:
        return fallback_iso


def _number_appears_in_source_text(metric_value: str, source_text: str) -> bool:
    """Verifica anti-allucinazione: le cifre del parametro vitale estratto devono comparire
    realmente nel testo dell'utente (es. '130/80' -> '130' e '80', '6.8' -> '6.8' o '6,8')."""
    if not source_text:
        return False
    norm_source = source_text.replace(",", ".")
    parts = re.findall(r"\d+(?:\.\d+)?", metric_value.replace(",", "."))
    if not parts:
        return False
    for p in parts:
        pattern = rf"(?<!\d){re.escape(p)}(?!\d)"
        if not re.search(pattern, norm_source):
            return False
    return True


def _match_active_therapy_id(
    drug_name: str,
    active_therapies: Sequence[dict[str, Any]],
) -> int | None:
    """Cerca una corrispondenza tra il farmaco nominato dall'utente e le terapie attive a DB."""
    norm_query = drug_name.strip().lower()
    if not norm_query or not active_therapies:
        return None
    for t in active_therapies:
        t_id = t.get("id")
        t_drug = str(t.get("drug_name") or "").strip().lower()
        if t_id is not None and t_drug and (norm_query in t_drug or t_drug in norm_query):
            try:
                return int(t_id)
            except (TypeError, ValueError):
                continue
    return None


def _clean_rehydrated_text(
    text: str | None,
    masking_map: PatientMaskingMap | None,
    *,
    allow_doctor_name: bool = False,
) -> str | None:
    """Re-idrata i token reversibili (`{{NOME}}`, `{{MEDICO}}`) ma rimuove sempre eventuali PII
    sensibili (CF, email, telefono, IBAN, tessera sanitaria, indirizzi)."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    # Prima puliamo i pattern sensibili (senza toccare il medico se stiamo re-idratando doctor_name)
    scrubbed = scrub_free_text(raw, None if allow_doctor_name else masking_map)
    rehydrated = rehydrate_assistant_reply(scrubbed, masking_map).strip()
    return rehydrated or None


def _build_proposed_entry(
    item: RawExtractedItem,
    *,
    index: int,
    fallback_iso: str,
    source_user_text: str | None,
    active_therapies: Sequence[dict[str, Any]],
    masking_map: PatientMaskingMap | None,
    source_report_id: int | None = None,
) -> ProposedEntry | None:
    """Valida e converte un `RawExtractedItem` in un `ProposedEntry` pronto per il FE Flutter.

    Restituisce `None` se l'elemento viola i guardrail clinici (es. valore vitale allucinato o fuori range).
    """
    table = item.target_table
    missing = [f.strip() for f in (item.missing_fields or []) if f and f.strip()]
    ts_iso = _normalize_iso_timestamp(item.timestamp_iso, fallback_iso=fallback_iso)
    proposal_id = f"prop_{index}"

    try:
        if table == "diabetes_data":
            if not item.metric_type or not item.metric_value:
                return None
            canon_metric = canonicalize_metric_type(item.metric_type)
            payload = DiabetesDataProposal(
                metric_type=canon_metric,
                metric_value=item.metric_value,
                measured_at=ts_iso,
                source_report_id=source_report_id,
            )
            # Guardrail anti-allucinazione numerica per la chat/voce
            if source_user_text is not None and not _number_appears_in_source_text(
                payload.metric_value, source_user_text
            ):
                logger.warning(
                    "extractor scartato valore diabetes_data non presente nel testo metric={}",
                    payload.metric_type,
                )
                return None

            label = _METRIC_TITLES.get(payload.metric_type, payload.metric_type)
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map)
                or f"{label}: {payload.metric_value} {payload.unit}".strip()
            )
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="diabetes_data",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=_clean_rehydrated_text(item.assistant_note, masking_map),
                data=payload.model_dump(),
            )

        if table == "meals":
            desc = _clean_rehydrated_text(item.meal_description or item.notes, masking_map)
            if not desc:
                return None
            meal_type_raw = item.meal_type or "SPUNTINO"
            is_est = True if item.is_estimated is None else bool(item.is_estimated)
            # Se i grammi di carboidrati non erano scritti dall'utente, forziamo is_estimated=True
            if (
                source_user_text is not None
                and item.carbs_grams is not None
                and not _number_appears_in_source_text(str(item.carbs_grams), source_user_text)
            ):
                is_est = True

            payload = MealProposal(
                meal_type=meal_type_raw,
                description=desc,
                carbs_grams=item.carbs_grams,
                calories=item.calories,
                is_estimated=is_est,
                meal_at=ts_iso,
            )
            note = _clean_rehydrated_text(item.assistant_note, masking_map)
            if payload.is_estimated and not note and payload.carbs_grams is not None:
                note = (
                    f"Valori nutrizionali stimati su porzione standard LARN "
                    f"(~{payload.carbs_grams} g carboidrati)."
                )
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map)
                or f"{payload.meal_type.capitalize()}: {payload.description[:50]}"
            )
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="meals",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=note,
                data=payload.model_dump(),
            )

        if table == "physical_activities":
            act_type_raw = item.activity_type or "CAMMINATA"
            dur = item.duration_minutes
            if dur is None and "duration_minutes" not in missing:
                missing.append("duration_minutes")
            payload = PhysicalActivityProposal(
                activity_type=act_type_raw,
                duration_minutes=dur,
                intensity=item.intensity or "MODERATA",
                steps=item.steps,
                notes=_clean_rehydrated_text(item.notes, masking_map),
                activity_at=ts_iso,
            )
            dur_str = f" ({dur} min)" if dur is not None else " (durata da confermare)"
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map)
                or f"Attività: {payload.activity_type.capitalize()}{dur_str}"
            )
            note = _clean_rehydrated_text(item.assistant_note, masking_map)
            if dur is None and not note:
                note = "Inserisci la durata in minuti nel riquadro prima di confermare."
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="physical_activities",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=note,
                data=payload.model_dump(),
            )

        if table == "appointments":
            appt_title = _clean_rehydrated_text(
                item.appointment_title or item.summary_title, masking_map
            )
            if not appt_title:
                return None
            doc_name = _clean_rehydrated_text(
                item.doctor_name, masking_map, allow_doctor_name=True
            )
            appt_at = (
                _normalize_iso_timestamp(item.timestamp_iso, fallback_iso=fallback_iso)
                if item.timestamp_iso
                else None
            )
            if appt_at is None and "appointment_at" not in missing:
                missing.append("appointment_at")
            elif item.has_explicit_time is False and "appointment_time" not in missing:
                missing.append("appointment_time")

            payload = AppointmentProposal(
                title=appt_title,
                doctor_name=doc_name,
                appointment_at=appt_at,
                notes=_clean_rehydrated_text(item.notes, masking_map),
            )
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map, allow_doctor_name=True)
                or f"Appuntamento: {appt_title}"
            )
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="appointments",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=_clean_rehydrated_text(item.assistant_note, masking_map),
                data=payload.model_dump(),
            )

        if table == "therapy_logs":
            drug = _clean_rehydrated_text(item.drug_name, masking_map)
            if not drug:
                return None
            resolved_id = item.therapy_id or _match_active_therapy_id(drug, active_therapies)
            if resolved_id is None and "therapy_id" not in missing:
                missing.append("therapy_id")
            status_raw = (item.therapy_status or "ASSUNTA").strip().upper()
            if status_raw not in ("ASSUNTA", "SALTATA", "RITARDATA"):
                status_raw = "ASSUNTA"

            payload = TherapyLogProposal(
                therapy_id=resolved_id,
                drug_name_mentioned=drug,
                taken_at=ts_iso,
                status=status_raw,  # type: ignore[arg-type]
            )
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map)
                or f"Terapia ({status_raw.lower()}): {drug}"
            )
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="therapy_logs",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=_clean_rehydrated_text(item.assistant_note, masking_map),
                data=payload.model_dump(),
            )

        if table == "therapies":
            drug = _clean_rehydrated_text(item.drug_name, masking_map)
            if not drug:
                return None
            dosage = _clean_rehydrated_text(item.dosage, masking_map)
            freq = _clean_rehydrated_text(item.frequency, masking_map)
            if not dosage and "dosage" not in missing:
                missing.append("dosage")
            if not freq and "frequency" not in missing:
                missing.append("frequency")

            payload = TherapyPlanProposal(
                therapy_type=item.therapy_type or "ALTRO",
                drug_name=drug,
                dosage=dosage,
                frequency=freq,
                scheduled_time=item.scheduled_time,
                notes=_clean_rehydrated_text(item.notes, masking_map),
            )
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map)
                or f"Piano terapeutico: {drug}"
            )
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="therapies",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=_clean_rehydrated_text(item.assistant_note, masking_map),
                data=payload.model_dump(),
            )

        if table == "profile":
            if (
                source_user_text is not None
                and item.peso_kg is not None
                and not _number_appears_in_source_text(str(item.peso_kg), source_user_text)
            ):
                return None
            payload = ProfileUpdateProposal(
                peso=item.peso_kg,
                altezza=item.altezza_cm,
            )
            parts = []
            if payload.peso is not None:
                parts.append(f"Peso {payload.peso} kg")
            if payload.altezza is not None:
                parts.append(f"Altezza {payload.altezza} cm")
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map)
                or "Aggiornamento profilo: " + ", ".join(parts)
            )
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="profile",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=_clean_rehydrated_text(item.assistant_note, masking_map),
                data=payload.model_dump(),
            )

    except Exception as exc:
        logger.warning("extractor item scartato table={} exc={}", table, type(exc).__name__)
        return None

    return None


async def extract_clinical_proposals(
    llm: LLMProvider,
    scrubbed_messages: Sequence[ChatMessage],
    *,
    masking_map: PatientMaskingMap | None = None,
    active_therapies: Sequence[dict[str, Any]] = (),
    now: datetime | None = None,
) -> list[ProposedEntry]:
    """Estrae in modo asincrono e sicuro le proposte di salvataggio dai messaggi della chat.

    Garantisce **degradazione graziosa totale**: qualsiasi errore del provider LLM o di
    validazione restituisce `[]` senza mai interrompere la risposta testuale della chat.
    """
    if not hasattr(llm, "generate_structured"):
        return []
    if not should_attempt_extraction(scrubbed_messages):
        return []

    iso_now, human_now = _format_rome_now(now)
    recent_turns = list(scrubbed_messages[-4:])
    user_turns_text = "\n".join(m.content for m in recent_turns if m.role == "user")

    therapies_block = "Nessuna terapia attiva registrata."
    if active_therapies:
        t_lines = [
            f"- id={t.get('id')}: {t.get('drug_name')} ({t.get('therapy_type')}, dosaggio: {t.get('dosage') or 'n.d.'})"
            for t in active_therapies
        ]
        therapies_block = "\n".join(t_lines)

    conv_lines = [
        f"[{'UTENTE' if m.role == 'user' else 'ASSISTENTE'}]: {m.content}"
        for m in recent_turns
    ]
    extraction_prompt = (
        f"<timestamp_corrente>\n{human_now}\n</timestamp_corrente>\n\n"
        f"<terapie_attive_paziente>\n{therapies_block}\n</terapie_attive_paziente>\n\n"
        f"<conversazione_recente>\n" + "\n".join(conv_lines) + "\n</conversazione_recente>"
    )

    try:
        batch: LLMExtractionBatch = await llm.generate_structured(
            [ChatMessage(role="user", content=extraction_prompt)],
            system=get_extraction_system_prompt(),
            response_schema=LLMExtractionBatch,
        )
    except Exception as exc:
        logger.warning("extractor chat fallito (degradazione graziosa) exc={}", type(exc).__name__)
        return []

    proposals: list[ProposedEntry] = []
    for idx, raw_item in enumerate(batch.items, start=1):
        entry = _build_proposed_entry(
            raw_item,
            index=idx,
            fallback_iso=iso_now,
            source_user_text=user_turns_text,
            active_therapies=active_therapies,
            masking_map=masking_map,
        )
        if entry is not None:
            proposals.append(entry)

    return proposals


async def extract_from_report(
    llm: LLMProvider,
    *,
    report_text: str | None = None,
    media_bytes: bytes | None = None,
    media_mime_type: str | None = None,
    masking_map: PatientMaskingMap | None = None,
    active_therapies: Sequence[dict[str, Any]] = (),
    source_report_id: int | None = None,
    now: datetime | None = None,
) -> ReportExtractionResult:
    """Estrae i valori di profilo glicemico (`GLICEMIA`, `HbA1c`), lipidico/colesterolo
    (`Colesterolo Totale`, `HDL`, `LDL`, `Trigliceridi`), renale (`eGFR`, `UACR`) e pressorio
    (`PRESSIONE`) da un referto medico testuale oppure da una foto/PDF (`media_bytes`).

    Applica sempre lo scrubbing dei dati sensibili prima dell'invio (per il testo) e sulla
    sintesi `clean_summary` prodotta in uscita.
    """
    if not hasattr(llm, "generate_structured"):
        return ReportExtractionResult()
    if not report_text and not media_bytes:
        return ReportExtractionResult()

    iso_now, human_now = _format_rome_now(now)
    scrubbed_text, updated_map = mask_user_text_with_dynamic_tokens(report_text, masking_map)

    prompt_parts = [
        f"<timestamp_corrente>\n{human_now}\n</timestamp_corrente>",
        "Estrai dal referto medico tutti i parametri clinici ed ematochimici "
        "(GLICEMIA, HbA1c, Colesterolo Totale, HDL, LDL, Trigliceridi, eGFR, UACR, PRESSIONE) "
        "e genera una sintesi clinica oggettiva completamente priva di dati identificativi.",
    ]
    if scrubbed_text:
        prompt_parts.append(f"<testo_referto>\n{scrubbed_text}\n</testo_referto>")

    try:
        batch: LLMReportExtractionBatch = await llm.generate_structured(
            [ChatMessage(role="user", content="\n\n".join(prompt_parts))],
            system=get_extraction_system_prompt(),
            response_schema=LLMReportExtractionBatch,
            media_bytes=media_bytes,
            media_mime_type=media_mime_type,
        )
    except Exception as exc:
        logger.warning("extractor report fallito exc={}", type(exc).__name__)
        return ReportExtractionResult()

    report_iso = (
        _normalize_iso_timestamp(batch.report_date_iso, fallback_iso=iso_now)
        if batch.report_date_iso
        else iso_now
    )

    proposals: list[ProposedEntry] = []
    for idx, raw_item in enumerate(batch.items, start=1):
        if not raw_item.timestamp_iso:
            raw_item.timestamp_iso = report_iso
        entry = _build_proposed_entry(
            raw_item,
            index=idx,
            fallback_iso=report_iso,
            source_user_text=scrubbed_text if scrubbed_text else None,
            active_therapies=active_therapies,
            masking_map=updated_map,
            source_report_id=source_report_id,
        )
        if entry is not None:
            proposals.append(entry)

    clean_summary = scrub_free_text(batch.clean_summary, updated_map)
    clean_summary = rehydrate_assistant_reply(clean_summary, updated_map)

    try:
        return ReportExtractionResult(
            report_type=batch.report_type or "ANALISI_SANGUE",
            report_at=report_iso,
            clean_summary=clean_summary,
            proposed_entries=proposals,
        )
    except Exception:
        return ReportExtractionResult(
            report_type="ALTRO",
            report_at=report_iso,
            clean_summary=clean_summary,
            proposed_entries=proposals,
        )
