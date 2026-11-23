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

from vitality_be.llm.base import ChatMessage, LLMError, LLMProvider, LLMUnavailable
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
from vitality_be.services.multimodal import inspect_pdf_in_ram
from vitality_be.services.patient_context import (
    PatientMaskingMap,
    mask_user_text_with_dynamic_tokens,
    rehydrate_assistant_reply,
    sanitize_xml_tag_content,
    scrub_free_text,
    scrub_report_text,
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
    realmente nel testo dell'utente (es. '130/80' -> '130' e '80', '6.8' -> '6.8', '6.80' o '6,8')."""
    if not source_text:
        return False
    norm_source = source_text.replace(",", ".")
    parts = re.findall(r"\d+(?:\.\d+)?", metric_value.replace(",", "."))
    if not parts:
        return False
    for p in parts:
        if "." in p:
            pattern = rf"(?<!\d){re.escape(p)}0*(?!\d)"
        else:
            pattern = rf"(?<!\d){re.escape(p)}(?:\.0+)?(?!\d)"
        if not re.search(pattern, norm_source):
            return False
    return True


_GENERIC_DRUG_TERMS = frozenset(
    {
        "farmaco",
        "il farmaco",
        "medicina",
        "la medicina",
        "medicinale",
        "il medicinale",
        "pillola",
        "la pillola",
        "pastiglia",
        "la pastiglia",
        "compressa",
        "la compressa",
        "terapia",
        "la terapia",
        "dose",
        "la dose",
        "farmaco da selezionare",
        "terapia da selezionare",
    }
)


def _parse_hhmm_minutes(raw_time: Any) -> int | None:
    if not raw_time:
        return None
    parts = str(raw_time).strip().split(":")
    if len(parts) < 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if 0 <= h <= 23 and 0 <= m <= 59:
        return h * 60 + m
    return None


def _iso_rome_minutes(ts_iso: str) -> int | None:
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ROME_TZ)
        local = dt.astimezone(ROME_TZ)
        return local.hour * 60 + local.minute
    except ValueError:
        return None


def _pick_best_therapy_by_time(
    candidates: Sequence[dict[str, Any]],
    ts_iso: str,
) -> dict[str, Any]:
    """Disambigua più dosi dello stesso farmaco (es. Metformina 08:00 e 20:00) usando `scheduled_time`."""
    if len(candidates) == 1:
        return candidates[0]
    target_min = _iso_rome_minutes(ts_iso)
    if target_min is None:
        return candidates[0]

    def _score(t: dict[str, Any]) -> tuple[int, int]:
        sched = _parse_hhmm_minutes(t.get("scheduled_time"))
        if sched is None:
            return (2, 10_000)
        # Preferisce una dose la cui finestra di -30 min è già aperta rispetto all'ora dichiarata
        window_open = max(0, sched - 30)
        is_future_locked = 1 if target_min < window_open else 0
        return (is_future_locked, abs(target_min - sched))

    return min(candidates, key=_score)


def _match_active_therapy_id(
    drug_name: str,
    active_therapies: Sequence[dict[str, Any]],
    *,
    ts_iso: str | None = None,
) -> int | None:
    """Cerca una corrispondenza tra il farmaco nominato dall'utente e le terapie attive a DB."""
    norm_query = drug_name.strip().lower()
    if not norm_query or norm_query in _GENERIC_DRUG_TERMS or not active_therapies:
        return None
    matches: list[dict[str, Any]] = []
    for t in active_therapies:
        t_id = t.get("id")
        t_drug = str(t.get("drug_name") or "").strip().lower()
        if t_id is not None and t_drug and (norm_query in t_drug or t_drug in norm_query):
            try:
                int(t_id)
                matches.append(t)
            except (TypeError, ValueError):
                continue
    if not matches:
        return None
    best = _pick_best_therapy_by_time(matches, ts_iso) if ts_iso else matches[0]
    return int(best["id"])


def _resolve_active_therapy(
    *,
    item_therapy_id: int | None,
    drug_name: str | None,
    active_therapies: Sequence[dict[str, Any]],
    ts_iso: str,
) -> tuple[int | None, str | None]:
    """Risolve e valida `(therapy_id, drug_name_canonico)` contro le terapie attive reali dell'utente."""
    valid_by_id: dict[int, dict[str, Any]] = {}
    for t in active_therapies:
        t_id = t.get("id")
        if t_id is None:
            continue
        try:
            valid_by_id[int(t_id)] = t
        except (TypeError, ValueError):
            continue

    clean_drug = (drug_name or "").strip()
    is_generic = (not clean_drug) or (clean_drug.lower() in _GENERIC_DRUG_TERMS)

    # 1. Valida sempre l'ID proposto dall'LLM contro gli ID reali in DB (anti-allucinazione FK)
    if item_therapy_id is not None:
        try:
            cand_id = int(item_therapy_id)
        except (TypeError, ValueError):
            cand_id = -1
        if cand_id in valid_by_id:
            matched_drug = str(valid_by_id[cand_id].get("drug_name") or "").strip() or clean_drug
            return cand_id, matched_drug

    # 2. Matching per nome farmaco + disambiguazione orario (es. Metformina 08:00 vs 20:00)
    if not is_generic:
        matched_id = _match_active_therapy_id(clean_drug, active_therapies, ts_iso=ts_iso)
        if matched_id is not None and matched_id in valid_by_id:
            matched_drug = str(valid_by_id[matched_id].get("drug_name") or "").strip() or clean_drug
            return matched_id, matched_drug
        return None, clean_drug

    # 3. Frase generica ("ho preso il farmaco"): se c'è 1 sola terapia attiva la seleziona subito
    if len(valid_by_id) == 1:
        only_id, only_t = next(iter(valid_by_id.items()))
        only_drug = str(only_t.get("drug_name") or "").strip() or "Terapia"
        return only_id, only_drug

    # Più terapie attive (o nessuna) e farmaco non nominato -> lascia scegliere dal menu a tendina FE
    return None, None


def _clean_rehydrated_text(
    text: str | None,
    masking_map: PatientMaskingMap | None,
    *,
    allow_doctor_name: bool = False,
) -> str | None:
    """Re-idrata i token reversibili (`[NOME_PAZIENTE]`, `[COGNOME_PAZIENTE]`, `[NOME_MEDICO]`, `[COGNOME_MEDICO]`)
    ma rimuove sempre eventuali PII sensibili (CF, email, telefono, IBAN, tessera sanitaria, indirizzi)."""
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
                steps=item.steps if str(act_type_raw).strip().upper() == "CAMMINATA" else None,
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
            raw_drug = _clean_rehydrated_text(item.drug_name, masking_map)
            resolved_id, resolved_drug = _resolve_active_therapy(
                item_therapy_id=item.therapy_id,
                drug_name=raw_drug,
                active_therapies=active_therapies,
                ts_iso=ts_iso,
            )
            drug_label = resolved_drug or raw_drug or "Terapia da selezionare"
            if resolved_id is None:
                if "therapy_id" not in missing:
                    missing.append("therapy_id")
            else:
                missing = [f for f in missing if f != "therapy_id"]

            status_raw = (item.therapy_status or "ASSUNTA").strip().upper()
            if status_raw not in ("ASSUNTA", "SALTATA", "RITARDATA"):
                status_raw = "ASSUNTA"

            payload = TherapyLogProposal(
                therapy_id=resolved_id,
                drug_name_mentioned=drug_label,
                taken_at=ts_iso,
                status=status_raw,  # type: ignore[arg-type]
            )
            title = (
                _clean_rehydrated_text(item.summary_title, masking_map)
                or (
                    f"Terapia ({status_raw.lower()}): {drug_label}"
                    if resolved_drug or raw_drug
                    else f"Registrazione terapia ({status_raw.lower()})"
                )
            )
            note = _clean_rehydrated_text(item.assistant_note, masking_map)
            if resolved_id is None and not note:
                note = (
                    "Seleziona il farmaco dal menu qui sotto prima di confermare."
                    if active_therapies
                    else "Configura prima almeno una terapia attiva nel Diario."
                )
            return ProposedEntry(
                proposal_id=proposal_id,
                target_table="therapy_logs",
                summary_title=title,
                is_complete=len(missing) == 0,
                missing_fields=missing,
                assistant_note=note,
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
    media_bytes: bytes | None = None,
    media_mime_type: str | None = None,
    attachment_kind: str | None = None,
    now: datetime | None = None,
) -> list[ProposedEntry]:
    """Estrae in modo asincrono e sicuro le proposte di salvataggio dai messaggi della chat
    (testo, trascrizione vocale oppure foto di pasti/alimenti, glucometri, misuratori pressione).

    Garantisce **degradazione graziosa totale**: qualsiasi errore del provider LLM o di
    validazione restituisce `[]` senza mai interrompere la risposta testuale della chat.
    """
    if not hasattr(llm, "generate_structured"):
        return []
    has_media = bool(media_bytes and media_mime_type)
    if not has_media and not should_attempt_extraction(scrubbed_messages):
        return []

    iso_now, human_now = _format_rome_now(now)
    recent_turns = list(scrubbed_messages[-4:])
    user_turns_text = "\n".join(m.content for m in recent_turns if m.role == "user")

    therapies_block = "Nessuna terapia attiva registrata."
    if active_therapies:
        t_lines = [
            f"- id={t.get('id')}: {t.get('drug_name')} ({t.get('therapy_type')}, "
            f"dosaggio: {t.get('dosage') or 'n.d.'}, "
            f"frequenza: {t.get('frequency') or 'Giornaliera'}, "
            f"orario: {t.get('scheduled_time') or 'n.d.'})"
            for t in active_therapies
        ]
        therapies_block = "\n".join(t_lines)

    conv_lines = [
        f"[{'UTENTE' if m.role == 'user' else 'ASSISTENTE'}]: {m.content}"
        for m in recent_turns
    ]
    prompt_parts = [
        f"<timestamp_corrente>\n{human_now}\n</timestamp_corrente>",
        f"<terapie_attive_paziente>\n{therapies_block}\n</terapie_attive_paziente>",
        "<conversazione_recente>\n" + "\n".join(conv_lines) + "\n</conversazione_recente>",
    ]
    if has_media:
        kind_label = attachment_kind or "generic_image"
        prompt_parts.append(
            f"<allegato_multimodale tipo=\"{kind_label}\">\n"
            "L'utente ha allegato un'immagine in questo turno. Esamina attentamente la foto:\n"
            "- Se mostra un alimento, una bevanda, uno snack o un prodotto confezionato (es. yogurt, pasta, frutta, etichetta nutrizionale), "
            "genera una proposta per `target_table=\"meals\"` con descrizione pulita dell'alimento visibile, `meal_type` coerente con "
            "`<timestamp_corrente>` (se non specificato dall'utente), carboidrati e calorie (letti dall'etichetta con `is_estimated=false` "
            "se ben visibili, oppure stimati su porzione standard LARN con `is_estimated=true`).\n"
            "- Se mostra il display di un glucometro o di un misuratore di pressione con cifre chiaramente leggibili, "
            "genera una proposta per `target_table=\"diabetes_data\"` (`GLICEMIA` o `PRESSIONE`) con il valore esatto letto sul display. "
            "Non inventare mai cifre se il display è sfocato o illeggibile.\n"
            "- Se mostra una confezione di farmaco e l'utente dichiara di averlo assunto o che gli è stato prescritto, "
            "genera la proposta per `therapy_logs` o `therapies`.\n"
            "</allegato_multimodale>"
        )
    extraction_prompt = "\n\n".join(prompt_parts)

    try:
        if has_media:
            try:
                batch: LLMExtractionBatch = await llm.generate_structured(
                    [ChatMessage(role="user", content=extraction_prompt)],
                    system=get_extraction_system_prompt(),
                    response_schema=LLMExtractionBatch,
                    media_bytes=media_bytes,
                    media_mime_type=media_mime_type,
                )
            except TypeError:
                batch = await llm.generate_structured(
                    [ChatMessage(role="user", content=extraction_prompt)],
                    system=get_extraction_system_prompt(),
                    response_schema=LLMExtractionBatch,
                )
        else:
            batch = await llm.generate_structured(
                [ChatMessage(role="user", content=extraction_prompt)],
                system=get_extraction_system_prompt(),
                response_schema=LLMExtractionBatch,
            )
    except Exception as exc:
        logger.warning("extractor chat fallito (degradazione graziosa) exc={}", type(exc).__name__)
        return []

    proposals: list[ProposedEntry] = []
    # Se i dati provengono da una foto (es. glucometro o etichetta nutrizionale), le cifre sono nei pixel
    # e non necessariamente scritte dall'utente nel testo del messaggio.
    source_text_for_check = None if has_media else user_turns_text
    for idx, raw_item in enumerate(batch.items, start=1):
        entry = _build_proposed_entry(
            raw_item,
            index=idx,
            fallback_iso=iso_now,
            source_user_text=source_text_for_check,
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

    Pipeline ibrida in 3 step:
    1. Ispezione PDF in RAM (PyMuPDF): se digitale nativo, estrae testo e tabelle a costo zero (media_bytes -> None).
    2. De-identificazione difensiva (GDPR Art. 9): rimozione CF, intestazioni anagrafiche, nomi e strutture sanitarie.
    3. Estrazione strutturata Pydantic v2 con sintesi clinica e guardrail anti-allucinazione.
    """
    if not hasattr(llm, "generate_structured"):
        return ReportExtractionResult()
    if not report_text and not media_bytes:
        return ReportExtractionResult()

    extraction_method: Literal["pymupdf_native", "gemini_vision", "text"] = "text"
    actual_media_bytes = media_bytes
    actual_mime_type = media_mime_type
    raw_source_text = report_text or ""

    # Step 1: Ispezione in RAM per PDF se report_text non è già stato estratto
    if actual_mime_type == "application/pdf" and actual_media_bytes and not raw_source_text:
        try:
            inspection = inspect_pdf_in_ram(actual_media_bytes)
            if inspection.is_native_digital:
                raw_source_text = inspection.extracted_text
                # Zero byte e zero PII inviate a Vertex AI Vision, elaborazione puramente testuale
                actual_media_bytes = None
                actual_mime_type = None
                extraction_method = "pymupdf_native"
                logger.info(
                    "extract_from_report uso testo nativo PyMuPDF chars={} pagine={}",
                    inspection.alnum_char_count,
                    inspection.page_count,
                )
            else:
                extraction_method = "gemini_vision"
                logger.info(
                    "extract_from_report PDF scansionato/raster: fallback su Gemini Multimodale pagine={}",
                    inspection.page_count,
                )
        except Exception as insp_exc:
            logger.warning("Ispezione PDF fallita, procedo con fallback multimodale exc={}", type(insp_exc).__name__)
            extraction_method = "gemini_vision"
    elif actual_media_bytes:
        extraction_method = "gemini_vision"
    else:
        extraction_method = "text"

    iso_now, human_now = _format_rome_now(now)

    # Step 2: De-identificazione e Scrubbing PII (GDPR Art. 9) sul testo prima dell'invio
    scrubbed_text, updated_map = scrub_report_text(raw_source_text, masking_map)

    prompt_parts = [
        f"<timestamp_corrente>\n{human_now}\n</timestamp_corrente>",
        "Verifica innanzitutto se il documento o l'immagine fornita è un vero referto medico "
        "(analisi del sangue/urine, referto di visita specialistica, esame strumentale): "
        "imposta 'is_medical_report' a true SOLO se è un referto medico; impostalo a false se è la foto di un pasto/cibo, "
        "di un glucometro, di una confezione di farmaci o di un documento non clinico.\n"
        "Se è un referto medico, estrai tutti i parametri clinici ed ematochimici "
        "(GLICEMIA, HbA1c, Colesterolo Totale, HDL, LDL, Trigliceridi, eGFR, UACR, PRESSIONE) "
        "e genera una sintesi clinica oggettiva ('clean_summary') completamente priva di dati identificativi "
        "(non includere MAI nomi di pazienti, codici fiscali, ospedali, indirizzi o date di nascita). "
        "In 'clinical_findings_text' trascrivi in modo schematico e de-identificato le principali conclusioni diagnostiche.",
    ]
    if scrubbed_text:
        prompt_parts.append(f"<testo_referto>\n{sanitize_xml_tag_content(scrubbed_text)}\n</testo_referto>")

    try:
        batch: LLMReportExtractionBatch = await llm.generate_structured(
            [ChatMessage(role="user", content="\n\n".join(prompt_parts))],
            system=get_extraction_system_prompt(),
            response_schema=LLMReportExtractionBatch,
            media_bytes=actual_media_bytes,
            media_mime_type=actual_mime_type,
        )
    except (LLMUnavailable, LLMError):
        # Propaga errori transienti/provider così POST /api/reports restituisce 502/503
        # invece di un falso 422 "documento non sembra un referto medico leggibile".
        raise
    except Exception as exc:
        logger.warning("extractor report fallito exc={}", type(exc).__name__)
        return ReportExtractionResult(is_medical_report=False)

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

    # Sanitizzazione difensiva di ritorno su clean_summary e note
    clean_summary_scrubbed, updated_map = scrub_report_text(batch.clean_summary, updated_map)
    clean_summary_deidentified = clean_summary_scrubbed.strip()
    clean_summary_rehydrated = rehydrate_assistant_reply(clean_summary_scrubbed, updated_map).strip()

    clean_findings_scrubbed, _ = scrub_report_text(batch.clinical_findings_text, updated_map)

    # Il testo de-identificato completo da archiviare e vettorizzare (GDPR Art. 9: zero PII nel DB o embedding)
    scrubbed_content = scrubbed_text or clean_findings_scrubbed or clean_summary_deidentified

    r_type = batch.report_type or "ANALISI_SANGUE"
    is_med = bool(batch.is_medical_report)

    try:
        return ReportExtractionResult(
            is_medical_report=is_med,
            report_type=r_type,  # type: ignore[arg-type]
            report_at=report_iso,
            clean_summary=clean_summary_deidentified,
            clean_summary_display=clean_summary_rehydrated or clean_summary_deidentified,
            scrubbed_content=scrubbed_content,
            extraction_method=extraction_method,
            proposed_entries=proposals,
        )
    except Exception:
        return ReportExtractionResult(
            is_medical_report=is_med,
            report_type="ALTRO",
            report_at=report_iso,
            clean_summary=clean_summary_deidentified,
            clean_summary_display=clean_summary_rehydrated or clean_summary_deidentified,
            scrubbed_content=scrubbed_content,
            extraction_method=extraction_method,
            proposed_entries=proposals,
        )
