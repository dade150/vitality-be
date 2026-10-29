"""Servizio di RAG Dinamico (Contesto Paziente Pseudonimizzato + Token Reversibili).

Responsabilità principali:
1. Recuperare in parallelo e in sicurezza (tramite il client RLS `user_db`) i dati
   clinici e di autogestione dell'utente autenticato da Supabase:
   - `user` (solo per età in anni e per il token reversibile `{{NOME}}` tenuto in RAM)
   - `profile` (sesso biologico, peso, altezza, IMC calcolato, note anamnestiche)
   - `diabetes_data` (diario glicemico SMBG con statistiche sintetiche, pressione arteriosa,
     esami ematochimici più recenti: HbA1c, eGFR, UACR, profilo lipidico)
   - `therapies` + `therapy_logs` (terapie attive e ultime assunzioni registrate)
   - `meals` (ultimi pasti registrati con carboidrati e calorie)
   - `physical_activities` (ultime sessioni di movimento e minuti negli ultimi 7 giorni)
   - `appointments` (prossime/ultime visite mediche con token reversibile `{{MEDICO}}`)
   - `patient_reports` (sintesi degli ultimi referti clinici caricati)
2. Garantire **Privacy & Data Minimization by Design (GDPR Art. 5 e 25)**:
   - Nessun identificativo diretto (nome, cognome, email, telefono, codice fiscale, UUID,
     data di nascita completa) viene MAI inserito in `<contesto_paziente>`.
   - Eventuali PII presenti in campi a testo libero (note, descrizioni, messaggi utente)
     vengono rimosse o sostituite con segnaposto prima dell'invio a Vertex AI.
3. Gestire la **Re-idratazione Post-LLM**:
   - Sostituire i segnaposto `{{NOME}}` e `{{MEDICO}}` nella risposta generata dal modello
     con i valori reali conservati esclusivamente in RAM per la singola richiesta.
4. Garantire **Degradazione Graziosa**:
   - Il fallimento o l'assenza di dati in una singola tabella non interrompe mai il recupero
     delle altre tabelle né il flusso della chat.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import re
import time
from typing import Any

from loguru import logger
from supabase import Client

from vitality_be.services.retriever import RetrievedChunk, format_rag_context

TOKEN_PATIENT_NAME = "{{NOME}}"
TOKEN_DOCTOR_NAME = "{{MEDICO}}"
RPC_PATIENT_CONTEXT_BUNDLE = "get_patient_context_bundle"

# Regex difensive per rimuovere PII accidentali nei testi liberi (input utente e referti)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CF_RE = re.compile(
    r"\b[A-Z]{6}[0-9LMNPQRSTUV]{2}[A-EHLMPR-T][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]\b",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?:(?:\+|00)39[\s.-]?)?(?:3\d{2}|0\d{1,3})[\s.-]?\d{3}[\s.-]?\d{3,4}\b"
)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_HEALTH_CARD_RE = re.compile(r"\b(?:8038000\d{13}|\d{16,20})\b")
_ADDRESS_RE = re.compile(
    r"\b(?:via|viale|piazza|corso|largo|vicolo)\s+[A-ZÀ-Úa-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,3}"
    r"(?:\s*,\s*|\s+(?:n\.?|civico\s+)?)\d{1,4}[A-Za-z]?\b",
    re.IGNORECASE,
)
_DOCTOR_MENTION_RE = re.compile(
    r"\b(?i:dott\.?(?:ssa|ore|oressa)?|dr\.?(?:ssa)?|prof\.?(?:ssa|essore|essoressa)?)"
    r"\s+[A-ZÀ-Ú][a-zà-ú']+(?:\s+[A-ZÀ-Ú][a-zà-ú']+)?\b"
)

# Normalizzazione dei tipi di esame in `diabetes_data.metric_type` (gestisce i duplicati dell'enum)
_LAB_METRIC_LABELS: dict[str, tuple[str, str]] = {
    "HBA1C": ("HbA1c (Emoglobina Glicata)", "%"),
    "EGFR": ("eGFR (Filtrato Glomerulare)", "mL/min/1.73m²"),
    "UACR": ("UACR (Rapporto Albumina/Creatinina urinaria)", "mg/g"),
    "COLESTEROLO_TOTALE": ("Colesterolo Totale", "mg/dL"),
    "HDL": ("Colesterolo HDL", "mg/dL"),
    "LDL": ("Colesterolo LDL", "mg/dL"),
    "Trigliceridi": ("Trigliceridi", "mg/dL"),
    "TRIGLICERIDI": ("Trigliceridi", "mg/dL"),
}

_THERAPY_TYPE_LABELS: dict[str, str] = {
    "COMPRESSE_ORALI": "Ipoglicemizzante orale / Compresse",
    "INSULINA_BASALE": "Insulina basale (lenta)",
    "INSULINA_RAPIDA": "Insulina rapida (ai pasti)",
    "GLP1_INIETTIVO": "Agonista recettoriale GLP-1 iniettivo",
    "ALTRO": "Altra terapia",
}


@dataclass(frozen=True)
class PatientMaskingMap:
    """Mappa dei segnaposto reversibili mantenuta esclusivamente in RAM durante la richiesta."""

    first_name: str | None = None
    last_name: str | None = None
    primary_doctor: str | None = None
    doctor_tokens: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PatientContextResult:
    """Risultato del recupero del contesto paziente."""

    context_xml: str
    masking_map: PatientMaskingMap
    sections_count: int = 0
    active_therapies: tuple[dict[str, Any], ...] = ()


def normalize_metric_type(raw_metric: str | None) -> str:
    """Normalizza i valori di `metric_enum` (es. `'Colesterolo Totale'` -> `'COLESTEROLO_TOTALE'`)."""
    if not raw_metric:
        return ""
    return re.sub(r"\s+", "_", str(raw_metric).strip()).upper()


def calculate_age_years(birth_value: str | date | None, *, today: date | None = None) -> int | None:
    """Calcola l'età in anni compiuti a partire dalla data di nascita (generalizzazione GDPR)."""
    if not birth_value:
        return None
    ref = today or datetime.now(timezone.utc).date()
    try:
        if isinstance(birth_value, datetime):
            bdate = birth_value.date()
        elif isinstance(birth_value, date):
            bdate = birth_value
        else:
            raw = str(birth_value).strip()[:10]
            bdate = date.fromisoformat(raw)
    except ValueError:
        return None

    age = ref.year - bdate.year - ((ref.month, ref.day) < (bdate.month, bdate.day))
    if age < 0 or age > 125:
        return None
    return age


def calculate_bmi(weight_kg: float | int | str | None, height_cm: float | int | str | None) -> tuple[float, str] | None:
    """Calcola l'IMC (BMI) e restituisce `(valore, categoria_clinica)`."""
    try:
        w = float(weight_kg)  # type: ignore[arg-type]
        h_cm = float(height_cm)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    if not (20.0 <= w <= 350.0 and 100.0 <= h_cm <= 250.0):
        return None

    h_m = h_cm / 100.0
    bmi = round(w / (h_m * h_m), 1)
    if bmi < 18.5:
        cat = "sottopeso"
    elif bmi < 25.0:
        cat = "normopeso"
    elif bmi < 30.0:
        cat = "sovrappeso"
    elif bmi < 35.0:
        cat = "obesità di grado I"
    else:
        cat = "obesità di grado II-III"
    return bmi, cat


def _parse_datetime(val: Any) -> datetime | None:
    if not val:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    raw = str(val).strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _format_timestamp(val: Any, *, now: datetime | None = None, include_time: bool = True) -> str:
    dt = _parse_datetime(val)
    if dt is None:
        return "data n.d."
    ref = now or datetime.now(timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    ref_utc = ref.astimezone(timezone.utc)
    days_diff = (ref_utc.date() - dt_utc.date()).days

    time_part = f" ore {dt_utc.strftime('%H:%M')}" if include_time else ""
    if days_diff == 0:
        return f"Oggi{time_part}"
    if days_diff == 1:
        return f"Ieri{time_part}"
    return f"{dt_utc.strftime('%d/%m/%Y')}{time_part}"


def _extract_first_float(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip().replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def scrub_free_text(text: str | None, masking_map: PatientMaskingMap | None = None) -> str:
    """Rimuove email, numeri di telefono, codici fiscali, IBAN, tessere sanitarie, indirizzi
    e il nome/cognome/medico reale dal testo libero prima di qualsiasi invio a Vertex AI."""
    if not text:
        return ""
    cleaned = str(text)
    cleaned = _EMAIL_RE.sub("[EMAIL_RIMOSSA]", cleaned)
    cleaned = _CF_RE.sub("[CF_RIMOSSO]", cleaned)
    cleaned = _IBAN_RE.sub("[IBAN_RIMOSSO]", cleaned)
    cleaned = _HEALTH_CARD_RE.sub("[TESSERA_SANITARIA_RIMOSSA]", cleaned)
    cleaned = _PHONE_RE.sub("[TELEFONO_RIMOSSO]", cleaned)
    cleaned = _ADDRESS_RE.sub("[INDIRIZZO_RIMOSSO]", cleaned)

    if masking_map is not None:
        # Se il cognome compare nel testo, lo oscuriamo prima del nome
        if masking_map.last_name and len(masking_map.last_name.strip()) >= 2:
            ln_escaped = re.escape(masking_map.last_name.strip())
            if masking_map.first_name and len(masking_map.first_name.strip()) >= 2:
                fn_escaped = re.escape(masking_map.first_name.strip())
                full_pattern = re.compile(rf"\b{fn_escaped}\s+{ln_escaped}\b", re.IGNORECASE)
                cleaned = full_pattern.sub(TOKEN_PATIENT_NAME, cleaned)
            ln_pattern = re.compile(rf"\b{ln_escaped}\b", re.IGNORECASE)
            cleaned = ln_pattern.sub("[COGNOME_RIMOSSO]", cleaned)

        if masking_map.first_name and len(masking_map.first_name.strip()) >= 2:
            fn_escaped = re.escape(masking_map.first_name.strip())
            fn_pattern = re.compile(rf"\b{fn_escaped}\b", re.IGNORECASE)
            cleaned = fn_pattern.sub(TOKEN_PATIENT_NAME, cleaned)

        # Sostituisce tutti i medici già mappati in doctor_tokens
        if masking_map.doctor_tokens:
            for token, doc_name in sorted(
                masking_map.doctor_tokens.items(),
                key=lambda kv: len(kv[1] or ""),
                reverse=True,
            ):
                if doc_name and len(doc_name.strip()) >= 3:
                    doc_escaped = re.escape(doc_name.strip())
                    cleaned = re.sub(doc_escaped, token, cleaned, flags=re.IGNORECASE)

        if masking_map.primary_doctor and len(masking_map.primary_doctor.strip()) >= 3:
            doc_escaped = re.escape(masking_map.primary_doctor.strip())
            doc_pattern = re.compile(rf"\b{doc_escaped}\b", re.IGNORECASE)
            cleaned = doc_pattern.sub(TOKEN_DOCTOR_NAME, cleaned)

    # Qualsiasi altro nome di medico residuo ("Dott. Cognome") non ancora tokenizzato viene oscurato
    cleaned = _DOCTOR_MENTION_RE.sub(TOKEN_DOCTOR_NAME, cleaned)

    return cleaned.strip()


def scrub_for_db_storage(text: str | None) -> str:
    """Scrubbing selettivo dei dati sensibili critici per il salvataggio su `public.chat_message`.

    A differenza di `scrub_free_text` (che de-identifica anche i nomi propri per Vertex AI),
    questa funzione:
    1. Mantiene il testo naturale leggibile dall'utente (incluso il proprio nome e il nome del medico).
    2. Rimuove e sostituisce SEMPRE identificativi critici scritti per errore (Codice Fiscale,
       IBAN, Tessera Sanitaria, Email, Numero di Telefono, Indirizzo di casa).
    """
    if not text:
        return ""
    cleaned = str(text)
    cleaned = _EMAIL_RE.sub("[EMAIL_RIMOSSA]", cleaned)
    cleaned = _CF_RE.sub("[CF_RIMOSSO]", cleaned)
    cleaned = _IBAN_RE.sub("[IBAN_RIMOSSO]", cleaned)
    cleaned = _HEALTH_CARD_RE.sub("[TESSERA_SANITARIA_RIMOSSA]", cleaned)
    cleaned = _PHONE_RE.sub("[TELEFONO_RIMOSSO]", cleaned)
    cleaned = _ADDRESS_RE.sub("[INDIRIZZO_RIMOSSO]", cleaned)
    return cleaned.strip()


def mask_user_text_with_dynamic_tokens(
    text: str | None,
    masking_map: PatientMaskingMap | None = None,
) -> tuple[str, PatientMaskingMap]:
    """Maschera i dati sensibili nel testo dell'utente (o di un referto) e registra dinamicamente
    eventuali nuovi nomi di medici menzionati (es. 'Dott. Marco Verdi') in `PatientMaskingMap`,
    così non lasciano mai il server verso Vertex AI ma vengono re-idratati per il Front-End."""
    current_map = masking_map or PatientMaskingMap()
    if not text:
        return "", current_map

    doctor_tokens = dict(current_map.doctor_tokens)
    primary_doctor = current_map.primary_doctor
    if primary_doctor and TOKEN_DOCTOR_NAME not in doctor_tokens:
        doctor_tokens[TOKEN_DOCTOR_NAME] = primary_doctor

    existing_by_norm = {v.strip().lower(): k for k, v in doctor_tokens.items() if v}

    def _replace_doc(match: re.Match[str]) -> str:
        nonlocal primary_doctor
        raw_doc = match.group(0).strip()
        norm_doc = raw_doc.lower()
        if norm_doc in existing_by_norm:
            return existing_by_norm[norm_doc]
        if primary_doctor is None:
            primary_doctor = raw_doc
            token = TOKEN_DOCTOR_NAME
        else:
            token = f"{{{{MEDICO_{len(existing_by_norm) + 1}}}}}"
        existing_by_norm[norm_doc] = token
        doctor_tokens[token] = raw_doc
        return token

    with_doc_tokens = _DOCTOR_MENTION_RE.sub(_replace_doc, str(text))
    updated_map = PatientMaskingMap(
        first_name=current_map.first_name,
        last_name=current_map.last_name,
        primary_doctor=primary_doctor,
        doctor_tokens=doctor_tokens,
    )
    return scrub_free_text(with_doc_tokens, updated_map), updated_map


def rehydrate_assistant_reply(reply_text: str, masking_map: PatientMaskingMap | None) -> str:
    """Sostituisce i token `{{NOME}}` e `{{MEDICO}}` nella risposta del modello con i valori reali."""
    if not reply_text:
        return reply_text

    result = reply_text
    first_name = (masking_map.first_name or "").strip() if masking_map else ""
    primary_doctor = (masking_map.primary_doctor or "").strip() if masking_map else ""

    # 1. Sostituzione dei token specifici dei medici ({{MEDICO_1}}, {{MEDICO_2}}, ...)
    if masking_map and masking_map.doctor_tokens:
        for token, doc_name in masking_map.doctor_tokens.items():
            if doc_name:
                result = re.sub(re.escape(token), doc_name, result, flags=re.IGNORECASE)

    # 2. Sostituzione o fallback del token {{MEDICO}}
    if re.search(r"\{\{\s*MEDICO(?:_\d+)?\s*\}\}", result, flags=re.IGNORECASE):
        replacement_doc = primary_doctor if primary_doctor else "il tuo medico di riferimento"
        result = re.sub(
            r"\{\{\s*MEDICO(?:_\d+)?\s*\}\}",
            replacement_doc,
            result,
            flags=re.IGNORECASE,
        )

    # 3. Sostituzione o rimozione elegante del token {{NOME}}
    if re.search(r"\{\{\s*NOME\s*\}\}", result, flags=re.IGNORECASE):
        if first_name:
            result = re.sub(r"\{\{\s*NOME\s*\}\}", first_name, result, flags=re.IGNORECASE)
        else:
            # Rimuove eventuali virgole/spazi superflui attorno al placeholder (es. "Ciao {{NOME}}, " -> "Ciao, ")
            result = re.sub(r",\s*\{\{\s*NOME\s*\}\}(?=[!?.:,;])", "", result, flags=re.IGNORECASE)
            result = re.sub(r"\s+\{\{\s*NOME\s*\}\}(?=[!?.:,;])", "", result, flags=re.IGNORECASE)
            result = re.sub(r"\{\{\s*NOME\s*\}\}\s*,\s*", "", result, flags=re.IGNORECASE)
            result = re.sub(r"\{\{\s*NOME\s*\}\}\s*", "", result, flags=re.IGNORECASE)

    return result


def _safe_query_sync(label: str, fn) -> list[dict[str, Any]]:
    """Esegue una query sincrona PostgREST con degradazione graziosa per singola tabella."""
    try:
        resp = fn()
        data = getattr(resp, "data", None)
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
        return []
    except Exception as exc:
        logger.warning("patient_context query fallita table={} exc={}", label, type(exc).__name__)
        return []


def _as_dict_list(val: Any) -> list[dict[str, Any]]:
    if isinstance(val, list):
        return [r for r in val if isinstance(r, dict)]
    return []


def _fetch_bundle_or_tables_sync(user_db: Client) -> tuple[list[dict[str, Any]], ...]:
    """Prova prima la RPC singola `get_patient_context_bundle` (1 solo round-trip HTTP);
    se non è ancora presente a DB o fallisce, ripiega automaticamente sulle query `.table(...)`."""
    if hasattr(user_db, "rpc"):
        try:
            resp = user_db.rpc(RPC_PATIENT_CONTEXT_BUNDLE, {}).execute()
            payload = getattr(resp, "data", None)
            if isinstance(payload, dict):
                return (
                    _as_dict_list(payload.get("user")),
                    _as_dict_list(payload.get("profile")),
                    _as_dict_list(payload.get("diabetes_data")),
                    _as_dict_list(payload.get("therapies")),
                    _as_dict_list(payload.get("therapy_logs")),
                    _as_dict_list(payload.get("meals")),
                    _as_dict_list(payload.get("physical_activities")),
                    _as_dict_list(payload.get("appointments")),
                    _as_dict_list(payload.get("patient_reports")),
                )
        except Exception as exc:
            logger.debug("patient_context rpc bundle fallback exc={}", type(exc).__name__)

    return _fetch_all_tables_sync(user_db)


def _fetch_all_tables_sync(user_db: Client) -> tuple[list[dict[str, Any]], ...]:
    """Esegue le query di contesto sullo stesso client sincrono `user_db` in modo thread-safe."""
    user_rows = _safe_query_sync(
        "user",
        lambda: user_db.table("user").select("name, surname, birth").limit(1).execute(),
    )
    profile_rows = _safe_query_sync(
        "profile",
        lambda: (
            user_db.table("profile")
            .select("peso, altezza, sesso, note_anamnestiche, created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ),
    )
    diabetes_rows = _safe_query_sync(
        "diabetes_data",
        lambda: (
            user_db.table("diabetes_data")
            .select("metric_type, metric_value, measured_at, created_at")
            .order("measured_at", desc=True)
            .limit(60)
            .execute()
        ),
    )
    therapies_rows = _safe_query_sync(
        "therapies",
        lambda: (
            user_db.table("therapies")
            .select("id, therapy_type, drug_name, dosage, frequency, scheduled_time, notes, is_active")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .limit(15)
            .execute()
        ),
    )
    therapy_logs_rows = _safe_query_sync(
        "therapy_logs",
        lambda: (
            user_db.table("therapy_logs")
            .select("therapy_id, taken_at, status")
            .order("taken_at", desc=True)
            .limit(10)
            .execute()
        ),
    )
    meals_rows = _safe_query_sync(
        "meals",
        lambda: (
            user_db.table("meals")
            .select("meal_type, description, carbs_grams, calories, meal_at, is_estimated, created_at")
            .order("meal_at", desc=True)
            .limit(7)
            .execute()
        ),
    )
    activities_rows = _safe_query_sync(
        "physical_activities",
        lambda: (
            user_db.table("physical_activities")
            .select("activity_type, duration_minutes, intensity, steps, notes, activity_at, created_at")
            .order("activity_at", desc=True)
            .limit(7)
            .execute()
        ),
    )
    appointments_rows = _safe_query_sync(
        "appointments",
        lambda: (
            user_db.table("appointments")
            .select("title, doctor_name, appointment_at, notes")
            .order("appointment_at", desc=True)
            .limit(5)
            .execute()
        ),
    )
    reports_rows = _safe_query_sync(
        "patient_reports",
        lambda: (
            user_db.table("patient_reports")
            .select("report_type, source_name, content, report_at, created_at")
            .order("report_at", desc=True)
            .limit(3)
            .execute()
        ),
    )
    return (
        user_rows,
        profile_rows,
        diabetes_rows,
        therapies_rows,
        therapy_logs_rows,
        meals_rows,
        activities_rows,
        appointments_rows,
        reports_rows,
    )


def _format_profile_section(
    user_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
    masking_map: PatientMaskingMap,
    *,
    now: datetime,
) -> str | None:
    parts: list[str] = []
    if masking_map.first_name:
        parts.append(f"Riferimento personale: {TOKEN_PATIENT_NAME} (usa questo segnaposto se vuoi chiamarlo per nome)")

    if user_rows:
        age = calculate_age_years(user_rows[0].get("birth"), today=now.date())
        if age is not None:
            parts.append(f"Età: {age} anni")

    if profile_rows:
        prof = profile_rows[0]
        sesso = str(prof.get("sesso") or "").strip()
        if sesso:
            parts.append(f"Sesso biologico: {sesso}")

        peso = prof.get("peso")
        altezza = prof.get("altezza")
        if peso is not None and altezza is not None:
            bmi_info = calculate_bmi(peso, altezza)
            if bmi_info is not None:
                bmi_val, bmi_cat = bmi_info
                parts.append(f"Antropometria: peso {peso} kg, altezza {altezza} cm (IMC {bmi_val} kg/m² - {bmi_cat})")
            else:
                parts.append(f"Antropometria: peso {peso} kg, altezza {altezza} cm")

        note = scrub_free_text(prof.get("note_anamnestiche"), masking_map)
        if note:
            parts.append(f"Note anamnestiche: {note[:400]}")

    if not parts:
        return None
    return "[Profilo Clinico]\n- " + "\n- ".join(parts)


def _format_diabetes_data_sections(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
) -> list[str]:
    if not rows:
        return []

    glucose_readings: list[tuple[Any, str, float | None]] = []
    bp_readings: list[tuple[Any, str]] = []
    latest_labs: dict[str, tuple[Any, str]] = {}

    for r in rows:
        m_type = normalize_metric_type(r.get("metric_type"))
        m_val = str(r.get("metric_value") or "").strip()
        m_at = r.get("measured_at") or r.get("created_at")
        if not m_type or not m_val:
            continue

        if m_type == "GLICEMIA":
            glucose_readings.append((m_at, m_val, _extract_first_float(m_val)))
        elif m_type == "PRESSIONE":
            bp_readings.append((m_at, m_val))
        elif m_type in _LAB_METRIC_LABELS and m_type not in latest_labs:
            latest_labs[m_type] = (m_at, m_val)

    sections: list[str] = []

    # 1. Diario glicemico SMBG + Statistiche pre-calcolate
    if glucose_readings:
        recent_g = glucose_readings[:10]
        numeric_vals = [v for _, _, v in recent_g if v is not None and 20.0 <= v <= 600.0]
        lines: list[str] = []
        if numeric_vals:
            avg_g = round(sum(numeric_vals) / len(numeric_vals))
            min_g = round(min(numeric_vals))
            max_g = round(max(numeric_vals))
            hypo_count = sum(1 for v in numeric_vals if v < 70.0)
            in_range_count = sum(1 for v in numeric_vals if 70.0 <= v <= 180.0)
            hyper_count = sum(1 for v in numeric_vals if v > 180.0)
            lines.append(
                f"Sintesi ultime {len(numeric_vals)} misurazioni: media {avg_g} mg/dL "
                f"(min {min_g}, max {max_g} mg/dL | in range 70-180: {in_range_count}, "
                f"ipoglicemie <70: {hypo_count}, iperglicemie >180: {hyper_count})"
            )
        for m_at, m_val, _ in recent_g:
            ts_str = _format_timestamp(m_at, now=now)
            unit = "" if "mg" in m_val.lower() else " mg/dL"
            lines.append(f"{ts_str}: {m_val}{unit}")
        sections.append("[Diario Glicemico Recente (SMBG)]\n- " + "\n- ".join(lines))

    # 2. Pressione arteriosa
    if bp_readings:
        bp_lines = [
            f"{_format_timestamp(m_at, now=now)}: {m_val}"
            + ("" if "mmhg" in m_val.lower() else " mmHg")
            for m_at, m_val in bp_readings[:5]
        ]
        sections.append("[Pressione Arteriosa Recente]\n- " + "\n- ".join(bp_lines))

    # 3. Esami ematochimici
    if latest_labs:
        lab_lines: list[str] = []
        for key in ("HBA1C", "EGFR", "UACR", "COLESTEROLO_TOTALE", "LDL", "HDL", "TRIGLICERIDI"):
            if key in latest_labs:
                m_at, m_val = latest_labs[key]
                label, default_unit = _LAB_METRIC_LABELS[key]
                date_str = _format_timestamp(m_at, now=now, include_time=False)
                has_unit = any(u in m_val.lower() for u in ("%", "mg", "ml", "mmol"))
                unit_str = "" if has_unit else f" {default_unit}"
                lab_lines.append(f"{label}: {m_val}{unit_str} (rilevato: {date_str})")
        if lab_lines:
            sections.append("[Esami Ematochimici Più Recenti]\n- " + "\n- ".join(lab_lines))

    return sections


def _format_therapies_section(
    therapies_rows: list[dict[str, Any]],
    logs_rows: list[dict[str, Any]],
    masking_map: PatientMaskingMap,
    *,
    now: datetime,
) -> str | None:
    if not therapies_rows:
        return None

    therapy_names_by_id: dict[Any, str] = {}
    lines: list[str] = []

    for t in therapies_rows:
        t_id = t.get("id")
        raw_type = str(t.get("therapy_type") or "ALTRO").strip().upper()
        type_label = _THERAPY_TYPE_LABELS.get(raw_type, raw_type)
        drug = scrub_free_text(t.get("drug_name"), masking_map) or "Farmaco non specificato"
        dosage = scrub_free_text(t.get("dosage"), masking_map)
        freq = scrub_free_text(t.get("frequency"), masking_map)
        sched = str(t.get("scheduled_time") or "").strip()[:5]
        notes = scrub_free_text(t.get("notes"), masking_map)

        if t_id is not None:
            therapy_names_by_id[t_id] = drug

        detail_parts = [f"{drug} ({type_label})"]
        if dosage:
            detail_parts.append(f"dosaggio: {dosage}")
        if freq:
            detail_parts.append(f"frequenza: {freq}")
        if sched:
            detail_parts.append(f"orario previsto: {sched}")
        if notes:
            detail_parts.append(f"note: {notes[:120]}")
        lines.append(" | ".join(detail_parts))

    if logs_rows:
        log_items: list[str] = []
        for lg in logs_rows[:5]:
            t_name = therapy_names_by_id.get(lg.get("therapy_id"), "Terapia")
            status = scrub_free_text(lg.get("status"), masking_map) or "registrata"
            ts_str = _format_timestamp(lg.get("taken_at"), now=now)
            log_items.append(f"{ts_str}: {t_name} ({status})")
        if log_items:
            lines.append("Ultime assunzioni registrate: " + "; ".join(log_items))

    return "[Terapia Farmacologica Attiva]\n- " + "\n- ".join(lines)


def _format_meals_section(
    meals_rows: list[dict[str, Any]],
    masking_map: PatientMaskingMap,
    *,
    now: datetime,
) -> str | None:
    if not meals_rows:
        return None

    lines: list[str] = []
    for m in meals_rows[:7]:
        m_type = str(m.get("meal_type") or "PASTO").strip().upper()
        desc = scrub_free_text(m.get("description"), masking_map)
        if not desc:
            continue
        ts_str = _format_timestamp(m.get("meal_at") or m.get("created_at"), now=now)
        carbs = m.get("carbs_grams")
        cals = m.get("calories")
        est = bool(m.get("is_estimated"))

        nutri_parts: list[str] = []
        if carbs is not None:
            nutri_parts.append(f"carboidrati: {carbs} g")
        if cals is not None:
            nutri_parts.append(f"calorie: {cals} kcal")
        if nutri_parts and est:
            nutri_parts.append("valori stimati")

        nutri_str = f" ({', '.join(nutri_parts)})" if nutri_parts else ""
        lines.append(f"{ts_str} [{m_type}]: {desc[:200]}{nutri_str}")

    if not lines:
        return None
    return "[Diario Alimentare Recente]\n- " + "\n- ".join(lines)


def _format_activities_section(
    act_rows: list[dict[str, Any]],
    masking_map: PatientMaskingMap,
    *,
    now: datetime,
) -> str | None:
    if not act_rows:
        return None

    seven_days_ago = now - timedelta(days=7)
    weekly_minutes = 0
    lines: list[str] = []

    for a in act_rows[:7]:
        act_type = str(a.get("activity_type") or "ATTIVITA").strip().upper()
        dur = a.get("duration_minutes")
        intensity = str(a.get("intensity") or "MODERATA").strip().lower()
        steps = a.get("steps")
        notes = scrub_free_text(a.get("notes"), masking_map)
        act_dt = _parse_datetime(a.get("activity_at") or a.get("created_at"))

        try:
            dur_int = int(dur) if dur is not None else 0
        except (TypeError, ValueError):
            dur_int = 0

        if act_dt is not None and act_dt >= seven_days_ago and dur_int > 0:
            weekly_minutes += dur_int

        ts_str = _format_timestamp(act_dt, now=now)
        extras = [f"{dur_int} min", f"intensità {intensity}"]
        if steps is not None:
            extras.append(f"{steps} passi")
        if notes:
            extras.append(f"note: {notes[:100]}")
        lines.append(f"{ts_str} [{act_type}]: " + ", ".join(extras))

    header_line = f"Totale movimento ultimi 7 giorni: {weekly_minutes} minuti (obiettivo linee guida: 150 min/settimana)"
    return "[Attività Fisica Recente]\n- " + "\n- ".join([header_line, *lines])


def _format_appointments_and_reports_section(
    appt_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    masking_map: PatientMaskingMap,
    *,
    now: datetime,
) -> tuple[str | None, PatientMaskingMap]:
    lines: list[str] = []
    doctor_tokens = dict(masking_map.doctor_tokens)
    primary_doctor = masking_map.primary_doctor
    doc_name_to_token: dict[str, str] = {}

    for idx, ap in enumerate(appt_rows[:5], start=1):
        title = scrub_free_text(ap.get("title"), masking_map)
        if not title:
            continue
        raw_doc = str(ap.get("doctor_name") or "").strip()
        doc_token_str = ""
        if raw_doc:
            norm_doc = raw_doc.lower()
            if norm_doc not in doc_name_to_token:
                if primary_doctor is None:
                    primary_doctor = raw_doc
                    token = TOKEN_DOCTOR_NAME
                else:
                    token = f"{{{{MEDICO_{len(doc_name_to_token) + 1}}}}}"
                doc_name_to_token[norm_doc] = token
                doctor_tokens[token] = raw_doc
            doc_token_str = f" con {doc_name_to_token[norm_doc]}"

        ts_str = _format_timestamp(ap.get("appointment_at"), now=now)
        notes = scrub_free_text(ap.get("notes"), masking_map)
        note_str = f" (note: {notes[:100]})" if notes else ""
        lines.append(f"Appuntamento {ts_str}: {title}{doc_token_str}{note_str}")

    updated_map = PatientMaskingMap(
        first_name=masking_map.first_name,
        last_name=masking_map.last_name,
        primary_doctor=primary_doctor,
        doctor_tokens=doctor_tokens,
    )

    for rep in report_rows[:3]:
        r_type = str(rep.get("report_type") or "REFERTO").strip().upper()
        r_date = _format_timestamp(rep.get("report_at") or rep.get("created_at"), now=now, include_time=False)
        r_content = scrub_free_text(rep.get("content"), updated_map)
        if r_content:
            lines.append(f"Referto [{r_type}] del {r_date}: {r_content[:240]}")

    if not lines:
        return None, updated_map
    return "[Visite Mediche e Referti Recenti]\n- " + "\n- ".join(lines), updated_map


async def fetch_patient_context(
    user_db: Client,
    *,
    now: datetime | None = None,
) -> PatientContextResult:
    """Recupera e formatta in modo pseudonimizzato il `<contesto_paziente>` da Supabase.

    Se `user_db` non espone `.table(...)` (es. mock minimale nei test del retriever statico)
    o se tutte le tabelle sono vuote/indisponibili, restituisce `context_xml=""` senza mai
    sollevare eccezioni.
    """
    if not hasattr(user_db, "table"):
        return PatientContextResult(context_xml="", masking_map=PatientMaskingMap(), sections_count=0)

    started = time.monotonic()
    ref_now = now or datetime.now(timezone.utc)

    (
        user_rows,
        profile_rows,
        diabetes_rows,
        therapies_rows,
        therapy_logs_rows,
        meals_rows,
        activities_rows,
        appointments_rows,
        reports_rows,
    ) = await asyncio.to_thread(_fetch_bundle_or_tables_sync, user_db)

    first_name: str | None = None
    last_name: str | None = None
    if user_rows:
        raw_fn = str(user_rows[0].get("name") or "").strip()
        raw_ln = str(user_rows[0].get("surname") or "").strip()
        first_name = raw_fn or None
        last_name = raw_ln or None

    masking_map = PatientMaskingMap(first_name=first_name, last_name=last_name)

    # Costruiamo prima la sezione appuntamenti/referti così popola eventuali token {{MEDICO}} in masking_map
    appt_section, masking_map = _format_appointments_and_reports_section(
        appointments_rows,
        reports_rows,
        masking_map,
        now=ref_now,
    )

    sections: list[str] = []
    prof_section = _format_profile_section(user_rows, profile_rows, masking_map, now=ref_now)
    if prof_section:
        sections.append(prof_section)

    sections.extend(_format_diabetes_data_sections(diabetes_rows, now=ref_now))

    ther_section = _format_therapies_section(therapies_rows, therapy_logs_rows, masking_map, now=ref_now)
    if ther_section:
        sections.append(ther_section)

    meals_section = _format_meals_section(meals_rows, masking_map, now=ref_now)
    if meals_section:
        sections.append(meals_section)

    act_section = _format_activities_section(activities_rows, masking_map, now=ref_now)
    if act_section:
        sections.append(act_section)

    if appt_section:
        sections.append(appt_section)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info("patient_context ok sections={} ms={}", len(sections), elapsed_ms)

    active_therapies_clean = tuple(
        {
            "id": t.get("id"),
            "therapy_type": str(t.get("therapy_type") or "ALTRO").strip().upper(),
            "drug_name": scrub_free_text(t.get("drug_name"), masking_map),
            "dosage": scrub_free_text(t.get("dosage"), masking_map),
        }
        for t in therapies_rows
        if t.get("drug_name")
    )

    if not sections:
        return PatientContextResult(
            context_xml="",
            masking_map=masking_map,
            sections_count=0,
            active_therapies=active_therapies_clean,
        )

    joined = "\n\n".join(sections)
    context_xml = f"<contesto_paziente>\n{joined}\n</contesto_paziente>"
    return PatientContextResult(
        context_xml=context_xml,
        masking_map=masking_map,
        sections_count=len(sections),
        active_therapies=active_therapies_clean,
    )


def build_dual_rag_message(
    user_message: str,
    *,
    patient_context_xml: str = "",
    clinical_chunks: list[RetrievedChunk] | None = None,
) -> str:
    """Combina `<contesto_paziente>`, `<contesto_clinico>` e `<messaggio_utente>` nell'ultimo turno utente."""
    clinical_xml = format_rag_context(clinical_chunks or [])
    blocks = [b for b in (patient_context_xml.strip(), clinical_xml.strip()) if b]
    if not blocks:
        return user_message

    joined_context = "\n\n".join(blocks)
    return (
        f"{joined_context}\n\n"
        f"<messaggio_utente>\n{user_message}\n</messaggio_utente>"
    )
