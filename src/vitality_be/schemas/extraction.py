"""Schemi Pydantic v2 per l'estrazione strutturata Human-in-the-Loop (Chat, Voce e Referti).

Allineati rigorosamente ai tipi ENUM e alle tabelle PostgreSQL/Supabase documentati in
`docs/db_schema_and_security.md`:
- `diabetes_data`: profilo glicemico (GLICEMIA, HbA1c), pressione (PRESSIONE), profilo renale
  (eGFR, UACR) e profilo lipidico (Colesterolo Totale, HDL, LDL, Trigliceridi).
- `meals`: diario alimentare con flag obbligatorio `is_estimated` per stime LARN.
- `physical_activities`: sessioni di movimento con gestione campi mancanti (es. durata).
- `appointments`: visite mediche con supporto a token reversibili `{{MEDICO}}`.
- `therapy_logs` e `therapies`: registro assunzioni e piano terapeutico.
- `profile`: aggiornamenti di peso e altezza.
- `patient_reports`: estrazione strutturata da testo o foto/PDF di referti clinici.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# 1. Tipi ENUM canonici di Supabase (`public`)
# ---------------------------------------------------------------------------

MetricEnum = Literal[
    "GLICEMIA",
    "PRESSIONE",
    "HbA1c",
    "eGFR",
    "UACR",
    "Colesterolo Totale",
    "HDL",
    "LDL",
    "Trigliceridi",
]

MealEnum = Literal[
    "COLAZIONE",
    "SPUNTINO_MATTINA",
    "PRANZO",
    "SPUNTINO_POMERIGGIO",
    "CENA",
    "FUORI_PASTO",
    "SPUNTINO",
]

ActivityEnum = Literal[
    "CAMMINATA",
    "CORSA",
    "BICICLETTA",
    "PALESTRA",
    "NUOTO",
    "ALTRO",
]

IntensityEnum = Literal["LEGGERA", "MODERATA", "INTENSA"]

TherapyTypeEnum = Literal[
    "COMPRESSE_ORALI",
    "INSULINA_BASALE",
    "INSULINA_RAPIDA",
    "GLP1_INIETTIVO",
    "ALTRO",
]

TherapyLogStatusEnum = Literal["ASSUNTA", "SALTATA", "RITARDATA"]

ReportEnum = Literal[
    "ANALISI_SANGUE",
    "VISITA_DIABETOLOGICA",
    "VISITA_CARDIOLOGICA",
    "RETINOPATIA",
    "NEFROPATIA",
    "ESAME_PODOLOGICO",
    "SALUTE_ORALE",
    "DISFUNZIONE_ERETTILE",
    "ALTRO",
]

TargetTableEnum = Literal[
    "diabetes_data",
    "meals",
    "physical_activities",
    "appointments",
    "therapy_logs",
    "therapies",
    "profile",
]

_CANONICAL_METRICS: dict[str, MetricEnum] = {
    "GLICEMIA": "GLICEMIA",
    "GLUCOSIO": "GLICEMIA",
    "PRESSIONE": "PRESSIONE",
    "PRESSIONE_ARTERIOSA": "PRESSIONE",
    "HBA1C": "HbA1c",
    "EMOGLOBINA_GLICATA": "HbA1c",
    "GLICATA": "HbA1c",
    "EGFR": "eGFR",
    "FILTRATO_GLOMERULARE": "eGFR",
    "UACR": "UACR",
    "ALBUMINURIA": "UACR",
    "MICROALBUMINURIA": "UACR",
    "COLESTEROLO_TOTALE": "Colesterolo Totale",
    "COLESTEROLO": "Colesterolo Totale",
    "HDL": "HDL",
    "COLESTEROLO_HDL": "HDL",
    "LDL": "LDL",
    "COLESTEROLO_LDL": "LDL",
    "TRIGLICERIDI": "Trigliceridi",
}

DEFAULT_METRIC_UNITS: dict[str, str] = {
    "GLICEMIA": "mg/dL",
    "PRESSIONE": "mmHg",
    "HbA1c": "%",
    "eGFR": "mL/min/1.73m²",
    "UACR": "mg/g",
    "Colesterolo Totale": "mg/dL",
    "HDL": "mg/dL",
    "LDL": "mg/dL",
    "Trigliceridi": "mg/dL",
}

# Range di plausibilità fisiologica/clinica per prevenire errori grossolani o refusi
_METRIC_NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "GLICEMIA": (20.0, 900.0),
    "HbA1c": (3.0, 22.0),
    "eGFR": (1.0, 200.0),
    "UACR": (0.1, 10000.0),
    "Colesterolo Totale": (40.0, 600.0),
    "HDL": (10.0, 150.0),
    "LDL": (10.0, 450.0),
    "Trigliceridi": (20.0, 3000.0),
}


def canonicalize_metric_type(v: Any) -> Any:
    """Converte varianti maiuscole/minuscole o con underscore nel valore canonico di `metric_enum`."""
    if not isinstance(v, str):
        return v
    key = re.sub(r"[\s\-]+", "_", v.strip()).upper()
    return _CANONICAL_METRICS.get(key, v.strip())


def _upper_strip(v: Any) -> Any:
    if isinstance(v, str):
        return re.sub(r"[\s\-]+", "_", v.strip()).upper()
    return v


CanonicalMetricType = Annotated[MetricEnum, BeforeValidator(canonicalize_metric_type)]
CanonicalMealType = Annotated[MealEnum, BeforeValidator(_upper_strip)]
CanonicalActivityType = Annotated[ActivityEnum, BeforeValidator(_upper_strip)]
CanonicalIntensityType = Annotated[IntensityEnum, BeforeValidator(_upper_strip)]
CanonicalTherapyType = Annotated[TherapyTypeEnum, BeforeValidator(_upper_strip)]
CanonicalReportType = Annotated[ReportEnum, BeforeValidator(_upper_strip)]


# ---------------------------------------------------------------------------
# 2. Validazione rigorosa dei valori clinici (`diabetes_data`)
# ---------------------------------------------------------------------------

_BP_RE = re.compile(r"^\s*(\d{2,3})\s*[/\\-]\s*(\d{2,3})\s*$")
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def normalize_and_validate_metric_value(metric_type: MetricEnum, raw_value: str | None) -> str:
    """Valida e normalizza `metric_value` secondo le regole cliniche di `diabetes_data`.

    Solleva `ValueError` se il valore manca, non è numerico o è fuori dai limiti biologici.
    """
    if raw_value is None:
        raise ValueError("Valore numerico mancante per il parametro clinico.")
    cleaned = str(raw_value).strip()
    if not cleaned:
        raise ValueError("Valore numerico vuoto.")

    # Rimuove eventuali unità di misura ripetute nel valore (il DB salva es. "118", "130/80", "6.8")
    cleaned_no_unit = re.sub(
        r"\s*(?:mg/dl|mmol/l|mmhg|ml/min(?:/1\.73m[²2])?|mg/g|%)\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    if metric_type == "PRESSIONE":
        m = _BP_RE.match(cleaned_no_unit)
        if not m:
            raise ValueError("La pressione deve essere nel formato Sistolica/Diastolica (es. '130/80').")
        sbp, dbp = int(m.group(1)), int(m.group(2))
        if not (50 <= sbp <= 280 and 30 <= dbp <= 180 and sbp > dbp):
            raise ValueError("Valori di pressione arteriosa fuori range fisiologico.")
        return f"{sbp}/{dbp}"

    num_match = _NUM_RE.fullmatch(cleaned_no_unit.replace(",", "."))
    if not num_match:
        raise ValueError(f"Il valore per {metric_type} deve essere un numero esplicito.")

    val = float(num_match.group(0))
    low, high = _METRIC_NUMERIC_BOUNDS[metric_type]
    if not (low <= val <= high):
        raise ValueError(f"Valore {val} per {metric_type} fuori dal range clinico ({low}-{high}).")

    # Formattazione pulita: intero per glicemia/colesterolo/trigliceridi se non ha decimali
    if val.is_integer() and metric_type != "HbA1c":
        return str(int(val))
    return f"{val:.1f}".rstrip("0").rstrip(".") if metric_type != "HbA1c" else f"{val:.1f}"


# ---------------------------------------------------------------------------
# 3. Modelli Payload Tipizzati per le Card di Conferma sul Front-End Flutter
# ---------------------------------------------------------------------------


class DiabetesDataProposal(BaseModel):
    """Payload per `public.diabetes_data` (Glicemia, Pressione, HbA1c, eGFR, UACR, Colesterolo, HDL, LDL, Trigliceridi)."""

    model_config = ConfigDict(extra="forbid")

    metric_type: CanonicalMetricType
    metric_value: str
    unit: str = ""
    measured_at: str  # ISO-8601 timestamptz
    source_report_id: int | None = None

    @model_validator(mode="after")
    def _validate_clinical_metric(self) -> "DiabetesDataProposal":
        self.metric_value = normalize_and_validate_metric_value(self.metric_type, self.metric_value)
        if not self.unit:
            self.unit = DEFAULT_METRIC_UNITS.get(self.metric_type, "")
        return self


class MealProposal(BaseModel):
    """Payload per `public.meals` (Diario alimentare con stime LARN dichiarate)."""

    model_config = ConfigDict(extra="forbid")

    meal_type: CanonicalMealType
    description: str = Field(min_length=1, max_length=500)
    carbs_grams: int | None = Field(default=None, ge=0, le=600)
    calories: int | None = Field(default=None, ge=0, le=5000)
    is_estimated: bool = True
    meal_at: str  # ISO-8601 timestamptz


class PhysicalActivityProposal(BaseModel):
    """Payload per `public.physical_activities` (Diario attività motoria)."""

    model_config = ConfigDict(extra="forbid")

    activity_type: CanonicalActivityType
    duration_minutes: int | None = Field(default=None, ge=1, le=720)
    intensity: CanonicalIntensityType = "MODERATA"
    steps: int | None = Field(default=None, ge=1, le=100000)
    notes: str | None = Field(default=None, max_length=500)
    activity_at: str  # ISO-8601 timestamptz


class AppointmentProposal(BaseModel):
    """Payload per `public.appointments` (Visite mediche e controlli)."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    doctor_name: str | None = Field(default=None, max_length=200)
    appointment_at: str | None = None  # ISO-8601 timestamptz (None se l'utente non ha detto la data)
    notes: str | None = Field(default=None, max_length=500)


class TherapyLogProposal(BaseModel):
    """Payload per `public.therapy_logs` (Registrazione assunzione farmaco)."""

    model_config = ConfigDict(extra="forbid")

    therapy_id: int | None = None
    drug_name_mentioned: str = Field(min_length=1, max_length=200)
    taken_at: str  # ISO-8601 timestamptz
    status: TherapyLogStatusEnum = "ASSUNTA"


class TherapyPlanProposal(BaseModel):
    """Payload per `public.therapies` (Segnalazione terapia prescritta dal medico)."""

    model_config = ConfigDict(extra="forbid")

    therapy_type: CanonicalTherapyType = "ALTRO"
    drug_name: str = Field(min_length=1, max_length=200)
    dosage: str | None = Field(default=None, max_length=150)
    frequency: str | None = Field(default=None, max_length=150)
    scheduled_time: str | None = Field(default=None, max_length=10)
    notes: str | None = Field(default=None, max_length=500)
    is_active: bool = True


class ProfileUpdateProposal(BaseModel):
    """Payload per `public.profile` (Aggiornamento peso e/o altezza)."""

    model_config = ConfigDict(extra="forbid")

    peso: float | None = Field(default=None, ge=20.0, le=350.0)
    altezza: int | None = Field(default=None, ge=100, le=250)

    @model_validator(mode="after")
    def _check_at_least_one(self) -> "ProfileUpdateProposal":
        if self.peso is None and self.altezza is None:
            raise ValueError("Almeno uno tra peso e altezza deve essere valorizzato.")
        return self


ProposalPayloadUnion = (
    DiabetesDataProposal
    | MealProposal
    | PhysicalActivityProposal
    | AppointmentProposal
    | TherapyLogProposal
    | TherapyPlanProposal
    | ProfileUpdateProposal
)


class ProposedEntry(BaseModel):
    """Singola proposta di salvataggio Human-in-the-Loop per il widget di conferma Flutter."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    target_table: TargetTableEnum
    summary_title: str
    is_complete: bool = True
    missing_fields: list[str] = Field(default_factory=list)
    assistant_note: str | None = None
    data: dict[str, Any]


class ReportExtractionResult(BaseModel):
    """Risultato dell'estrazione strutturata da un referto medico (testo o foto/PDF).

    Include sia i metadati per `public.patient_reports` (con `clean_summary` già
    privo di PII), sia la lista di `proposed_entries` (glicemia, HbA1c, colesterolo
    totale, HDL, LDL, trigliceridi, eGFR, UACR, pressione, visite di controllo)
    pronta per il widget di conferma sul Front-End Flutter.
    """

    model_config = ConfigDict(extra="forbid")

    report_type: CanonicalReportType = "ANALISI_SANGUE"
    report_at: str | None = None
    clean_summary: str = ""
    proposed_entries: list[ProposedEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. Schemi intermedi per Structured Outputs di Vertex AI (`response_schema`)
# ---------------------------------------------------------------------------


class RawExtractedItem(BaseModel):
    """Elemento grezzo restituito da Vertex AI in JSON mode."""

    target_table: TargetTableEnum
    summary_title: str = ""
    assistant_note: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    timestamp_iso: str | None = None

    # Campi per `diabetes_data` (profilo glicemico, lipidico, colesterolo, renale, pressione)
    metric_type: str | None = None
    metric_value: str | None = None

    # Campi per `meals`
    meal_type: str | None = None
    meal_description: str | None = None
    carbs_grams: int | None = None
    calories: int | None = None
    is_estimated: bool | None = None

    # Campi per `physical_activities`
    activity_type: str | None = None
    duration_minutes: int | None = None
    intensity: str | None = None
    steps: int | None = None

    # Campi per `appointments`
    appointment_title: str | None = None
    doctor_name: str | None = None
    has_explicit_time: bool | None = None

    # Campi per `therapy_logs` e `therapies`
    therapy_id: int | None = None
    drug_name: str | None = None
    therapy_type: str | None = None
    dosage: str | None = None
    frequency: str | None = None
    scheduled_time: str | None = None
    therapy_status: str | None = None

    # Campi per `profile`
    peso_kg: float | None = None
    altezza_cm: int | None = None

    # Note libere comuni
    notes: str | None = None


class LLMExtractionBatch(BaseModel):
    """Schema radice passato a `GenerateContentConfig.response_schema` per la chat/voce."""

    items: list[RawExtractedItem] = Field(default_factory=list)


class LLMReportExtractionBatch(BaseModel):
    """Schema radice passato a `GenerateContentConfig.response_schema` per i referti (testo o foto)."""

    report_type: str = "ANALISI_SANGUE"
    report_date_iso: str | None = None
    clean_summary: str = ""
    items: list[RawExtractedItem] = Field(default_factory=list)
