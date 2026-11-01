"""Schemi Pydantic v2 per il caricamento, parsing e ingestion dei referti clinici."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vitality_be.schemas.extraction import ProposedEntry


class ExtractedParamItem(BaseModel):
    """Singolo parametro clinico estratto per la visualizzazione/conferma nel Front-End."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="Codice parametro clinico (es. HBA1C, GLICEMIA, LDL, EGFR)")
    value: float = Field(..., description="Valore numerico normalizzato")


class ReportUploadResponse(BaseModel):
    """Risposta del backend all'upload di un referto (conforme a ReportService in Flutter)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="ID del referto salvato in public.patient_reports")
    file_path: str | None = Field(None, description="Path del file se salvato su Supabase Storage")
    report_type: str = Field("ANALISI_SANGUE", description="Tipologia referto da public.report_enum")
    report_at: str | None = Field(None, description="Data dell'esame in formato ISO")
    clean_summary: str = Field("", description="Sintesi clinica oggettiva de-identificata")
    extraction_method: Literal["pymupdf_native", "gemini_vision", "text"] = Field(
        "pymupdf_native", description="Metodo usato per estrarre le informazioni"
    )
    extracted: list[ExtractedParamItem] = Field(
        default_factory=list, description="Lista parametri per retrocompatibilità AddReportScreen"
    )
    proposed_entries: list[ProposedEntry] = Field(
        default_factory=list, description="Proposte complete Human-in-the-Loop per il salvataggio"
    )
