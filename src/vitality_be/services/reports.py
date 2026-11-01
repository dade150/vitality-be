"""Servizio di Ingestion e Vettorizzazione Referti Medici (Pipeline a 4 Step).

Step 1: Ispezione PDF in RAM (PyMuPDF): estrazione testo/tabelle vettoriali a costo 0 e zero rete.
Step 2: De-identificazione & Scrubbing PII (GDPR Art. 9): rimozione CF, intestazioni anagrafiche, nomi e strutture sanitarie.
Step 3: Estrazione Strutturata (Pydantic v2): estrazione parametri ematochimici e sintesi de-identificata.
Step 4: Embedding Vettoriale (Gemini Embedding 2 a 1536 dim) e persistenza in public.patient_reports sotto RLS.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
import re
import time
from typing import Any

from loguru import logger
from supabase import Client

from vitality_be.llm.base import LLMProvider
from vitality_be.llm.embedder import EmbedderProvider
from vitality_be.schemas.extraction import ProposedEntry, ReportExtractionResult
from vitality_be.schemas.reports import ExtractedParamItem, ReportUploadResponse
from vitality_be.services.extractor import extract_from_report
from vitality_be.services.multimodal import ProcessedAttachment
from vitality_be.services.patient_context import (
    PatientMaskingMap,
    mask_user_text_with_dynamic_tokens,
    scrub_report_text,
)
from vitality_be.services.retriever import vector_literal

# Mapping per normalizzare i tipi di parametro da ProposedEntry verso ExtractedParamItem (codici DB)
_METRIC_TO_CODE = {
    "GLICEMIA": "GLICEMIA",
    "HBA1C": "HBA1C",
    "HbA1c": "HBA1C",
    "EGFR": "EGFR",
    "eGFR": "EGFR",
    "UACR": "UACR",
    "COLESTEROLO_TOTALE": "COLESTEROLO_TOTALE",
    "Colesterolo Totale": "COLESTEROLO_TOTALE",
    "HDL": "HDL",
    "LDL": "LDL",
    "TRIGLICERIDI": "TRIGLICERIDI",
    "Trigliceridi": "TRIGLICERIDI",
}


def _extract_number(val: Any) -> float | None:
    if val is None:
        return None
    cleaned = str(val).strip().replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


async def _run_sync_or_thread(func):
    """Esegue I/O sincrono in thread pool se c'è un loop asincrono attivo, o direttamente (es. test sincroni)."""
    try:
        return await asyncio.to_thread(func)
    except RuntimeError:
        return func()


async def _resolve_user_masking_map(user_db: Client, existing_map: PatientMaskingMap | None) -> PatientMaskingMap:
    """Recupera nome e cognome dell'utente per garantire lo scrubbing dei dati anagrafici dal PDF."""
    if existing_map and existing_map.first_name:
        return existing_map

    first_name = None
    last_name = None
    if hasattr(user_db, "table"):
        try:
            resp = await _run_sync_or_thread(
                lambda: user_db.table("user").select("name, surname").limit(1).execute()
            )
            data = getattr(resp, "data", None)
            if data and isinstance(data, list) and len(data) > 0:
                first_name = str(data[0].get("name") or "").strip() or None
                last_name = str(data[0].get("surname") or "").strip() or None
        except Exception as exc:
            logger.debug("Recupero nome utente per masking map ignorato exc={}", type(exc).__name__)

    return PatientMaskingMap(first_name=first_name, last_name=last_name)


def _build_embedding_document_text(
    clean_summary: str,
    extracted_params: dict[str, float],
    scrubbed_content: str,
) -> str:
    """Costruisce il payload testuale de-identificato per il calcolo dell'embedding a 1536 dimensioni."""
    parts: list[str] = []
    if clean_summary:
        parts.append(f"Sintesi clinica: {clean_summary.strip()}")

    if extracted_params:
        params_str = ", ".join(f"{k}: {v}" for k, v in extracted_params.items())
        parts.append(f"Parametri di laboratorio rilevati: {params_str}")

    if scrubbed_content:
        # Tronca a massimo 3500 caratteri per stare comodamente nel contesto di embedding
        clean_excerpt = scrubbed_content.strip()[:3500]
        parts.append(f"Estratto clinico de-identificato:\n{clean_excerpt}")

    return "\n\n".join(parts).strip()


async def process_and_ingest_report(
    *,
    attachment: ProcessedAttachment,
    title: str | None = None,
    report_date_hint: str | None = None,
    user_db: Client,
    llm: LLMProvider,
    embedder: EmbedderProvider,
    masking_map: PatientMaskingMap | None = None,
    active_therapies: Sequence[dict[str, Any]] = (),
) -> ReportUploadResponse:
    """Esegue la pipeline end-to-end a 4 step per l'analisi e l'ingestion vettoriale del referto."""
    started = time.monotonic()
    clean_title = (title or "Referto medico").strip()
    if not clean_title:
        clean_title = "Referto medico"

    # Step 2 prep: assicura che il nome/cognome del paziente siano presenti nella mappa di de-identificazione
    resolved_map = await _resolve_user_masking_map(user_db, masking_map)

    # Step 1, 2, 3: Ispezione in RAM, de-identificazione PII ed estrazione strutturata
    extraction: ReportExtractionResult = await extract_from_report(
        llm,
        media_bytes=attachment.clean_bytes,
        media_mime_type=attachment.mime_type,
        masking_map=resolved_map,
        active_therapies=active_therapies,
    )

    # Normalizza data referto: usa il suggerimento esplicito se fornito, altrimenti la data estratta
    report_at = report_date_hint or extraction.report_at
    if not report_at:
        report_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Mappa e lista dei parametri estratti
    extracted_params_map: dict[str, float] = {}
    extracted_items: list[ExtractedParamItem] = []

    for prop in extraction.proposed_entries:
        if prop.target_table == "diabetes_data":
            m_type = str(prop.data.get("metric_type") or "").strip()
            canon_code = _METRIC_TO_CODE.get(m_type, m_type.upper().replace(" ", "_"))
            num_val = _extract_number(prop.data.get("metric_value"))
            if num_val is not None:
                extracted_params_map[canon_code] = num_val
                extracted_items.append(ExtractedParamItem(type=canon_code, value=num_val))

    # Step 4: Calcolo dell'embedding a 1536 dimensioni sul testo clinico de-identificato
    text_for_embedding = _build_embedding_document_text(
        clean_summary=extraction.clean_summary,
        extracted_params=extracted_params_map,
        scrubbed_content=extraction.scrubbed_content,
    )

    vector: list[float] | None = None
    if text_for_embedding and hasattr(embedder, "embed_document"):
        try:
            vector = await embedder.embed_document(title=clean_title, text=text_for_embedding)
        except Exception as embed_exc:
            logger.warning(
                "Calcolo embedding referto fallito (degradazione graziosa) exc={}",
                type(embed_exc).__name__,
            )
            vector = None

    # Step 4 bis: Persistenza su Supabase (public.patient_reports) con RLS SECURITY INVOKER
    report_id_str = f"rep_{int(time.time() * 1000)}"
    if hasattr(user_db, "table"):
        row_payload: dict[str, Any] = {
            "source_name": clean_title,
            "report_type": extraction.report_type,
            "report_at": report_at,
            "content": extraction.scrubbed_content or extraction.clean_summary or clean_title,
            "clean_summary": extraction.clean_summary or None,
            "extracted_params": extracted_params_map,
        }
        if vector is not None:
            row_payload["embedding"] = vector_literal(vector)

        try:
            insert_resp = await _run_sync_or_thread(
                lambda: user_db.table("patient_reports").insert(row_payload).execute()
            )
            inserted_rows = getattr(insert_resp, "data", None)
            if inserted_rows and isinstance(inserted_rows, list) and len(inserted_rows) > 0:
                real_id = inserted_rows[0].get("id")
                if real_id is not None:
                    report_id_str = str(real_id)
        except Exception as db_exc:
            logger.error("Salvataggio referto su DB fallito exc={}", type(db_exc).__name__)

    # Collega source_report_id nelle proposte emesse
    resolved_report_id_int: int | None = None
    try:
        resolved_report_id_int = int(report_id_str)
    except (ValueError, TypeError):
        pass

    final_proposals: list[ProposedEntry] = []
    for prop in extraction.proposed_entries:
        if resolved_report_id_int is not None and prop.target_table == "diabetes_data":
            data_copy = dict(prop.data)
            data_copy["source_report_id"] = resolved_report_id_int
            final_proposals.append(prop.model_copy(update={"data": data_copy}))
        else:
            final_proposals.append(prop)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "process_and_ingest_report completato id={} method={} params={} embedded={} ms={}",
        report_id_str,
        extraction.extraction_method,
        len(extracted_items),
        vector is not None,
        elapsed_ms,
    )

    return ReportUploadResponse(
        id=report_id_str,
        file_path=None,
        report_type=extraction.report_type,
        report_at=report_at,
        clean_summary=extraction.clean_summary,
        extraction_method=extraction.extraction_method,
        extracted=extracted_items,
        proposed_entries=final_proposals,
    )
