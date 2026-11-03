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


class NonMedicalDocumentError(ValueError):
    """Sollevata quando il file caricato non è un referto medico valido o è illeggibile."""


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
    user_id: str | None = None,
    filename: str | None = None,
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

    if not extraction.is_medical_report or (
        not extraction.proposed_entries
        and not (extraction.clean_summary or "").strip()
        and not (extraction.scrubbed_content or "").strip()
    ):
        logger.info("Upload referto rifiutato: documento non sanitario o illeggibile")
        raise NonMedicalDocumentError(
            "Il documento caricato non sembra essere un referto medico leggibile. "
            "Assicurati che l'immagine sia nitida e contenga valori clinici."
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

    # Step 4 bis: Archiviazione su Supabase Storage (bucket privato patient-reports) se disponibile
    saved_file_path: str | None = None
    if user_id and hasattr(user_db, "storage") and attachment.clean_bytes:
        ext_map = {
            "application/pdf": "pdf",
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }
        ext = ext_map.get(attachment.mime_type, "pdf")
        base_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", clean_title)[:40].strip("_") or "referto"
        storage_path = f"{user_id}/{int(time.time() * 1000)}_{base_slug}.{ext}"
        try:
            await _run_sync_or_thread(
                lambda: user_db.storage.from_("patient-reports").upload(
                    storage_path,
                    attachment.clean_bytes,
                    {"content-type": attachment.mime_type, "upsert": "true"},
                )
            )
            saved_file_path = storage_path
        except Exception as st_exc:
            logger.debug("Archiviazione Storage opzionale non disponibile exc={}", type(st_exc).__name__)

    # Step 4 ter: Persistenza su Supabase (public.patient_reports) con RLS SECURITY INVOKER
    report_id_str = f"rep_{int(time.time() * 1000)}"
    if hasattr(user_db, "table"):
        resolved_uid = user_id
        if not resolved_uid:
            try:
                u_resp = await _run_sync_or_thread(
                    lambda: user_db.table("user").select("id").limit(1).execute()
                )
                u_rows = getattr(u_resp, "data", None)
                if u_rows and isinstance(u_rows, list) and len(u_rows) > 0 and u_rows[0].get("id"):
                    resolved_uid = str(u_rows[0]["id"])
            except Exception:
                resolved_uid = None

        row_payload: dict[str, Any] = {
            "source_name": clean_title,
            "report_type": extraction.report_type,
            "report_at": report_at,
            "content": extraction.scrubbed_content or extraction.clean_summary or clean_title,
            "clean_summary": extraction.clean_summary or None,
            "extracted_params": extracted_params_map,
        }
        if resolved_uid:
            row_payload["user_id"] = resolved_uid
        if saved_file_path:
            row_payload["file_path"] = saved_file_path
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
            logger.warning(
                "Salvataggio esteso referto fallito ({}): tentativo senza embedding",
                db_exc,
            )
            no_vec_payload = {k: v for k, v in row_payload.items() if k != "embedding"}
            try:
                insert_resp = await _run_sync_or_thread(
                    lambda: user_db.table("patient_reports").insert(no_vec_payload).execute()
                )
                inserted_rows = getattr(insert_resp, "data", None)
                if inserted_rows and isinstance(inserted_rows, list) and len(inserted_rows) > 0:
                    real_id = inserted_rows[0].get("id")
                    if real_id is not None:
                        report_id_str = str(real_id)
            except Exception:
                fallback_payload: dict[str, Any] = {
                    "source_name": clean_title,
                    "report_type": "ANALISI_SANGUE",
                    "report_at": report_at,
                    "content": extraction.scrubbed_content or extraction.clean_summary or clean_title,
                }
                if resolved_uid:
                    fallback_payload["user_id"] = resolved_uid
                try:
                    insert_resp = await _run_sync_or_thread(
                        lambda: user_db.table("patient_reports").insert(fallback_payload).execute()
                    )
                    inserted_rows = getattr(insert_resp, "data", None)
                    if inserted_rows and isinstance(inserted_rows, list) and len(inserted_rows) > 0:
                        real_id = inserted_rows[0].get("id")
                        if real_id is not None:
                            report_id_str = str(real_id)
                except Exception as fb_exc:
                    logger.error("Salvataggio referto su DB fallito anche in fallback exc={}: {}", type(fb_exc).__name__, fb_exc)

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
        file_path=saved_file_path,
        report_type=extraction.report_type,
        report_at=report_at,
        clean_summary=extraction.clean_summary,
        extraction_method=extraction.extraction_method,
        extracted=extracted_items,
        proposed_entries=final_proposals,
    )


def render_single_report_pdf(row: dict[str, Any]) -> bytes:
    """Genera un PDF consultabile con PyMuPDF a partire dai dati salvati in `patient_reports`."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    title = str(row.get("source_name") or "Referto medico").strip()
    rep_type = str(row.get("report_type") or "ANALISI_SANGUE").replace("_", " ")
    raw_date = str(row.get("report_at") or row.get("created_at") or "")[:10]
    summary = str(row.get("clean_summary") or "").strip()
    content = str(row.get("content") or "").strip()
    params = row.get("extracted_params") if isinstance(row.get("extracted_params"), dict) else {}

    page.draw_rect(fitz.Rect(40, 35, 555, 90), color=(0.9, 0.32, 0.0), fill=(1.0, 0.96, 0.92), width=1)
    page.insert_text(fitz.Point(55, 60), f"REFERTO CLINICO: {title.upper()}", fontsize=11, fontname="helv", color=(0.8, 0.25, 0.0))
    page.insert_text(fitz.Point(55, 77), f"Tipologia: {rep_type}   |   Data: {raw_date}", fontsize=9, fontname="helv", color=(0.3, 0.3, 0.3))

    y = 115
    if summary:
        page.insert_text(fitz.Point(40, y), "SINTESI CLINICA:", fontsize=10, fontname="helv", color=(0.15, 0.15, 0.15))
        y += 8
        rect = fitz.Rect(40, y, 555, y + 80)
        page.draw_rect(rect, color=(0.85, 0.85, 0.85), fill=(0.98, 0.98, 0.98), width=0.5)
        page.insert_textbox(fitz.Rect(48, y + 6, 547, y + 74), summary, fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 95

    if params:
        page.insert_text(fitz.Point(40, y), "PARAMETRI RILEVATI:", fontsize=10, fontname="helv", color=(0.15, 0.15, 0.15))
        y += 14
        for k, v in params.items():
            page.insert_text(fitz.Point(50, y), f"- {k}: {v}", fontsize=9, fontname="helv", color=(0.1, 0.1, 0.1))
            y += 15
        y += 10

    if content:
        page.insert_text(fitz.Point(40, y), "CONTENUTO DEL DOCUMENTO:", fontsize=10, fontname="helv", color=(0.15, 0.15, 0.15))
        y += 8
        rem_height = max(120, 800 - y)
        rect = fitz.Rect(40, y, 555, y + rem_height)
        page.draw_rect(rect, color=(0.85, 0.85, 0.85), fill=(0.99, 0.99, 0.99), width=0.5)
        page.insert_textbox(fitz.Rect(48, y + 8, 547, y + rem_height - 8), content[:3500], fontsize=8.5, fontname="helv", color=(0.2, 0.2, 0.2))

    return doc.tobytes()

