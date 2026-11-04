"""Router FastAPI per l'upload, estrazione e ingestion dei referti clinici (POST /api/reports)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from loguru import logger

from vitality_be.core.config import get_settings
from vitality_be.core.consent import require_ai_health_consent
from vitality_be.core.limits import GateBusy, RateLimitExceeded, get_gate, get_rate_limiter
from vitality_be.core.security import CurrentUserDep
from vitality_be.db.supabase1 import UserDbDep
from vitality_be.llm.base import LLMError, LLMProvider, LLMUnavailable
from vitality_be.llm.embedder import EmbedderProvider, get_embedder
from vitality_be.llm.factory import get_llm
from vitality_be.schemas.reports import ReportUploadResponse
from vitality_be.services.multimodal import MultimodalValidationError, process_raw_attachment_bytes
from vitality_be.services.pdf_summary import generate_patient_summary_pdf
from vitality_be.services.reports import (
    NonMedicalDocumentError,
    process_and_ingest_report,
    render_single_report_pdf,
)

router = APIRouter(tags=["reports"])


def enforce_rate_limit(user: CurrentUserDep) -> None:
    try:
        get_rate_limiter().check(user.id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Troppe richieste. Riprova tra poco.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from None


@router.post(
    "/reports",
    response_model=ReportUploadResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(enforce_rate_limit), Depends(require_ai_health_consent)],
)
async def upload_report(
    user: CurrentUserDep,
    user_db: UserDbDep,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    embedder: Annotated[EmbedderProvider, Depends(get_embedder)],
    file: UploadFile = File(..., description="File PDF o immagine del referto"),
    title: str = Form(default="Referto medico", description="Titolo o descrizione del referto"),
    report_date: str | None = Form(default=None, description="Data dell'esame (YYYY-MM-DD o ISO)"),
) -> ReportUploadResponse:
    """Caricamento di un referto clinico con estrazione strutturata dei parametri e ingestion vettoriale.

    - PDF digitali nativi: testo e tabelle estratti in memoria RAM con PyMuPDF, con scrubbing PII (GDPR Art. 9)
      prima dell'invio al modello linguistico (zero byte binari inviati).
    - PDF scansionati e foto: inoltrati in modalità multimodale protetta a Vertex AI (regione Unione Europea).
    - Storage privato: una copia del documento (con metadati EXIF/GPS rimossi) viene conservata nel bucket privato
      Supabase 'patient-reports' isolato da RLS (auth.uid()), per consentirne la consultazione e il download al paziente.
    - Vector DB: nel database vettoriale e nei dati clinici di testo viene memorizzato esclusivamente il contenuto
      clinico de-identificato e la sintesi oggettiva.
    """
    s = get_settings()

    if not file.content_type:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Tipo MIME del file mancante.")

    try:
        raw_bytes = await file.read()
    except Exception as exc:
        logger.warning("Upload referto read fallito user={} exc={}", user.id, type(exc).__name__)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Impossibile leggere il file caricato. Verifica il documento e riprova.",
        ) from None

    if not raw_bytes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Il file inviato è vuoto.")

    try:
        processed = process_raw_attachment_bytes(
            raw_bytes=raw_bytes,
            mime_type=file.content_type,
            kind="medical_report",
            max_bytes=s.max_attachment_bytes,
        )
    except MultimodalValidationError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Documento non valido: {exc}") from None

    try:
        async with asyncio.timeout(s.chat_request_timeout_s):
            async with get_gate().slot():
                result = await process_and_ingest_report(
                    attachment=processed,
                    title=title,
                    report_date_hint=report_date,
                    user_db=user_db,
                    llm=llm,
                    embedder=embedder,
                    user_id=user.id,
                    filename=file.filename,
                )
                return result
    except NonMedicalDocumentError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            str(exc),
        ) from None
    except GateBusy:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Il servizio è molto richiesto. Riprova tra poco.",
            headers={"Retry-After": "5"},
        ) from None
    except TimeoutError:
        logger.warning("upload_report user={} timeout elaborazione", user.id)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "L'elaborazione del referto ha richiesto troppo tempo. Riprova.",
        ) from None
    except (LLMUnavailable, LLMError) as exc:
        logger.error("upload_report user={} errore LLM exc={} code={}", user.id, type(exc).__name__, exc.code)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Servizio di comprensione referti momentaneamente non disponibile.",
        ) from None


@router.get(
    "/reports/summary-pdf",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(enforce_rate_limit)],
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Report clinico PDF multi-mensile per il medico curante o diabetologo",
        }
    },
)
async def export_summary_pdf(
    user: CurrentUserDep,
    user_db: UserDbDep,
    months: Annotated[int, Query(ge=2, le=4, description="Numero di mesi da includere (2, 3 o 4)")] = 3,
) -> Response:
    """Genera ed esporta il report clinico multi-mensile in PDF per il medico.

    Include grafici settimanali aggregati di glicemia e pressione con note fuori target,
    diario pasti e attività registrate, ultimi 3 esami ematochimici per profilo e
    sintesi degli ultimi referti differenziati per tipologia di visita.
    Nessun dato transita verso Vertex AI (generazione deterministica in RAM).
    """
    try:
        pdf_bytes = await generate_patient_summary_pdf(user_db=user_db, months=months)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None
    except Exception as exc:
        logger.error("export_summary_pdf user={} errore generazione exc={}", user.id, type(exc).__name__)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Impossibile generare il report PDF riepilogativo. Riprova più tardi.",
        ) from None

    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    filename = f"Report_Clinico_{months}M_{today_str}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/reports/{report_id}/document",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(enforce_rate_limit)],
)
async def download_report_document(
    report_id: int,
    user: CurrentUserDep,
    user_db: UserDbDep,
) -> Response:
    """Restituisce il file originale del referto da Storage o ne genera il PDF dai dati salvati."""
    try:
        resp = await asyncio.to_thread(
            lambda: user_db.table("patient_reports").select("*").eq("id", report_id).limit(1).execute()
        )
        rows = getattr(resp, "data", None)
    except Exception as exc:
        logger.error("download_report_document query fallita user={} exc={}", user.id, type(exc).__name__)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Impossibile recuperare il referto.") from None

    if not rows or not isinstance(rows, list):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Referto non trovato.")

    row = rows[0]
    file_path = row.get("file_path")

    # 1. Prova a scaricare il file originale da Supabase Storage se presente
    if file_path and hasattr(user_db, "storage"):
        try:
            file_bytes = await asyncio.to_thread(
                lambda: user_db.storage.from_("patient-reports").download(str(file_path))
            )
            if file_bytes:
                lower_path = str(file_path).lower()
                if lower_path.endswith(".png"):
                    mime = "image/png"
                    ext = "png"
                elif lower_path.endswith((".jpg", ".jpeg")):
                    mime = "image/jpeg"
                    ext = "jpg"
                elif lower_path.endswith(".webp"):
                    mime = "image/webp"
                    ext = "webp"
                else:
                    mime = "application/pdf"
                    ext = "pdf"
                return Response(
                    content=file_bytes,
                    media_type=mime,
                    headers={
                        "Content-Disposition": f'inline; filename="referto_{report_id}.{ext}"',
                        "Cache-Control": "no-store, no-cache, must-revalidate",
                        "X-Content-Type-Options": "nosniff",
                    },
                )
        except Exception as st_exc:
            logger.debug("Download da Storage non riuscito, genero PDF di fallback exc={}", type(st_exc).__name__)

    # 2. Fallback: genera al volo un PDF pulito con i dati clinici del referto
    pdf_bytes = await asyncio.to_thread(render_single_report_pdf, row)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="referto_{report_id}.pdf"',
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )

