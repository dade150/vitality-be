"""Router FastAPI per l'upload, estrazione e ingestion dei referti clinici (POST /api/reports)."""
from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger

from vitality_be.core.config import get_settings
from vitality_be.core.limits import GateBusy, RateLimitExceeded, get_gate, get_rate_limiter
from vitality_be.core.security import CurrentUserDep
from vitality_be.db.supabase1 import UserDbDep
from vitality_be.llm.base import LLMError, LLMProvider, LLMUnavailable
from vitality_be.llm.embedder import EmbedderProvider, get_embedder
from vitality_be.llm.factory import get_llm
from vitality_be.schemas.reports import ReportUploadResponse
from vitality_be.services.multimodal import MultimodalValidationError, process_raw_attachment_bytes
from vitality_be.services.reports import process_and_ingest_report

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
    dependencies=[Depends(enforce_rate_limit)],
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

    Supporta PDF digitali nativi (PyMuPDF in RAM, zero rete), PDF scansionati e foto tramite Vertex AI EU.
    I dati identificativi (GDPR Art. 9) vengono sempre rimossi prima del salvataggio nel Vector DB.
    """
    s = get_settings()

    if not file.content_type:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Tipo MIME del file mancante.")

    try:
        raw_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Impossibile leggere il file caricato: {exc}") from None

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
                )
                return result
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
