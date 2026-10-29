import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from vitality_be.core.config import get_settings
from vitality_be.core.limits import (
    GateBusy,
    RateLimitExceeded,
    get_gate,
    get_rate_limiter,
)
from vitality_be.core.security import CurrentUserDep
from vitality_be.db.supabase1 import UserDbDep
from vitality_be.llm.base import ChatMessage, LLMError, LLMProvider, LLMUnavailable
from vitality_be.llm.embedder import EmbedderProvider, get_embedder
from vitality_be.llm.factory import get_llm
from vitality_be.schemas.chat import (
    AttachmentInfoOut,
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ChatSessionDetailResponse,
    ChatSessionListResponse,
    ChatSessionOut,
    UpdateSessionTitleIn,
)
from vitality_be.services import chat as chat_service
from vitality_be.services import chat_history
from vitality_be.services.chat_history import SessionNotFoundError
from vitality_be.services.multimodal import (
    MultimodalValidationError,
    process_and_sanitize_attachment,
)

router = APIRouter(tags=["chat"])


def enforce_rate_limit(user: CurrentUserDep) -> None:
    """Rate limit per user.id (non per IP: dietro proxy l'IP non è affidabile).
    Si applica dopo l'autenticazione: richieste anonime non consumano quote di nessuno."""
    try:
        get_rate_limiter().check(user.id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            429, "Troppe richieste. Riprova tra poco.", headers={"Retry-After": str(exc.retry_after)}
        ) from None


@router.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_unset=True,
    dependencies=[Depends(enforce_rate_limit)],
)
async def chat(
    body: ChatRequest,
    user: CurrentUserDep,
    user_db: UserDbDep,
    llm: Annotated[LLMProvider, Depends(get_llm)],
    embedder: Annotated[EmbedderProvider, Depends(get_embedder)],
):
    started = time.monotonic()
    s = get_settings()

    # 1. Validazione e sanitizzazione in RAM dell'eventuale allegato multimodale
    media_bytes: bytes | None = None
    media_mime: str | None = None
    attachment_kind = None
    attachment_meta: dict | None = None
    attachment_out: AttachmentInfoOut | None = None

    if body.attachment:
        try:
            processed = process_and_sanitize_attachment(
                data_base64=body.attachment.data_base64,
                mime_type=body.attachment.mime_type,
                kind=body.attachment.kind,
                max_bytes=s.max_attachment_bytes,
            )
            media_bytes = processed.clean_bytes
            media_mime = processed.mime_type
            attachment_kind = processed.kind
            attachment_meta = processed.to_rag_context_dict()
            attachment_out = AttachmentInfoOut(
                has_attachment=True,
                kind=processed.kind,
                mime_type=processed.mime_type,
                size_bytes=processed.size_bytes,
                exif_stripped=processed.exif_stripped,
                sha256_prefix=processed.sha256_prefix,
            )
        except MultimodalValidationError as exc:
            raise HTTPException(400, f"Allegato non valido: {exc}") from None

    # 2. Gestione sessione e caricamento cronologia server-side (se user_db è disponibile)
    session_id: int | None = body.session_id
    history_for_llm: list[ChatMessage] = []

    if hasattr(user_db, "table"):
        try:
            session_row = await chat_history.get_or_create_session(
                user_db,
                user_id=user.id,
                session_id=session_id,
                title_seed=body.message or "",
                attachment_kind=attachment_kind,
            )
            session_id = session_row.get("id")

            # Se il client ha inviato session_id (o creata nuova sessione) e non una lista legacy messages[]
            if session_id and not body.messages:
                history_for_llm = await chat_history.load_session_messages_for_llm(
                    user_db,
                    session_id=session_id,
                    limit=s.chat_history_window,
                )
            elif body.messages and len(body.messages) > 1:
                # Modalità legacy: usa la cronologia inviata dal client (escludendo l'ultimo turno già in body.message)
                history_for_llm = [ChatMessage(m.role, m.content) for m in body.messages[:-1]]
        except SessionNotFoundError:
            raise HTTPException(404, "Sessione di chat non trovata.") from None
        except Exception as exc:
            logger.warning("Recupero sessione degradato user={} exc={}", user.id, type(exc).__name__)
    elif body.messages and len(body.messages) > 1:
        history_for_llm = [ChatMessage(m.role, m.content) for m in body.messages[:-1]]

    user_text = body.message or ""
    logger.info("chat user={} session={} history_turns={}", user.id, session_id, len(history_for_llm))

    # 3. Esecuzione pipeline LLM Dual-RAG
    try:
        async with asyncio.timeout(s.chat_request_timeout_s):
            async with get_gate().slot():
                pipeline_res = await chat_service.run_chat_pipeline(
                    llm,
                    user_message=user_text,
                    history_messages=history_for_llm,
                    media_bytes=media_bytes,
                    media_mime_type=media_mime,
                    attachment_kind=attachment_kind,
                    user_db=user_db,
                    embedder=embedder,
                )
    except GateBusy:
        raise HTTPException(
            503, "Il servizio è molto richiesto. Riprova tra poco.", headers={"Retry-After": "5"}
        ) from None
    except TimeoutError:
        logger.warning("chat user={} timeout totale", user.id)
        raise HTTPException(503, "Il servizio di chat ci sta mettendo troppo. Riprova.") from None
    except LLMUnavailable as exc:
        logger.warning("chat user={} llm non disponibile exc={} code={}", user.id, type(exc).__name__, exc.code)
        raise HTTPException(503, "Il servizio di chat non è al momento disponibile. Riprova più tardi.") from None
    except LLMError as exc:
        logger.error("chat user={} llm errore exc={} code={}", user.id, type(exc).__name__, exc.code)
        raise HTTPException(502, "Il servizio di chat non è al momento disponibile.") from None

    # 4. Salvataggio della coppia di turni (user + assistant) nel database
    user_msg_id: int | None = None
    asst_msg_id: int | None = None
    if session_id and hasattr(user_db, "table"):
        proposals_dicts = [p.model_dump() for p in pipeline_res.proposed_entries]
        user_msg_id, asst_msg_id = await chat_history.save_chat_turn(
            user_db,
            session_id=session_id,
            user_content=user_text,
            assistant_content=pipeline_res.reply,
            attachment_meta=attachment_meta,
            rag_chunks=pipeline_res.rag_chunks,
            proposed_entries=proposals_dicts,
            tokens_used=pipeline_res.tokens_used,
        )

    logger.info(
        "chat user={} ok session={} proposals={} ms={}",
        user.id,
        session_id,
        len(pipeline_res.proposed_entries),
        int((time.monotonic() - started) * 1000),
    )

    response_kwargs: dict = {"reply": pipeline_res.reply}
    if pipeline_res.proposed_entries:
        response_kwargs["proposed_entries"] = pipeline_res.proposed_entries
    if session_id is not None:
        response_kwargs["session_id"] = session_id
    if user_msg_id is not None:
        response_kwargs["user_message_id"] = user_msg_id
    if asst_msg_id is not None:
        response_kwargs["assistant_message_id"] = asst_msg_id
    if attachment_out is not None:
        response_kwargs["attachment_info"] = attachment_out

    return ChatResponse(**response_kwargs)


@router.get("/chat/sessions", response_model=ChatSessionListResponse)
async def list_sessions(
    user: CurrentUserDep,
    user_db: UserDbDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Elenca le sessioni di chat dell'utente autenticato in ordine cronologico inverso."""
    rows = await chat_history.list_user_sessions(user_db, limit=limit, offset=offset)
    return ChatSessionListResponse(
        sessions=[ChatSessionOut(id=r["id"], title=r.get("title"), created_at=r["created_at"]) for r in rows]
    )


@router.get("/chat/sessions/{session_id}", response_model=ChatSessionDetailResponse)
async def get_session(
    session_id: int,
    user: CurrentUserDep,
    user_db: UserDbDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
):
    """Recupera i dettagli di una sessione e i messaggi scambiati (escludendo dettagli RAG interni)."""
    try:
        session_row, messages = await chat_history.get_session_with_messages(
            user_db, session_id=session_id, limit=limit
        )
    except SessionNotFoundError:
        raise HTTPException(404, "Sessione non trovata.") from None

    return ChatSessionDetailResponse(
        session=ChatSessionOut(
            id=session_row["id"],
            title=session_row.get("title"),
            created_at=session_row["created_at"],
        ),
        messages=[ChatMessageOut(**m) for m in messages],
    )


@router.patch("/chat/sessions/{session_id}", response_model=ChatSessionOut)
async def update_session_title(
    session_id: int,
    body: UpdateSessionTitleIn,
    user: CurrentUserDep,
    user_db: UserDbDep,
):
    """Aggiorna il titolo di una sessione."""
    try:
        updated = await chat_history.update_session_title(user_db, session_id=session_id, new_title=body.title)
    except SessionNotFoundError:
        raise HTTPException(404, "Sessione non trovata.") from None

    return ChatSessionOut(id=updated["id"], title=updated.get("title"), created_at=updated["created_at"])


@router.delete("/chat/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int,
    user: CurrentUserDep,
    user_db: UserDbDep,
):
    """Elimina una sessione e a cascata tutti i messaggi associati (GDPR Art. 17)."""
    try:
        await chat_history.delete_user_session(user_db, session_id=session_id)
    except SessionNotFoundError:
        raise HTTPException(404, "Sessione non trovata.") from None
    return None
