import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from vitality_be.core.config import get_settings
from vitality_be.core.limits import (
    GateBusy,
    RateLimitExceeded,
    get_gate,
    get_rate_limiter,
)
from vitality_be.core.security import CurrentUserDep
from vitality_be.llm.base import LLMError, LLMProvider, LLMUnavailable
from vitality_be.llm.factory import get_llm
from vitality_be.schemas.chat import ChatRequest, ChatResponse
from vitality_be.services import chat as chat_service

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


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
async def chat(
    body: ChatRequest,
    user: CurrentUserDep,
    llm: Annotated[LLMProvider, Depends(get_llm)],
):
    started = time.monotonic()
    # mai il contenuto: solo id utente e numero di turni
    logger.info("chat user={} turns={}", user.id, len(body.messages))
    try:
        # Scadenza TOTALE (coda + provider) < 60 s del FE
        async with asyncio.timeout(get_settings().chat_request_timeout_s):
            async with get_gate().slot():
                text = await chat_service.reply(llm, body.messages)
    except GateBusy:
        raise HTTPException(503, "Il servizio è molto richiesto. Riprova tra poco.",
                            headers={"Retry-After": "5"}) from None
    except TimeoutError:
        logger.warning("chat user={} timeout totale", user.id)
        raise HTTPException(503, "Il servizio di chat ci sta mettendo troppo. Riprova.") from None
    except LLMUnavailable as exc:
        logger.warning("chat user={} llm non disponibile exc={} code={}", user.id, type(exc).__name__, exc.code)
        raise HTTPException(503, "Il servizio di chat non è al momento disponibile. Riprova più tardi.") from None
    except LLMError as exc:
        # niente logger.exception: la traceback potrebbe contenere il corpo della risposta del provider
        logger.error("chat user={} llm errore exc={} code={}", user.id, type(exc).__name__, exc.code)
        raise HTTPException(502, "Il servizio di chat non è al momento disponibile.") from None

    logger.info("chat user={} ok ms={}", user.id, int((time.monotonic() - started) * 1000))
    return ChatResponse(reply=text)
