from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from vitality_be.core.security import CurrentUserDep
from vitality_be.llm.base import LLMError, LLMProvider
from vitality_be.llm.factory import get_llm
from vitality_be.schemas.chat import ChatRequest, ChatResponse
from vitality_be.services import chat as chat_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: CurrentUserDep,
    llm: Annotated[LLMProvider, Depends(get_llm)],
):
    logger.info("chat user={} turns={}", user.id, len(body.messages))  # mai il contenuto
    try:
        text = await chat_service.reply(llm, body.messages)
    except LLMError:
        logger.exception("Errore provider LLM")
        raise HTTPException(502, "Il servizio di chat non è al momento disponibile.")
    return ChatResponse(reply=text)