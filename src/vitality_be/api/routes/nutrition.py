import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from vitality_be.core.limits import GateBusy, RateLimitExceeded, get_gate, get_rate_limiter
from vitality_be.core.security import CurrentUserDep
from vitality_be.db.supabase1 import UserDbDep
from vitality_be.llm.base import LLMProvider
from vitality_be.llm.factory import get_llm
from vitality_be.schemas.nutrition import MealEstimateRequest, MealEstimateResponse
from vitality_be.services.nutrition import estimate_meal_nutrition

router = APIRouter(tags=["nutrition"])


def enforce_rate_limit(user: CurrentUserDep) -> None:
    """Rate limit per user.id."""
    try:
        get_rate_limiter().check(user.id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            429, "Troppe richieste. Riprova tra poco.", headers={"Retry-After": str(exc.retry_after)}
        ) from None


@router.post(
    "/nutrition/estimate-meal",
    response_model=MealEstimateResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def estimate_meal(
    body: MealEstimateRequest,
    user: CurrentUserDep,
    user_db: UserDbDep,
    llm: Annotated[LLMProvider, Depends(get_llm)],
):
    """Stima nutrizionale intelligente di un pasto basata su tabelle ufficiali CREA e Gemini."""
    try:
        async with asyncio.timeout(20.0):
            async with get_gate().slot():
                return await estimate_meal_nutrition(body, user_db=user_db, llm=llm)
    except GateBusy:
        raise HTTPException(
            503, "Il servizio è molto richiesto. Riprova tra poco.", headers={"Retry-After": "5"}
        ) from None
    except TimeoutError:
        logger.warning("nutrition estimate-meal user={} timeout", user.id)
        raise HTTPException(503, "Il servizio di stima nutrizionale ci sta mettendo troppo. Riprova.") from None
    except Exception as exc:
        logger.error("nutrition estimate-meal errore non gestito user={} exc={}", user.id, type(exc).__name__)
        raise HTTPException(500, "Errore interno durante il calcolo nutrizionale.") from None
