from fastapi import APIRouter, Depends

from vitality_be.api.routes import chat, health, me, nutrition
from vitality_be.core.security import get_current_user

# Pubblico: SOLO health, fuori dal router protetto.
public_router = APIRouter(prefix="/api")
public_router.include_router(health.router)

# Protetto: l'auth è a livello di router, un endpoint dimenticato non resta aperto.
api_router = APIRouter(prefix="/api", dependencies=[Depends(get_current_user)])
api_router.include_router(me.router)
api_router.include_router(chat.router)
api_router.include_router(nutrition.router)
