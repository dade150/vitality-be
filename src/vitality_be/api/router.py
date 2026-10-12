from fastapi import APIRouter
from vitality_be.api.routes import me
from vitality_be.api.routes import chat, me

api_router = APIRouter(prefix="/api")
api_router.include_router(me.router)
api_router.include_router(chat.router)