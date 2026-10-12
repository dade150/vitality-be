from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from vitality_be.core.config import get_settings
from vitality_be.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()  # fail-fast: se manca una env il processo non parte
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(api_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Fallita richiesta {request.method} {request.url.path}")
    return JSONResponse(500, {"detail": "Si è verificato un errore interno. Riprova più tardi."})