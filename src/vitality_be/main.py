from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.trustedhost import TrustedHostMiddleware

from vitality_be.api.router import api_router, public_router
from vitality_be.core.config import get_settings
from vitality_be.core.logging import configure_logging
from vitality_be.core.middleware import BodySizeLimitMiddleware, RequestContextMiddleware
from vitality_be.llm.embedder import get_embedder
from vitality_be.llm.factory import get_llm
from vitality_be.llm.prompts import get_chat_system_prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fail-fast: config non valida, prompt mancante o provider mal configurato -> il processo non parte
    s = get_settings()
    configure_logging(s.is_prod)
    get_chat_system_prompt()
    llm = get_llm()
    embedder = get_embedder()
    logger.info("Avvio env={} llm_provider={}", s.environment, s.llm_provider)
    yield
    if hasattr(embedder, "aclose"):
        await embedder.aclose()
    if hasattr(llm, "aclose"):
        await llm.aclose()


def create_app() -> FastAPI:
    s = get_settings()
    # /docs, /redoc, /openapi.json disattivati in prod (riduce la superficie esposta)
    docs = {} if not s.is_prod else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(lifespan=lifespan, **docs)

    # Nessun CORS: l'app è mobile. Se servirà, allowlist esplicita da config.
    # Ordine: l'ultimo aggiunto è il più esterno.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=s.max_body_bytes)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=s.allowed_hosts)
    app.add_middleware(RequestContextMiddleware)

    app.include_router(public_router)
    app.include_router(api_router)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Il FE si aspetta {"detail": str}. Il default di FastAPI restituirebbe una lista e
        # RIPETEREBBE l'input dell'utente (chat): qui solo i nomi dei campi.
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
        return JSONResponse(status_code=422,
                            content={"detail": "Richiesta non valida (campi: " + ", ".join(fields) + ")."})

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # solo il tipo: il messaggio dell'eccezione potrebbe contenere testo della chat
        logger.error("Fallita richiesta {} {} exc={}", request.method, request.url.path, type(exc).__name__)
        return JSONResponse(status_code=500,
                            content={"detail": "Si è verificato un errore interno. Riprova più tardi."})

    return app


app = create_app()
