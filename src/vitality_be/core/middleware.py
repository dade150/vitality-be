"""Middleware ASGI puri (niente BaseHTTPMiddleware: non bufferizzano il body)."""
import json
import uuid

from fastapi import HTTPException
from loguru import logger
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestContextMiddleware:
    """Request-id per richiesta (nei log e nell'header X-Request-ID) + header di sicurezza.
    Minaccia: dati sanitari in cache di proxy/browser (Cache-Control), sniffing, leak del referrer."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        rid = uuid.uuid4().hex[:12]
        is_api = scope["path"].startswith("/api")

        async def send_wrapper(message: Message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-request-id", rid.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                ]
                if is_api:
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        with logger.contextualize(request_id=rid):
            await self.app(scope, receive, send_wrapper)


class BodySizeLimitMiddleware:
    """Minaccia: body enormi (memoria/CPU). Controlla Content-Length E conta i byte del flusso
    (chunked o Content-Length falso). Oltre il limite -> 413 {"detail": ...}."""

    def __init__(self, app: ASGIApp, max_bytes: int, chat_max_bytes: int | None = None):
        self.app, self.max_bytes = app, max_bytes
        self.chat_max_bytes = chat_max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        method = scope.get("method", "")
        limit = (
            self.chat_max_bytes
            if (self.chat_max_bytes is not None and method == "POST" and path in ("/api/chat", "/api/chat/stream"))
            else self.max_bytes
        )

        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    too_big = int(value) > limit
                except ValueError:
                    too_big = True
                if too_big:
                    return await self._reject(send)

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    # HTTPException: FastAPI la lascia passare e diventa un 413 pulito
                    raise HTTPException(413, "Richiesta troppo grande.")
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(send: Send):
        body = json.dumps({"detail": "Richiesta troppo grande."}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"cache-control", b"no-store"),
                                (b"x-content-type-options", b"nosniff")]})
        await send({"type": "http.response.body", "body": body})
