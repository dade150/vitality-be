# llm/base.py
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol


class LLMError(Exception):
    """Errore del provider (risposta non valida, config, 4xx permanenti...) -> HTTP 502.

    Regola: il messaggio NON contiene mai testo della conversazione né corpi di
    risposta del provider; solo un `code` breve e stabile.
    """

    def __init__(self, message: str = "Errore del provider LLM", *, code: str | None = None):
        super().__init__(message)
        self.code = code


class LLMUnavailable(LLMError):
    """Provider temporaneamente non disponibile (429/5xx/timeout/rete/credenziali) -> HTTP 503."""


class LLMBlocked(LLMError):
    """Risposta o prompt bloccati dai filtri di sicurezza del provider."""


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMChatResult:
    text: str
    tokens_used: int | None = None


class LLMProvider(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ) -> str: ...

    async def chat_with_usage(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ) -> LLMChatResult: ...

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ) -> AsyncIterator[str]: ...
