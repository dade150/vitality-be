# llm/base.py
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


class LLMProvider(Protocol):
    async def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str: ...
