# llm/base.py
from dataclasses import dataclass
from typing import Literal, Protocol


class LLMError(Exception):
    """Errore del provider (rete, quota, risposta bloccata...)."""


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


class LLMProvider(Protocol):
    async def chat(self, messages: list[ChatMessage]) -> str: ...