# llm/echo.py
from vitality_be.llm.base import ChatMessage


class EchoProvider:
    """Provider finto per sviluppo/test: nessuna credenziale, nessuna rete."""

    async def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str:
        return f"[echo] {messages[-1].content}"
