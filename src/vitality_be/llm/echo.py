# llm/echo.py
from vitality_be.llm.base import ChatMessage


class EchoProvider:
    """Provider finto per sviluppo/test: nessuna credenziale, nessuna rete."""

    async def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str:
        return f"[echo] {messages[-1].content}"

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        response_schema: type,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ):
        return response_schema()
