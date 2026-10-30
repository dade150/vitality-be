# llm/echo.py
from collections.abc import AsyncIterator
from vitality_be.llm.base import ChatMessage, LLMChatResult


class EchoProvider:
    """Provider finto per sviluppo/test: nessuna credenziale, nessuna rete."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ) -> str:
        attachment_tag = f" [allegato: {media_mime_type}]" if media_bytes and media_mime_type else ""
        return f"[echo] {messages[-1].content}{attachment_tag}"

    async def chat_with_usage(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ) -> LLMChatResult:
        text = await self.chat(
            messages,
            system=system,
            media_bytes=media_bytes,
            media_mime_type=media_mime_type,
        )
        # Stima fittizia token per i test
        estimated_tokens = sum(len(m.content) for m in messages) // 4 + len(text) // 4
        return LLMChatResult(text=text, tokens_used=estimated_tokens)

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        media_bytes: bytes | None = None,
        media_mime_type: str | None = None,
    ) -> AsyncIterator[str]:
        full_text = await self.chat(
            messages,
            system=system,
            media_bytes=media_bytes,
            media_mime_type=media_mime_type,
        )
        # Emette il testo in piccoli chunk per simulare lo streaming
        chunk_size = 12
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i : i + chunk_size]

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
