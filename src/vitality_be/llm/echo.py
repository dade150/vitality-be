# llm/echo.py
from vitality_be.llm.base import ChatMessage


class EchoProvider:
    async def chat(self, messages: list[ChatMessage]) -> str:
        return f"[echo] {messages[-1].content}"