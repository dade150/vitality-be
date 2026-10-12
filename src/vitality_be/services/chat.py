from vitality_be.llm.base import ChatMessage, LLMProvider
from vitality_be.schemas.chat import ChatMessageIn


async def reply(llm: LLMProvider, messages: list[ChatMessageIn]) -> str:
    return await llm.chat([ChatMessage(m.role, m.content) for m in messages])