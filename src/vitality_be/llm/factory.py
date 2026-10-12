# llm/factory.py
from functools import lru_cache

from vitality_be.core.config import get_settings
from vitality_be.llm.base import LLMProvider
from vitality_be.llm.echo import EchoProvider

@lru_cache
def get_llm() -> LLMProvider:
    name = get_settings().llm_provider
    if name == "echo":
        return EchoProvider()
    raise ValueError(f"LLM provider non implementato: {name}")