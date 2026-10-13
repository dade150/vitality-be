# llm/factory.py
from functools import lru_cache

from vitality_be.core.config import get_settings
from vitality_be.llm.base import LLMProvider


@lru_cache
def get_llm() -> LLMProvider:
    """Singleton del provider scelto da config. Gli import sono lazy: l'SDK di Google
    si carica solo se serve davvero."""
    s = get_settings()
    if s.llm_provider == "echo":
        from vitality_be.llm.echo import EchoProvider

        return EchoProvider()

    if s.llm_provider == "vertex":
        missing = [n for n, v in (("GCP_PROJECT", s.gcp_project), ("GCP_LOCATION", s.gcp_location),
                                  ("LLM_MODEL", s.llm_model)) if not v]
        if missing:
            raise ValueError(f"Config Vertex incompleta, mancano: {', '.join(missing)}")
        from vitality_be.llm.vertex import VertexProvider

        return VertexProvider(
            project=s.gcp_project,
            location=s.gcp_location,
            model=s.llm_model,
            timeout_s=s.llm_timeout_s,
            max_output_tokens=s.llm_max_output_tokens,
            temperature=s.llm_temperature,
            thinking_level=s.llm_thinking_level,
        )

    raise ValueError(f"LLM provider non implementato: {s.llm_provider}")
