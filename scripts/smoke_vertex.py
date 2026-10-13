"""Prova reale di Vertex con prompt FINTI (nessun dato di pazienti).

Uso:  LLM_PROVIDER=vertex GCP_PROJECT=... LLM_MODEL=... uv run python scripts/smoke_vertex.py
Richiede Application Default Credentials (gcloud auth application-default login).
Serve anche a verificare che il modello sia disponibile su location 'eu'.
"""
import asyncio
import os
import sys
import time

os.environ.setdefault("SUPABASE_URL", "http://unused")
os.environ.setdefault("SUPABASE_ANON_KEY", "unused")
os.environ["LLM_PROVIDER"] = "vertex"

from vitality_be.core.config import get_settings  # noqa: E402
from vitality_be.llm.base import ChatMessage, LLMError  # noqa: E402
from vitality_be.llm.factory import get_llm  # noqa: E402
from vitality_be.llm.prompts import get_chat_system_prompt  # noqa: E402

PROMPTS = [
    "Ciao, come stai oggi?",
    "Che cosa posso mangiare a cena se ho il diabete?",
    "Mi ricordi di bere acqua durante il giorno?",
]


async def main() -> int:
    s = get_settings()
    print(f"modello={s.llm_model} location={s.gcp_location} timeout={s.llm_timeout_s}s")
    llm, system, failures = get_llm(), get_chat_system_prompt(), 0
    for p in PROMPTS:
        t = time.monotonic()
        try:
            out = await llm.chat([ChatMessage("user", p)], system=system)
            print(f"\n[{time.monotonic() - t:.1f}s] {p}\n-> {out}")
        except LLMError as e:
            failures += 1
            print(f"\n[{time.monotonic() - t:.1f}s] {p}\n!! {type(e).__name__} code={e.code}")
    await llm.aclose()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
