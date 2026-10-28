# llm/prompts/__init__.py
import re
from functools import lru_cache
from importlib import resources


@lru_cache
def load_prompt(name: str) -> str:
    """Carica un prompt .md incluso nel package. Se manca solleva subito:
    chiamata in lifespan, così l'avvio fallisce invece della prima richiesta."""
    raw = resources.files(__package__).joinpath(name).read_text(encoding="utf-8")
    # i commenti HTML sono note per chi mantiene il file: non vanno inviati al modello
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL).strip()
    if not text:
        raise ValueError(f"Prompt vuoto: {name}")
    return text


def get_chat_system_prompt() -> str:
    return load_prompt("chat_system.md")


def get_extraction_system_prompt() -> str:
    return load_prompt("extraction_system.md")
