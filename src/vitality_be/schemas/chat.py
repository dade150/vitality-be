import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

MAX_MESSAGES = 30  # coerente con il FE
MAX_CONTENT_CHARS = 4000
MAX_TOTAL_CHARS = 16000  # minaccia: denial of wallet (token in ingresso) e cronologia finta enorme


def _normalize(v):
    """Normalizza il testo (NFC), toglie caratteri di controllo e invisibili, poi strip.
    Minaccia: caratteri nascosti/bidi usati per offuscare prompt injection; input vuoto
    mascherato da spazi. Mantiene newline e tab, e lo ZWJ (serve alle emoji composte)."""
    if not isinstance(v, str):
        return v
    v = unicodedata.normalize("NFC", v).replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for ch in v:
        cat = unicodedata.category(ch)
        if ch in ("\n", "\t") or ch == "\u200d":
            out.append(ch)
        elif cat in ("Cc", "Cf", "Cs"):
            continue
        else:
            out.append(ch)
    return "".join(out).strip()


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Il ruolo "system" non è ammesso: il system prompt vive solo sul server.
    role: Literal["user", "assistant"]
    content: Annotated[
        str,
        BeforeValidator(_normalize),  # gira PRIMA dei limiti: "   " diventa "" e viene rifiutato
        Field(min_length=1, max_length=MAX_CONTENT_CHARS),
    ]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessageIn] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def _check(self):
        if self.messages[-1].role != "user":
            raise ValueError("L'ultimo messaggio deve essere dell'utente")
        if sum(len(m.content) for m in self.messages) > MAX_TOTAL_CHARS:
            raise ValueError(f"Conversazione troppo lunga (max {MAX_TOTAL_CHARS} caratteri in totale)")
        return self


class ChatResponse(BaseModel):
    reply: str
