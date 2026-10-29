from datetime import datetime
import unicodedata
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from vitality_be.schemas.extraction import ProposedEntry

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


class ChatAttachmentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "meal_photo",
        "glucometer_photo",
        "medical_report",
        "medication_photo",
        "generic_image",
        "pdf_document",
    ] = "generic_image"
    mime_type: Literal["image/jpeg", "image/png", "image/webp", "application/pdf"]
    data_base64: str = Field(min_length=16, max_length=10 * 1024 * 1024)


class AttachmentInfoOut(BaseModel):
    has_attachment: bool = True
    kind: str
    mime_type: str
    size_bytes: int
    exif_stripped: bool = True
    sha256_prefix: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Supporto persistenza server-side
    session_id: int | None = Field(default=None, ge=1)
    message: Annotated[
        str | None,
        BeforeValidator(_normalize),
        Field(default=None, max_length=MAX_CONTENT_CHARS),
    ] = None
    attachment: ChatAttachmentIn | None = None

    # Supporto legacy con cronologia inviata dal client (transizione zero-downtime)
    messages: list[ChatMessageIn] | None = Field(default=None, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def _validate_request(self):
        # Retrocompatibilità: se il FE invia ancora messages[]
        if self.messages is not None:
            if len(self.messages) == 0:
                raise ValueError("La lista messages non può essere vuota")
            if self.messages[-1].role != "user":
                raise ValueError("L'ultimo messaggio deve essere dell'utente")
            if sum(len(m.content) for m in self.messages) > MAX_TOTAL_CHARS:
                raise ValueError(f"Conversazione troppo lunga (max {MAX_TOTAL_CHARS} caratteri in totale)")
            if not self.message:
                self.message = self.messages[-1].content

        if not self.message and not self.attachment:
            raise ValueError("Devi fornire almeno un messaggio di testo o un allegato.")

        if not self.message and self.attachment:
            default_prompts = {
                "meal_photo": "Ecco la foto del mio pasto. Puoi analizzarlo dal punto di vista nutrizionale?",
                "glucometer_photo": "Ecco la lettura del mio glucometro.",
                "medical_report": "Ecco il mio referto medico. Puoi analizzarlo?",
                "medication_photo": "Ecco la foto del mio farmaco.",
                "pdf_document": "Ecco il documento/referto allegato.",
                "generic_image": "Ecco l'immagine allegata.",
            }
            self.message = default_prompts.get(self.attachment.kind, "Analizza l'allegato fornito.")

        return self


class ChatResponse(BaseModel):
    reply: str
    proposed_entries: list[ProposedEntry] = Field(default_factory=list)
    session_id: int | None = None
    user_message_id: int | None = None
    assistant_message_id: int | None = None
    attachment_info: AttachmentInfoOut | None = None


class ChatSessionOut(BaseModel):
    id: int
    title: str | None
    created_at: datetime


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionOut]


class ChatMessageOut(BaseModel):
    id: int
    session_id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    attachment_info: AttachmentInfoOut | None = None
    proposed_entries: list[ProposedEntry] = Field(default_factory=list)


class ChatSessionDetailResponse(BaseModel):
    session: ChatSessionOut
    messages: list[ChatMessageOut]


class UpdateSessionTitleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Annotated[
        str,
        BeforeValidator(_normalize),
        Field(min_length=1, max_length=100),
    ]
