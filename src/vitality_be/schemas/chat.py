from typing import Literal
from pydantic import BaseModel, Field, model_validator

MAX_MESSAGES = 30
MAX_CONTENT_CHARS = 4000


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)


class ChatRequest(BaseModel):
    messages: list[ChatMessageIn] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def last_must_be_user(self):
        if self.messages[-1].role != "user":
            raise ValueError("L'ultimo messaggio deve essere dell'utente")
        return self


class ChatResponse(BaseModel):
    reply: str