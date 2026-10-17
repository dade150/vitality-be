from functools import lru_cache
from typing import Annotated, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    environment: Literal["dev", "prod"] = "dev"

    supabase_url: str
    supabase_anon_key: str

    # --- LLM ---
    llm_provider: Literal["echo", "vertex"] = "echo"
    gcp_project: str | None = None
    # "eu" = multi-regione UE (host aiplatform.eu.rep.googleapis.com).
    # "global" è sempre rifiutato: nessuna garanzia sul luogo di elaborazione.
    gcp_location: str = "eu"
    llm_model: str | None = None
    llm_timeout_s: float = 40.0  # scadenza TOTALE verso il provider (retry inclusi)
    llm_max_output_tokens: int = 2048  # i token di ragionamento Gemini 3 contano qui dentro
    llm_temperature: float = 0.3
    llm_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"

    # --- Richiesta / abuso (minaccia: denial of wallet) ---
    chat_request_timeout_s: float = 50.0  # deve restare < 60 s (timeout del FE)
    rate_limit_requests: int = 20
    rate_limit_window_s: int = 60
    llm_max_concurrency: int = 8
    llm_queue_timeout_s: float = 5.0
    max_body_bytes: int = 64 * 1024

    # --- Superficie HTTP ---
    allowed_hosts: Annotated[list[str], NoDecode] = ["localhost", "127.0.0.1", "testserver"]

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, v):
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
        return v

    @model_validator(mode="after")
    def _validate(self) -> "Settings":
        errors: list[str] = []

        if self.gcp_location.strip().lower() == "global":
            errors.append("GCP_LOCATION=global non è ammesso (nessuna garanzia sul luogo di elaborazione): usa 'eu'.")

        # CONTROLLO DISABILITATO TEMPORANEAMENTE PER TESTARE SU EUROPE-WEST1
        # if self.llm_model and self.llm_model.lower().startswith("gemini-2.5"):
        #     errors.append("I modelli gemini-2.5-* sono in ritiro: scegli un modello Gemini 3.x.")

        if self.llm_provider == "vertex":
            if not self.gcp_project:
                errors.append("GCP_PROJECT obbligatorio con LLM_PROVIDER=vertex.")
            if not self.gcp_location:
                errors.append("GCP_LOCATION obbligatorio con LLM_PROVIDER=vertex.")
            if not self.llm_model:
                errors.append("LLM_MODEL obbligatorio con LLM_PROVIDER=vertex.")
        if self.environment == "prod":
            if self.llm_model and "preview" in self.llm_model.lower():
                errors.append("In prod non sono ammessi modelli 'preview'.")
            if "allowed_hosts" not in self.model_fields_set or not self.allowed_hosts or "*" in self.allowed_hosts:
                errors.append("In prod ALLOWED_HOSTS va impostato esplicitamente (no default, no '*').")

        if not 0 < self.chat_request_timeout_s < 60:
            errors.append("CHAT_REQUEST_TIMEOUT_S deve essere > 0 e < 60 (timeout del FE).")
        if not 0 < self.llm_timeout_s < self.chat_request_timeout_s:
            errors.append("LLM_TIMEOUT_S deve essere > 0 e < CHAT_REQUEST_TIMEOUT_S.")
        if self.llm_max_output_tokens < 256:
            errors.append("LLM_MAX_OUTPUT_TOKENS troppo basso (< 256): le risposte verrebbero tagliate.")
        for name in ("rate_limit_requests", "rate_limit_window_s", "llm_max_concurrency", "max_body_bytes"):
            if getattr(self, name) < 1:
                errors.append(f"{name.upper()} deve essere >= 1.")

        if errors:
            raise ValueError("Configurazione non valida: " + " | ".join(errors))
        return self

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()