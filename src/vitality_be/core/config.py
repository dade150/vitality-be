from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_anon_key: str

    # in futuro: llm_provider, gcp_project, gcp_location, llm_model...
    llm_provider: Literal["echo", "vertex"] = "echo"

@lru_cache
def get_settings() -> Settings:
    return Settings()