"""Provider di embedding per il retrieval runtime (Gemini Embedding 2 su Vertex AI EU).

Convenzione di Gemini Embedding 2 (coerente con scripts/chunk_markdown.py):
- Documenti indicizzati: `title: <source_name> | text: <content>`
- Query di ricerca:      `task: search result | query: <domanda>`

Regola di sicurezza: nessun testo della domanda né corpi di risposta del provider
finiscono mai nei log o nei messaggi delle eccezioni.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
import math
import random
import time
from typing import Any, Protocol

import httpx
from loguru import logger

from vitality_be.core.config import get_settings
from vitality_be.llm.base import LLMError, LLMUnavailable

CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def format_query_for_embedding(query: str) -> str:
    """Prefisso asimmetrico per le query di ricerca in Gemini Embedding 2."""
    cleaned = " ".join(query.split())
    return f"task: search result | query: {cleaned}"


def validate_embedding_vector(values: Any, dim: int) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("l'embedding non è una lista")
    if len(values) != dim:
        raise ValueError(f"dimensione attesa {dim}, ricevuta {len(values)}")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("l'embedding contiene valori non numerici o non finiti")
    return [float(v) for v in values]


class TokenProvider(Protocol):
    def get_token(self, force_refresh: bool = False) -> str: ...


class AdcTokenProvider:
    """Token OAuth2 da Application Default Credentials, rinfrescato automaticamente."""

    def __init__(self) -> None:
        self._credentials: Any = None

    def get_token(self, force_refresh: bool = False) -> str:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("Libreria google-auth non disponibile", code="embed_auth_lib") from exc

        try:
            if self._credentials is None:
                self._credentials, _ = google.auth.default(scopes=[CLOUD_SCOPE])
                force_refresh = True
            if force_refresh or not self._credentials.valid:
                self._credentials.refresh(Request())
            token = self._credentials.token
        except Exception:
            raise LLMUnavailable("Credenziali Google ADC non disponibili", code="embed_credentials") from None

        if not token:
            raise LLMUnavailable("Token Google ADC vuoto", code="embed_empty_token")
        return str(token)


class EmbedderProvider(Protocol):
    async def embed_query(self, query: str) -> list[float]: ...


class EchoEmbedder:
    """Embedder finto per sviluppo/test offline (nessuna credenziale, nessuna rete)."""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    async def embed_query(self, query: str) -> list[float]:
        # Vettore unitario deterministico non nullo
        val = round(1.0 / math.sqrt(self.dim), 6)
        return [val] * self.dim


class VertexEmbedder:
    """Client asincrono per l'endpoint EU `:embedContent` di Vertex AI."""

    def __init__(
        self,
        *,
        project: str,
        location: str = "eu",
        endpoint: str = "https://aiplatform.eu.rep.googleapis.com",
        model: str = "gemini-embedding-2",
        dim: int = 1536,
        timeout_s: float = 10.0,
        max_retries: int = 1,
        tokens: TokenProvider | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.project = project
        self.location = location
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.dim = dim
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._tokens = tokens or AdcTokenProvider()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._sleep = sleep

    @property
    def url(self) -> str:
        return (
            f"{self.endpoint}/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/google/models/{self.model}:embedContent"
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed_query(self, query: str) -> list[float]:
        formatted = format_query_for_embedding(query)
        payload = {
            "content": {"parts": [{"text": formatted}]},
            "outputDimensionality": self.dim,
        }

        started = time.monotonic()
        refreshed = False
        attempt = 0

        while True:
            token = await asyncio.to_thread(self._tokens.get_token, refreshed and attempt > 0)
            try:
                response = await self._client.post(
                    self.url,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self._timeout_s,
                )
            except httpx.TimeoutException:
                if attempt >= self._max_retries:
                    raise LLMUnavailable("Timeout durante il calcolo dell'embedding", code="embed_timeout") from None
                await self._backoff(attempt)
                attempt += 1
                continue
            except httpx.TransportError:
                if attempt >= self._max_retries:
                    raise LLMUnavailable("Errore di rete verso il provider di embedding", code="embed_network") from None
                await self._backoff(attempt)
                attempt += 1
                continue

            status = response.status_code
            if status == 200:
                vector = self._parse_response(response)
                logger.info(
                    "embed model={} dim={} ms={} attempts={}",
                    self.model,
                    self.dim,
                    int((time.monotonic() - started) * 1000),
                    attempt + 1,
                )
                return vector

            if status == 401 and not refreshed:
                refreshed = True
                await asyncio.to_thread(self._tokens.get_token, True)
                continue

            if status in RETRYABLE_STATUS and attempt < self._max_retries:
                logger.warning("Vertex embed retry status={}", status)
                await self._backoff(attempt)
                attempt += 1
                continue

            if status in (401, 403) or status in RETRYABLE_STATUS:
                raise LLMUnavailable("Provider di embedding non disponibile", code=f"embed_http_{status}")
            raise LLMError("Errore dal provider di embedding", code=f"embed_http_{status}")

    def _parse_response(self, response: httpx.Response) -> list[float]:
        try:
            values = response.json()["embedding"]["values"]
            return validate_embedding_vector(values, self.dim)
        except Exception:
            raise LLMError("Risposta di embedding non valida dal provider", code="embed_invalid_payload") from None

    async def _backoff(self, attempt: int) -> None:
        delay = min(0.4 * (2**attempt), 2.0) + random.uniform(0, 0.2)
        await self._sleep(delay)


@lru_cache
def get_embedder() -> EmbedderProvider:
    """Singleton del provider di embedding scelto da config."""
    s = get_settings()
    if s.llm_provider == "echo":
        return EchoEmbedder(dim=s.embedding_dim)

    if s.llm_provider == "vertex":
        project = s.embedding_gcp_project or s.gcp_project
        if not project:
            raise ValueError("Configurazione embedding incompleta: manca EMBEDDING_GCP_PROJECT / GCP_PROJECT.")
        return VertexEmbedder(
            project=project,
            location=s.embedding_location,
            endpoint=s.embedding_api_endpoint,
            model=s.embedding_model,
            dim=s.embedding_dim,
            timeout_s=s.embedding_timeout_s,
        )

    raise ValueError(f"Provider di embedding non implementato per: {s.llm_provider}")
