#!/usr/bin/env python3
"""Fase 2 della pipeline RAG: embedding dei chunk con Gemini Embedding 2 (Vertex AI, EU).

Legge output/chunks/chunks.jsonl, chiama l'endpoint regionale EU di Vertex AI
(un testo per richiesta, `:embedContent`) e scrive output/embeddings/embeddings.jsonl.
Autenticazione: Application Default Credentials (gcloud auth application-default login).
Non tocca il database. Riesecuzioni: i vettori gia' calcolati con lo stesso
content_hash, modello e dimensione vengono riusati, non ricalcolati.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol
from urllib.parse import urlparse

import httpx
from dotenv import find_dotenv, load_dotenv
from loguru import logger

DEFAULT_MODEL = "gemini-embedding-2"
DEFAULT_LOCATION = "eu"
EU_ENDPOINT_HOST = "aiplatform.eu.rep.googleapis.com"
DEFAULT_ENDPOINT = f"https://{EU_ENDPOINT_HOST}"
DEFAULT_DIM = 1536
MAX_MODEL_DIM = 3072
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

Key = tuple[str, int]


class ConfigError(Exception):
    pass


class EmbeddingError(Exception):
    """Errore su un singolo chunk: gli altri chunk proseguono."""


class FatalEmbeddingError(EmbeddingError):
    """Errore che rende inutile proseguire (credenziali, permessi)."""


# --------------------------------------------------------------------------
# Configurazione
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    project: str
    endpoint: str
    location: str
    model: str
    dim: int
    max_concurrency: int
    max_retries: int
    request_timeout: float = 60.0

    @property
    def url(self) -> str:
        return (
            f"{self.endpoint.rstrip('/')}/v1/projects/{self.project}"
            f"/locations/{self.location}/publishers/google/models/{self.model}:embedContent"
        )


def validate_config(config: EmbeddingConfig) -> EmbeddingConfig:
    if not config.project or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]*", config.project):
        raise ConfigError("EMBEDDING_GCP_PROJECT mancante o non valido")
    parsed = urlparse(config.endpoint)
    if parsed.scheme != "https" or parsed.hostname != EU_ENDPOINT_HOST:
        raise ConfigError(
            f"endpoint non consentito: gli embedding devono usare https://{EU_ENDPOINT_HOST}"
        )
    if config.location != DEFAULT_LOCATION:
        raise ConfigError(f"location non consentita: deve essere '{DEFAULT_LOCATION}'")
    if not config.model.startswith(DEFAULT_MODEL):
        raise ConfigError(
            f"modello non consentito: il formato dei prompt e' specifico di {DEFAULT_MODEL}"
        )
    if not 1 <= config.dim <= MAX_MODEL_DIM:
        raise ConfigError(f"EMBEDDING_DIM deve essere tra 1 e {MAX_MODEL_DIM}")
    if config.max_concurrency < 1 or config.max_retries < 0:
        raise ConfigError("concorrenza o numero di retry non validi")
    return config


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} deve essere un intero") from exc


# --------------------------------------------------------------------------
# Validazione vettori
# --------------------------------------------------------------------------


def validate_vector(values: Any, dim: int) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("l'embedding non e' una lista")
    if len(values) != dim:
        raise ValueError(f"dimensione attesa {dim}, ricevuta {len(values)}")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("l'embedding contiene valori non numerici")
        if not math.isfinite(value):
            raise ValueError("l'embedding contiene valori non finiti")
    return [float(v) for v in values]


# --------------------------------------------------------------------------
# Autenticazione (ADC)
# --------------------------------------------------------------------------


class TokenProvider(Protocol):
    def get_token(self, force_refresh: bool = False) -> str: ...


class AdcTokenProvider:
    """Token OAuth da Application Default Credentials, rinfrescato su richiesta."""

    def __init__(self) -> None:
        self._credentials: Any = None

    def get_token(self, force_refresh: bool = False) -> str:
        try:
            import google.auth
            from google.auth.transport.requests import Request
        except ImportError as exc:  # pragma: no cover
            raise FatalEmbeddingError("pacchetto google-auth non installato") from exc
        try:
            if self._credentials is None:
                self._credentials, _ = google.auth.default(scopes=[CLOUD_SCOPE])
                force_refresh = True
            if force_refresh or not self._credentials.valid:
                self._credentials.refresh(Request())
            token = self._credentials.token
        except Exception as exc:
            raise FatalEmbeddingError(
                "credenziali ADC non disponibili: esegui "
                "'gcloud auth application-default login'"
            ) from exc
        if not token:
            raise FatalEmbeddingError("token ADC vuoto")
        return token


# --------------------------------------------------------------------------
# Client Vertex
# --------------------------------------------------------------------------


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 120.0))
    except ValueError:
        return None


def _error_summary(response: httpx.Response) -> str:
    """Solo codice e stato: il corpo puo' contenere il testo inviato, non lo riportiamo."""
    status = ""
    try:
        status = str(response.json().get("error", {}).get("status", ""))
    except Exception:
        pass
    return f"HTTP {response.status_code}" + (f" ({status})" if status else "")


class VertexEmbedder:
    def __init__(
        self,
        config: EmbeddingConfig,
        tokens: TokenProvider,
        client: httpx.AsyncClient,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.config = config
        self.tokens = tokens
        self.client = client
        self._sleep = sleep

    def _payload(self, text: str) -> dict[str, Any]:
        return {
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.config.dim,
        }

    async def embed(self, text: str) -> list[float]:
        refreshed = False
        attempt = 0
        while True:
            token = await asyncio.to_thread(self.tokens.get_token, refreshed and attempt > 0)
            try:
                response = await self.client.post(
                    self.config.url,
                    json=self._payload(text),
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self.config.request_timeout,
                )
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise EmbeddingError(f"errore di rete: {type(exc).__name__}") from exc
                await self._backoff(attempt, None)
                attempt += 1
                continue

            status = response.status_code
            if status == 200:
                return self._parse(response)
            if status == 401:
                if refreshed:
                    raise FatalEmbeddingError("autenticazione rifiutata (HTTP 401)")
                refreshed = True
                self.tokens.get_token(True)
                continue
            if status == 403:
                raise FatalEmbeddingError("permesso negato (HTTP 403): controlla ruolo IAM e API")
            if status in RETRYABLE_STATUS:
                if attempt >= self.config.max_retries:
                    raise EmbeddingError(f"{_error_summary(response)} dopo {attempt + 1} tentativi")
                await self._backoff(attempt, _retry_after_seconds(response))
                attempt += 1
                continue
            raise EmbeddingError(_error_summary(response))

    def _parse(self, response: httpx.Response) -> list[float]:
        try:
            values = response.json()["embedding"]["values"]
            return validate_vector(values, self.config.dim)
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"risposta non valida: {exc}") from exc

    async def _backoff(self, attempt: int, retry_after: float | None) -> None:
        delay = retry_after if retry_after is not None else min(2.0**attempt, 30.0)
        await self._sleep(delay + random.uniform(0, 0.25))


# --------------------------------------------------------------------------
# I/O JSONL
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} riga {number}: JSON non valido") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path.name} riga {number}: non e' un oggetto")
            yield number, obj


def load_chunks(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    chunks: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: set[Key] = set()
    for number, obj in read_jsonl(path):
        doc_id, index = obj.get("doc_id"), obj.get("chunk_index")
        text, digest = obj.get("embedding_input"), obj.get("content_hash")
        if not (
            isinstance(doc_id, str)
            and doc_id
            and isinstance(index, int)
            and not isinstance(index, bool)
            and isinstance(text, str)
            and text.strip()
            and isinstance(digest, str)
            and digest
        ):
            problems.append(f"riga {number}: campi mancanti o non validi")
            continue
        if (doc_id, index) in seen:
            problems.append(f"riga {number}: chiave duplicata {doc_id}#{index}")
            continue
        seen.add((doc_id, index))
        chunks.append(obj)
    return chunks, problems


def load_existing(path: Path, model: str, dim: int) -> dict[Key, dict[str, Any]]:
    """Vettori gia' calcolati e ancora validi per modello e dimensione."""
    existing: dict[Key, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for _, obj in read_jsonl(path):
        try:
            validate_vector(obj.get("embedding"), dim)
        except ValueError:
            continue
        if obj.get("embedding_model") != model or obj.get("embedding_dim") != dim:
            continue
        existing[(obj["doc_id"], obj["chunk_index"])] = obj
    return existing


def plan_work(
    chunks: list[dict[str, Any]], existing: dict[Key, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restituisce (da calcolare, riusabili)."""
    pending, reusable = [], []
    for chunk in chunks:
        prior = existing.get((chunk["doc_id"], chunk["chunk_index"]))
        if prior is not None and prior.get("content_hash") == chunk["content_hash"]:
            reusable.append(prior)
        else:
            pending.append(chunk)
    return pending, reusable


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------
# Generazione
# --------------------------------------------------------------------------


async def embed_pending(
    embedder: VertexEmbedder,
    pending: list[dict[str, Any]],
    config: EmbeddingConfig,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Restituisce (record riusciti, errori, interrotto per errore fatale)."""
    semaphore = asyncio.Semaphore(config.max_concurrency)
    stop = asyncio.Event()
    done: list[dict[str, Any]] = []
    errors: list[str] = []

    async def work(chunk: dict[str, Any]) -> None:
        label = f"{chunk['doc_id']}#{chunk['chunk_index']}"
        async with semaphore:
            if stop.is_set():
                return
            try:
                vector = await embedder.embed(chunk["embedding_input"])
            except FatalEmbeddingError as exc:
                stop.set()
                errors.append(f"{label}: {exc}")
                return
            except EmbeddingError as exc:
                errors.append(f"{label}: {exc}")
                return
        done.append(
            {
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
                "content_hash": chunk["content_hash"],
                "embedding_model": config.model,
                "embedding_dim": config.dim,
                "embedding": vector,
            }
        )

    await asyncio.gather(*(work(c) for c in pending))
    return done, errors, stop.is_set()


def confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Terminale non interattivo: usa --yes per procedere.", file=sys.stderr)
        return False
    return input(f"{message} [s/N] ").strip().lower() in {"s", "si", "y", "yes"}


async def generate(
    config: EmbeddingConfig,
    chunks_path: Path,
    output_path: Path,
    dry_run: bool,
    assume_yes: bool,
    tokens: TokenProvider | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> int:
    try:
        chunks, problems = load_chunks(chunks_path)
    except (OSError, ValueError) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 2
    existing = load_existing(output_path, config.model, config.dim)
    pending, reusable = plan_work(chunks, existing)

    print("\nPIANO EMBEDDING")
    print("=" * 60)
    print(f"Modello:               {config.model} (dim {config.dim})")
    print(f"Endpoint:              {config.endpoint} / location {config.location}")
    print(f"Chunk letti:           {len(chunks)}")
    print(f"Chunk scartati:        {len(problems)}")
    print(f"Vettori riusati:       {len(reusable)}")
    print(f"Da calcolare:          {len(pending)}")
    for line in problems:
        print(f"  scartato: {line}")

    if dry_run:
        print("\nDRY-RUN: nessuna chiamata API, nessun file scritto.\n")
        return 1 if problems else 0
    if pending and not confirm(f"Calcolare {len(pending)} embedding con chiamate a pagamento?", assume_yes):
        print("Annullato.")
        return 1

    errors: list[str] = []
    fatal = False
    fresh: list[dict[str, Any]] = []
    if pending:
        async with httpx.AsyncClient(transport=transport) as client:
            embedder = VertexEmbedder(config, tokens or AdcTokenProvider(), client)
            fresh, errors, fatal = await embed_pending(embedder, pending, config)

    merged = {(r["doc_id"], r["chunk_index"]): r for r in reusable}
    merged.update({(r["doc_id"], r["chunk_index"]): r for r in fresh})
    write_jsonl_atomic(output_path, [merged[k] for k in sorted(merged)])

    print(f"\nCalcolati:             {len(fresh)}")
    print(f"Falliti:               {len(errors)}")
    print(f"Totale in output:      {len(merged)}  ->  {output_path}")
    if fatal:
        print("\nINTERROTTO per errore non recuperabile (credenziali o permessi).")
    for line in errors:
        print(f"  errore: {line}")
    print()
    return 1 if (errors or problems or fatal) else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{level: <8} {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calcola gli embedding dei chunk (Vertex AI, EU).")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path(os.environ.get("CHUNKS_OUTPUT_DIR", "output/chunks")) / "chunks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("EMBEDDINGS_OUTPUT_DIR", "output/embeddings")) / "embeddings.jsonl",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Mostra il piano senza chiamare l'API.")
    parser.add_argument("--yes", action="store_true", help="Non chiedere conferma.")
    return parser


def config_from_env(args: argparse.Namespace) -> EmbeddingConfig:
    return validate_config(
        EmbeddingConfig(
            project=os.environ.get("EMBEDDING_GCP_PROJECT", ""),
            endpoint=os.environ.get("EMBEDDING_API_ENDPOINT", DEFAULT_ENDPOINT),
            location=os.environ.get("EMBEDDING_LOCATION", DEFAULT_LOCATION),
            model=os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL),
            dim=_env_int("EMBEDDING_DIM", DEFAULT_DIM),
            max_concurrency=_env_int("EMBEDDING_MAX_CONCURRENCY", 2),
            max_retries=args.max_retries,
        )
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        config = config_from_env(args)
    except ConfigError as exc:
        print(f"Errore di configurazione: {exc}", file=sys.stderr)
        return 2
    if not args.chunks.is_file():
        print(f"Errore: file dei chunk non trovato: {args.chunks}", file=sys.stderr)
        return 2
    return asyncio.run(generate(config, args.chunks, args.output, args.dry_run, args.yes))


if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    raise SystemExit(main())
