#!/usr/bin/env python3
"""Fase 3 della pipeline RAG: caricamento dei chunk e degli embedding su Supabase.

Unisce chunks.jsonl ed embeddings.jsonl (chiave: doc_id + chunk_index, con
controllo del content_hash), costruisce le righe per public.document_chunks e le
scrive con upsert su (doc_id, chunk_index). Richiede un indice UNIQUE su quelle
due colonne. Non scrive mai: id, created_at, fts (generata dal DB), summary,
topic, source_priority. Usa SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY dal .env.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from dotenv import find_dotenv, load_dotenv
from loguru import logger

Key = tuple[str, int]
DEFAULT_TABLE = "document_chunks"
DEFAULT_BATCH = 100
COMPARE_FIELDS = ("content_hash", "embedding_model", "embedding_dim")
# Colonne che lo script non deve mai inviare.
FORBIDDEN_COLUMNS = frozenset({"id", "created_at", "fts", "summary", "topic", "source_priority"})


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_vector(values: Any, dim: int) -> list[float]:
    if not isinstance(values, list):
        raise ValueError("l'embedding non e' una lista")
    if len(values) != dim:
        raise ValueError(f"dimensione attesa {dim}, ricevuta {len(values)}")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("l'embedding contiene valori non numerici o non finiti")
    return [float(v) for v in values]


def vector_literal(vector: list[float]) -> str:
    """Formato testo accettato da pgvector: [0.1,0.2,...]."""
    return "[" + ",".join(repr(v) for v in vector) + "]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
            records.append(obj)
    return records


def load_embeddings(path: Path) -> dict[Key, dict[str, Any]]:
    return {(r["doc_id"], r["chunk_index"]): r for r in read_jsonl(path)}


def build_row(
    chunk: dict[str, Any], emb: dict[str, Any], expected_dim: int
) -> dict[str, Any]:
    """Riga pronta per l'upsert. Solleva ValueError se i due record non coincidono."""
    if emb.get("content_hash") != chunk.get("content_hash"):
        raise ValueError("content_hash diverso: l'embedding e' di una versione precedente del chunk")
    dim = emb.get("embedding_dim")
    if dim != expected_dim:
        raise ValueError(f"embedding_dim {dim} diverso da quello atteso {expected_dim}")
    vector = validate_vector(emb.get("embedding"), expected_dim)
    columns = chunk.get("columns") or {}
    for required in ("source_name", "document_type"):
        if not str(columns.get(required) or "").strip():
            raise ValueError(f"colonna obbligatoria mancante: {required}")
    if not str(chunk.get("content") or "").strip():
        raise ValueError("content vuoto")
    row: dict[str, Any] = {
        **columns,
        "doc_id": chunk["doc_id"],
        "chunk_index": chunk["chunk_index"],
        "heading_path": chunk.get("heading_path"),
        "content": chunk["content"],
        "content_hash": chunk["content_hash"],
        "metadata": chunk.get("metadata") or {},
        "embedding_model": emb["embedding_model"],
        "embedding_dim": dim,
        "embedding": vector_literal(vector),
    }
    clash = FORBIDDEN_COLUMNS & row.keys()
    if clash:
        raise ValueError(f"colonne non scrivibili presenti: {', '.join(sorted(clash))}")
    return row


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class Store(Protocol):
    def fetch_state(self, doc_ids: list[str]) -> dict[Key, dict[str, Any]]: ...
    def upsert(self, rows: list[dict[str, Any]]) -> None: ...


class NullStore:
    """Usato in dry-run: non contatta il database, tutto risulta nuovo."""

    def fetch_state(self, doc_ids: list[str]) -> dict[Key, dict[str, Any]]:
        return {}

    def upsert(self, rows: list[dict[str, Any]]) -> None:
        raise RuntimeError("NullStore non scrive")


class SupabaseStore:
    def __init__(self, client: Any, table: str) -> None:
        self.client = client
        self.table = table

    def fetch_state(self, doc_ids: list[str]) -> dict[Key, dict[str, Any]]:
        state: dict[Key, dict[str, Any]] = {}
        columns = "doc_id,chunk_index," + ",".join(COMPARE_FIELDS)
        for start in range(0, len(doc_ids), 50):
            part = doc_ids[start : start + 50]
            offset = 0
            while True:
                result = (
                    self.client.table(self.table)
                    .select(columns)
                    .in_("doc_id", part)
                    .order("doc_id")
                    .order("chunk_index")
                    .range(offset, offset + 999)
                    .execute()
                )
                data = result.data or []
                for item in data:
                    state[(item["doc_id"], item["chunk_index"])] = item
                if len(data) < 1000:
                    break
                offset += 1000
        return state

    def upsert(self, rows: list[dict[str, Any]]) -> None:
        self.client.table(self.table).upsert(rows, on_conflict="doc_id,chunk_index").execute()


def create_store(table: str) -> SupabaseStore:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY devono essere nel .env")
    from supabase import create_client

    return SupabaseStore(create_client(url, key), table)


# --------------------------------------------------------------------------
# Logica
# --------------------------------------------------------------------------


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(t in text for t in ("timeout", "timed out", "connection", "502", "503", "504", "429"))


def with_retry(
    action: Callable[[], Any], attempts: int = 4, sleep: Callable[[float], None] = time.sleep
) -> Any:
    for attempt in range(attempts):
        try:
            return action()
        except Exception as exc:
            if attempt == attempts - 1 or not _is_transient(exc):
                raise
            sleep(min(2.0**attempt, 20.0))


def prepare(
    chunks: list[dict[str, Any]],
    embeddings: dict[Key, dict[str, Any]],
    expected_dim: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for chunk in chunks:
        key = (chunk.get("doc_id"), chunk.get("chunk_index"))
        label = f"{key[0]}#{key[1]}"
        emb = embeddings.get(key)
        if emb is None:
            problems.append(f"{label}: embedding mancante")
            continue
        try:
            rows.append(build_row(chunk, emb, expected_dim))
        except (ValueError, KeyError) as exc:
            problems.append(f"{label}: {exc}")
    return rows, problems


def classify(
    rows: list[dict[str, Any]], state: dict[Key, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Restituisce (nuovi, modificati, invariati)."""
    new, changed, same = [], [], []
    for row in rows:
        current = state.get((row["doc_id"], row["chunk_index"]))
        if current is None:
            new.append(row)
        elif all(current.get(f) == row[f] for f in COMPARE_FIELDS):
            same.append(row)
        else:
            changed.append(row)
    return new, changed, same


def stale_keys(chunks: list[dict[str, Any]], state: dict[Key, dict[str, Any]]) -> list[Key]:
    """Righe presenti nel DB per i doc_id caricati ma non piu' prodotte dal chunking."""
    wanted = {(c["doc_id"], c["chunk_index"]) for c in chunks}
    return sorted(k for k in state if k not in wanted)


def upload(store: Store, rows: list[dict[str, Any]], batch_size: int) -> tuple[int, list[str]]:
    written = 0
    errors: list[str] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        try:
            with_retry(lambda b=batch: store.upsert(b))
            written += len(batch)
        except Exception as exc:
            first, last = batch[0], batch[-1]
            errors.append(
                f"batch {first['doc_id']}#{first['chunk_index']} .. "
                f"{last['doc_id']}#{last['chunk_index']}: {type(exc).__name__}"
            )
    return written, errors


def verify_upload(store: Store, uploaded: list[dict[str, Any]]) -> list[str]:
    if not uploaded:
        return []
    state = store.fetch_state(sorted({r["doc_id"] for r in uploaded}))
    bad = []
    for row in uploaded:
        current = state.get((row["doc_id"], row["chunk_index"]))
        if current is None or any(current.get(f) != row[f] for f in COMPARE_FIELDS):
            bad.append(f"{row['doc_id']}#{row['chunk_index']}")
    return bad


def confirm(message: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Terminale non interattivo: usa --yes per procedere.", file=sys.stderr)
        return False
    return input(f"{message} [s/N] ").strip().lower() in {"s", "si", "y", "yes"}


def run_index(
    chunks: list[dict[str, Any]],
    embeddings: dict[Key, dict[str, Any]],
    store: Store,
    expected_dim: int,
    batch_size: int,
    dry_run: bool,
    assume_yes: bool,
) -> int:
    rows, problems = prepare(chunks, embeddings, expected_dim)
    doc_ids = sorted({r["doc_id"] for r in rows})
    state = with_retry(lambda: store.fetch_state(doc_ids)) if doc_ids else {}
    new, changed, same = classify(rows, state)
    stale = stale_keys([c for c in chunks if c.get("doc_id") in set(doc_ids)], state)

    print("\nPIANO INDICIZZAZIONE")
    print("=" * 60)
    print(f"Chunk letti:             {len(chunks)}")
    print(f"Righe valide:            {len(rows)}")
    print(f"Non caricabili:          {len(problems)}")
    print(f"Nuove:                   {len(new)}")
    print(f"Modificate:              {len(changed)}")
    print(f"Invariate (saltate):     {len(same)}")
    print(f"Orfane nel DB:           {len(stale)}  (solo segnalate, mai cancellate)")
    for line in problems:
        print(f"  saltato: {line}")
    for doc_id, index in stale[:20]:
        print(f"  orfana: {doc_id}#{index}")

    todo = new + changed
    if dry_run:
        print("\nDRY-RUN: nessuna scrittura.\n")
        return 1 if problems else 0
    if not todo:
        print("\nNiente da scrivere.\n")
        return 1 if problems else 0
    if not confirm(f"Scrivere {len(todo)} righe su Supabase?", assume_yes):
        print("Annullato.")
        return 1

    written, errors = upload(store, todo, batch_size)
    mismatches = verify_upload(store, todo[:written] if not errors else todo) if written else []
    print(f"\nScritte:                 {written}")
    print(f"Batch falliti:           {len(errors)}")
    print(f"Verifica post-scrittura: {'OK' if not mismatches else str(len(mismatches)) + ' righe non corrispondono'}")
    for line in errors:
        print(f"  errore: {line}")
    for label in mismatches[:20]:
        print(f"  non verificata: {label}")
    print()
    return 1 if (problems or errors or mismatches) else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{level: <8} {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Carica chunk ed embedding su Supabase.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path(os.environ.get("CHUNKS_OUTPUT_DIR", "output/chunks")) / "chunks.jsonl",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path(os.environ.get("EMBEDDINGS_OUTPUT_DIR", "output/embeddings")) / "embeddings.jsonl",
    )
    parser.add_argument("--table", default=os.environ.get("SUPABASE_TABLE", DEFAULT_TABLE))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("SUPABASE_UPSERT_BATCH_SIZE", DEFAULT_BATCH)),
    )
    parser.add_argument("--dim", type=int, default=int(os.environ.get("EMBEDDING_DIM", 1536)))
    parser.add_argument("--dry-run", action="store_true", help="Mostra il piano senza scrivere.")
    parser.add_argument("--yes", action="store_true", help="Non chiedere conferma.")
    return parser


def main(argv: list[str] | None = None, store: Store | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        print("Errore: --batch-size deve essere maggiore di zero", file=sys.stderr)
        return 2
    for path in (args.chunks, args.embeddings):
        if not path.is_file():
            print(f"Errore: file non trovato: {path}", file=sys.stderr)
            return 2
    try:
        chunks = read_jsonl(args.chunks)
        embeddings = load_embeddings(args.embeddings)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 2
    if store is None:
        try:
            store = NullStore() if args.dry_run and not os.environ.get("SUPABASE_URL") else create_store(args.table)
        except Exception as exc:
            print(f"Errore: {exc}", file=sys.stderr)
            return 2
    try:
        return run_index(chunks, embeddings, store, args.dim, args.batch_size, args.dry_run, args.yes)
    except Exception as exc:
        print(f"Errore durante l'indicizzazione: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    raise SystemExit(main())
