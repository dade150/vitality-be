#!/usr/bin/env python3
"""Fase 1 della pipeline RAG: chunking dei Markdown curati.

Legge i file .md del corpus, interpreta il front matter YAML, divide il testo
per intestazioni H2 (stessa logica di analisi-chunk.py) e scrive un file JSONL
con un record per chunk. Non chiama servizi esterni e non tocca il database.

Contratto di output (una riga JSON per chunk):
    doc_id, chunk_index, heading_path, content,
    embedding_input  -> testo esatto che verra' inviato al modello di embedding
    content_hash     -> SHA-256 di embedding_input
    columns          -> valori per le colonne di public.document_chunks
    metadata         -> tutto il resto del front matter + provenienza (md_file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv
from loguru import logger

FRONT_MATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S
)
H2_SPLIT_RE = re.compile(r"(?m)(?=^##[ \t]+)")
H1_RE = re.compile(r"(?m)^#[ \t]+(\S.*?)[ \t]*$")
H2_LINE_RE = re.compile(r"^##[ \t]+(\S.*?)[ \t]*$")
ANY_HEADING_RE = re.compile(r"^#{1,6}(?:[ \t]+|$)")

REQUIRED_KEYS = ("doc_id", "source_name", "doc_type")

# chiave del front matter -> colonna della tabella
COLUMN_MAP = {
    "source_name": "source_name",
    "doc_type": "document_type",
    "language": "language",
    "jurisdiction": "jurisdiction",
    "audience": "audience",
    "population": "population",
    "publication_year": "publication_year",
    "last_verified": "last_verified",
}

DEFAULT_MAX_CHARS = 3000


class ChunkingError(ValueError):
    """Errore di un singolo file: il file viene saltato e segnalato."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        raise ChunkingError("front matter YAML assente")
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ChunkingError("front matter YAML non valido") from exc
    if not isinstance(data, dict):
        raise ChunkingError("front matter YAML non e' una mappa chiave/valore")
    return data, text[match.end():]


def _to_year(value: Any) -> int:
    if isinstance(value, bool):
        raise ChunkingError("publication_year non valido")
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ChunkingError("publication_year non e' un intero") from exc


def _to_date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()).isoformat()
        except ValueError:
            pass
    raise ChunkingError("last_verified non e' una data ISO (AAAA-MM-GG)")


def build_columns(meta: dict[str, Any]) -> dict[str, Any]:
    missing = [
        key for key in REQUIRED_KEYS if not str(meta.get(key) or "").strip()
    ]
    if missing:
        raise ChunkingError(
            "campi obbligatori mancanti nel front matter: " + ", ".join(missing)
        )
    columns: dict[str, Any] = {}
    for source_key, column in COLUMN_MAP.items():
        value = meta.get(source_key)
        if value is None or (isinstance(value, str) and not value.strip()):
            columns[column] = None
        elif column == "publication_year":
            columns[column] = _to_year(value)
        elif column == "last_verified":
            columns[column] = _to_date(value)
        else:
            columns[column] = str(value).strip()
    return columns


def split_h2(body: str) -> list[str]:
    return [piece for piece in H2_SPLIT_RE.split(body) if piece != ""]


def is_heading_only(chunk: str) -> bool:
    lines = [line for line in chunk.splitlines() if line.strip()]
    return bool(lines) and all(ANY_HEADING_RE.match(line) for line in lines)


def build_heading_path(h1: str, chunk: str) -> str:
    first_line = chunk.splitlines()[0] if chunk else ""
    match = H2_LINE_RE.match(first_line)
    parts = [h1]
    if match and match.group(1) != h1:
        parts.append(match.group(1))
    return " > ".join(part for part in parts if part)


def build_embedding_input(source_name: str, content: str) -> str:
    """Formato documento di Gemini Embedding 2: `title: ... | text: ...`.

    Le query (fase di retrieval) dovranno usare `task: search result | query: ...`.
    """
    title = " ".join(source_name.split()) or "none"
    return f"title: {title} | text: {content}"


@dataclass
class FileResult:
    doc_id: str
    records: list[dict[str, Any]] = field(default_factory=list)
    counters: Counter = field(default_factory=Counter)


def chunk_file(path: Path, root: Path, max_chars: int) -> FileResult:
    text = path.read_text(encoding="utf-8-sig")
    meta, body = parse_front_matter(text)
    columns = build_columns(meta)
    doc_id = str(meta["doc_id"]).strip()
    source_name = columns["source_name"]

    extras = {
        key: to_json_safe(value)
        for key, value in meta.items()
        if key != "doc_id" and key not in COLUMN_MAP
    }
    md_file = path.relative_to(root).as_posix()

    h1_match = H1_RE.search(body)
    h1 = h1_match.group(1) if h1_match else source_name

    result = FileResult(doc_id=doc_id)
    seen_hashes: set[str] = set()

    for piece in split_h2(body):
        content = piece.strip()
        if not content:
            result.counters["skipped_empty"] += 1
            continue
        if is_heading_only(content):
            result.counters["skipped_heading_only"] += 1
            continue

        embedding_input = build_embedding_input(source_name, content)
        content_hash = sha256_text(embedding_input)
        if content_hash in seen_hashes:
            result.counters["skipped_duplicate"] += 1
            continue
        seen_hashes.add(content_hash)

        chunk_index = len(result.records)
        if len(content) > max_chars:
            result.counters["oversize"] += 1
            logger.warning(
                "Chunk oltre {} caratteri: doc_id={} chunk_index={} ({} caratteri)",
                max_chars,
                doc_id,
                chunk_index,
                len(content),
            )

        result.records.append(
            {
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "heading_path": build_heading_path(h1, content),
                "content": content,
                "embedding_input": embedding_input,
                "content_hash": content_hash,
                "columns": dict(columns),
                "metadata": {
                    **extras,
                    "md_file": md_file,
                    "char_count": len(content),
                },
            }
        )
    return result


def collect(
    root: Path, max_chars: int
) -> tuple[list[dict[str, Any]], Counter, list[str]]:
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".md"),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    counters: Counter = Counter()
    counters["files_found"] = len(files)
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    doc_owner: dict[str, str] = {}

    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            result = chunk_file(path, root, max_chars)
        except (OSError, UnicodeError, ChunkingError) as exc:
            counters["files_failed"] += 1
            errors.append(f"{rel}: {exc}")
            continue
        if result.doc_id in doc_owner:
            counters["files_failed"] += 1
            errors.append(
                f"{rel}: doc_id duplicato (gia' usato da {doc_owner[result.doc_id]})"
            )
            continue
        doc_owner[result.doc_id] = rel
        counters["files_ok"] += 1
        counters.update(result.counters)
        records.extend(result.records)

    counters["chunks"] = len(records)
    return records, counters, errors


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{level: <8} {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Divide i Markdown del corpus in chunk (H2) e scrive chunks.jsonl."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(os.environ.get("CORPUS_DIR", "curated")),
        help="Cartella del corpus (env CORPUS_DIR, default: curated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("CHUNKS_OUTPUT_DIR", "output/chunks")),
        help="Cartella di output (env CHUNKS_OUTPUT_DIR, default: output/chunks)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=int(os.environ.get("CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS)),
        help="Soglia di segnalazione per chunk grandi (env CHUNK_MAX_CHARS, default: 3000). "
        "I chunk oltre soglia vengono segnalati, non tagliati.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analizza il corpus e stampa il riepilogo senza scrivere file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)

    if args.max_chars < 1:
        print("Errore: --max-chars deve essere maggiore di zero", file=sys.stderr)
        return 2
    if not args.corpus_dir.is_dir():
        print(f"Errore: cartella del corpus non trovata: {args.corpus_dir}", file=sys.stderr)
        return 2

    records, counters, errors = collect(args.corpus_dir, args.max_chars)
    out_path = args.output_dir / "chunks.jsonl"

    if not args.dry_run:
        if errors:
            logger.warning(
                "Ci sono file con errori: i loro chunk NON saranno scritti. "
                "Correggili e rilancia per includerli."
            )
        write_jsonl_atomic(out_path, records)

    print("\nRIEPILOGO CHUNKING")
    print("=" * 60)
    print(f"Modalita':                {'DRY-RUN (nessun file scritto)' if args.dry_run else 'scrittura'}")
    print(f"File trovati:             {counters['files_found']}")
    print(f"File elaborati:           {counters['files_ok']}")
    print(f"File con errori:          {counters['files_failed']}")
    print(f"Chunk creati:             {counters['chunks']}")
    print(f"Chunk saltati (vuoti):    {counters['skipped_empty']}")
    print(f"Chunk saltati (solo intestazione): {counters['skipped_heading_only']}")
    print(f"Chunk saltati (duplicati): {counters['skipped_duplicate']}")
    print(f"Chunk oltre {args.max_chars} caratteri: {counters['oversize']}")
    if not args.dry_run:
        print(f"Output:                   {out_path}")
    if errors:
        print("\nERRORI")
        for line in errors:
            print(f"  {line}")
    print()
    return 1 if errors else 0


if __name__ == "__main__":
    load_dotenv(find_dotenv(usecwd=True))
    raise SystemExit(main())