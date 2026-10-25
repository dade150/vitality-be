"""Servizio di Retrieval Ibrido (Semantico + Full-Text Search con fusione RRF).

Flusso:
1. Genera l'embedding a 1536 dimensioni della domanda utente (`task: search result | query: ...`).
2. Invoca la RPC PostgreSQL `public.hybrid_search_chunks` su Supabase tramite il client RLS utente:
   - Top-K semantico (default 10) su indice HNSW (`vector_cosine_ops`)
   - Top-K lessicale/FTS (default 10) su indice GIN (`fts` italiano + inglese)
   - Fusione Reciprocal Rank Fusion (RRF, k=60) e selezione dei Top-N migliori (default 8).
3. Formatta i chunk recuperati in un blocco dati `<contesto_clinico>` separato dal system prompt.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any

from loguru import logger
from supabase import Client

from vitality_be.core.config import get_settings
from vitality_be.llm.embedder import EmbedderProvider

RPC_HYBRID_SEARCH = "hybrid_search_chunks"


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    doc_id: str | None
    chunk_index: int | None
    source_name: str
    document_type: str
    heading_path: str | None
    content: str
    language: str | None
    similarity: float | None
    fts_rank: float | None
    rrf_score: float


def vector_literal(vector: list[float]) -> str:
    """Serializza il vettore nel formato testuale atteso da pgvector: `[0.1,0.2,...]`."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _parse_chunk(row: dict[str, Any]) -> RetrievedChunk | None:
    content = str(row.get("content") or "").strip()
    source_name = str(row.get("source_name") or "").strip()
    if not content or not source_name:
        return None
    return RetrievedChunk(
        id=int(row.get("id") or 0),
        doc_id=row.get("doc_id"),
        chunk_index=int(row["chunk_index"]) if row.get("chunk_index") is not None else None,
        source_name=source_name,
        document_type=str(row.get("document_type") or ""),
        heading_path=str(row["heading_path"]).strip() if row.get("heading_path") else None,
        content=content,
        language=row.get("language"),
        similarity=float(row["similarity"]) if row.get("similarity") is not None else None,
        fts_rank=float(row["fts_rank"]) if row.get("fts_rank") is not None else None,
        rrf_score=float(row.get("rrf_score") or 0.0),
    )


async def retrieve_chunks(
    query: str,
    *,
    user_db: Client,
    embedder: EmbedderProvider,
    semantic_top_k: int | None = None,
    fts_top_k: int | None = None,
    match_count: int | None = None,
    rrf_k: int | None = None,
) -> list[RetrievedChunk]:
    """Esegue il retrieval ibrido per la domanda dell'utente.

    In caso di indisponibilità temporanea dell'embedder o del database, registra un warning
    (senza mai loggare il testo della domanda) e restituisce una lista vuota per consentire
    alla chat di proseguire in modalità degradata sicura.
    """
    s = get_settings()
    sem_k = semantic_top_k if semantic_top_k is not None else s.rag_semantic_top_k
    lex_k = fts_top_k if fts_top_k is not None else s.rag_fts_top_k
    top_n = match_count if match_count is not None else s.rag_top_k
    k_rrf = rrf_k if rrf_k is not None else s.rag_rrf_k

    started = time.monotonic()
    try:
        query_vector = await embedder.embed_query(query)
    except Exception as exc:
        logger.warning(
            "rag embedding fallito exc={} code={}",
            type(exc).__name__,
            getattr(exc, "code", None),
        )
        return []

    params = {
        "query_text": query,
        "query_embedding": vector_literal(query_vector),
        "semantic_top_k": sem_k,
        "fts_top_k": lex_k,
        "match_count": top_n,
        "rrf_k": k_rrf,
    }

    try:
        resp = await asyncio.to_thread(
            lambda: user_db.rpc(RPC_HYBRID_SEARCH, params).execute()
        )
        rows = resp.data if resp and isinstance(resp.data, list) else []
    except Exception as exc:
        logger.warning("rag rpc fallita exc={}", type(exc).__name__)
        return []

    chunks = [c for r in rows if isinstance(r, dict) and (c := _parse_chunk(r)) is not None]
    elapsed_ms = int((time.monotonic() - started) * 1000)
    top_rrf = round(chunks[0].rrf_score, 4) if chunks else 0.0
    top_sim = (
        round(chunks[0].similarity, 4)
        if chunks and chunks[0].similarity is not None
        else None
    )
    logger.info(
        "rag ok chunks={} top_rrf={} top_sim={} ms={}",
        len(chunks),
        top_rrf,
        top_sim,
        elapsed_ms,
    )
    return chunks


def format_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Formatta i chunk recuperati in un blocco strutturato da allegare come dati."""
    if not chunks:
        return ""

    blocks: list[str] = []
    for idx, c in enumerate(chunks, start=1):
        header_parts = [f"Fonte {idx}: {c.source_name}"]
        if c.heading_path:
            header_parts.append(f"Sezione: {c.heading_path}")
        header = " | ".join(header_parts)
        blocks.append(f"[{header}]\n{c.content}")

    joined = "\n\n---\n\n".join(blocks)
    return f"<contesto_clinico>\n{joined}\n</contesto_clinico>"


def attach_context_to_message(user_message: str, chunks: list[RetrievedChunk]) -> str:
    """Unisce il blocco `<contesto_clinico>` al messaggio utente senza toccare il system prompt."""
    if not chunks:
        return user_message
    context_block = format_rag_context(chunks)
    return (
        f"{context_block}\n\n"
        f"<messaggio_utente>\n{user_message}\n</messaggio_utente>"
    )
