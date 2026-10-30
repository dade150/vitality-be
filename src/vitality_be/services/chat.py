from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from supabase import Client

from vitality_be.llm.base import ChatMessage, LLMBlocked, LLMChatResult, LLMProvider
from vitality_be.llm.embedder import EmbedderProvider
from vitality_be.llm.prompts import get_chat_system_prompt
from vitality_be.schemas.chat import ChatMessageIn
from vitality_be.schemas.extraction import ProposedEntry
from vitality_be.services.extractor import extract_clinical_proposals, extract_from_report
from vitality_be.services.multimodal import AttachmentKind
from vitality_be.services.patient_context import (
    PatientContextResult,
    PatientMaskingMap,
    build_dual_rag_message,
    fetch_patient_context,
    mask_user_text_with_dynamic_tokens,
    rehydrate_assistant_reply,
)
from vitality_be.services.retriever import RetrievedChunk, retrieve_chunks

# Risposta fissa e sicura quando i filtri del provider bloccano (HTTP 200, non un errore).
BLOCKED_REPLY = (
    "Non posso rispondere a questa richiesta. "
    "Se hai un problema di salute urgente, chiama subito il 112. "
    "Per altre domande sulla tua salute, parlane con il tuo medico."
)


@dataclass(frozen=True)
class ChatPipelineResult:
    reply: str
    proposed_entries: list[ProposedEntry] = field(default_factory=list)
    tokens_used: int | None = None
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)


async def run_chat_pipeline(
    llm: LLMProvider,
    *,
    user_message: str,
    history_messages: list[ChatMessage] | None = None,
    media_bytes: bytes | None = None,
    media_mime_type: str | None = None,
    attachment_kind: AttachmentKind | None = None,
    user_db: Client | None = None,
    embedder: EmbedderProvider | None = None,
) -> ChatPipelineResult:
    """Esegue la pipeline Dual-RAG completa con supporto multimodale ed estrazione clinica in parallelo."""
    history: list[ChatMessage] = list(history_messages) if history_messages else []
    # Assicura che l'ultimo messaggio sia l'input corrente dell'utente
    if not history or history[-1].role != "user" or history[-1].content != user_message:
        history.append(ChatMessage(role="user", content=user_message))

    masking_map = PatientMaskingMap()
    active_therapies: tuple[dict, ...] = ()
    chunks: list[RetrievedChunk] = []
    patient_context_xml = ""

    # --- PIPELINE CONVERSAZIONALE (DUAL-RAG) ---
    if user_db is not None:
        last_user_query = user_message
        if embedder is not None:
            chunks, patient_ctx = await asyncio.gather(
                retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder),
                fetch_patient_context(user_db),
            )
        else:
            patient_ctx: PatientContextResult = await fetch_patient_context(user_db)

        masking_map = patient_ctx.masking_map
        active_therapies = patient_ctx.active_therapies
        patient_context_xml = patient_ctx.context_xml

    # De-identificazione sui turni utente prima dell'invio al provider
    scrubbed_history: list[ChatMessage] = []
    for m in history:
        if m.role == "user":
            scrubbed_text, masking_map = mask_user_text_with_dynamic_tokens(m.content, masking_map)
            scrubbed_history.append(ChatMessage(role="user", content=scrubbed_text))
        else:
            scrubbed_history.append(m)

    chat_history = list(scrubbed_history)
    if chunks or patient_context_xml:
        chat_history[-1] = ChatMessage(
            role="user",
            content=build_dual_rag_message(
                scrubbed_history[-1].content,
                patient_context_xml=patient_context_xml,
                clinical_chunks=chunks,
            ),
        )

    # Preparazione task paralleli: generazione risposta + estrazione proposte
    async def _generate_llm() -> LLMChatResult:
        if hasattr(llm, "chat_with_usage"):
            try:
                return await llm.chat_with_usage(
                    chat_history,
                    system=get_chat_system_prompt(),
                    media_bytes=media_bytes,
                    media_mime_type=media_mime_type,
                )
            except TypeError:
                return await llm.chat_with_usage(
                    chat_history,
                    system=get_chat_system_prompt(),
                )

        try:
            text = await llm.chat(
                chat_history,
                system=get_chat_system_prompt(),
                media_bytes=media_bytes,
                media_mime_type=media_mime_type,
            )
        except TypeError:
            text = await llm.chat(chat_history, system=get_chat_system_prompt())
        return LLMChatResult(text=text, tokens_used=None)

    async def _extract() -> list[ProposedEntry]:
        # Se l'allegato è un referto clinico o un PDF, prova anche l'estrazione dedicata per referti
        is_report = attachment_kind in ("medical_report", "pdf_document") or media_mime_type == "application/pdf"
        if not is_report or not (media_bytes and media_mime_type):
            return await extract_clinical_proposals(
                llm,
                scrubbed_history,
                masking_map=masking_map,
                active_therapies=active_therapies,
            )

        res_proposals = await extract_clinical_proposals(
            llm,
            scrubbed_history,
            masking_map=masking_map,
            active_therapies=active_therapies,
        )
        try:
            rep = await extract_from_report(
                llm,
                media_bytes=media_bytes,
                media_mime_type=media_mime_type,
                masking_map=masking_map,
                active_therapies=active_therapies,
            )
            if rep and rep.proposed_entries:
                res_proposals.extend(rep.proposed_entries)
        except Exception as exc:
            logger.warning("extract_from_report fallito exc={}", type(exc).__name__)

        return res_proposals

    try:
        llm_result, proposals = await asyncio.gather(_generate_llm(), _extract())
        rehydrated = rehydrate_assistant_reply(llm_result.text, masking_map)

        chunks_dicts = [
            {
                "chunk_id": c.id,
                "source_name": c.source_name,
                "heading_path": c.heading_path,
                "rrf_score": c.rrf_score,
                "similarity": c.similarity,
            }
            for c in chunks
        ]

        return ChatPipelineResult(
            reply=rehydrated,
            proposed_entries=proposals,
            tokens_used=llm_result.tokens_used,
            rag_chunks=chunks_dicts,
        )
    except LLMBlocked as exc:
        logger.warning("Risposta bloccata dai filtri code={}", exc.code)
        return ChatPipelineResult(reply=BLOCKED_REPLY, proposed_entries=[], tokens_used=None)


async def stream_chat_pipeline(
    llm: LLMProvider,
    *,
    user_message: str,
    history_messages: list[ChatMessage] | None = None,
    media_bytes: bytes | None = None,
    media_mime_type: str | None = None,
    attachment_kind: AttachmentKind | None = None,
    user_db: Client | None = None,
    embedder: EmbedderProvider | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Esegue la pipeline Dual-RAG con generazione del testo in streaming SSE e task HITL in parallelo.

    Emette:
    - {'type': 'delta', 'text': str} durante la generazione (con safe buffering dei placeholder reversibili)
    - {'type': 'done', 'result': ChatPipelineResult} al termine di testo ed estrazione clinica
    """
    history: list[ChatMessage] = list(history_messages) if history_messages else []
    if not history or history[-1].role != "user" or history[-1].content != user_message:
        history.append(ChatMessage(role="user", content=user_message))

    masking_map = PatientMaskingMap()
    active_therapies: tuple[dict, ...] = ()
    chunks: list[RetrievedChunk] = []
    patient_context_xml = ""

    if user_db is not None:
        last_user_query = user_message
        if embedder is not None:
            chunks, patient_ctx = await asyncio.gather(
                retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder),
                fetch_patient_context(user_db),
            )
        else:
            patient_ctx: PatientContextResult = await fetch_patient_context(user_db)

        masking_map = patient_ctx.masking_map
        active_therapies = patient_ctx.active_therapies
        patient_context_xml = patient_ctx.context_xml

    scrubbed_history: list[ChatMessage] = []
    for m in history:
        if m.role == "user":
            scrubbed_text, masking_map = mask_user_text_with_dynamic_tokens(m.content, masking_map)
            scrubbed_history.append(ChatMessage(role="user", content=scrubbed_text))
        else:
            scrubbed_history.append(m)

    chat_history = list(scrubbed_history)
    if chunks or patient_context_xml:
        chat_history[-1] = ChatMessage(
            role="user",
            content=build_dual_rag_message(
                scrubbed_history[-1].content,
                patient_context_xml=patient_context_xml,
                clinical_chunks=chunks,
            ),
        )

    async def _extract() -> list[ProposedEntry]:
        is_report = attachment_kind in ("medical_report", "pdf_document") or media_mime_type == "application/pdf"
        if not is_report or not (media_bytes and media_mime_type):
            return await extract_clinical_proposals(
                llm,
                scrubbed_history,
                masking_map=masking_map,
                active_therapies=active_therapies,
            )

        res_proposals = await extract_clinical_proposals(
            llm,
            scrubbed_history,
            masking_map=masking_map,
            active_therapies=active_therapies,
        )
        try:
            rep = await extract_from_report(
                llm,
                media_bytes=media_bytes,
                media_mime_type=media_mime_type,
                masking_map=masking_map,
                active_therapies=active_therapies,
            )
            if rep and rep.proposed_entries:
                res_proposals.extend(rep.proposed_entries)
        except Exception as exc:
            logger.warning("extract_from_report fallito exc={}", type(exc).__name__)

        return res_proposals

    # Avvia l'estrazione HITL in background contemporaneamente allo stream
    extract_task = asyncio.create_task(_extract())
    accumulated_raw: list[str] = []
    buffer = ""

    try:
        if hasattr(llm, "stream_chat"):
            raw_stream = llm.stream_chat(
                chat_history,
                system=get_chat_system_prompt(),
                media_bytes=media_bytes,
                media_mime_type=media_mime_type,
            )
        else:
            # Fallback se il provider non ha stream_chat
            async def _single_chunk_stream():
                res = await llm.chat(
                    chat_history,
                    system=get_chat_system_prompt(),
                    media_bytes=media_bytes,
                    media_mime_type=media_mime_type,
                )
                yield res

            raw_stream = _single_chunk_stream()

        async for chunk in raw_stream:
            accumulated_raw.append(chunk)
            buffer += chunk

            # Buffer di sicurezza anti-spezzamento placeholder (es: '{{NO' ... 'ME}}')
            # Se nel buffer c'è un '{{' senza un '}}', trattieni la porzione finale
            if "{{" in buffer and "}}" not in buffer[buffer.rfind("{{") :]:
                safe_pos = buffer.rfind("{{")
                ready_to_emit = buffer[:safe_pos]
                buffer = buffer[safe_pos:]
            else:
                ready_to_emit = buffer
                buffer = ""

            if ready_to_emit:
                clean_chunk = rehydrate_assistant_reply(ready_to_emit, masking_map)
                yield {"type": "delta", "text": clean_chunk}

        if buffer:
            clean_chunk = rehydrate_assistant_reply(buffer, masking_map)
            yield {"type": "delta", "text": clean_chunk}

        raw_full_reply = "".join(accumulated_raw)
        rehydrated_full = rehydrate_assistant_reply(raw_full_reply, masking_map)

    except LLMBlocked as exc:
        logger.warning("Risposta bloccata dai filtri durante stream code={}", exc.code)
        if not extract_task.done():
            extract_task.cancel()
        yield {"type": "delta", "text": BLOCKED_REPLY}
        yield {
            "type": "done",
            "result": ChatPipelineResult(reply=BLOCKED_REPLY, proposed_entries=[], tokens_used=None),
        }
        return

    # Attendi l'estrazione HITL
    try:
        proposals = await extract_task
    except Exception as exc:
        logger.warning("Estrazione HITL fallita durante streaming exc={}", type(exc).__name__)
        proposals = []

    chunks_dicts = [
        {
            "chunk_id": c.id,
            "source_name": c.source_name,
            "heading_path": c.heading_path,
            "rrf_score": c.rrf_score,
            "similarity": c.similarity,
        }
        for c in chunks
    ]

    pipeline_res = ChatPipelineResult(
        reply=rehydrated_full,
        proposed_entries=proposals,
        tokens_used=None,
        rag_chunks=chunks_dicts,
    )
    yield {"type": "done", "result": pipeline_res}


async def reply_with_proposals(
    llm: LLMProvider,
    messages: list[ChatMessageIn],
    *,
    user_db: Client | None = None,
    embedder: EmbedderProvider | None = None,
) -> tuple[str, list[ProposedEntry]]:
    """Funzione di compatibilità con i test e con il formato `messages: list[ChatMessageIn]`."""
    history = [ChatMessage(m.role, m.content) for m in messages]
    last_msg = history[-1].content if history else ""
    res = await run_chat_pipeline(
        llm,
        user_message=last_msg,
        history_messages=history[:-1] if len(history) > 1 else [],
        user_db=user_db,
        embedder=embedder,
    )
    return res.reply, res.proposed_entries


async def reply(
    llm: LLMProvider,
    messages: list[ChatMessageIn],
    *,
    user_db: Client | None = None,
    embedder: EmbedderProvider | None = None,
) -> str:
    text, _ = await reply_with_proposals(
        llm,
        messages,
        user_db=user_db,
        embedder=embedder,
    )
    return text
