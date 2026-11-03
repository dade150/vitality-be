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
from vitality_be.services.guardrails import (
    EMERGENCY_REPLY_GENERAL,
    EMERGENCY_REPLY_HYPO,
    JAILBREAK_REPLY,
    OFF_TOPIC_REPLY,
    PRESCRIPTION_DISCLAIMER,
    GuardrailCategory,
    check_and_sanitize_output,
    check_emergency_input,
    check_jailbreak_input,
    classify_user_input,
)
from vitality_be.services.multimodal import AttachmentKind, inspect_pdf_in_ram
from vitality_be.services.patient_context import (
    PatientContextResult,
    PatientMaskingMap,
    build_dual_rag_message,
    fetch_patient_context,
    mask_user_text_with_dynamic_tokens,
    rehydrate_assistant_reply,
    scrub_free_text,
    scrub_report_text,
)
from vitality_be.services.retriever import (
    RetrievedChunk,
    retrieve_chunks,
    retrieve_patient_reports,
    vector_literal,
)

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
    is_emergency: bool = False
    guardrail_category: GuardrailCategory | None = "in_topic"


async def _archive_verified_chat_report(
    *,
    rep,
    user_db: Client | None,
    embedder: EmbedderProvider | None,
    raw_media_bytes: bytes | None = None,
    raw_media_mime: str | None = None,
    user_id: str | None = None,
    fallback_proposals: list[ProposedEntry] | None = None,
    fallback_text: str | None = None,
) -> list[ProposedEntry]:
    """Archivia in `public.patient_reports` un referto medico verificato allegato in chat."""
    is_med = bool(rep and rep.is_medical_report)
    proposals = list((rep.proposed_entries if rep else None) or fallback_proposals or [])
    clean_sum = (rep.clean_summary if rep else None) or ""
    scrub_txt = (rep.scrubbed_content if rep else None) or fallback_text or ""
    rep_type = (rep.report_type if rep else None) or "ANALISI_SANGUE"
    rep_at = (rep.report_at if rep else None)

    has_clinical_proposals = any(p.target_table == "diabetes_data" for p in proposals)
    if not is_med and not (raw_media_mime == "application/pdf" and has_clinical_proposals):
        return []

    if user_db is None or not hasattr(user_db, "table") or not (clean_sum or proposals or scrub_txt):
        return proposals

    try:
        from datetime import datetime, timezone
        import re
        import time

        if not rep_at:
            rep_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        uid = user_id
        if not uid:
            try:
                u_resp = await asyncio.to_thread(
                    lambda: user_db.table("user").select("id").limit(1).execute()
                )
                u_rows = getattr(u_resp, "data", None)
                uid = str(u_rows[0].get("id")) if (u_rows and isinstance(u_rows, list) and len(u_rows) > 0 and u_rows[0].get("id")) else None
            except Exception:
                uid = None

        rep_title = f"Referto {str(rep_type).replace('_', ' ').title()}"
        text_for_embed = f"{clean_sum}\n\n{scrub_txt[:3000]}".strip()
        vector = None
        if embedder is not None and text_for_embed and hasattr(embedder, "embed_document"):
            try:
                vector = await embedder.embed_document(title=rep_title, text=text_for_embed)
            except Exception:
                vector = None

        saved_file_path: str | None = None
        if uid and raw_media_bytes and raw_media_mime and hasattr(user_db, "storage"):
            try:
                ext_map = {
                    "application/pdf": "pdf",
                    "image/jpeg": "jpg",
                    "image/png": "png",
                    "image/webp": "webp",
                }
                ext = ext_map.get(raw_media_mime, "pdf")
                slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", rep_title)[:40].strip("_") or "referto"
                storage_path = f"{uid}/{int(time.time() * 1000)}_{slug}.{ext}"
                await asyncio.to_thread(
                    lambda: user_db.storage.from_("patient-reports").upload(
                        storage_path,
                        raw_media_bytes,
                        {"content-type": raw_media_mime, "upsert": "true"},
                    )
                )
                saved_file_path = storage_path
            except Exception as st_exc:
                logger.debug("Storage upload referto chat opzionale ignorato exc={}", type(st_exc).__name__)

        extracted_params: dict[str, Any] = {}
        for p in proposals:
            if p.target_table == "diabetes_data":
                m_key = str(p.data.get("metric_type") or "MISURA").upper().replace(" ", "_")
                extracted_params[m_key] = p.data.get("metric_value")

        row_data: dict[str, Any] = {
            "source_name": rep_title,
            "report_type": rep_type,
            "report_at": rep_at,
            "content": scrub_txt or clean_sum or rep_title,
            "clean_summary": clean_sum or None,
            "extracted_params": extracted_params,
        }
        if uid:
            row_data["user_id"] = uid
        if saved_file_path:
            row_data["file_path"] = saved_file_path
        if vector is not None:
            row_data["embedding"] = vector_literal(vector)

        try:
            inserted = await asyncio.to_thread(
                lambda: user_db.table("patient_reports").insert(row_data).execute()
            )
        except Exception as ins_exc:
            logger.warning("Insert esteso referto chat fallito ({}): fallback schema base", ins_exc)
            fb_row: dict[str, Any] = {
                "source_name": rep_title,
                "report_type": "ANALISI_SANGUE",
                "report_at": rep_at,
                "content": scrub_txt or clean_sum or rep_title,
            }
            if uid:
                fb_row["user_id"] = uid
            if vector is not None:
                fb_row["embedding"] = vector_literal(vector)
            inserted = await asyncio.to_thread(
                lambda: user_db.table("patient_reports").insert(fb_row).execute()
            )

        rows = getattr(inserted, "data", None)
        if rows and isinstance(rows, list) and len(rows) > 0:
            rep_id = rows[0].get("id")
            if rep_id is not None:
                try:
                    rep_id_int = int(rep_id)
                    updated_props: list[ProposedEntry] = []
                    for p in proposals:
                        if p.target_table == "diabetes_data":
                            d_copy = dict(p.data)
                            d_copy["source_report_id"] = rep_id_int
                            updated_props.append(p.model_copy(update={"data": d_copy}))
                        else:
                            updated_props.append(p)
                    proposals = updated_props
                except (ValueError, TypeError):
                    pass
    except Exception as save_exc:
        logger.warning("Auto-archiviazione referto chat fallita exc={}: {}", type(save_exc).__name__, save_exc)

    return proposals


def _merge_unique_proposals(
    base_props: list[ProposedEntry],
    report_props: list[ProposedEntry],
) -> list[ProposedEntry]:
    """Unisce le proposte evitando duplicati sullo stesso parametro clinico (privilegia report_props con source_report_id)."""
    if not report_props:
        return base_props
    if not base_props:
        return report_props
    seen_metrics: set[str] = set()
    merged: list[ProposedEntry] = []
    for p in report_props:
        merged.append(p)
        if p.target_table == "diabetes_data":
            m_key = str(p.data.get("metric_type") or "").strip().upper().replace(" ", "_")
            if m_key:
                seen_metrics.add(m_key)
    for p in base_props:
        if p.target_table == "diabetes_data":
            m_key = str(p.data.get("metric_type") or "").strip().upper().replace(" ", "_")
            if m_key and m_key in seen_metrics:
                continue
        merged.append(p)
    return merged


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
    user_id: str | None = None,
) -> ChatPipelineResult:
    """Esegue la pipeline Dual-RAG completa con supporto multimodale ed estrazione clinica in parallelo."""
    # --- LIVELLO 1: INPUT DETERMINISTICO (< 1 ms, 0 ms LLM) ---
    emer_match = check_emergency_input(user_message)
    if emer_match:
        logger.info("guardrails L1 emergency triggered kind={}", emer_match.trigger_kind)
        return ChatPipelineResult(
            reply=emer_match.reply,
            proposed_entries=[],
            tokens_used=None,
            rag_chunks=[],
            is_emergency=True,
            guardrail_category="emergency",
        )

    if check_jailbreak_input(user_message):
        logger.info("guardrails L1 jailbreak blocked")
        return ChatPipelineResult(
            reply=JAILBREAK_REPLY,
            proposed_entries=[],
            tokens_used=None,
            rag_chunks=[],
            is_emergency=False,
            guardrail_category="jailbreak_blocked",
        )

    history: list[ChatMessage] = list(history_messages) if history_messages else []

    # Step 1 (Ispezione PDF in RAM): se PDF nativo con testo, estrae in RAM e azzera media_bytes (zero rete/costi)
    actual_media_bytes = media_bytes
    actual_media_mime = media_mime_type
    extracted_pdf_text = ""

    if media_mime_type == "application/pdf" and media_bytes:
        try:
            inspection = inspect_pdf_in_ram(media_bytes)
            if inspection.is_native_digital:
                extracted_pdf_text = inspection.extracted_text
                actual_media_bytes = None
                actual_media_mime = None
                logger.info("chat_pipeline PDF nativo convertito in testo RAM chars={}", inspection.alnum_char_count)
        except Exception as insp_exc:
            logger.debug("Ispezione PDF chat ignorata exc={}", type(insp_exc).__name__)

    full_user_input = user_message
    if extracted_pdf_text:
        full_user_input = (
            f"{user_message}\n\n[Documento PDF allegato estratto in memoria]\n{extracted_pdf_text}"
        ).strip()

    # Assicura che l'ultimo messaggio sia l'input corrente dell'utente
    if not history or history[-1].role != "user" or history[-1].content != full_user_input:
        history.append(ChatMessage(role="user", content=full_user_input))

    masking_map = PatientMaskingMap()
    active_therapies: tuple[dict, ...] = ()
    chunks: list[RetrievedChunk] = []
    patient_context_xml = ""

    # --- LIVELLO 2 + DUAL-RAG + PATIENT CONTEXT IN PARALLELO (asyncio.gather) ---
    scrubbed_query_for_classifier = scrub_free_text(user_message)
    has_attachment_flag = bool(actual_media_bytes or extracted_pdf_text)
    classify_task = classify_user_input(
        llm,
        scrubbed_query_for_classifier,
        has_attachment=has_attachment_flag,
    )

    if user_db is not None:
        last_user_query = scrub_free_text(user_message) if user_message else "referti e salute"

        async def _fetch_rag():
            if embedder is None:
                return [], []
            try:
                query_vector = await embedder.embed_query(last_user_query)
            except Exception:
                query_vector = None

            if query_vector is not None:
                chunks_res, reports_res = await asyncio.gather(
                    retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder, query_vector=query_vector),
                    retrieve_patient_reports(user_db=user_db, query_vector=query_vector, query_text=last_user_query),
                )
                return chunks_res, reports_res
            chunks_res = await retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder)
            return chunks_res, []

        classification, (chunks, reports_res), patient_ctx = await asyncio.gather(
            classify_task,
            _fetch_rag(),
            fetch_patient_context(user_db),
        )
        patient_context_xml = patient_ctx.context_xml
        masking_map = patient_ctx.masking_map
        active_therapies = patient_ctx.active_therapies

        if reports_res:
            rep_lines = [
                f"- Referto [{r.get('report_type')}] - {r.get('source_name')} ({str(r.get('report_at') or '')[:10]}): "
                f"{str(r.get('clean_summary') or r.get('content') or '')[:300]}"
                for r in reports_res
            ]
            patient_context_xml = (
                patient_context_xml
                + "\n\n[Referti Storici Pertinenti Recuperati]\n"
                + "\n".join(rep_lines)
            )
    else:
        classification = await classify_task

    # Valutazione Guardrail Livello 2: Emergenza Acuta Sfumata
    if classification.emergency_risk:
        logger.info("guardrails L2 emergency risk triggered")
        return ChatPipelineResult(
            reply=EMERGENCY_REPLY_GENERAL,
            proposed_entries=[],
            tokens_used=None,
            rag_chunks=[],
            is_emergency=True,
            guardrail_category="emergency",
        )

    # Valutazione Guardrail Livello 2: Richiesta Off-Topic
    if not classification.in_topic:
        logger.info("guardrails L2 off-topic request blocked intent={}", classification.intent)
        return ChatPipelineResult(
            reply=OFF_TOPIC_REPLY,
            proposed_entries=[],
            tokens_used=None,
            rag_chunks=[],
            is_emergency=False,
            guardrail_category="off_topic",
        )

    guardrail_category: GuardrailCategory = "in_topic"
    if classification.is_medical_prescription_request:
        guardrail_category = "prescription_redirect"

    # De-identificazione sui turni utente prima dell'invio al provider
    scrubbed_history: list[ChatMessage] = []
    for m in history:
        if m.role == "user":
            scrubbed_text, masking_map = scrub_report_text(m.content, masking_map)
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

    # Se c'è richiesta di prescrizione, iniettiamo la direttiva clinica nel prompt
    if classification.is_medical_prescription_request:
        prescription_directive = (
            "\n\n[DIRETTIVA DI SICUREZZA CLINICA: L'utente sta chiedendo un'indicazione posologica, "
            "prescrittiva o una modifica della terapia. NON suggerire dosaggi numerici o cambi di farmaco; "
            "spiega i principi educativi generali e raccomanda di concordare qualsiasi decisione "
            "con il proprio medico curante o diabetologo {{MEDICO}}.]"
        )
        chat_history[-1] = ChatMessage(
            role="user",
            content=chat_history[-1].content + prescription_directive,
        )

    # Preparazione task paralleli: generazione risposta + estrazione proposte
    async def _generate_llm() -> LLMChatResult:
        if hasattr(llm, "chat_with_usage"):
            try:
                return await llm.chat_with_usage(
                    chat_history,
                    system=get_chat_system_prompt(),
                    media_bytes=actual_media_bytes,
                    media_mime_type=actual_media_mime,
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
                media_bytes=actual_media_bytes,
                media_mime_type=actual_media_mime,
            )
        except TypeError:
            text = await llm.chat(chat_history, system=get_chat_system_prompt())
        return LLMChatResult(text=text, tokens_used=None)

    async def _extract() -> list[ProposedEntry]:
        # Ottimizzazione Latenza: se non ci sono dati loggabili e non ci sono allegati, risparmiamo la chiamata LLM
        if not classification.has_loggable_data and not (actual_media_bytes or extracted_pdf_text):
            return []

        # Se l'allegato è un referto clinico, un PDF o una foto di un documento, prova l'estrazione referti
        is_report = (
            attachment_kind in ("medical_report", "pdf_document", "generic_image")
            or media_mime_type == "application/pdf"
            or bool(extracted_pdf_text)
        )
        if not is_report or not (actual_media_bytes or extracted_pdf_text):
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
        rep = None
        try:
            rep = await extract_from_report(
                llm,
                report_text=extracted_pdf_text or None,
                media_bytes=actual_media_bytes,
                media_mime_type=actual_media_mime,
                masking_map=masking_map,
                active_therapies=active_therapies,
            )
        except Exception as exc:
            logger.warning("extract_from_report fallito exc={}", type(exc).__name__)

        rep_props = await _archive_verified_chat_report(
            rep=rep,
            user_db=user_db,
            embedder=embedder,
            raw_media_bytes=media_bytes,
            raw_media_mime=media_mime_type,
            user_id=user_id,
            fallback_proposals=res_proposals,
            fallback_text=extracted_pdf_text or None,
        )
        return _merge_unique_proposals(res_proposals, rep_props)

    try:
        llm_result, proposals = await asyncio.gather(_generate_llm(), _extract())
        rehydrated = rehydrate_assistant_reply(llm_result.text, masking_map)

        # --- LIVELLO 3: OUTPUT SANITIZATION GUARDRAIL ---
        out_check = check_and_sanitize_output(rehydrated)
        if out_check.triggered:
            logger.warning("guardrails L3 triggered violation={}", out_check.violation_type)
            rehydrated = out_check.sanitized_text
            guardrail_category = "prescription_redirect"

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
            is_emergency=False,
            guardrail_category=guardrail_category,
        )
    except LLMBlocked as exc:
        logger.warning("Risposta bloccata dai filtri code={}", exc.code)
        return ChatPipelineResult(
            reply=BLOCKED_REPLY,
            proposed_entries=[],
            tokens_used=None,
            is_emergency=False,
            guardrail_category="in_topic",
        )


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
    user_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Esegue la pipeline Dual-RAG con generazione del testo in streaming SSE e task HITL in parallelo.

    Emette:
    - {'type': 'delta', 'text': str} durante la generazione (con safe buffering dei placeholder reversibili)
    - {'type': 'done', 'result': ChatPipelineResult} al termine di testo ed estrazione clinica
    """
    # --- LIVELLO 1: INPUT DETERMINISTICO (< 1 ms, 0 ms LLM) ---
    emer_match = check_emergency_input(user_message)
    if emer_match:
        logger.info("guardrails L1 emergency triggered (stream) kind={}", emer_match.trigger_kind)
        yield {"type": "delta", "text": emer_match.reply}
        yield {
            "type": "done",
            "result": ChatPipelineResult(
                reply=emer_match.reply,
                proposed_entries=[],
                tokens_used=None,
                rag_chunks=[],
                is_emergency=True,
                guardrail_category="emergency",
            ),
        }
        return

    if check_jailbreak_input(user_message):
        logger.info("guardrails L1 jailbreak blocked (stream)")
        yield {"type": "delta", "text": JAILBREAK_REPLY}
        yield {
            "type": "done",
            "result": ChatPipelineResult(
                reply=JAILBREAK_REPLY,
                proposed_entries=[],
                tokens_used=None,
                rag_chunks=[],
                is_emergency=False,
                guardrail_category="jailbreak_blocked",
            ),
        }
        return

    history: list[ChatMessage] = list(history_messages) if history_messages else []

    actual_media_bytes = media_bytes
    actual_media_mime = media_mime_type
    extracted_pdf_text = ""

    if media_mime_type == "application/pdf" and media_bytes:
        try:
            inspection = inspect_pdf_in_ram(media_bytes)
            if inspection.is_native_digital:
                extracted_pdf_text = inspection.extracted_text
                actual_media_bytes = None
                actual_media_mime = None
                logger.info("stream_chat_pipeline PDF nativo convertito in testo RAM chars={}", inspection.alnum_char_count)
        except Exception as insp_exc:
            logger.debug("Ispezione PDF stream ignorata exc={}", type(insp_exc).__name__)

    full_user_input = user_message
    if extracted_pdf_text:
        full_user_input = (
            f"{user_message}\n\n[Documento PDF allegato estratto in memoria]\n{extracted_pdf_text}"
        ).strip()

    if not history or history[-1].role != "user" or history[-1].content != full_user_input:
        history.append(ChatMessage(role="user", content=full_user_input))

    masking_map = PatientMaskingMap()
    active_therapies: tuple[dict, ...] = ()
    chunks: list[RetrievedChunk] = []
    patient_context_xml = ""

    # --- LIVELLO 2 + DUAL-RAG + PATIENT CONTEXT IN PARALLELO (asyncio.gather) ---
    scrubbed_query_for_classifier = scrub_free_text(user_message)
    has_attachment_flag = bool(actual_media_bytes or extracted_pdf_text)
    classify_task = classify_user_input(
        llm,
        scrubbed_query_for_classifier,
        has_attachment=has_attachment_flag,
    )

    if user_db is not None:
        last_user_query = scrub_free_text(user_message) if user_message else "referti e salute"

        async def _fetch_rag():
            if embedder is None:
                return [], []
            try:
                query_vector = await embedder.embed_query(last_user_query)
            except Exception:
                query_vector = None

            if query_vector is not None:
                chunks_res, reports_res = await asyncio.gather(
                    retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder, query_vector=query_vector),
                    retrieve_patient_reports(user_db=user_db, query_vector=query_vector, query_text=last_user_query),
                )
                return chunks_res, reports_res
            chunks_res = await retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder)
            return chunks_res, []

        classification, (chunks, reports_res), patient_ctx = await asyncio.gather(
            classify_task,
            _fetch_rag(),
            fetch_patient_context(user_db),
        )
        patient_context_xml = patient_ctx.context_xml
        masking_map = patient_ctx.masking_map
        active_therapies = patient_ctx.active_therapies

        if reports_res:
            rep_lines = [
                f"- Referto [{r.get('report_type')}] - {r.get('source_name')} ({str(r.get('report_at') or '')[:10]}): "
                f"{str(r.get('clean_summary') or r.get('content') or '')[:300]}"
                for r in reports_res
            ]
            patient_context_xml = (
                patient_context_xml
                + "\n\n[Referti Storici Pertinenti Recuperati]\n"
                + "\n".join(rep_lines)
            )
    else:
        classification = await classify_task

    # Valutazione Guardrail Livello 2: Rischio Emergenza Acuta
    if classification.emergency_risk:
        logger.info("guardrails L2 emergency risk triggered (stream)")
        yield {"type": "delta", "text": EMERGENCY_REPLY_GENERAL}
        yield {
            "type": "done",
            "result": ChatPipelineResult(
                reply=EMERGENCY_REPLY_GENERAL,
                proposed_entries=[],
                tokens_used=None,
                rag_chunks=[],
                is_emergency=True,
                guardrail_category="emergency",
            ),
        }
        return

    # Valutazione Guardrail Livello 2: Richiesta Off-Topic
    if not classification.in_topic:
        logger.info("guardrails L2 off-topic request blocked (stream) intent={}", classification.intent)
        yield {"type": "delta", "text": OFF_TOPIC_REPLY}
        yield {
            "type": "done",
            "result": ChatPipelineResult(
                reply=OFF_TOPIC_REPLY,
                proposed_entries=[],
                tokens_used=None,
                rag_chunks=[],
                is_emergency=False,
                guardrail_category="off_topic",
            ),
        }
        return

    guardrail_category: GuardrailCategory = "in_topic"
    if classification.is_medical_prescription_request:
        guardrail_category = "prescription_redirect"

    scrubbed_history: list[ChatMessage] = []
    for m in history:
        if m.role == "user":
            scrubbed_text, masking_map = scrub_report_text(m.content, masking_map)
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

    # Iniezione direttiva clinica se c'è richiesta di prescrizione
    if classification.is_medical_prescription_request:
        prescription_directive = (
            "\n\n[DIRETTIVA DI SICUREZZA CLINICA: L'utente sta chiedendo un'indicazione posologica, "
            "prescrittiva o una modifica della terapia. NON suggerire dosaggi numerici o cambi di farmaco; "
            "spiega i principi educativi generali e raccomanda di concordare qualsiasi decisione "
            "con il proprio medico curante o diabetologo {{MEDICO}}.]"
        )
        chat_history[-1] = ChatMessage(
            role="user",
            content=chat_history[-1].content + prescription_directive,
        )

    async def _extract() -> list[ProposedEntry]:
        is_report = (
            attachment_kind in ("medical_report", "pdf_document", "generic_image")
            or media_mime_type == "application/pdf"
            or bool(extracted_pdf_text)
        )
        if not is_report or not (actual_media_bytes or extracted_pdf_text):
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
        rep = None
        try:
            rep = await extract_from_report(
                llm,
                report_text=extracted_pdf_text or None,
                media_bytes=actual_media_bytes,
                media_mime_type=actual_media_mime,
                masking_map=masking_map,
                active_therapies=active_therapies,
            )
        except Exception as exc:
            logger.warning("extract_from_report fallito exc={}", type(exc).__name__)

        rep_props = await _archive_verified_chat_report(
            rep=rep,
            user_db=user_db,
            embedder=embedder,
            raw_media_bytes=media_bytes,
            raw_media_mime=media_mime_type,
            user_id=user_id,
            fallback_proposals=res_proposals,
            fallback_text=extracted_pdf_text or None,
        )
        return _merge_unique_proposals(res_proposals, rep_props)

    # Ottimizzazione Latenza: avvia l'estrazione HITL in background SOLO se ci sono dati loggabili o allegati
    extract_task = None
    should_extract = classification.has_loggable_data or bool(actual_media_bytes or extracted_pdf_text)
    if should_extract:
        try:
            asyncio.get_running_loop()
            extract_task = asyncio.create_task(_extract())
        except RuntimeError:
            extract_task = None

    accumulated_raw: list[str] = []
    buffer = ""

    try:
        if hasattr(llm, "stream_chat"):
            raw_stream = llm.stream_chat(
                chat_history,
                system=get_chat_system_prompt(),
                media_bytes=actual_media_bytes,
                media_mime_type=actual_media_mime,
            )
        else:
            # Fallback se il provider non ha stream_chat
            async def _single_chunk_stream():
                res = await llm.chat(
                    chat_history,
                    system=get_chat_system_prompt(),
                    media_bytes=actual_media_bytes,
                    media_mime_type=actual_media_mime,
                )
                yield res

            raw_stream = _single_chunk_stream()

        async for chunk in raw_stream:
            accumulated_raw.append(chunk)
            buffer += chunk

            # Buffer di sicurezza anti-spezzamento placeholder (es: '{{NO' ... 'ME}}')
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
        if extract_task and not extract_task.done():
            extract_task.cancel()
        yield {"type": "delta", "text": BLOCKED_REPLY}
        yield {
            "type": "done",
            "result": ChatPipelineResult(
                reply=BLOCKED_REPLY,
                proposed_entries=[],
                tokens_used=None,
                is_emergency=False,
                guardrail_category="in_topic",
            ),
        }
        return

    # Attendi l'estrazione HITL se era stata avviata
    if extract_task is not None:
        try:
            proposals = await extract_task
        except Exception as exc:
            logger.warning("Estrazione HITL fallita durante streaming exc={}", type(exc).__name__)
            proposals = []
    else:
        proposals = []

    # --- LIVELLO 3: OUTPUT SANITIZATION GUARDRAIL ---
    out_check = check_and_sanitize_output(rehydrated_full)
    if out_check.triggered:
        logger.warning("guardrails L3 triggered in stream violation={}", out_check.violation_type)
        if PRESCRIPTION_DISCLAIMER in out_check.sanitized_text and PRESCRIPTION_DISCLAIMER not in rehydrated_full:
            yield {"type": "delta", "text": PRESCRIPTION_DISCLAIMER}
        rehydrated_full = out_check.sanitized_text
        guardrail_category = "prescription_redirect"

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
        is_emergency=False,
        guardrail_category=guardrail_category,
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
