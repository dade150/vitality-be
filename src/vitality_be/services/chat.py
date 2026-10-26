import asyncio

from loguru import logger
from supabase import Client

from vitality_be.llm.base import ChatMessage, LLMBlocked, LLMProvider
from vitality_be.llm.embedder import EmbedderProvider
from vitality_be.llm.prompts import get_chat_system_prompt
from vitality_be.schemas.chat import ChatMessageIn
from vitality_be.services.patient_context import (
    PatientContextResult,
    PatientMaskingMap,
    build_dual_rag_message,
    fetch_patient_context,
    rehydrate_assistant_reply,
    scrub_free_text,
)
from vitality_be.services.retriever import RetrievedChunk, retrieve_chunks

# Risposta fissa e sicura quando i filtri del provider bloccano (HTTP 200, non un errore).
BLOCKED_REPLY = (
    "Non posso rispondere a questa richiesta. "
    "Se hai un problema di salute urgente, chiama subito il 112. "
    "Per altre domande sulla tua salute, parlane con il tuo medico."
)

# RISCHIO RESIDUO (prompt injection tramite cronologia): i turni `assistant` arrivano dal
# client, quindi sono FALSIFICABILI: un utente può inventare risposte "dell'assistente" per
# spingere il modello. Mitigazione ora: limiti stretti (30 messaggi, 4000 caratteri ciascuno,
# 16000 in totale), system prompt solo in system_instruction e modello senza accesso a
# DB, tool o dati altrui (la vera difesa). Soluzione futura: cronologia salvata lato server
# (tabelle chat_session/chat_message) e il client invia solo il nuovo messaggio.


async def reply(
    llm: LLMProvider,
    messages: list[ChatMessageIn],
    *,
    user_db: Client | None = None,
    embedder: EmbedderProvider | None = None,
) -> str:
    history = [ChatMessage(m.role, m.content) for m in messages]
    masking_map = PatientMaskingMap()

    # --- PIPELINE CONVERSAZIONALE (DUAL-RAG) ---
    # 1. analisi della richiesta (intento, emergenza, fuori tema) [futuro]
    # 2. guardrail in ingresso [futuro]
    # 3. recupero parallelo del contesto:
    #    - RAG Statico Ibrido (Top 10 semantici + Top 10 FTS -> Top 8 RRF -> <contesto_clinico>)
    #    - RAG Dinamico Paziente (profilo, diario SMBG, esami, terapie, pasti, attività -> <contesto_paziente>)
    #    Passati come dati nell'ultimo turno utente, MAI nel system prompt.
    if user_db is not None:
        last_user_query = history[-1].content
        if embedder is not None:
            chunks, patient_ctx = await asyncio.gather(
                retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder),
                fetch_patient_context(user_db),
            )
        else:
            chunks: list[RetrievedChunk] = []
            patient_ctx: PatientContextResult = await fetch_patient_context(user_db)

        masking_map = patient_ctx.masking_map

        # De-identificazione del testo utente prima dell'invio a Vertex AI
        history = [
            ChatMessage(
                role=m.role,
                content=scrub_free_text(m.content, masking_map) if m.role == "user" else m.content,
            )
            for m in history
        ]
        scrubbed_last_query = history[-1].content

        if chunks or patient_ctx.context_xml:
            history[-1] = ChatMessage(
                role="user",
                content=build_dual_rag_message(
                    scrubbed_last_query,
                    patient_context_xml=patient_ctx.context_xml,
                    clinical_chunks=chunks,
                ),
            )

    # 4. generazione
    # 5. re-idratazione token reversibili ({{NOME}}, {{MEDICO}}) e guardrail in uscita [futuro]
    # Nessun identificativo diretto del paziente (nome reale, cognome, email...) va mai al provider LLM.

    try:
        raw_reply = await llm.chat(history, system=get_chat_system_prompt())
        return rehydrate_assistant_reply(raw_reply, masking_map)
    except LLMBlocked as exc:
        logger.warning("Risposta bloccata dai filtri code={}", exc.code)
        return BLOCKED_REPLY
