import asyncio

from loguru import logger
from supabase import Client

from vitality_be.llm.base import ChatMessage, LLMBlocked, LLMProvider
from vitality_be.llm.embedder import EmbedderProvider
from vitality_be.llm.prompts import get_chat_system_prompt
from vitality_be.schemas.chat import ChatMessageIn
from vitality_be.schemas.extraction import ProposedEntry
from vitality_be.services.extractor import extract_clinical_proposals
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

# RISCHIO RESIDUO (prompt injection tramite cronologia): i turni `assistant` arrivano dal
# client, quindi sono FALSIFICABILI: un utente può inventare risposte "dell'assistente" per
# spingere il modello. Mitigazione ora: limiti stretti (30 messaggi, 4000 caratteri ciascuno,
# 16000 in totale), system prompt solo in system_instruction e modello senza accesso a
# DB, tool o dati altrui (la vera difesa). Soluzione futura: cronologia salvata lato server
# (tabelle chat_session/chat_message) e il client invia solo il nuovo messaggio.


async def reply_with_proposals(
    llm: LLMProvider,
    messages: list[ChatMessageIn],
    *,
    user_db: Client | None = None,
    embedder: EmbedderProvider | None = None,
) -> tuple[str, list[ProposedEntry]]:
    """Esegue la pipeline Dual-RAG + Estrazione Strutturata Human-in-the-Loop in parallelo.

    Garantisce che:
    1. Tutti i dati sensibili eventualmente scritti dall'utente (CF, email, telefono, IBAN,
       tessera sanitaria, indirizzi, nome/cognome e nomi di medici) vengano mascherati PRIMA
       di lasciare il server verso Vertex AI.
    2. L'estrazione delle proposte di salvataggio (`proposed_entries`) giri in parallelo
       alla generazione conversazionale con degradazione graziosa totale.
    """
    raw_history = [ChatMessage(m.role, m.content) for m in messages]
    masking_map = PatientMaskingMap()
    active_therapies: tuple[dict, ...] = ()
    chunks: list[RetrievedChunk] = []
    patient_context_xml = ""

    # --- PIPELINE CONVERSAZIONALE (DUAL-RAG) ---
    if user_db is not None:
        last_user_query = raw_history[-1].content
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

    # De-identificazione SEMPRE attiva sui messaggi dell'utente prima dell'invio a Vertex AI
    # (anche se user_db è None, rimuove email, CF, telefono, tessera sanitaria, indirizzi e tokenizza i medici)
    scrubbed_history: list[ChatMessage] = []
    for m in raw_history:
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

    # 4. Generazione risposta conversazionale + Estrazione proposte Human-in-the-Loop in parallelo
    try:
        raw_reply, proposals = await asyncio.gather(
            llm.chat(chat_history, system=get_chat_system_prompt()),
            extract_clinical_proposals(
                llm,
                scrubbed_history,
                masking_map=masking_map,
                active_therapies=active_therapies,
            ),
        )
        return rehydrate_assistant_reply(raw_reply, masking_map), proposals
    except LLMBlocked as exc:
        logger.warning("Risposta bloccata dai filtri code={}", exc.code)
        return BLOCKED_REPLY, []


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
