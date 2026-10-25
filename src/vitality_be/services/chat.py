from loguru import logger
from supabase import Client

from vitality_be.llm.base import ChatMessage, LLMBlocked, LLMProvider
from vitality_be.llm.embedder import EmbedderProvider
from vitality_be.llm.prompts import get_chat_system_prompt
from vitality_be.schemas.chat import ChatMessageIn
from vitality_be.services.retriever import attach_context_to_message, retrieve_chunks

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

    # --- PIPELINE CONVERSAZIONALE ---
    # 1. analisi della richiesta (intento, emergenza, fuori tema) [futuro]
    # 2. guardrail in ingresso [futuro]
    # 3. recupero del contesto (RAG ibrido: Top 10 semantici + Top 10 FTS -> Top 8 RRF),
    #    passato come dati nell'ultimo turno utente, MAI nel system prompt.
    if user_db is not None and embedder is not None:
        last_user_query = history[-1].content
        chunks = await retrieve_chunks(last_user_query, user_db=user_db, embedder=embedder)
        if chunks:
            history[-1] = ChatMessage(
                role="user",
                content=attach_context_to_message(last_user_query, chunks),
            )

    # 4. generazione
    # 5. guardrail in uscita [futuro]
    # Nessun identificativo del paziente (nome, email...) va mai incluso nella richiesta.

    try:
        return await llm.chat(history, system=get_chat_system_prompt())
    except LLMBlocked as exc:
        logger.warning("Risposta bloccata dai filtri code={}", exc.code)
        return BLOCKED_REPLY
