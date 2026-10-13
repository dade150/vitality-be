from loguru import logger

from vitality_be.llm.base import ChatMessage, LLMBlocked, LLMProvider
from vitality_be.llm.prompts import get_chat_system_prompt
from vitality_be.schemas.chat import ChatMessageIn

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


async def reply(llm: LLMProvider, messages: list[ChatMessageIn]) -> str:
    history = [ChatMessage(m.role, m.content) for m in messages]

    # --- PUNTI D'INNESTO FUTURI (fuori ambito in questa iterazione) ---
    # 1. analisi della richiesta (intento, emergenza, fuori tema)
    # 2. guardrail in ingresso
    # 3. recupero del contesto (RAG): da passare come dati, MAI nel system prompt
    # 4. generazione  <-- unico passo attivo ora
    # 5. guardrail in uscita
    # Nessun identificativo del paziente (nome, email...) va mai incluso nella richiesta.

    try:
        return await llm.chat(history, system=get_chat_system_prompt())
    except LLMBlocked as exc:
        logger.warning("Risposta bloccata dai filtri code={}", exc.code)
        return BLOCKED_REPLY
