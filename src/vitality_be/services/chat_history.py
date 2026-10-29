"""Servizio di Persistenza Server-Side per Sessioni e Messaggi Chat su Supabase.

Garantisce:
1. Integrità dei turni 'assistant' contro manipolazioni e prompt injection da client.
2. Scrubbing selettivo dei dati sensibili critici prima del salvataggio nel database.
3. Tracciamento strutturato degli allegati multimodali e dei chunk RAG in `rag_context` (JSONB).
4. Protezione Anti-IDOR: tutte le query operano tramite il client `user_db` con RLS dell'utente.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from supabase import Client

from vitality_be.llm.base import ChatMessage
from vitality_be.services.patient_context import scrub_for_db_storage


class SessionNotFoundError(Exception):
    """Sollevata quando una sessione richiesta non esiste o appartiene a un altro utente (RLS)."""


def generate_session_title(first_message: str, attachment_kind: str | None = None) -> str:
    """Genera un titolo compatto e pulito (max 60 caratteri) per la nuova sessione senza costo token."""
    clean = scrub_for_db_storage(first_message).strip()

    if attachment_kind:
        kind_labels = {
            "meal_photo": "Foto pasto",
            "glucometer_photo": "Lettura glicemia",
            "medical_report": "Referto medico",
            "medication_photo": "Foto terapia",
            "pdf_document": "Documento referto",
            "generic_image": "Immagine",
        }
        prefix = kind_labels.get(attachment_kind, "Allegato")
        if not clean or clean.lower().startswith("ecco") or clean.lower().startswith("analizza"):
            return prefix
        return f"{prefix}: {clean[:45]}..." if len(clean) > 45 else f"{prefix}: {clean}"

    if not clean:
        return "Nuova conversazione"

    # Taglia alla prima frase o a max 50 caratteri
    first_sentence = clean.split("\n")[0].split(".")[0].strip()
    if len(first_sentence) > 50:
        return first_sentence[:47].strip() + "..."
    return first_sentence or clean[:50]


async def get_or_create_session(
    user_db: Client,
    user_id: str,
    session_id: int | None,
    title_seed: str,
    attachment_kind: str | None = None,
) -> dict[str, Any]:
    """Recupera la sessione esistente o ne crea una nuova."""
    if session_id is not None:
        def _fetch_session():
            resp = (
                user_db.table("chat_session")
                .select("id, user_id, title, created_at")
                .eq("id", session_id)
                .limit(1)
                .execute()
            )
            data = getattr(resp, "data", None)
            return data[0] if data and isinstance(data, list) and len(data) > 0 else None

        row = await asyncio.to_thread(_fetch_session)
        if not row:
            raise SessionNotFoundError(f"Sessione {session_id} non trovata.")
        return row

    title = generate_session_title(title_seed, attachment_kind)

    def _insert_session():
        resp = (
            user_db.table("chat_session")
            .insert({"user_id": user_id, "title": title})
            .execute()
        )
        data = getattr(resp, "data", None)
        return data[0] if data and isinstance(data, list) and len(data) > 0 else {"id": None, "title": title}

    new_row = await asyncio.to_thread(_insert_session)
    logger.info("chat_session creata id={} user={}", new_row.get("id"), user_id)
    return new_row


async def load_session_messages_for_llm(
    user_db: Client,
    session_id: int,
    limit: int = 20,
) -> list[ChatMessage]:
    """Carica gli ultimi `limit` messaggi da `chat_message` e li converte in formato per Gemini.

    I turni recuperati sono garantiti autentici e immutabili. Se un messaggio `user` storico
    conteneva una foto o un PDF, include un promemoria di contesto contestuale per il modello.
    """
    def _fetch_history():
        resp = (
            user_db.table("chat_message")
            .select("id, role, content, rag_context, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        data = getattr(resp, "data", None)
        return data if data and isinstance(data, list) else []

    rows = await asyncio.to_thread(_fetch_history)
    # Rovesciamo per avere l'ordine cronologico naturale (dal più vecchio al più recente)
    rows.reverse()

    messages: list[ChatMessage] = []
    for r in rows:
        role = "assistant" if r.get("role") == "assistant" else "user"
        content = str(r.get("content") or "").strip()
        rag_ctx = r.get("rag_context") or {}

        # Memoria del mezzo per l'LLM: se in passato è stato allegato un file, informa il modello
        if role == "user" and isinstance(rag_ctx, dict):
            att = rag_ctx.get("attachment")
            if isinstance(att, dict) and att.get("has_attachment"):
                kind = att.get("kind", "file")
                mime = att.get("mime_type", "")
                content = f"[In questo turno l'utente ha allegato: {kind} ({mime})]\n{content}"

        messages.append(ChatMessage(role=role, content=content))

    return messages


async def save_chat_turn(
    user_db: Client,
    *,
    session_id: int,
    user_content: str,
    assistant_content: str,
    attachment_meta: dict[str, Any] | None = None,
    rag_chunks: list[dict[str, Any]] | None = None,
    proposed_entries: list[dict[str, Any]] | None = None,
    tokens_used: int | None = None,
) -> tuple[int | None, int | None]:
    """Salva la coppia di messaggi (user + assistant) in `public.chat_message`.

    Applica lo scrubbing selettivo anti-PII prima della persistenza.
    Degrada con grazia se il database o la tabella non è disponibile (es. test unitari).
    """
    scrubbed_user = scrub_for_db_storage(user_content)
    scrubbed_reply = scrub_for_db_storage(assistant_content)

    user_rag_ctx = {"attachment": attachment_meta} if attachment_meta else None
    asst_rag_ctx: dict[str, Any] = {}
    if rag_chunks:
        asst_rag_ctx["rag_chunks"] = rag_chunks
    if proposed_entries:
        asst_rag_ctx["proposed_entries"] = proposed_entries

    payload_user = {
        "session_id": session_id,
        "role": "user",
        "content": scrubbed_user,
        "rag_context": user_rag_ctx,
    }
    payload_asst = {
        "session_id": session_id,
        "role": "assistant",
        "content": scrubbed_reply,
        "rag_context": asst_rag_ctx if asst_rag_ctx else None,
        "tokens_used": tokens_used,
    }

    def _persist():
        try:
            resp = user_db.table("chat_message").insert([payload_user, payload_asst]).execute()
            data = getattr(resp, "data", None)
            if data and isinstance(data, list) and len(data) >= 2:
                return data[0].get("id"), data[1].get("id")
            return None, None
        except Exception as exc:
            logger.warning("save_chat_turn fallito session_id={} exc={}", session_id, type(exc).__name__)
            return None, None

    u_id, a_id = await asyncio.to_thread(_persist)
    return u_id, a_id


async def list_user_sessions(
    user_db: Client,
    limit: int = 30,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Elenca le sessioni di chat dell'utente autenticato (protette da RLS)."""
    def _list():
        resp = (
            user_db.table("chat_session")
            .select("id, title, created_at")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        data = getattr(resp, "data", None)
        return data if data and isinstance(data, list) else []

    return await asyncio.to_thread(_list)


async def get_session_with_messages(
    user_db: Client,
    session_id: int,
    limit: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recupera i dettagli di una sessione e i relativi messaggi in ordine cronologico.

    Formatta `attachment_info` e `proposed_entries` per il client ed esclude dettagli RAG interni.
    """
    def _fetch():
        s_resp = (
            user_db.table("chat_session")
            .select("id, title, created_at")
            .eq("id", session_id)
            .limit(1)
            .execute()
        )
        s_data = getattr(s_resp, "data", None)
        if not s_data or not isinstance(s_data, list):
            return None, []

        m_resp = (
            user_db.table("chat_message")
            .select("id, session_id, role, content, rag_context, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=True)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        m_data = getattr(m_resp, "data", None)
        rows = m_data if m_data and isinstance(m_data, list) else []
        rows.reverse()
        return s_data[0], rows

    session, raw_messages = await asyncio.to_thread(_fetch)
    if not session:
        raise SessionNotFoundError(f"Sessione {session_id} non trovata.")

    cleaned_messages: list[dict[str, Any]] = []
    for r in raw_messages:
        rag_ctx = r.get("rag_context") or {}
        att_info = None
        props = []

        if isinstance(rag_ctx, dict):
            raw_att = rag_ctx.get("attachment")
            if isinstance(raw_att, dict) and raw_att.get("has_attachment"):
                att_info = {
                    "has_attachment": True,
                    "kind": raw_att.get("kind", "generic_image"),
                    "mime_type": raw_att.get("mime_type", ""),
                    "size_bytes": raw_att.get("size_bytes", 0),
                    "exif_stripped": bool(raw_att.get("exif_stripped")),
                    "sha256_prefix": raw_att.get("sha256_prefix"),
                }
            raw_props = rag_ctx.get("proposed_entries")
            if isinstance(raw_props, list):
                props = raw_props

        cleaned_messages.append(
            {
                "id": r.get("id"),
                "session_id": r.get("session_id"),
                "role": r.get("role"),
                "content": r.get("content"),
                "created_at": r.get("created_at"),
                "attachment_info": att_info,
                "proposed_entries": props,
            }
        )

    return session, cleaned_messages


async def update_session_title(
    user_db: Client,
    session_id: int,
    new_title: str,
) -> dict[str, Any]:
    """Aggiorna il titolo di una sessione."""
    clean_title = scrub_for_db_storage(new_title)[:100].strip()

    def _update():
        resp = (
            user_db.table("chat_session")
            .update({"title": clean_title})
            .eq("id", session_id)
            .execute()
        )
        data = getattr(resp, "data", None)
        return data[0] if data and isinstance(data, list) and len(data) > 0 else None

    updated = await asyncio.to_thread(_update)
    if not updated:
        raise SessionNotFoundError(f"Sessione {session_id} non trovata.")
    return updated


async def delete_user_session(
    user_db: Client,
    session_id: int,
) -> None:
    """Elimina una sessione (e a cascata tutti i messaggi associati)."""
    def _delete():
        resp = (
            user_db.table("chat_session")
            .delete()
            .eq("id", session_id)
            .execute()
        )
        data = getattr(resp, "data", None)
        return data if data and isinstance(data, list) and len(data) > 0 else None

    res = await asyncio.to_thread(_delete)
    if not res:
        raise SessionNotFoundError(f"Sessione {session_id} non trovata.")
