import base64
from datetime import datetime, timezone
import pytest

from tests.conftest import FakeProvider, GOOD_TOKEN
from vitality_be.db.supabase1 import get_user_db
from vitality_be.services.chat_history import (
    SessionNotFoundError,
    delete_user_session,
    generate_session_title,
    get_or_create_session,
    get_session_with_messages,
    list_user_sessions,
    load_session_messages_for_llm,
    save_chat_turn,
    update_session_title,
)
from vitality_be.services.patient_context import scrub_for_db_storage

MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00\x43\x00" + b"\x01" * 64 +
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00"
    b"\xff\xd9"
)


class MockTableQuery:
    def __init__(self, db: "MockChatDb", table_name: str):
        self.db = db
        self.table_name = table_name
        self.filters = []
        self.orders = []
        self._limit = None
        self._range = None
        self._insert_data = None
        self._update_data = None
        self._is_delete = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, col: str, val):
        self.filters.append((col, val))
        return self

    def order(self, col: str, desc: bool = False):
        self.orders.append((col, desc))
        return self

    def limit(self, count: int):
        self._limit = count
        return self

    def range(self, start: int, end: int):
        self._range = (start, end)
        return self

    def insert(self, data):
        self._insert_data = data
        return self

    def update(self, data):
        self._update_data = data
        return self

    def delete(self):
        self._is_delete = True
        return self

    def execute(self):
        table = self.db.tables[self.table_name]

        if self._insert_data is not None:
            items = self._insert_data if isinstance(self._insert_data, list) else [self._insert_data]
            inserted = []
            for item in items:
                row = dict(item)
                row["id"] = self.db.next_id
                self.db.next_id += 1
                row["created_at"] = datetime.now(timezone.utc).isoformat()
                table.append(row)
                inserted.append(row)
            class Res:
                data = inserted
            return Res()

        if self._update_data is not None:
            updated = []
            for row in table:
                match = all(row.get(c) == v for c, v in self.filters)
                if match:
                    row.update(self._update_data)
                    updated.append(dict(row))
            class Res:
                data = updated
            return Res()

        if self._is_delete:
            deleted = []
            kept = []
            for row in table:
                match = all(row.get(c) == v for c, v in self.filters)
                if match:
                    deleted.append(row)
                    # Se eliminiamo una sessione, eliminiamo a cascata i suoi messaggi
                    if self.table_name == "chat_session":
                        s_id = row["id"]
                        self.db.tables["chat_message"] = [
                            m for m in self.db.tables["chat_message"] if m["session_id"] != s_id
                        ]
                else:
                    kept.append(row)
            self.db.tables[self.table_name] = kept
            class Res:
                data = deleted
            return Res()

        # Select
        results = [dict(r) for r in table]
        for col, val in self.filters:
            results = [r for r in results if r.get(col) == val]

        for col, desc in reversed(self.orders):
            results.sort(key=lambda r: r.get(col) or "", reverse=desc)

        if self._range:
            start, end = self._range
            results = results[start : end + 1]
        elif self._limit:
            results = results[: self._limit]

        class Res:
            data = results
        return Res()


class MockChatDb:
    def __init__(self):
        self.next_id = 1
        self.tables = {
            "chat_session": [],
            "chat_message": [],
        }

    def table(self, name: str) -> MockTableQuery:
        if name not in self.tables:
            self.tables[name] = []
        return MockTableQuery(self, name)


def test_scrub_for_db_storage():
    text = (
        "Ciao sono Mario Rossi, il mio CF è RSSMRA60E20H501Z, la mia email è test@email.com, "
        "telefono 333 1234567, abito in via Roma 12. Mi segue il Dott. Marco Ferri."
    )
    cleaned = scrub_for_db_storage(text)
    # Il nome del paziente e del medico rimangono per naturalezza
    assert "Mario Rossi" in cleaned
    assert "Dott. Marco Ferri" in cleaned
    # I dati critici vengono oscurati
    assert "RSSMRA60E20H501Z" not in cleaned
    assert "[CF_RIMOSSO]" in cleaned
    assert "test@email.com" not in cleaned
    assert "[EMAIL_RIMOSSA]" in cleaned
    assert "333 1234567" not in cleaned
    assert "[TELEFONO_RIMOSSO]" in cleaned
    assert "via Roma 12" not in cleaned
    assert "[INDIRIZZO_RIMOSSO]" in cleaned


def test_generate_session_title():
    assert generate_session_title("Ho mangiato una pasta al pesto") == "Ho mangiato una pasta al pesto"
    long_msg = "A" * 80
    assert len(generate_session_title(long_msg)) <= 50
    assert generate_session_title("Foto", attachment_kind="meal_photo").startswith("Foto pasto")


@pytest.mark.asyncio
async def test_chat_history_service_crud():
    db = MockChatDb()

    # 1. Creazione sessione
    session = await get_or_create_session(db, "user-1", None, "Domanda sui pasti")
    s_id = session["id"]
    assert s_id == 1
    assert "Domanda" in session["title"]

    # 2. Salvataggio coppia turni (con PII da oscurare)
    u_id, a_id = await save_chat_turn(
        db,
        session_id=s_id,
        user_content="La mia email è test@email.com, posso mangiare la pizza?",
        assistant_content="Ciao Mario, puoi mangiarla con moderazione.",
        attachment_meta=None,
        rag_chunks=[{"source_name": "ADA 2026"}],
        tokens_used=120,
    )
    assert u_id is not None
    assert a_id is not None

    # Verifica contenuto salvato a DB
    stored_msgs = db.tables["chat_message"]
    assert len(stored_msgs) == 2
    assert "[EMAIL_RIMOSSA]" in stored_msgs[0]["content"]
    assert stored_msgs[1]["tokens_used"] == 120

    # 3. Caricamento messaggi per LLM
    llm_history = await load_session_messages_for_llm(db, s_id)
    assert len(llm_history) == 2
    assert llm_history[0].role == "user"
    assert llm_history[1].role == "assistant"

    # 4. Elenco sessioni
    sessions = await list_user_sessions(db)
    assert len(sessions) == 1
    assert sessions[0]["id"] == s_id

    # 5. Dettaglio sessione con messaggi
    s_detail, msgs = await get_session_with_messages(db, s_id)
    assert s_detail["id"] == s_id
    assert len(msgs) == 2

    # 6. Aggiornamento titolo
    updated = await update_session_title(db, s_id, "Nuovo Titolo Chat")
    assert updated["title"] == "Nuovo Titolo Chat"

    # 7. Cancellazione sessione e cascade
    await delete_user_session(db, s_id)
    assert len(db.tables["chat_session"]) == 0
    assert len(db.tables["chat_message"]) == 0

    with pytest.raises(SessionNotFoundError):
        await get_session_with_messages(db, s_id)


class FakeEmbedder:
    async def embed_query(self, _query: str):
        return [0.0] * 1536


@pytest.mark.asyncio
async def test_api_chat_flow_and_multimodal_attachment(make_client, auth):
    from vitality_be.llm.embedder import get_embedder

    mock_db = MockChatDb()
    provider = FakeProvider(reply="Ecco i consigli nutrizionali.")
    c, app = make_client(provider)
    app.dependency_overrides[get_user_db] = lambda: mock_db
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()

    # 1. POST /api/chat senza session_id -> crea nuova sessione e salva messaggi
    b64_img = base64.b64encode(MINIMAL_JPEG).decode()
    payload = {
        "message": "Ho mangiato questo piatto a pranzo",
        "attachment": {
            "kind": "meal_photo",
            "mime_type": "image/jpeg",
            "data_base64": b64_img,
        },
    }
    r = await c.post("/api/chat", json=payload, headers=auth)
    assert r.status_code == 200, r.json()
    data = r.json()
    assert data["reply"] == "Ecco i consigli nutrizionali."
    assert data["session_id"] == 1
    assert data["user_message_id"] is not None
    assert data["assistant_message_id"] is not None
    assert data["attachment_info"]["has_attachment"] is True
    assert data["attachment_info"]["kind"] == "meal_photo"

    # 2. Secondo turno sulla stessa sessione
    payload_2 = {
        "session_id": 1,
        "message": "Quanti carboidrati conteneva?",
    }
    r2 = await c.post("/api/chat", json=payload_2, headers=auth)
    assert r2.status_code == 200
    assert r2.json()["session_id"] == 1

    # 3. GET /api/chat/sessions
    r_list = await c.get("/api/chat/sessions", headers=auth)
    assert r_list.status_code == 200
    sessions = r_list.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == 1

    # 4. GET /api/chat/sessions/1
    r_detail = await c.get("/api/chat/sessions/1", headers=auth)
    assert r_detail.status_code == 200
    detail = r_detail.json()
    assert len(detail["messages"]) == 4  # 2 turni x 2 messaggi (user + assistant)
    # Il primo messaggio utente ha l'attachment_info
    assert detail["messages"][0]["attachment_info"] is not None
    assert detail["messages"][0]["attachment_info"]["kind"] == "meal_photo"

    # 5. PATCH /api/chat/sessions/1 (rinomina)
    r_patch = await c.patch("/api/chat/sessions/1", json={"title": "Pranzo carboidrati"}, headers=auth)
    assert r_patch.status_code == 200
    assert r_patch.json()["title"] == "Pranzo carboidrati"

    # 6. DELETE /api/chat/sessions/1
    r_del = await c.delete("/api/chat/sessions/1", headers=auth)
    assert r_del.status_code == 204

    # 7. GET su sessione eliminata -> 404
    r_not_found = await c.get("/api/chat/sessions/1", headers=auth)
    assert r_not_found.status_code == 404
