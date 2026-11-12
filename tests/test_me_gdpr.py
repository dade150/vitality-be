import json
import pytest
from fastapi import HTTPException

from vitality_be.api.routes.me import (
    ConsentUpdateRequest,
    delete_my_account_and_data,
    export_my_personal_data,
    update_my_ai_health_consent,
)
from vitality_be.core.consent import require_ai_health_consent
from vitality_be.core.security import CurrentUser


def run_coro(coro):
    """Driver sincrono per coroutine lineari nei test Windows (evita freeze IOCP Proactor)."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value


class FakeStorageBucket:
    def __init__(self):
        self.removed = []
        self.listed = []

    def list(self, path):
        self.listed.append(path)
        return [{"name": "report_1.pdf"}, {"name": "report_2.jpg"}]

    def remove(self, paths):
        self.removed.extend(paths)
        return [{"name": p} for p in paths]


class FakeStorageClient:
    def __init__(self, bucket):
        self._bucket = bucket

    def from_(self, name):
        return self._bucket


class FakeUserDb:
    def __init__(self, rpc_success: bool = True, table_data: dict | None = None):
        self.rpc_success = rpc_success
        self.rpc_called = False
        self.bucket = FakeStorageBucket()
        self.storage = FakeStorageClient(self.bucket)
        self.deleted_tables: list[str] = []
        self.table_data = table_data or {}

    def rpc(self, name: str, params: dict):
        self.rpc_called = True

        class Q:
            def execute(_self):
                if not self.rpc_success:
                    raise RuntimeError("RPC delete_own_account failed (simulazione storage.protect_delete)")
                return None

        return Q()

    def table(self, name: str):
        self.deleted_tables.append(name)
        current_data = self.table_data.get(name, [])

        class Q:
            def select(_self, *_a, **_k):
                return _self

            def eq(_self, *_a, **_k):
                return _self

            def limit(_self, *_a, **_k):
                return _self

            def delete(_self):
                return _self

            def execute(_self):
                class R:
                    data = current_data

                return R()

        return Q()


def test_delete_account_success_with_storage_cleanup():
    user = CurrentUser(id="user-123", email="user@test.it", token="test-token")
    db = FakeUserDb(rpc_success=True)

    resp = run_coro(delete_my_account_and_data(user=user, user_db=db))
    assert resp.status_code == 204
    assert db.rpc_called is True
    # B1: Verifica che i file nello storage siano stati rimossi prima dell'RPC
    assert "user-123/report_1.pdf" in db.bucket.removed
    assert "user-123/report_2.jpg" in db.bucket.removed


def test_delete_account_fallback_500_when_rpc_fails():
    user = CurrentUser(id="user-456", email="user2@test.it", token="test-token")
    db = FakeUserDb(rpc_success=False)

    with pytest.raises(HTTPException) as exc_info:
        run_coro(delete_my_account_and_data(user=user, user_db=db))

    assert exc_info.value.status_code == 500
    assert db.rpc_called is True
    # B1 & B2: I file storage sono stati rimossi e il fallback tabellare include app_logs
    assert "user-456/report_1.pdf" in db.bucket.removed
    assert "app_logs" in db.deleted_tables
    assert "chat_session" in db.deleted_tables
    assert "patient_reports" in db.deleted_tables
    assert "user" in db.deleted_tables


def test_export_my_personal_data_includes_dialog2_reports_content_and_app_logs():
    user = CurrentUser(id="user-789", email="export@test.it", token="test-token")
    mock_data = {
        "user": [{"id": "user-789", "email": "export@test.it"}],
        "patient_reports": [
            {
                "id": 1,
                "source_name": "esami.pdf",
                "source_path": "user-789/esami.pdf",
                "clean_summary": "Sintesi referto",
                "content": "Testo referto completo originale",
            }
        ],
        "app_logs": [
            {
                "id": 10,
                "level": "INFO",
                "message": "Caricamento referto",
                "route": "/reports",
                "created_at": "2026-11-04T08:00:00Z",
            }
        ],
    }
    db = FakeUserDb(rpc_success=True, table_data=mock_data)

    resp = run_coro(export_my_personal_data(user=user, user_db=db))
    assert resp.status_code == 200
    assert resp.headers["Content-Disposition"].startswith('attachment; filename="diaLog2_Dati_GDPR_')

    bundle = json.loads(resp.body.decode("utf-8"))
    # B4: Applicazione rinominata in diaLog2 e schema_version 1.1
    assert bundle["metadata"]["application"] == "diaLog2"
    assert bundle["metadata"]["schema_version"] == "1.1"
    # B4: Inclusione content nei referti e application_logs
    assert bundle["health_diary"]["medical_reports"][0]["content"] == "Testo referto completo originale"
    assert bundle["application_logs"][0]["message"] == "Caricamento referto"


def test_require_ai_health_consent_allowed_by_token():
    user = CurrentUser(id="user-ai-1", email="ai1@test.it", token="tok", ai_health_consent=True)
    db = FakeUserDb()

    # Consenso presente nel token: passa immediatamente senza interrogare DB
    res = run_coro(require_ai_health_consent(user=user, user_db=db))
    assert res is None


def test_require_ai_health_consent_allowed_by_db_fallback():
    user = CurrentUser(id="user-ai-2", email="ai2@test.it", token="tok", ai_health_consent=False)
    db = FakeUserDb(table_data={"user": [{"ai_health_consent": True}]})

    # Consenso assente nel token ma presente in DB: passa
    res = run_coro(require_ai_health_consent(user=user, user_db=db))
    assert res is None


def test_require_ai_health_consent_forbidden_when_false():
    user = CurrentUser(id="user-ai-3", email="ai3@test.it", token="tok", ai_health_consent=False)
    db = FakeUserDb(table_data={"user": [{"ai_health_consent": False}]})

    # B5: Consenso assente sia in token sia in DB: HTTP 403 Forbidden
    with pytest.raises(HTTPException) as exc_info:
        run_coro(require_ai_health_consent(user=user, user_db=db))

    assert exc_info.value.status_code == 403
    assert "GDPR Art. 9" in exc_info.value.detail


def test_require_ai_health_consent_revoked_in_db_overrides_token():
    # Anche se il JWT user_metadata ha ancora ai_health_consent=True (es. revoca Art. 7.3 non riallineata sul client),
    # la tabella public.user è la fonte autoritativa e blocca l'accesso con 403.
    user = CurrentUser(id="user-ai-revoked", email="revoked@test.it", token="tok", ai_health_consent=True)
    db = FakeUserDb(table_data={"user": [{"ai_health_consent": False}]})

    with pytest.raises(HTTPException) as exc_info:
        run_coro(require_ai_health_consent(user=user, user_db=db))

    assert exc_info.value.status_code == 403
    assert "GDPR Art. 9" in exc_info.value.detail


def test_update_ai_health_consent_success():
    user = CurrentUser(id="user-toggle", email="toggle@test.it", token="tok", ai_health_consent=True)
    db = FakeUserDb(rpc_success=True)

    # Revoca del consenso
    req = ConsentUpdateRequest(ai_health_consent=False)
    res = run_coro(update_my_ai_health_consent(body=req, user=user, user_db=db))
    assert res["status"] == "ok"
    assert res["ai_health_consent"] is False
    assert db.rpc_called is True



