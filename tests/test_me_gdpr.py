import pytest
from fastapi import HTTPException

from vitality_be.api.routes.me import delete_my_account_and_data
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
    def __init__(self, rpc_success: bool = True):
        self.rpc_success = rpc_success
        self.rpc_called = False
        self.bucket = FakeStorageBucket()
        self.storage = FakeStorageClient(self.bucket)
        self.deleted_tables: list[str] = []

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

        class Q:
            def select(_self, *_a, **_k):
                return _self

            def eq(_self, *_a, **_k):
                return _self

            def delete(_self):
                return _self

            def execute(_self):
                class R:
                    data = []

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

