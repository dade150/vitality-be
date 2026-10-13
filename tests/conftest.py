import os
import types as pytypes

import httpx
import pytest
from loguru import logger
from supabase import AuthApiError

os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("SUPABASE_ANON_KEY", "anon-test")

from vitality_be.core import limits, security  # noqa: E402
from vitality_be.core.config import get_settings  # noqa: E402
from vitality_be.llm.base import ChatMessage  # noqa: E402
from vitality_be.llm.factory import get_llm  # noqa: E402

_orig_get_auth_client = security.get_auth_client

GOOD_TOKEN = "token-valido-SEGRETO-123"
USER_ID = "user-1"


class FakeAuth:
    """Finto client Supabase auth: accetta solo GOOD_TOKEN."""

    class auth:  # noqa: N801
        @staticmethod
        def get_user(jwt: str):
            if jwt != GOOD_TOKEN:
                raise AuthApiError("invalid", 401, None)
            return pytypes.SimpleNamespace(user=pytypes.SimpleNamespace(id=USER_ID, email="a@b.it"))


class FakeProvider:
    def __init__(self, reply: str = "ok", exc: Exception | None = None):
        self.reply, self.exc = reply, exc
        self.calls: list[tuple[list[ChatMessage], str | None]] = []

    async def chat(self, messages, *, system=None):
        self.calls.append((messages, system))
        if self.exc:
            raise self.exc
        return self.reply


def _clear_caches():
    get_settings.cache_clear()
    get_llm.cache_clear()
    limits.get_rate_limiter.cache_clear()
    limits.get_gate.cache_clear()
    _orig_get_auth_client.cache_clear()


@pytest.fixture
def make_client(monkeypatch):
    """make_client(provider=FakeProvider(), **env) -> (httpx.AsyncClient, app)"""
    clients = []

    def _make(provider=None, **env):
        for k, v in env.items():
            monkeypatch.setenv(k.upper(), str(v))
        _clear_caches()
        monkeypatch.setattr(security, "get_auth_client", lambda: FakeAuth)
        from vitality_be.main import create_app

        app = create_app()
        app.dependency_overrides[get_llm] = lambda: provider or FakeProvider()
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://testserver",
        )
        clients.append(client)
        return client, app

    yield _make
    _clear_caches()


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {GOOD_TOKEN}"}


@pytest.fixture
def log_sink():
    """Cattura tutto ciò che Loguru scrive (messaggio + eventuale eccezione), come in prod."""
    lines: list[str] = []
    hid = logger.add(lambda m: lines.append(str(m)), level="DEBUG", diagnose=False, backtrace=False)
    yield lines
    logger.remove(hid)
