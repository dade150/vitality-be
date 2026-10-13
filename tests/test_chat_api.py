import pytest

from tests.conftest import FakeProvider, GOOD_TOKEN
from vitality_be.llm.base import LLMBlocked, LLMError, LLMUnavailable
from vitality_be.services.chat import BLOCKED_REPLY

OK = {"messages": [{"role": "user", "content": "Ciao"}]}


async def test_health_senza_auth(make_client):
    c, _ = make_client()
    r = await c.get("/api/health")
    assert r.status_code == 200


async def test_chat_ok_con_system_prompt(make_client, auth):
    prov = FakeProvider("Buongiorno")
    c, _ = make_client(prov)
    r = await c.post("/api/chat", json=OK, headers=auth)
    assert r.status_code == 200 and r.json() == {"reply": "Buongiorno"}
    assert prov.calls[0][1] and "Vitality Assist" in prov.calls[0][1]
    assert "<!--" not in prov.calls[0][1]  # la nota HTML non va al modello
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "no-referrer"


async def test_401_senza_token(make_client):
    c, _ = make_client()
    r = await c.post("/api/chat", json=OK)
    assert r.status_code == 401 and isinstance(r.json()["detail"], str)


async def test_401_token_falso(make_client):
    c, _ = make_client()
    r = await c.post("/api/chat", json=OK, headers={"Authorization": "Bearer falso"})
    assert r.status_code == 401


async def test_401_prima_di_422(make_client):
    c, _ = make_client()
    r = await c.post("/api/chat", json={"x": 1})
    assert r.status_code == 401


async def test_nessuna_rotta_api_non_protetta(make_client):
    """Enumera le rotte /api/* dallo schema OpenAPI (indipendente dai dettagli interni di
    FastAPI) e verifica che ognuna risponda 401 senza token. Unica eccezione: /api/health."""
    c, app = make_client()
    pubbliche = {"/api/health"}
    verificate = 0
    for path, methods in app.openapi()["paths"].items():
        if not path.startswith("/api") or path in pubbliche:
            continue
        for method in methods:
            url = path.replace("{", "").replace("}", "") or path
            r = await c.request(method.upper(), url)
            assert r.status_code == 401, f"{method.upper()} {path} senza autenticazione ({r.status_code})"
            verificate += 1
    assert verificate >= 2  # almeno me + chat


@pytest.mark.parametrize("payload", [
    {"messages": [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]},
    {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
    {"messages": [{"role": "user", "content": ""}]},
    {"messages": [{"role": "user", "content": "   \n\t  "}]},
    {"messages": [{"role": "user", "content": "\u200b\u202e"}]},
    {"messages": [{"role": "user", "content": "a"}], "extra": 1},
    {"messages": [{"role": "user", "content": "a", "name": "x"}]},
    {"messages": []},
    {"messages": [{"role": "user", "content": "x" * 4001}]},
    {"messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": "a"} for i in range(31)]},
    # totale > 16000 con ogni messaggio <= 4000
    {"messages": [{"role": "user", "content": "x" * 3900}] * 5},
])
async def test_422(make_client, auth, payload):
    c, _ = make_client()
    r = await c.post("/api/chat", json=payload, headers=auth)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str)  # il FE si aspetta una stringa
    assert "xxxx" not in r.text  # l'input non viene ripetuto nella risposta


async def test_normalizzazione(make_client, auth):
    prov = FakeProvider()
    c, _ = make_client(prov)
    body = {"messages": [{"role": "user", "content": "  Ciao\u200b\x00 mondo\r\nok  "}]}
    assert (await c.post("/api/chat", json=body, headers=auth)).status_code == 200
    assert prov.calls[0][0][0].content == "Ciao mondo\nok"


async def test_413_content_length(make_client, auth):
    c, _ = make_client(max_body_bytes=1024)
    big = {"messages": [{"role": "user", "content": "a" * 2000}]}
    r = await c.post("/api/chat", json=big, headers=auth)
    assert r.status_code == 413 and isinstance(r.json()["detail"], str)


async def test_413_flusso_senza_content_length(make_client, auth):
    c, _ = make_client(max_body_bytes=1024)

    async def gen():  # chunked: niente Content-Length
        for _ in range(5):
            yield b"a" * 500

    r = await c.post("/api/chat", content=gen(), headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == 413


async def test_429_con_retry_after(make_client, auth):
    c, _ = make_client(rate_limit_requests=2, rate_limit_window_s=60)
    assert (await c.post("/api/chat", json=OK, headers=auth)).status_code == 200
    assert (await c.post("/api/chat", json=OK, headers=auth)).status_code == 200
    r = await c.post("/api/chat", json=OK, headers=auth)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1
    assert isinstance(r.json()["detail"], str)


async def test_502_su_llm_error(make_client, auth):
    c, _ = make_client(FakeProvider(exc=LLMError("x", code="http_400")))
    r = await c.post("/api/chat", json=OK, headers=auth)
    assert r.status_code == 502 and isinstance(r.json()["detail"], str)


async def test_503_su_llm_unavailable(make_client, auth):
    c, _ = make_client(FakeProvider(exc=LLMUnavailable("x", code="http_429")))
    r = await c.post("/api/chat", json=OK, headers=auth)
    assert r.status_code == 503 and isinstance(r.json()["detail"], str)


async def test_risposta_fissa_su_blocked(make_client, auth):
    c, _ = make_client(FakeProvider(exc=LLMBlocked("x", code="finish_safety")))
    r = await c.post("/api/chat", json=OK, headers=auth)
    assert r.status_code == 200 and r.json()["reply"] == BLOCKED_REPLY


async def test_500_generico(make_client, auth):
    c, _ = make_client(FakeProvider(exc=RuntimeError("boom")))
    r = await c.post("/api/chat", json=OK, headers=auth)
    assert r.status_code == 500 and isinstance(r.json()["detail"], str)
    assert "boom" not in r.text


async def test_timeout_totale_503(make_client, auth):
    import asyncio

    class Lento:
        async def chat(self, messages, *, system=None):
            await asyncio.sleep(5)

    c, _ = make_client(Lento(), chat_request_timeout_s=0.3, llm_timeout_s=0.1)
    r = await c.post("/api/chat", json=OK, headers=auth)
    assert r.status_code == 503


async def test_coda_piena_503(make_client, auth):
    import asyncio

    class Lento:
        async def chat(self, messages, *, system=None):
            await asyncio.sleep(0.6)
            return "ok"

    c, _ = make_client(Lento(), llm_max_concurrency=1, llm_queue_timeout_s=0.1)
    r1, r2 = await asyncio.gather(
        c.post("/api/chat", json=OK, headers=auth), c.post("/api/chat", json=OK, headers=auth)
    )
    assert sorted([r1.status_code, r2.status_code]) == [200, 503]


async def test_host_non_ammesso(make_client, auth):
    c, _ = make_client(allowed_hosts="api.example.com")
    r = await c.get("/api/health")  # Host: testserver
    assert r.status_code == 400


async def test_docs_chiusi_in_prod(make_client, monkeypatch):
    c, _ = make_client(environment="prod", allowed_hosts="testserver")
    for p in ("/docs", "/redoc", "/openapi.json"):
        assert (await c.get(p)).status_code == 404


async def test_log_senza_contenuto_ne_token(make_client, auth, log_sink):
    secret = "FRASE-RISERVATA-DEL-PAZIENTE"
    for exc in (None, LLMError("x", code="c"), LLMUnavailable("x", code="c"),
                LLMBlocked("x", code="c"), RuntimeError(secret)):
        c, _ = make_client(FakeProvider(reply=f"risposta {secret}", exc=exc))
        await c.post("/api/chat", json={"messages": [{"role": "user", "content": secret}]}, headers=auth)
    await c.post("/api/chat", json={"messages": [{"role": "user", "content": secret}]},
                 headers={"Authorization": "Bearer falso-TOKEN-XYZ"})
    tutto = "\n".join(log_sink)
    assert tutto  # il sink ha davvero catturato qualcosa
    assert secret not in tutto
    assert GOOD_TOKEN not in tutto and "falso-TOKEN-XYZ" not in tutto
    assert "Authorization" not in tutto and "Bearer" not in tutto
    assert "user=user-1" in tutto  # l'id utente invece sì
