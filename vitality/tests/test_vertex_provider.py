import httpx
import pytest
from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors, types

from vitality_be.llm.base import ChatMessage, LLMBlocked, LLMError, LLMUnavailable
from vitality_be.llm.vertex import VertexProvider

MSGS = [ChatMessage("user", "Ciao"), ChatMessage("assistant", "Salve"), ChatMessage("user", "Come stai?")]


def resp(text="Risposta", finish=types.FinishReason.STOP, parts=None, block=None, no_candidates=False):
    if no_candidates:
        cands = []
    else:
        parts = parts if parts is not None else ([types.Part(text=text)] if text is not None else [])
        cands = [types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)]
    fb = types.GenerateContentResponsePromptFeedback(block_reason=block) if block else None
    return types.GenerateContentResponse(
        candidates=cands, prompt_feedback=fb,
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, thoughts_token_count=3),
    )


class FakeModels:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    async def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        if callable(o):
            return await o()
        return o


class FakeClient:
    def __init__(self, *outcomes):
        self.models = FakeModels(*outcomes)

        class _Aio:
            pass

        self.aio = _Aio()
        self.aio.models = self.models


def make(*outcomes, timeout_s=10.0):
    client = FakeClient(*outcomes)
    p = VertexProvider(project="p", location="eu", model="m", timeout_s=timeout_s,
                       max_output_tokens=1024, temperature=0.3, client=client,
                       retry_backoff_s=(0.0, 0.0))
    return p, client.models


async def test_ok_e_richiesta_corretta():
    p, models = make(resp("Ciao a te"))
    assert await p.chat(MSGS, system="SYS") == "Ciao a te"
    model, contents, cfg = models.calls[0]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert cfg.system_instruction == "SYS"
    assert "SYS" not in "".join(pt.text for c in contents for pt in c.parts)
    assert cfg.max_output_tokens == 1024 and cfg.safety_settings and not cfg.tools
    assert cfg.thinking_config.thinking_level == types.ThinkingLevel.LOW


async def test_scarta_i_thought():
    p, _ = make(resp(parts=[types.Part(text="ragiono", thought=True), types.Part(text="Visibile")]))
    assert await p.chat(MSGS) == "Visibile"


async def test_vuoto():
    for r in (resp(text=None), resp(no_candidates=True), resp(text="   ")):
        p, _ = make(r)
        with pytest.raises(LLMError) as e:
            await p.chat(MSGS)
        assert not isinstance(e.value, (LLMBlocked, LLMUnavailable))


@pytest.mark.parametrize("finish", [types.FinishReason.SAFETY, types.FinishReason.PROHIBITED_CONTENT,
                                    types.FinishReason.BLOCKLIST, types.FinishReason.SPII])
async def test_blocco_sicurezza(finish):
    p, _ = make(resp(text=None, finish=finish))
    with pytest.raises(LLMBlocked):
        await p.chat(MSGS)


async def test_prompt_feedback_bloccato():
    p, _ = make(resp(no_candidates=True, block=types.BlockedReason.SAFETY))
    with pytest.raises(LLMBlocked):
        await p.chat(MSGS)


async def test_max_tokens_tronca_all_ultima_frase():
    p, _ = make(resp("Prima frase. Seconda frase tagliata a me", finish=types.FinishReason.MAX_TOKENS))
    assert await p.chat(MSGS) == "Prima frase."


async def test_max_tokens_senza_testo_errore():
    p, _ = make(resp(text=None, finish=types.FinishReason.MAX_TOKENS))
    with pytest.raises(LLMError) as e:
        await p.chat(MSGS)
    assert e.value.code == "max_tokens_empty"


@pytest.mark.parametrize("code,exc", [
    (400, LLMError), (401, LLMError), (403, LLMError), (404, LLMError),
    (429, LLMUnavailable), (500, LLMUnavailable), (503, LLMUnavailable),
])
async def test_mappatura_api_error(code, exc):
    err = (errors.ClientError if code < 500 else errors.ServerError)(code, {"error": {"message": "SEGRETO-CORPO"}})
    p, models = make(err, err)
    with pytest.raises(LLMError) as e:
        await p.chat(MSGS)
    assert type(e.value) is exc
    assert "SEGRETO-CORPO" not in str(e.value) and e.value.__cause__ is None and e.value.__suppress_context__
    expected_calls = 2 if code in (429, 503) else 1  # al massimo 1 retry, solo 429/503
    assert len(models.calls) == expected_calls


async def test_retry_su_429_poi_ok():
    p, models = make(errors.ClientError(429, {}), resp("Ce l'ho fatta"))
    assert await p.chat(MSGS) == "Ce l'ho fatta"
    assert len(models.calls) == 2


async def test_nessun_retry_se_manca_il_tempo():
    p, models = make(errors.ClientError(429, {}), resp("x"), timeout_s=1.5)
    with pytest.raises(LLMUnavailable):
        await p.chat(MSGS)
    assert len(models.calls) == 1


async def test_timeout():
    import asyncio

    async def lento():
        await asyncio.sleep(5)

    p, _ = make(lento, timeout_s=0.2)
    with pytest.raises(LLMUnavailable) as e:
        await p.chat(MSGS)
    assert e.value.code == "timeout"


async def test_credenziali_e_rete():
    p, _ = make(DefaultCredentialsError("niente creds"))
    with pytest.raises(LLMUnavailable) as e:
        await p.chat(MSGS)
    assert e.value.code == "credentials"
    p, _ = make(httpx.ConnectError("giù"))
    with pytest.raises(LLMUnavailable) as e:
        await p.chat(MSGS)
    assert e.value.code == "network"
