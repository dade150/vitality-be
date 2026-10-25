import json
import types as pytypes

import httpx
import pytest

from vitality_be.llm.base import ChatMessage, LLMError, LLMUnavailable
from vitality_be.llm.embedder import (
    EchoEmbedder,
    VertexEmbedder,
    format_query_for_embedding,
    validate_embedding_vector,
)
from vitality_be.schemas.chat import ChatMessageIn
from vitality_be.services import chat as chat_service
from vitality_be.services.retriever import (
    RPC_HYBRID_SEARCH,
    RetrievedChunk,
    attach_context_to_message,
    format_rag_context,
    retrieve_chunks,
    vector_literal,
)


class FakeTokens:
    def __init__(self):
        self.calls: list[bool] = []

    def get_token(self, force_refresh: bool = False) -> str:
        self.calls.append(force_refresh)
        return "tok-refreshed" if force_refresh else "tok-initial"


class FakeRpcCall:
    def __init__(self, data=None, exc: Exception | None = None):
        self._data = data
        self._exc = exc

    def execute(self):
        if self._exc:
            raise self._exc
        return pytypes.SimpleNamespace(data=self._data)


class FakeSupabaseClient:
    def __init__(self, rows=None, exc: Exception | None = None):
        self.rows = rows if rows is not None else []
        self.exc = exc
        self.calls: list[tuple[str, dict]] = []

    def rpc(self, fn_name: str, params: dict):
        self.calls.append((fn_name, params))
        return FakeRpcCall(data=self.rows, exc=self.exc)


class FakeEmbedder:
    def __init__(self, vector: list[float] | None = None, exc: Exception | None = None):
        self.vector = vector or [0.25, -0.5, 0.75, 0.1]
        self.exc = exc
        self.queries: list[str] = []

    async def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        if self.exc:
            raise self.exc
        return self.vector


def test_format_query_for_embedding():
    q = format_query_for_embedding("  Cosa   posso mangiare\na colazione? ")
    assert q == "task: search result | query: Cosa posso mangiare a colazione?"


def test_validate_embedding_vector():
    assert validate_embedding_vector([1, -0.5], 2) == [1.0, -0.5]
    for invalid in ("non-lista", [1.0], [1.0, "x"], [True, 0.5], [1.0, float("nan")]):
        with pytest.raises(ValueError):
            validate_embedding_vector(invalid, 2)


async def test_echo_embedder():
    emb = EchoEmbedder(dim=1536)
    vec = await emb.embed_query("Domanda di prova")
    assert len(vec) == 1536
    assert all(isinstance(x, float) for x in vec)


async def test_vertex_embedder_success():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": {"values": [0.1, 0.2, 0.3, 0.4]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        embedder = VertexEmbedder(
            project="vitality-assist",
            location="eu",
            endpoint="https://aiplatform.eu.rep.googleapis.com",
            model="gemini-embedding-2",
            dim=4,
            tokens=FakeTokens(),
            client=client,
        )
        vec = await embedder.embed_query("Ho la glicemia a 180")

    assert vec == [0.1, 0.2, 0.3, 0.4]
    assert seen["url"] == (
        "https://aiplatform.eu.rep.googleapis.com/v1/projects/vitality-assist"
        "/locations/eu/publishers/google/models/gemini-embedding-2:embedContent"
    )
    assert seen["auth"] == "Bearer tok-initial"
    assert seen["body"] == {
        "content": {"parts": [{"text": "task: search result | query: Ho la glicemia a 180"}]},
        "outputDimensionality": 4,
    }


async def test_vertex_embedder_retry_and_401_refresh():
    tokens = FakeTokens()
    statuses = iter([401, 503, 200])

    async def no_sleep(_delay: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        st = next(statuses)
        if st == 200:
            return httpx.Response(200, json={"embedding": {"values": [0.5, 0.5]}})
        return httpx.Response(st)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        embedder = VertexEmbedder(
            project="vitality-assist",
            dim=2,
            max_retries=2,
            tokens=tokens,
            client=client,
            sleep=no_sleep,
        )
        vec = await embedder.embed_query("test")

    assert vec == [0.5, 0.5]
    assert True in tokens.calls


async def test_vertex_embedder_errors_do_not_leak_sensitive_data():
    secret = "DATO-CLINICO-SEGRETO-999"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": secret}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        embedder = VertexEmbedder(
            project="vitality-assist",
            dim=2,
            max_retries=0,
            tokens=FakeTokens(),
            client=client,
        )
        with pytest.raises(LLMError) as ctx:
            await embedder.embed_query(secret)

    assert secret not in str(ctx.value)
    assert ctx.value.code == "embed_http_400"


async def test_retrieve_chunks_calls_hybrid_rpc_and_parses_results():
    sample_rows = [
        {
            "id": 10,
            "doc_id": "ada-2026-ch6",
            "chunk_index": 2,
            "source_name": "ADA Standards of Care 2026",
            "document_type": "clinical_guideline",
            "heading_path": "Glycemic Goals > Hypoglycemia",
            "content": "Treat hypoglycemia (<70 mg/dL) with 15g of fast-acting carbohydrates.",
            "language": "en",
            "similarity": 0.84,
            "fts_rank": 0.42,
            "rrf_score": 0.0325,
        },
        {
            "id": 22,
            "doc_id": "iss-ipoglicemia",
            "chunk_index": 0,
            "source_name": "ISSalute - Ipoglicemia",
            "document_type": "patient_information",
            "heading_path": "Ipoglicemia > Cosa fare",
            "content": "Assumere 15 grammi di zuccheri semplici e ricontrollare dopo 15 minuti.",
            "language": "it",
            "similarity": 0.81,
            "fts_rank": None,
            "rrf_score": 0.0158,
        },
    ]
    db = FakeSupabaseClient(rows=sample_rows)
    emb = FakeEmbedder(vector=[0.1, -0.2, 0.3, 0.4])

    chunks = await retrieve_chunks(
        "Cosa faccio se ho la glicemia sotto 70?",
        user_db=db,
        embedder=emb,
        semantic_top_k=10,
        fts_top_k=10,
        match_count=8,
        rrf_k=60,
    )

    assert len(chunks) == 2
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].source_name == "ADA Standards of Care 2026"
    assert chunks[0].similarity == 0.84
    assert chunks[1].fts_rank is None

    assert len(db.calls) == 1
    fn_name, params = db.calls[0]
    assert fn_name == RPC_HYBRID_SEARCH
    assert params == {
        "query_text": "Cosa faccio se ho la glicemia sotto 70?",
        "query_embedding": vector_literal([0.1, -0.2, 0.3, 0.4]),
        "semantic_top_k": 10,
        "fts_top_k": 10,
        "match_count": 8,
        "rrf_k": 60,
    }


async def test_retrieve_chunks_graceful_degradation_on_failure(log_sink):
    secret = "DOMANDA-RISERVATA-PAZIENTE"
    # 1. Fallimento embedder
    chunks_1 = await retrieve_chunks(
        secret,
        user_db=FakeSupabaseClient(),
        embedder=FakeEmbedder(exc=LLMUnavailable("down", code="embed_timeout")),
    )
    assert chunks_1 == []

    # 2. Fallimento RPC Supabase
    chunks_2 = await retrieve_chunks(
        secret,
        user_db=FakeSupabaseClient(exc=RuntimeError(secret)),
        embedder=FakeEmbedder(),
    )
    assert chunks_2 == []

    logs = "\n".join(log_sink)
    assert secret not in logs


def test_format_and_attach_rag_context():
    chunk = RetrievedChunk(
        id=1,
        doc_id="sid-2025",
        chunk_index=0,
        source_name="Linee Guida SID/AMD 2025",
        document_type="clinical_guideline",
        heading_path="Obiettivi glicemici",
        content="L'obiettivo di HbA1c generale è inferiore al 7%.",
        language="it",
        similarity=0.88,
        fts_rank=0.5,
        rrf_score=0.032,
    )
    ctx = format_rag_context([chunk])
    assert "<contesto_clinico>" in ctx
    assert "[Fonte 1: Linee Guida SID/AMD 2025 | Sezione: Obiettivi glicemici]" in ctx
    assert "L'obiettivo di HbA1c generale è inferiore al 7%." in ctx

    attached = attach_context_to_message("Qual è il valore target di emoglobina glicata?", [chunk])
    assert attached.startswith("<contesto_clinico>")
    assert "<messaggio_utente>\nQual è il valore target di emoglobina glicata?\n</messaggio_utente>" in attached


async def test_chat_service_reply_injects_rag_context_in_user_turn_not_system():
    captured: list[tuple[list[ChatMessage], str | None]] = []

    class CapturingLLM:
        async def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str:
            captured.append((messages, system))
            return "Risposta basata sulle linee guida."

    sample_rows = [
        {
            "id": 5,
            "doc_id": "sinu-2025",
            "chunk_index": 1,
            "source_name": "SINU LARN Porzioni",
            "document_type": "nutrition_guideline",
            "heading_path": "Cereali e derivati",
            "content": "La porzione standard di pane è di 50 grammi.",
            "language": "it",
            "similarity": 0.79,
            "fts_rank": 0.31,
            "rrf_score": 0.031,
        }
    ]
    db = FakeSupabaseClient(rows=sample_rows)
    emb = FakeEmbedder()

    res = await chat_service.reply(
        CapturingLLM(),
        [
            ChatMessageIn(role="user", content="Buongiorno"),
            ChatMessageIn(role="assistant", content="Buongiorno! Come posso aiutarti?"),
            ChatMessageIn(role="user", content="Quanti grammi è una porzione di pane?"),
        ],
        user_db=db,
        embedder=emb,
    )

    assert res == "Risposta basata sulle linee guida."
    messages_sent, system_sent = captured[0]
    # I turni precedenti restano intatti
    assert messages_sent[0].content == "Buongiorno"
    assert messages_sent[1].content == "Buongiorno! Come posso aiutarti?"
    # L'ultimo turno utente contiene il contesto clinico + la domanda
    assert "<contesto_clinico>" in messages_sent[2].content
    assert "SINU LARN Porzioni" in messages_sent[2].content
    assert "Quanti grammi è una porzione di pane?" in messages_sent[2].content
    # Il system prompt NON contiene i chunk recuperati (difesa da prompt injection)
    assert system_sent is not None
    assert "SINU LARN Porzioni" not in system_sent
