import json
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import embed_chunks as ec  # noqa: E402

DIM = 4


def make_config(**overrides):
    values = dict(
        project="vitality-assist",
        endpoint=ec.DEFAULT_ENDPOINT,
        location="eu",
        model="gemini-embedding-2",
        dim=DIM,
        max_concurrency=2,
        max_retries=2,
    )
    values.update(overrides)
    return ec.EmbeddingConfig(**values)


class FakeTokens:
    def __init__(self):
        self.calls = []

    def get_token(self, force_refresh=False):
        self.calls.append(force_refresh)
        return "tok-refreshed" if force_refresh else "tok"


def ok_response(vector=None):
    return httpx.Response(200, json={"embedding": {"values": vector or [0.1, 0.2, 0.3, 0.4]}})


async def no_sleep(_delay):
    return None


def make_embedder(handler, config=None, tokens=None):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ec.VertexEmbedder(config or make_config(), tokens or FakeTokens(), client, sleep=no_sleep), client


class ConfigTests(unittest.TestCase):
    def test_url_uses_regional_eu_endpoint(self):
        url = make_config().url
        self.assertEqual(
            url,
            "https://aiplatform.eu.rep.googleapis.com/v1/projects/vitality-assist/locations/eu"
            "/publishers/google/models/gemini-embedding-2:embedContent",
        )

    def test_rejects_wrong_endpoint_location_model_and_dim(self):
        for override in (
            {"endpoint": "https://aiplatform.googleapis.com"},
            {"endpoint": "http://aiplatform.eu.rep.googleapis.com"},
            {"location": "europe-west1"},
            {"model": "text-embedding-005"},
            {"dim": 0},
            {"dim": 4000},
            {"project": ""},
        ):
            with self.assertRaises(ec.ConfigError, msg=str(override)):
                ec.validate_config(make_config(**override))

    def test_validate_vector(self):
        self.assertEqual(ec.validate_vector([1, 2.5], 2), [1.0, 2.5])
        for bad in ("x", [1], [1, "a"], [1, float("nan")], [True, 1]):
            with self.assertRaises(ValueError):
                ec.validate_vector(bad, 2)


class EmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_shape_and_success(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return ok_response()

        embedder, client = make_embedder(handler)
        async with client:
            vector = await embedder.embed("title: T | text: ciao")
        self.assertEqual(vector, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(seen["url"].endswith(":embedContent"))
        self.assertEqual(seen["auth"], "Bearer tok")
        self.assertEqual(seen["body"]["content"]["parts"], [{"text": "title: T | text: ciao"}])
        self.assertEqual(seen["body"]["outputDimensionality"], DIM)

    async def test_retries_transient_errors_honouring_retry_after(self):
        delays = []

        async def record_sleep(delay):
            delays.append(delay)

        statuses = iter([429, 503])

        def handler(request):
            try:
                status = next(statuses)
            except StopIteration:
                return ok_response()
            return httpx.Response(status, headers={"retry-after": "3"} if status == 429 else {})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        embedder = ec.VertexEmbedder(make_config(), FakeTokens(), client, sleep=record_sleep)
        async with client:
            await embedder.embed("x")
        self.assertEqual(len(delays), 2)
        self.assertGreaterEqual(delays[0], 3.0)

    async def test_gives_up_after_max_retries(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(503)

        embedder, client = make_embedder(handler, make_config(max_retries=2))
        async with client:
            with self.assertRaises(ec.EmbeddingError):
                await embedder.embed("x")
        self.assertEqual(len(calls), 3)

    async def test_network_errors_are_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("boom")
            return ok_response()

        embedder, client = make_embedder(handler)
        async with client:
            self.assertEqual(len(await embedder.embed("x")), DIM)
        self.assertEqual(len(calls), 2)

    async def test_non_retryable_error_does_not_leak_body(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT", "message": "SEGRETO"}})

        embedder, client = make_embedder(handler)
        async with client:
            with self.assertRaises(ec.EmbeddingError) as ctx:
                await embedder.embed("x")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("SEGRETO", str(ctx.exception))
        self.assertIn("400", str(ctx.exception))

    async def test_expired_token_is_refreshed_once(self):
        tokens = FakeTokens()
        auths = []

        def handler(request):
            auths.append(request.headers["authorization"])
            return httpx.Response(401) if len(auths) == 1 else ok_response()

        embedder, client = make_embedder(handler, tokens=tokens)
        async with client:
            await embedder.embed("x")
        self.assertEqual(auths, ["Bearer tok", "Bearer tok"])
        self.assertIn(True, tokens.calls)

    async def test_persistent_401_and_403_are_fatal(self):
        for status in (401, 403):
            embedder, client = make_embedder(lambda r, s=status: httpx.Response(s))
            async with client:
                with self.assertRaises(ec.FatalEmbeddingError):
                    await embedder.embed("x")

    async def test_invalid_success_payload_is_error_without_retry(self):
        for payload in ({}, {"embedding": {"values": [1, 2]}}, {"embedding": {"values": "x"}}):
            calls = []

            def handler(request, p=payload, c=calls):
                c.append(1)
                return httpx.Response(200, json=p)

            embedder, client = make_embedder(handler)
            async with client:
                with self.assertRaises(ec.EmbeddingError):
                    await embedder.embed("x")
            self.assertEqual(len(calls), 1)


def chunk(doc, index, text="t", digest=None):
    return {
        "doc_id": doc,
        "chunk_index": index,
        "embedding_input": f"title: x | text: {text}",
        "content_hash": digest or f"h-{doc}-{index}-{text}",
    }


class PlanAndRunTests(unittest.IsolatedAsyncioTestCase):
    def test_plan_work_reuses_only_matching_hash(self):
        a, b = chunk("d", 0), chunk("d", 1, "nuovo")
        existing = {
            ("d", 0): {"doc_id": "d", "chunk_index": 0, "content_hash": a["content_hash"]},
            ("d", 1): {"doc_id": "d", "chunk_index": 1, "content_hash": "vecchio"},
        }
        pending, reusable = ec.plan_work([a, b], existing)
        self.assertEqual([c["chunk_index"] for c in pending], [1])
        self.assertEqual(len(reusable), 1)

    def test_load_existing_ignores_other_model_or_dim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.jsonl"
            good = {"doc_id": "d", "chunk_index": 0, "content_hash": "h",
                    "embedding_model": "gemini-embedding-2", "embedding_dim": DIM, "embedding": [0.0] * DIM}
            other = {**good, "chunk_index": 1, "embedding_dim": 8, "embedding": [0.0] * 8}
            path.write_text("\n".join(json.dumps(x) for x in (good, other)), encoding="utf-8")
            loaded = ec.load_existing(path, "gemini-embedding-2", DIM)
        self.assertEqual(list(loaded), [("d", 0)])

    def test_load_chunks_reports_invalid_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            rows = [chunk("d", 0), chunk("d", 0), {"doc_id": "d"}]
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            chunks, problems = ec.load_chunks(path)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(problems), 2)

    async def test_embed_pending_continues_after_single_failure(self):
        def handler(request):
            text = json.loads(request.content)["content"]["parts"][0]["text"]
            if "rotto" in text:
                return httpx.Response(400)
            seed = float(len(text))
            return ok_response([seed, seed, seed, seed])

        embedder, client = make_embedder(handler)
        pending = [chunk("d", 0, "uno"), chunk("d", 1, "rotto"), chunk("d", 2, "tre tre")]
        async with client:
            done, errors, fatal = await ec.embed_pending(embedder, pending, make_config())
        self.assertFalse(fatal)
        self.assertEqual(sorted(r["chunk_index"] for r in done), [0, 2])
        self.assertEqual(len(errors), 1)
        self.assertNotEqual(done[0]["embedding"], done[1]["embedding"])

    async def test_fatal_error_stops_remaining_work(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(403)

        embedder, client = make_embedder(handler)
        pending = [chunk("d", i) for i in range(6)]
        async with client:
            done, errors, fatal = await ec.embed_pending(embedder, pending, make_config(max_concurrency=1))
        self.assertTrue(fatal)
        self.assertEqual(done, [])
        self.assertEqual(len(calls), 1)

    async def test_generate_end_to_end_reuses_on_second_run(self):
        def handler(request):
            return ok_response()

        with tempfile.TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            out = Path(tmp) / "emb" / "embeddings.jsonl"
            chunks_path.write_text(
                "\n".join(json.dumps(chunk("d", i)) for i in range(3)), encoding="utf-8"
            )
            transport = httpx.MockTransport(handler)
            code = await ec.generate(make_config(), chunks_path, out, False, True, FakeTokens(), transport)
            self.assertEqual(code, 0)
            self.assertEqual(len(out.read_text().splitlines()), 3)

            def must_not_be_called(request):
                raise AssertionError("nessuna chiamata attesa alla seconda esecuzione")

            code = await ec.generate(
                make_config(), chunks_path, out, False, True, FakeTokens(), httpx.MockTransport(must_not_be_called)
            )
            self.assertEqual(code, 0)

    async def test_generate_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.jsonl"
            out = Path(tmp) / "embeddings.jsonl"
            chunks_path.write_text(json.dumps(chunk("d", 0)), encoding="utf-8")
            code = await ec.generate(make_config(), chunks_path, out, True, True)
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
