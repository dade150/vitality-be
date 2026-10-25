import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scripts import index_chunks as ix

DIM = 3


def chunk(doc="d", index=0, digest="h0"):
    return {
        "doc_id": doc,
        "chunk_index": index,
        "heading_path": "T > S",
        "content": "## S\n\ntesto",
        "content_hash": digest,
        "columns": {
            "source_name": "Fonte",
            "document_type": "patient_information",
            "language": "it",
            "publication_year": 2026,
            "last_verified": "2026-10-24",
        },
        "metadata": {"source_org": "ISS"},
    }


def emb(doc="d", index=0, digest="h0", dim=DIM):
    return {
        "doc_id": doc,
        "chunk_index": index,
        "content_hash": digest,
        "embedding_model": "gemini-embedding-2",
        "embedding_dim": dim,
        "embedding": [0.1] * dim,
    }


class FakeStore:
    def __init__(self, state=None, fail_times=0, drop_writes=False):
        self.state = dict(state or {})
        self.fail_times = fail_times
        self.drop_writes = drop_writes
        self.batches = []

    def fetch_state(self, doc_ids):
        return {k: v for k, v in self.state.items() if k[0] in doc_ids}

    def upsert(self, rows):
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("503 service unavailable")
        self.batches.append(rows)
        if not self.drop_writes:
            for r in rows:
                self.state[(r["doc_id"], r["chunk_index"])] = {
                    "doc_id": r["doc_id"], "chunk_index": r["chunk_index"],
                    "content_hash": r["content_hash"],
                    "embedding_model": r["embedding_model"], "embedding_dim": r["embedding_dim"],
                }


class BuildRowTests(unittest.TestCase):
    def test_row_has_expected_columns_and_no_forbidden_ones(self):
        row = ix.build_row(chunk(), emb(), DIM)
        self.assertEqual(row["embedding"], "[0.1,0.1,0.1]")
        self.assertEqual(row["document_type"], "patient_information")
        self.assertEqual(row["metadata"], {"source_org": "ISS"})
        self.assertFalse(ix.FORBIDDEN_COLUMNS & row.keys())

    def test_hash_dim_vector_and_required_checks(self):
        with self.assertRaises(ValueError):
            ix.build_row(chunk(), emb(digest="altro"), DIM)
        with self.assertRaises(ValueError):
            ix.build_row(chunk(), emb(dim=5), DIM)
        bad = emb()
        bad["embedding"] = [0.1, float("nan"), 0.2]
        with self.assertRaises(ValueError):
            ix.build_row(chunk(), bad, DIM)
        no_type = chunk()
        no_type["columns"]["document_type"] = ""
        with self.assertRaises(ValueError):
            ix.build_row(no_type, emb(), DIM)

    def test_prepare_reports_missing_embedding(self):
        rows, problems = ix.prepare([chunk(), chunk(index=1, digest="h1")], {("d", 0): emb()}, DIM)
        self.assertEqual(len(rows), 1)
        self.assertIn("d#1", problems[0])


class ClassifyTests(unittest.TestCase):
    def test_new_changed_same(self):
        rows = [ix.build_row(chunk(index=i, digest=f"h{i}"), emb(index=i, digest=f"h{i}"), DIM) for i in range(3)]
        state = {
            ("d", 1): {"content_hash": "h1", "embedding_model": "gemini-embedding-2", "embedding_dim": DIM},
            ("d", 2): {"content_hash": "vecchio", "embedding_model": "gemini-embedding-2", "embedding_dim": DIM},
            ("d", 9): {"content_hash": "x", "embedding_model": "m", "embedding_dim": DIM},
        }
        new, changed, same = ix.classify(rows, state)
        self.assertEqual([r["chunk_index"] for r in new], [0])
        self.assertEqual([r["chunk_index"] for r in changed], [2])
        self.assertEqual([r["chunk_index"] for r in same], [1])
        self.assertEqual(ix.stale_keys([chunk(index=i) for i in range(3)], state), [("d", 9)])


class RunTests(unittest.TestCase):
    def run_index(self, store, chunks, embeddings, **kw):
        args = dict(expected_dim=DIM, batch_size=2, dry_run=False, assume_yes=True)
        args.update(kw)
        return ix.run_index(chunks, embeddings, store, **args)

    def data(self, n=5):
        chunks = [chunk(index=i, digest=f"h{i}") for i in range(n)]
        embs = {("d", i): emb(index=i, digest=f"h{i}") for i in range(n)}
        return chunks, embs

    def test_writes_in_batches_and_second_run_skips_everything(self):
        store = FakeStore()
        chunks, embs = self.data()
        self.assertEqual(self.run_index(store, chunks, embs), 0)
        self.assertEqual([len(b) for b in store.batches], [2, 2, 1])
        self.assertEqual(self.run_index(store, chunks, embs), 0)
        self.assertEqual(len(store.batches), 3)

    def test_dry_run_does_not_write(self):
        store = FakeStore()
        chunks, embs = self.data()
        self.assertEqual(self.run_index(store, chunks, embs, dry_run=True), 0)
        self.assertEqual(store.batches, [])

    def test_transient_error_is_retried(self):
        original = ix.time.sleep
        ix.time.sleep = lambda s: None
        try:
            store = FakeStore(fail_times=2)
            chunks, embs = self.data(2)
            self.assertEqual(self.run_index(store, chunks, embs), 0)
        finally:
            ix.time.sleep = original
        self.assertEqual(len(store.state), 2)

    def test_failed_verification_gives_nonzero_exit(self):
        store = FakeStore(drop_writes=True)
        chunks, embs = self.data(2)
        self.assertEqual(self.run_index(store, chunks, embs), 1)

    def test_problems_give_nonzero_exit_but_valid_rows_are_written(self):
        store = FakeStore()
        chunks, embs = self.data(3)
        del embs[("d", 1)]
        self.assertEqual(self.run_index(store, chunks, embs), 1)
        self.assertEqual(sorted(k[1] for k in store.state), [0, 2])

    def test_main_reads_files_and_uses_injected_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, e = Path(tmp) / "c.jsonl", Path(tmp) / "e.jsonl"
            chunks, embs = self.data(2)
            c.write_text("\n".join(json.dumps(x) for x in chunks), encoding="utf-8")
            e.write_text("\n".join(json.dumps(x) for x in embs.values()), encoding="utf-8")
            store = FakeStore()
            code = ix.main(["--chunks", str(c), "--embeddings", str(e), "--dim", str(DIM), "--yes"], store=store)
        self.assertEqual(code, 0)
        self.assertEqual(len(store.state), 2)


if __name__ == "__main__":
    unittest.main()
