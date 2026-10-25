import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import chunk_markdown as cm  # noqa: E402

FM = """---
doc_id: "iss-test"
source_name: "Iperglicemia nel diabete"
source_org: "ISS"
doc_type: "patient_information"
audience: "general_public"
jurisdiction: "IT"
language: "it"
population: "Persone con diabete"
publication_year: 2026
last_verified: 2026-10-24
can_quote_numbers: true
---
"""

BODY = """# Iperglicemia

Introduzione.

## Sintomi

Sete intensa e urinazione frequente.

## Cosa fare

Misurare la glicemia.
"""


def write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_front_matter_maps_to_columns_and_metadata(self):
        path = write(self.root, "a.md", FM + BODY)
        result = cm.chunk_file(path, self.root, 3000)
        first = result.records[0]
        cols = first["columns"]
        self.assertEqual(cols["document_type"], "patient_information")
        self.assertEqual(cols["publication_year"], 2026)
        self.assertEqual(cols["last_verified"], "2026-10-24")
        self.assertNotIn("source_org", cols)
        self.assertEqual(first["metadata"]["source_org"], "ISS")
        self.assertIs(first["metadata"]["can_quote_numbers"], True)
        self.assertEqual(first["metadata"]["md_file"], "a.md")

    def test_h2_split_indexes_and_heading_path(self):
        path = write(self.root, "a.md", FM + BODY)
        records = cm.chunk_file(path, self.root, 3000).records
        self.assertEqual([r["chunk_index"] for r in records], [0, 1, 2])
        self.assertEqual(records[1]["heading_path"], "Iperglicemia > Sintomi")
        self.assertEqual(records[0]["heading_path"], "Iperglicemia")

    def test_embedding_input_and_hash_are_consistent(self):
        path = write(self.root, "a.md", FM + BODY)
        record = cm.chunk_file(path, self.root, 3000).records[1]
        self.assertTrue(record["embedding_input"].startswith("title: Iperglicemia nel diabete | text: "))
        self.assertEqual(record["content_hash"], cm.sha256_text(record["embedding_input"]))
        self.assertTrue(record["content"].startswith("## Sintomi"))

    def test_missing_required_field_is_reported_not_invented(self):
        text = FM.replace('doc_id: "iss-test"\n', "") + BODY
        path = write(self.root, "a.md", text)
        records, counters, errors = cm.collect(self.root, 3000)
        self.assertEqual(records, [])
        self.assertEqual(counters["files_failed"], 1)
        self.assertIn("doc_id", errors[0])

    def test_invalid_date_is_rejected(self):
        text = FM.replace("last_verified: 2026-10-24", "last_verified: ieri") + BODY
        write(self.root, "a.md", text)
        _, counters, errors = cm.collect(self.root, 3000)
        self.assertEqual(counters["files_failed"], 1)
        self.assertIn("last_verified", errors[0])

    def test_duplicate_doc_id_across_files(self):
        write(self.root, "a.md", FM + BODY)
        write(self.root, "b.md", FM + BODY)
        _, counters, errors = cm.collect(self.root, 3000)
        self.assertEqual(counters["files_ok"], 1)
        self.assertEqual(counters["files_failed"], 1)
        self.assertIn("duplicato", errors[0])

    def test_heading_only_and_duplicate_chunks_are_skipped(self):
        body = "# T\n\n## Vuoto\n\n## Uguale\n\nTesto.\n\n## Uguale\n\nTesto.\n"
        path = write(self.root, "a.md", FM + body)
        result = cm.chunk_file(path, self.root, 3000)
        # il solo "# T" iniziale e "## Vuoto" non hanno testo
        self.assertEqual(result.counters["skipped_heading_only"], 2)
        self.assertEqual(result.counters["skipped_duplicate"], 1)
        self.assertEqual([r["chunk_index"] for r in result.records], [0])

    def test_oversize_is_flagged_not_cut(self):
        body = "# T\n\n## Lungo\n\n" + ("parola " * 100)
        path = write(self.root, "a.md", FM + body)
        result = cm.chunk_file(path, self.root, 100)
        self.assertEqual(result.counters["oversize"], 1)
        self.assertGreater(len(result.records[-1]["content"]), 100)

    def test_output_is_written_atomically_as_jsonl(self):
        write(self.root, "a.md", FM + BODY)
        out = self.root / "out" / "chunks.jsonl"
        records, _, _ = cm.collect(self.root, 3000)
        cm.write_jsonl_atomic(out, records)
        lines = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), len(records))
        self.assertEqual(json.loads(lines[0])["doc_id"], "iss-test")
        self.assertEqual(list(out.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
