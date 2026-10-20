#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estrae gli Standard Quantitativi delle Porzioni (SINU 2025) dal PDF in Markdown.

A differenza delle linee guida CREA, questo documento si basa pesantemente
su TABELLE. Questo script utilizza il modulo di estrazione tabelle nativo di PyMuPDF
per preservare la struttura tabellare in Markdown, fondamentale per il RAG.

Uso:
  python sinu_pdf_extract.py Standard-Quantitativi-delle-PorzioniB_2.pdf -o sinu_porzioni_rag.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit(
        "Dipendenza mancante. Installa PyMuPDF con: python -m pip install pymupdf>=1.23.0"
    ) from exc

DEFAULT_SOURCE_NAME = "Standard quantitativi delle porzioni - Revisione 2025 (LARN)"
DEFAULT_DOC_ID = "sinu-standard-porzioni-2025"


def normalize_space(text: str) -> str:
    """Pulisce gli spazi e i ritorni a capo dal testo."""
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\n", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_noise(text: str, bbox: tuple[float, float, float, float], page_height: float) -> bool:
    """Filtra intestazioni, piè di pagina e numeri di pagina."""
    t = text.strip()
    if not t:
        return True

    # Numeri di pagina isolati
    if re.fullmatch(r"\d{1,3}", t):
        return True

    # Intestazioni/Footer ricorrenti del documento SINU
    upper_t = t.upper()
    noise_phrases = [
        "LARN",
        "LARN V REVISIONE",
        "STANDARD QUANTITATIVI DELLE PORZIONI",
        "STANDARD QUANTITATIVI DELLE",
        "PORZIONI"
    ]
    if upper_t in noise_phrases:
        return True

    # Margini: ignora i testi troppo in alto o troppo in basso
    y0, _, _, y1 = bbox
    if y0 < 45 or y1 > page_height - 40:
        return True

    return False


def format_table_to_markdown(table: fitz.Table) -> str:
    """Converte una tabella estratta in una stringa di tabella Markdown valida."""
    rows = table.extract()
    if not rows:
        return ""

    md_lines = []
    for i, row in enumerate(rows):
        # Pulisce i ritorni a capo dentro le celle (il Markdown non li supporta nelle tabelle)
        clean_row = []
        for cell in row:
            if cell is None:
                clean_row.append("")
            else:
                clean_row.append(str(cell).replace("\n", " ").replace("|", "\\|").strip())

        md_lines.append("| " + " | ".join(clean_row) + " |")

        # Aggiunge la riga di separazione dopo l'intestazione (prima riga)
        if i == 0:
            md_lines.append("|" + "|".join(["---"] * len(clean_row)) + "|")

    return "\n".join(md_lines) + "\n"


def process_text_block(block: dict[str, Any]) -> str:
    """Estrae testo formattato (titoli, elenchi) da un blocco di testo PyMuPDF."""
    text = ""
    is_bold = False
    is_upper = False

    lines = block.get("lines", [])
    if not lines:
        return ""

    # Controlla la formattazione della prima riga per dedurre se è un titolo
    first_line_spans = lines[0].get("spans", [])
    if first_line_spans:
        font = str(first_line_spans[0].get("font", "")).lower()
        flags = int(first_line_spans[0].get("flags", 0))
        if "bold" in font or (flags & 16):
            is_bold = True

    raw_text = "".join(span.get("text", "") for line in lines for span in line.get("spans", []))
    text = normalize_space(raw_text)

    if not text:
        return ""

    if text.isupper() and len(text) > 5 and len(text) < 100:
        is_upper = True

    # Formattazione in Markdown
    if is_upper:
        return f"\n## {text}\n"
    elif is_bold and len(text) < 100:
        return f"\n### {text}\n"
    elif text.startswith(("■", "•", "-")):
        return f"- {text.lstrip('■•- ')}"
    else:
        return f"{text}"


def build_frontmatter(args: argparse.Namespace, pdf_path: Path) -> str:
    """Genera lo YAML frontmatter per il RAG."""
    lines = ["---"]
    lines.append(f"doc_id: \"{args.doc_id}\"")
    lines.append(f"source_name: \"{args.source_name}\"")
    lines.append(f"source_org: \"{args.source_org}\"")
    lines.append(f"source_file: \"{pdf_path.name}\"")
    lines.append(f"doc_type: \"nutrition_guideline\"")
    lines.append(f"jurisdiction: \"IT\"")
    lines.append(f"publication_year: {args.publication_year}")
    lines.append("can_quote_numbers: false")
    lines.append("---")
    return "\n".join(lines)


def extract(pdf_path: Path, output_path: Path, args: argparse.Namespace) -> dict[str, int]:
    doc = fitz.open(pdf_path)
    output: list[str] = [build_frontmatter(args, pdf_path), "", f"# {args.source_name}", ""]

    stats = {"pages_processed": 0, "tables_extracted": 0, "text_blocks": 0}

    for page_idx in range(args.start_page - 1, min(len(doc), args.end_page) if args.end_page else len(doc)):
        page = doc[page_idx]
        stats["pages_processed"] += 1
        page_height = page.rect.height

        # 1. Trova le tabelle nella pagina
        tables = page.find_tables()
        table_bboxes = [fitz.Rect(t.bbox) for t in tables]

        # 2. Estrai i blocchi di testo
        blocks = page.get_text("dict")["blocks"]

        # 3. Combiniamo testo e tabelle ordinandoli per coordinata Y (dall'alto in basso)
        page_elements = []

        # Aggiungiamo le tabelle
        for tab in tables:
            md_table = format_table_to_markdown(tab)
            if md_table.strip():
                page_elements.append((md_table, tab.bbox[1]))  # Usa Y0 per l'ordinamento
                stats["tables_extracted"] += 1

        # Aggiungiamo i testi (escludendo quelli che cadono dentro le tabelle)
        for block in blocks:
            if block.get("type") != 0:  # Ignora le immagini
                continue

            bbox = block["bbox"]
            rect = fitz.Rect(bbox)

            # Salta il testo se fa parte di una tabella (già processata)
            in_table = any(rect.intersects(t_rect) for t_rect in table_bboxes)
            if in_table:
                continue

            # Salta header/footer
            raw_text = "".join(
                span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", []))
            if is_noise(raw_text, bbox, page_height):
                continue

            md_text = process_text_block(block)
            if md_text:
                page_elements.append((md_text, bbox[1]))
                stats["text_blocks"] += 1

        # Ordina gli elementi dall'alto al basso
        page_elements.sort(key=lambda x: x[1])

        # Aggiungi al documento finale
        for el, _ in page_elements:
            output.append(el)

        output.append("\n<!-- ------------------------- -->\n")  # Separatore di pagina visivo

    doc.close()

    # Pulizia righe vuote multiple
    final_text = re.sub(r'\n{3,}', '\n\n', "\n".join(output))

    output_path.write_text(final_text, encoding="utf-8")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estrae PDF SINU in Markdown preservando le tabelle.")
    parser.add_argument("pdf", nargs="?", default="Standard-Quantitativi-delle-PorzioniB_2.pdf")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--source-org", default="SINU")
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--publication-year", default="2024")
    parser.add_argument("--start-page", type=int, default=7, help="Inizia da pagina 7 (inizio del documento utile)")
    parser.add_argument("--end-page", type=int, default=0, help="0 = fino alla fine")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"Errore: File non trovato {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.out).resolve() if args.out else pdf_path.with_suffix(".md")

    try:
        stats = extract(pdf_path, output_path, args)
        print(f"Markdown creato con successo: {output_path}")
        print(f"Statistiche: {json.dumps(stats, indent=2)}")
    except Exception as e:
        print(f"Errore durante l'estrazione: {e}", file=sys.stderr)
        sys.exit(1)