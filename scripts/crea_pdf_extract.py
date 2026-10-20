#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estrae le Linee guida CREA dal PDF e le converte in Markdown pronto per RAG.

Uso rapido:
  python crea_pdf_to_rag.py LINEE-GUIDA-DEFINITIVO.pdf -o crea_linee_guida_rag.md

Dipendenza:
  python -m pip install pymupdf

Il programma fa estrazione e normalizzazione deterministica: NON inventa raccomandazioni,
livelli di evidenza o codici. Le intestazioni sono riconosciute con euristiche tipografiche,
quindi il Markdown va comunque verificato soprattutto nelle tabelle e nelle figure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise SystemExit(
        "Dipendenza mancante. Installa PyMuPDF con: python -m pip install pymupdf"
    ) from exc

DEFAULT_SOURCE_NAME = "Linee guida per una sana alimentazione - Revisione 2018"
DEFAULT_DOC_ID = "crea-linee-guida-sana-alimentazione-2018"

BULLET_CHARS = "•▪●◦‣⁃"
BULLET_SPLIT_RE = re.compile(r"\s*[" + re.escape(BULLET_CHARS) + r"]\s*")


def normalize_space(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\xad", "")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def join_pdf_lines(lines: list[str]) -> str:
    """Unisce righe spezzate dal layout PDF, ricomponendo parole sillabate a fine riga."""
    text = " ".join(normalize_space(line) for line in lines if normalize_space(line))
    # Ricompone parole spezzate dal ritorno a capo, per esempio "alimen- tazione".
    text = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])[-‐]\s+([a-zà-öø-ÿ])", r"\1\2", text)
    return normalize_space(text)


def is_mostly_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) >= 0.82


def is_noise_block(text: str, bbox: tuple[float, float, float, float], page_height: float) -> bool:
    if not text:
        return True
    x0, y0, x1, y1 = bbox
    # Numero di pagina / elementi di bordo tipici di questa pubblicazione.
    if y0 > page_height - 32 or y1 < 22:
        return True
    if re.fullmatch(r"\d{1,3}", text):
        return True
    if re.fullmatch(r"(?:pagina|page)\s*\d+(?:\s+di\s+\d+)?", text, re.I):
        return True
    # Elimina linee decorative isolate, non il testo scientifico che le contiene.
    if re.fullmatch(r"[_—–\-─\s]{5,}", text):
        return True
    return False


def get_block_text(block: dict[str, Any]) -> tuple[str, float, bool, tuple[float, float, float, float], str | None, str]:
    """Estrae il testo e, quando possibile, separa il titolo in grassetto dal corpo."""
    if "lines" not in block:
        return None  # type: ignore[return-value]
    raw_lines: list[str] = []
    sizes: list[float] = []
    bold = False
    line_spans = block.get("lines", [])
    for line in line_spans:
        line_parts = []
        for span in line.get("spans", []):
            line_parts.append(str(span.get("text", "")))
            try:
                sizes.append(float(span.get("size", 10.0)))
            except (TypeError, ValueError):
                pass
            font = str(span.get("font", "")).lower()
            flags = int(span.get("flags", 0) or 0)
            if "bold" in font or flags & 16:
                bold = True
        line_text = "".join(line_parts)
        if line_text.strip():
            raw_lines.append(line_text)
    text = join_pdf_lines(raw_lines)
    if not text:
        return None  # type: ignore[return-value]
    bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))

    # Nel PDF CREA talvolta titolo di capitolo e primo periodo condividono lo stesso
    # blocco: separiamo il prefisso numerato in grassetto dal testo normale seguente.
    heading_prefix = None
    remainder = text
    if line_spans:
        first_line_spans = line_spans[0].get("spans", [])
        title_parts: list[str] = []
        rest_parts: list[str] = []
        still_title = True
        for span in first_line_spans:
            st = str(span.get("text", ""))
            font = str(span.get("font", "")).lower()
            flags = int(span.get("flags", 0) or 0)
            span_bold = "bold" in font or bool(flags & 16)
            try:
                span_size = float(span.get("size", 10.0))
            except (TypeError, ValueError):
                span_size = 10.0
            if still_title and span_bold and span_size >= 10.5:
                title_parts.append(st)
            else:
                still_title = False
                rest_parts.append(st)
        candidate = normalize_space("".join(title_parts))
        candidate = re.sub(r"^ϰ\.\s*", "4. ", candidate)
        if re.match(r"^\d{1,2}\.\s*\S", candidate) and len(candidate) >= 8:
            heading_prefix = candidate
            trailing = "".join(rest_parts)
            following = []
            for line in line_spans[1:]:
                line_text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
                if line_text.strip():
                    following.append(line_text)
            remainder = join_pdf_lines(([trailing] if trailing.strip() else []) + following)

    text = text.replace("ϰ. BEVI", "4. BEVI").replace("ϰ. Bevi", "4. Bevi")
    remainder = remainder.replace("ϰ. BEVI", "4. BEVI").replace("ϰ. Bevi", "4. Bevi")
    return text, max(sizes or [10.0]), bold, bbox, heading_prefix, remainder


def classify_heading(text: str, font_size: float, bold: bool) -> int | None:
    """Restituisce livello Markdown H2/H3 per titoli probabili, altrimenti None."""
    t = normalize_space(text).strip(" .")
    if not t or len(t) > 160 or font_size < 9.5:
        return None
    t = re.sub(r"^ϰ\.\s*", "4. ", t)

    # Titoli di capitolo CREA, ad es. "1. CONTROLLA IL PESO...".
    numbered_caps = re.match(r"^(\d{1,2})\.\s+(.+)$", t)
    if numbered_caps and is_mostly_upper(numbered_caps.group(2)) and (bold or font_size >= 11.5):
        return 2 if font_size >= 12.5 else 3

    # Numerazioni delle sottosezioni, spesso in grassetto nel documento.
    if re.match(r"^\d{1,2}\.\s+\S", t) and bold and len(t) <= 105 and not re.search(r"[.!?;:]$", t):
        return 3

    # Titoli di intermezzo/sezione in maiuscolo.
    if is_mostly_upper(t) and len(t) <= 115 and (bold or font_size >= 11.4):
        if t not in {"LINEE GUIDA PER UNA SANA ALIMENTAZIONE", "REVISIONE 2018"}:
            if t in {"INTRODUZIONE", "PRESENTAZIONE", "PREFACE"}:
                return 2
            return 2 if font_size >= 12 or t.startswith(("CAPITOLO", "ALLEGATO")) else 3

    # Pagine divisorie con titolo molto grande in Title Case.
    if font_size >= 22 and len(t) <= 110:
        return 2

    return None


def split_content_block(text: str) -> list[tuple[str, bool, bool]]:
    """Restituisce (testo, è_titolo_inline, inizia_con_bullet_esplicito)."""
    text = normalize_space(text)
    if not text:
        return []

    result: list[tuple[str, bool, bool]] = []
    m = re.match(r"^(COME COMPORTARSI|COME COMPORTARSI\s+NELLA\s+PRATICA)\s*(.*)$", text, re.I)
    if m:
        result.append((m.group(1).title(), True, False))
        text = m.group(2).strip()
        if not text:
            return result

    # Mantiene l'informazione su quali frammenti fossero introdotti da un vero bullet.
    # La prima parte, se presente prima del primo bullet, può essere la continuazione
    # della frase emessa dal blocco PDF precedente.
    if re.search(r"[" + re.escape(BULLET_CHARS) + r"]", text):
        tokens = re.split(r"([" + re.escape(BULLET_CHARS) + r"])", text)
        explicit_next = False
        for token in tokens:
            if token in BULLET_CHARS:
                explicit_next = True
                continue
            part = normalize_space(token).strip(" \t•▪●◦‣⁃")
            if part:
                result.append((part, False, explicit_next))
            explicit_next = False
        return result

    result.append((text, False, False))
    return result


def merge_with_previous_bullet(output: list[str], atom: str, page_no: int, explicit_bullet: bool) -> bool:
    """Unisce una continuazione di frase al bullet precedente; restituisce True se unita."""
    if explicit_bullet or not output or not atom:
        return False
    previous = output[-1]
    m = re.match(r"^- (.*?) <!-- (pagina PDF: (\d+)|pagine PDF: (\d+)-(\d+)) -->$", previous)
    if not m:
        return False
    previous_text = m.group(1)
    prev_start = int(m.group(3) or m.group(4))
    prev_end = int(m.group(3) or m.group(5) or m.group(4))
    # Se la frase precedente è già conclusa e la nuova parte inizia con maiuscola,
    # consideriamo il testo un nuovo punto e non una continuazione.
    if re.search(r"[.!?…]?[\"’”)]?$", previous_text) and previous_text.rstrip().endswith((".", "!", "?", "…")):
        return False
    if not atom[0].islower() and not previous_text.endswith(("-", "‐")):
        return False
    if previous_text.endswith(("-", "‐")):
        combined = previous_text[:-1] + atom
    else:
        combined = previous_text + " " + atom
    first_page = min(prev_start, page_no)
    last_page = max(prev_end, page_no)
    if first_page == last_page:
        comment = f"pagina PDF: {first_page}"
    else:
        comment = f"pagine PDF: {first_page}-{last_page}"
    output[-1] = f"- {combined} <!-- {comment} -->"
    return True


def yaml_scalar(value: str | None) -> str:
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False)


def build_frontmatter(args: argparse.Namespace, pdf_path: Path) -> str:
    source_name = args.source_name or DEFAULT_SOURCE_NAME
    metadata: list[tuple[str, str | None]] = [
        ("doc_id", args.doc_id or DEFAULT_DOC_ID),
        ("source_name", source_name),
        ("source_org", args.source_org),
        ("source_file", pdf_path.name),
        ("doc_type", args.doc_type),
        ("audience", args.audience),
        ("jurisdiction", args.jurisdiction),
        ("language", args.language),
        ("population", args.population),
        ("publication_year", str(args.publication_year) if args.publication_year else None),
        ("last_verified", args.last_verified),
        ("can_quote_numbers", "false"),
        ("license_note", args.license_note),
    ]
    lines = ["---"]
    for key, value in metadata:
        if key == "can_quote_numbers":
            lines.append(f"{key}: false")
        elif key == "publication_year" and value is not None:
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    if args.source_url:
        lines.append(f"source_url: {yaml_scalar(args.source_url)}")
    lines.append("---")
    return "\n".join(lines)


def extract(pdf_path: Path, output_path: Path, args: argparse.Namespace) -> dict[str, int]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"File PDF non trovato: {pdf_path}")

    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        # Consente l'accesso solo se il documento è già apribile senza password; non tenta password arbitrarie.
        if not doc.authenticate(""):
            raise RuntimeError("Il PDF richiede una password per l'apertura.")

    total_pages = len(doc)
    start_page = max(1, args.start_page)
    end_page = min(total_pages, args.end_page) if args.end_page else total_pages
    if start_page > end_page:
        raise ValueError(f"Intervallo pagine non valido: {start_page}-{end_page} su {total_pages} pagine")

    output: list[str] = [build_frontmatter(args, pdf_path), "", f"# {args.title or args.source_name or DEFAULT_SOURCE_NAME}", ""]
    stats = {"pages_total": total_pages, "pages_processed": 0, "blocks_emitted": 0, "headings_emitted": 0, "bullet_items_emitted": 0}
    last_heading: dict[int, str] = {}

    for page_idx in range(start_page - 1, end_page):
        page = doc[page_idx]
        page_no = page_idx + 1
        stats["pages_processed"] += 1
        page_height = float(page.rect.height)
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        for block in blocks:
            parsed = get_block_text(block)
            if not parsed:
                continue
            text, font_size, bold, bbox, heading_prefix, remainder = parsed
            if is_noise_block(text, bbox, page_height):
                continue

            # Evita di reingestire la nota di copyright come contenuto alimentare.
            if re.match(r"^Tutti i diritti sono riservati\b", text, re.I):
                continue

            if heading_prefix:
                heading = normalize_space(heading_prefix).rstrip(" .")
                level = 2 if font_size >= 12.5 else 3
                if last_heading.get(level) != heading:
                    output.extend(["", f"{'#' * level} {heading}", ""])
                    last_heading[level] = heading
                    if level == 2:
                        last_heading.pop(3, None)
                    stats["headings_emitted"] += 1
                text = remainder
                if not text:
                    continue
                font_size = 10.5
                bold = False

            heading_level = classify_heading(text, font_size, bold) if not heading_prefix else None
            if heading_level:
                heading = normalize_space(text).rstrip(" .")
                if last_heading.get(heading_level) == heading:
                    continue
                if heading_level == 2:
                    last_heading.pop(3, None)
                output.extend(["", f"{'#' * heading_level} {heading}", ""])
                last_heading[heading_level] = heading
                stats["headings_emitted"] += 1
                continue

            # Ignora la ripetizione del titolo di copertina nel corpo.
            if text.upper() in {"LINEE GUIDA PER UNA SANA ALIMENTAZIONE", "REVISIONE 2018"}:
                continue

            for atom, inline_heading, explicit_bullet in split_content_block(text):
                if inline_heading:
                    heading = atom.rstrip(" .")
                    if last_heading.get(3) != heading:
                        output.extend([f"### {heading}", ""])
                        last_heading[3] = heading
                        stats["headings_emitted"] += 1
                    continue
                atom = normalize_space(atom)
                if not atom:
                    continue
                # Ricongiunge frammenti di frase spezzati dal layout, ma non punti
                # introdotti da un bullet esplicito. I riferimenti pagina diventano un intervallo.
                if not merge_with_previous_bullet(output, atom, page_no, explicit_bullet):
                    line = f"- {atom} <!-- pagina PDF: {page_no} -->"
                    output.append(line)
                    stats["blocks_emitted"] += 1
                    stats["bullet_items_emitted"] += 1

    # Normalizza righe bianche evitando di alterare il contenuto testuale.
    final_lines: list[str] = []
    blank_count = 0
    for line in output:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                final_lines.append("")
        else:
            blank_count = 0
            final_lines.append(line.rstrip())
    markdown = "\n".join(final_lines).strip() + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    doc.close()
    stats["output_bytes"] = len(markdown.encode("utf-8"))
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estrae le Linee guida CREA da PDF in Markdown per RAG.")
    parser.add_argument("pdf", nargs="?", default="LINEE-GUIDA-DEFINITIVO.pdf", help="PDF di input")
    parser.add_argument("-o", "--out", default=None, help="Markdown di output (default: nome PDF con estensione .md)")
    parser.add_argument("--title", default=DEFAULT_SOURCE_NAME, help="Titolo Markdown principale")
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--source-org", default="CREA")
    parser.add_argument("--doc-type", default="nutrition_guideline")
    parser.add_argument("--audience", default="general_public")
    parser.add_argument("--jurisdiction", default="IT")
    parser.add_argument("--language", default="it")
    parser.add_argument("--population", default="Popolazione generale; consultare la fonte per le indicazioni rivolte a gruppi specifici")
    parser.add_argument("--publication-year", default="2018")
    parser.add_argument("--last-verified", default=None, help="Data ISO effettiva della verifica, es. 2026-10-20")
    parser.add_argument("--source-url", default=None, help="URL ufficiale della fonte, se verificato")
    parser.add_argument("--license-note", default="Verificare le condizioni di riutilizzo e autorizzazione per estrazione automatizzata prima della pubblicazione")
    parser.add_argument("--start-page", type=int, default=16, help="Prima pagina fisica PDF da estrarre (default: 16, dopo copertina e indice)")
    parser.add_argument("--end-page", type=int, default=0, help="Ultima pagina fisica PDF da estrarre; 0 = ultima pagina")
    parser.add_argument("--stats-json", default=None, help="Facoltativo: salva statistiche di estrazione in JSON")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.start_page < 1 or args.end_page < 0:
        parser.error("--start-page deve essere >= 1 e --end-page >= 0")
    pdf_path = Path(args.pdf).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve() if args.out else pdf_path.with_suffix(".md")
    try:
        stats = extract(pdf_path, output_path, args)
    except Exception as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        return 1
    print(f"Markdown creato: {output_path}")
    print("Statistiche: " + json.dumps(stats, ensure_ascii=False))
    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Statistiche JSON: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
