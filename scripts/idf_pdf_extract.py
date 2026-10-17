#!/usr/bin/env python3
"""
Estrae dal PDF "IDF Global Clinical Practice Recommendations for Managing Type 2 Diabetes - 2025":

  idf2025-chN_recs.md   le raccomandazioni di ogni capitolo (tabelle "RECOMMENDATIONS":
                        colonne Optimal Care / Basic Care)                  [da curare e indicizzare]
  idf2025-chN_full.md   il testo del capitolo senza bibliografia, citazioni e intestazioni [archivio]

Stile uguale agli script ADA e SID-AMD: front matter YAML, nessuna AI, controlli finali.
Nota: "Optimal Care" = contesti con risorse adeguate (e' la colonna rilevante per l'Italia);
      "Basic Care" = contesti con risorse limitate.

Uso:
  uv run --with pymupdf python idf_pdf_extract.py IDF_Rec_2025.pdf --outdir extracted/idf
  uv run --with pymupdf python idf_pdf_extract.py IDF_Rec_2025.pdf --outdir extracted/idf --optimal-only --no-full
"""
import argparse
import datetime
import pathlib
import re
import sys
from collections import Counter

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

CHAPTERS = {
    1: "Detection of diabetes and intermediate hyperglycaemia, and prevention of type 2 diabetes",
    2: "Glycaemic control assessment and targets in type 2 diabetes",
    3: "Blood glucose-lowering therapies - non-insulin options for type 2 diabetes",
    4: "Blood glucose-lowering therapies - insulin options for type 2 diabetes",
    5: "Weight control in type 2 diabetes mellitus management",
    6: "Cardio-renal protection in type 2 diabetes",
    7: "Management of metabolic dysfunction-associated steatotic liver disease (MASLD) in type 2 diabetes",
}
HEADER_PREFIX = "IDF Global Clinical Practice Recommendations"
TERMINAL = (".", ":", ";", "?", "!")
BULLETS = ("•", "o", "–", "-")


# ------------------------------------------------------------------ utilità
def squash(s):
    return re.sub(r"\s+", " ", s).strip()


def inside(word_or_rect, rect, pad=1.5):
    x0, y0, x1, y1 = word_or_rect[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rect.x0 - pad <= cx <= rect.x1 + pad and rect.y0 - pad <= cy <= rect.y1 + pad


def cell_items(page, rect, exclude=()):
    """Testo di una cella come lista di (tipo, testo): 'b' = voce di elenco, 'p' = paragrafo."""
    words = [w for w in page.get_text("words", clip=rect) if not any(inside(w, ex) for ex in exclude)]
    words.sort(key=lambda w: (w[5], w[6], w[7]))
    lines, key = [], None
    for w in words:
        k = (w[5], w[6])
        if k != key:
            lines.append([])
            key = k
        lines[-1].append(w[4])
    items = []
    for ln in lines:
        txt = " ".join(ln).strip()
        if not txt:
            continue
        if txt[0] in "•" or txt.startswith("• "):
            items.append(["b", txt.lstrip("• ").strip()])
        elif items and not items[-1][1].rstrip().endswith(TERMINAL) and (txt[0].islower() or txt[0].isdigit()):
            items[-1][1] += " " + txt                  # riga di continuazione
        elif items and items[-1][0] == "b" and not items[-1][1].rstrip().endswith(TERMINAL):
            items[-1][1] += " " + txt
        else:
            items.append(["p", txt])
    return [(t, squash(x)) for t, x in items]


def items_md(items, indent=""):
    out = []
    for t, x in items:
        out.append(f"{indent}- {x}" if t == "b" else f"{indent}{x}")
    return out


def inner_table_md(page, table):
    rows = []
    for row in table.rows:
        cells = []
        for c in row.cells:
            cells.append(squash(" ".join(x for _, x in cell_items(page, fitz.Rect(c)))) if c else "")
        rows.append(cells)
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return []
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    md += ["| " + " | ".join(c.replace("|", "/") for c in r) + " |" for r in rows[1:]]
    return md


# ------------------------------------------------------------------ individuazione capitoli
def find_chapters(doc):
    """{n: pagina (1-based) con la tabella RECOMMENDATIONS}. Numerazione per ordine di apparizione."""
    pages = [i + 1 for i, p in enumerate(doc)
             if i > 8 and re.search(r"^RECOMMENDATIONS\s*$", p.get_text(), re.M)]
    if len(pages) != len(CHAPTERS):
        print(f"[AVVISO] trovate {len(pages)} pagine 'RECOMMENDATIONS' invece di {len(CHAPTERS)}: "
              "controllare la numerazione dei capitoli")
    return {n + 1: pg for n, pg in enumerate(pages)}


def chapter_ranges(doc, recs_pages):
    ns = sorted(recs_pages)
    ranges = {}
    for j, n in enumerate(ns):
        start = recs_pages[n] + 1
        end = (recs_pages[ns[j + 1]] - 2) if j + 1 < len(ns) else len(doc)
        ranges[n] = (start, end)
    return ranges


# ------------------------------------------------------------------ tabella delle raccomandazioni
def parse_recs_table(doc, page_no):
    page = doc[page_no - 1]
    tabs = page.find_tables().tables
    if not tabs:
        return [], ["tabella non trovata"]
    outer = max(tabs, key=lambda t: fitz.Rect(t.bbox).get_area())
    ob = fitz.Rect(outer.bbox)
    inners = [t for t in tabs if t is not outer and fitz.Rect(t.bbox) in ob]
    inner_rects = [fitz.Rect(t.bbox) for t in inners]

    rows_out, warns = [], []
    header = outer.rows[0].cells if outer.rows else []
    w_opt = (header[1][2] - header[1][0]) if len(header) > 1 and header[1] else 0
    prev_label = ""
    for r in outer.rows[1:]:
        cells = r.cells
        if len(cells) < 2:
            continue

        def col(i, ex=inner_rects):
            c = cells[i] if i < len(cells) else None
            if not c:
                return None, []
            rect = fitz.Rect(c)
            return rect, cell_items(page, rect, exclude=ex)

        _, label_items = col(0, ())
        label = squash(" ".join(x for _, x in label_items)) or prev_label
        if label:
            prev_label = label
        r1, opt = col(1)
        r2, basic = col(2)
        shared = bool(r1 is not None and w_opt and (r1.x1 - r1.x0) > 1.4 * w_opt and not basic)

        # tabelle annidate: le rendo come tabella markdown dentro la cella Optimal
        extra = []
        if r1 is not None:
            for it, ir in zip(inners, inner_rects):
                if inside(ir, r1):
                    extra += inner_table_md(page, it)
        rows_out.append({"label": label, "optimal": opt, "basic": basic, "shared": shared, "extra": extra})
    if not rows_out:
        warns.append("nessuna riga letta")
    return rows_out, warns


# ------------------------------------------------------------------ testo del capitolo
def chapter_text(doc, start, end):
    pages = list(range(start, end + 1))
    # intestazioni correnti: righe ripetute nelle prime 5 righe di >=3 pagine
    first = Counter()
    for pn in pages:
        top = [l.strip() for l in doc[pn - 1].get_text().split("\n") if l.strip()][:5]
        for l in set(top):
            first[l] += 1
    running = {l for l, c in first.items() if c >= 3 and not l.isdigit()}

    out = []
    for pn in pages:
        page = doc[pn - 1]
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            block_lines = []
            for ln in b["lines"]:
                spans = [s for s in ln["spans"] if s["size"] >= 7.0]   # toglie i numeri di citazione
                txt = squash("".join(s["text"] for s in spans))
                if not txt:
                    continue
                bold = all("Bold" in s["font"] for s in spans if s["text"].strip())
                block_lines.append((txt, bold))
            if not block_lines:
                continue
            texts = [t for t, _ in block_lines]
            joined = squash(" ".join(texts))
            if joined.upper() == "REFERENCES":
                return out, True
            if (joined.startswith(HEADER_PREFIX) or re.fullmatch(r"\d{1,3}", joined)
                    or re.fullmatch(r"Chapter \d", joined) or joined in running):
                continue
            # elimina righe di intestazione dentro il blocco
            block_lines = [(t, bd) for t, bd in block_lines
                           if not (t.startswith(HEADER_PREFIX) or re.fullmatch(r"\d{1,3}", t)
                                   or re.fullmatch(r"Chapter \d", t) or t in running)]
            if not block_lines:
                continue
            if all(bd for _, bd in block_lines):
                head = squash(" ".join(t for t, _ in block_lines))
                out.append(("h", 2 if head.isupper() else 3, head))
            else:
                out.append(("p", squash(" ".join(t for t, _ in block_lines))))
    return out, False


# ------------------------------------------------------------------ scrittura
def front(name, n, title, kind, extra=""):
    today = datetime.date.today().isoformat()
    return ("---\n"
            f"doc_id: {name}-{kind}\n"
            f"source_name: \"IDF Global Clinical Practice Recommendations for Managing Type 2 Diabetes 2025 - Chapter {n}: {title}\"\n"
            "source_org: IDF\n"
            "source_file: IDF_Rec_2025.pdf\n"
            "doc_type: clinical_guideline\n"
            "audience: clinician\n"
            "jurisdiction: INT\n"
            "language: en\n"
            "diabetes_type: [T2]\n"
            "population: adulti (documento globale, non specifico per anziani fragili)\n"
            "resource_setting_note: \"Optimal Care = contesti con risorse adeguate (rilevante per l'Italia); Basic Care = risorse limitate\"\n"
            f"last_verified: {today}\n"
            "can_quote_numbers: false\n"
            "license_note: \"IDF: verificare condizioni di riuso e se serve il permesso scritto prima di pubblicare\"\n"
            f"{extra}---\n\n")


def write_recs(path, n, title, rows, optimal_only):
    L = [front("idf2025-ch%d" % n, n, title, "recs"), f"# Capitolo {n} - {title} - raccomandazioni\n"]
    for r in rows:
        L.append(f"\n## {r['label'] or '(senza etichetta)'}\n")
        if r["shared"]:
            L.append("**Optimal Care e Basic Care (indicazione unica)**")
            L += items_md(r["optimal"])
        else:
            L.append("**Optimal Care**")
            L += items_md(r["optimal"]) if r["optimal"] else ["- (nessuna indicazione)"]
            if not optimal_only:
                L.append("\n**Basic Care**")
                L += items_md(r["basic"]) if r["basic"] else ["- (nessuna indicazione / vedi Optimal Care)"]
        if r["extra"]:
            L.append("")
            L += r["extra"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_full(path, n, title, blocks):
    L = [front("idf2025-ch%d" % n, n, title, "full", extra="archive_only: true\n"), f"# Capitolo {n} - {title}\n"]
    for b in blocks:
        if b[0] == "h":
            L.append("\n" + "#" * (b[1] if b[1] > 1 else 2) + " " + b[2] + "\n")
        else:
            L.append(b[1] + "\n")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--outdir", default="idf_out")
    ap.add_argument("--no-full", action="store_true", help="non scrivere i file _full")
    ap.add_argument("--optimal-only", action="store_true", help="nei _recs tieni solo la colonna Optimal Care")
    args = ap.parse_args()

    path = pathlib.Path(args.pdf)
    if not path.exists():
        sys.exit(f"[ERRORE] File non trovato: {path}")
    doc = fitz.open(path)
    recs_pages = find_chapters(doc)
    if not recs_pages:
        sys.exit("[ERRORE] Nessuna pagina 'RECOMMENDATIONS' trovata: e' il PDF IDF 2025 giusto?")
    ranges = chapter_ranges(doc, recs_pages)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for n in sorted(recs_pages):
        title = CHAPTERS.get(n, f"Chapter {n}")
        rows, warns = parse_recs_table(doc, recs_pages[n])
        write_recs(outdir / f"idf2025-ch{n}_recs.md", n, title, rows, args.optimal_only)
        msg = f"[OK] cap. {n}: {len(rows)} righe di raccomandazioni (pag. {recs_pages[n]})"
        if not args.no_full:
            start, end = ranges[n]
            blocks, found_refs = chapter_text(doc, start, end)
            write_full(outdir / f"idf2025-ch{n}_full.md", n, title, blocks)
            msg += f"; testo {sum(len(b[-1].split()) for b in blocks)} parole"
            if not found_refs:
                msg += " (ATTENZIONE: bibliografia non individuata, controllare la fine del file)"
        print(msg)
        for w in warns:
            print("     ATTENZIONE:", w)


if __name__ == "__main__":
    main()