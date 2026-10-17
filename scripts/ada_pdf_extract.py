#!/usr/bin/env python3
"""
Estrae dai PDF ADA "Standards of Care in Diabetes 2026" (dc26sNNN.pdf):
  <nome>_recs.md  -> SOLO le raccomandazioni numerate con il grado di evidenza [A/B/C/E]
  <nome>_full.md  -> testo completo ordinato per colonne, senza bibliografia/note,
                     con le raccomandazioni nel punto in cui compaiono e le tabelle in markdown

Nessuna AI: usa la struttura del PDF (font, colori, riquadro rosa delle raccomandazioni).

Uso:
  pip install pymupdf
  python ada_pdf_extract.py dc26s006.pdf dc26s012.pdf --outdir ada_out
  python ada_pdf_extract.py cartella_con_i_pdf/ --outdir ada_out

ATTENZIONE: i PDF ADA riportano che l'uso per text/data mining o machine learning
richiede il permesso scritto di ADA (permissions@diabetes.org). Vedi note in chat.
"""
import argparse
import datetime
import pathlib
import re
import sys

try:
    import pymupdf as fitz
except ImportError:  # versioni vecchie
    import fitz  # type: ignore

PINK = (0.91, 0.80, 0.81)          # riquadro "Recommendations"
NUM_RE = re.compile(r"\d{1,2}\.\d{1,2}[a-z]?")
GRADE_RE = re.compile(r"[ABCE]")
TERMINAL = tuple(".!?:;)]”\"")
STOP_HEADINGS = ("references",)


# ---------------------------------------------------------------- utilità testo
def clean(s):
    s = s.replace("\u200b", "").replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.;:)])", r"\1", s)
    return s


def join_lines(lines):
    """Unisce righe gestendo i trattini a fine riga (soft hyphen e '-')."""
    out = ""
    for raw in lines:
        raw = raw.replace("\u200b", "").strip()
        if not raw:
            continue
        if not out:
            out = raw
        elif out.endswith("\u00ad"):
            out = out[:-1] + raw
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha():
            out += raw
        else:
            out += " " + raw
    return clean(out)


def near(c, ref, tol=0.04):
    return all(abs(a - b) <= tol for a, b in zip(c, ref))


def line_text(line):
    return "".join(s["text"] for s in line["spans"])


# ---------------------------------------------------------------- classificazione
# Indipendente dal nome del font (cambia da capitolo a capitolo): usa dimensione e stile.
BOLD_RE = re.compile(r"(Bold|\.B$|-B$|Black|Semibold)", re.I)
ITAL_RE = re.compile(r"(Italic|\.I$|-I$|Oblique)", re.I)
CAP_RE = re.compile(r"(Table|Figure|Box) \d+\.\d+")


def is_bold(font):
    return bool(BOLD_RE.search(font))


def is_italic(font):
    return bool(ITAL_RE.search(font))


def kind_of(line, page_no):
    sp = line["spans"]
    if not sp:
        return "skip"
    s0 = sp[0]
    f, size = s0["font"], s0["size"]
    x0 = line["bbox"][0]
    txt = line_text(line).strip()
    if not txt:
        return "skip"
    if f.startswith("ArialMT") and size <= 6.5:
        return "skip"                      # filigrana "Downloaded from..."
    if x0 > 600:
        return "skip"                      # colonna laterale e linguetta
    if page_no == 0 and x0 >= 370:
        return "skip"                      # colonna laterale (copyright) nel layout a 2 colonne
    if size >= 20:
        return "title"
    if CAP_RE.match(txt) and size <= 8.6:
        return "cap"
    if is_italic(f) and size <= 8.2 and txt.lower().startswith("recommendation"):
        return "label"
    if page_no == 0 and txt.startswith("Diabetes Care 20"):
        return "skip"
    if page_no == 0 and txt.startswith("The American Diabetes Association (ADA)"):
        return "skipblock"                 # paragrafo introduttivo standard
    if size >= 9.2:
        return "body"
    if size >= 8.3:                        # titoli: 9.0 = livello 2, 8.5 = livello 3
        if is_italic(f):
            return "h4"
        return "h2" if size >= 8.9 else "h3"
    return "skip"                          # note, bibliografia, celle di tabella, header/footer


def cluster_columns(xs, tol=20, min_lines=3):
    xs = sorted(xs)
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    centers = [sum(g) / len(g) for g in groups if len(g) >= min_lines]
    return centers or [60.0]


def col_of(x, centers):
    return min(range(len(centers)), key=lambda i: abs(centers[i] - x))


# ---------------------------------------------------------------- estrazione pagina
def page_items(page, pn):
    d = page.get_text("dict")
    lines = []
    for b in d["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            lines.append(l)

    # riquadri rosa
    boxes = []
    for dr in page.get_drawings():
        fill = dr.get("fill")
        r = dr["rect"]
        if fill and near(fill, PINK) and r.width > 100 and r.height > 20:
            boxes.append(r)

    # tabelle (solo se non sovrapposte ai riquadri)
    tables = []
    try:
        for t in page.find_tables().tables:
            tb = fitz.Rect(t.bbox)
            if any((tb & bx).get_area() > 0.3 * tb.get_area() for bx in boxes):
                continue
            rows = t.extract()
            if len(rows) >= 3 and max(len(r) for r in rows) >= 2:
                tables.append((tb, rows))
    except Exception:
        pass

    def inside(rect, ln, pad=2):
        cx = (ln["bbox"][0] + ln["bbox"][2]) / 2
        cy = (ln["bbox"][1] + ln["bbox"][3]) / 2
        return (rect.x0 - pad <= cx <= rect.x1 + pad) and (rect.y0 - pad <= cy <= rect.y1 + pad)

    box_lines = [[] for _ in boxes]
    body = []
    for ln in lines:
        placed = False
        for i, bx in enumerate(boxes):
            if inside(bx, ln):
                box_lines[i].append(ln)
                placed = True
                break
        if placed:
            continue
        if any(inside(tb, ln) for tb, _ in tables):
            continue
        k = kind_of(ln, pn)
        if k in ("skip",):
            continue
        body.append((k, ln))

    centers = cluster_columns(
        [ln["bbox"][0] for k, ln in body if k in ("body", "h2", "h3", "title")]
    )
    items = []

    # righe di testo -> paragrafi
    body.sort(key=lambda kl: (col_of(kl[1]["bbox"][0], centers), kl[1]["bbox"][1]))
    cur = None
    for k, ln in body:
        c = col_of(ln["bbox"][0], centers)
        y0, y1 = ln["bbox"][1], ln["bbox"][3]
        txt = line_text(ln)
        if k == "skipblock":
            cur = None
            continue
        gap_limit = 6 if k != "body" else 2.5
        new = (
            cur is None
            or cur["kind"] != k
            or cur["col"] != c
            or (y0 - cur["y1"]) > gap_limit
            or (k == "body" and txt.lstrip().startswith("•"))
        )
        if new:
            cur = {"kind": k, "page": pn, "col": c, "y0": y0, "y1": y1, "lines": [txt],
                   "break_by_col": False}
            items.append(cur)
        else:
            cur["lines"].append(txt)
            cur["y1"] = y1
    # riquadri
    for bx, bl in zip(boxes, box_lines):
        c = col_of(bx.x0 + 6, centers)
        bl.sort(key=lambda l: (round(l["bbox"][1]), l["bbox"][0]))
        items.append({"kind": "box", "page": pn, "col": c, "y0": bx.y0, "lines": bl})
    # tabelle
    for tb, rows in tables:
        items.append({"kind": "table", "page": pn, "col": col_of(tb.x0 + 6, centers),
                      "y0": tb.y0, "rows": rows})
    items.sort(key=lambda it: (it["col"], it["y0"]))
    return items


# ---------------------------------------------------------------- raccomandazioni
class Rec:
    def __init__(self, num):
        self.num = num
        self.lines = []
        self.cur = ""
        self.section = ""

    def flush(self):
        if self.cur.strip():
            self.lines.append(self.cur)
        self.cur = ""

    @property
    def text(self):
        self.flush()
        t = join_lines(self.lines)
        t = re.sub(r"\s+(\[[ABCE]\])", r" \1", t)
        return t


def parse_box(lines, last_rec, section):
    """Restituisce (recs_nuove, last_rec). Le righe senza numero iniziale proseguono l'ultima rec."""
    new = []
    cur = last_rec
    for ln in lines:
        for sp in ln["spans"]:
            t, f = sp["text"], sp["font"]
            if is_italic(f) and re.fullmatch(r"Recommendations?", t.strip()):
                continue
            bold = is_bold(f)
            if bold and NUM_RE.fullmatch(t.strip()):
                if cur is not None:
                    cur.flush()
                cur = Rec(t.strip())
                cur.section = section
                new.append(cur)
                continue
            if bold and GRADE_RE.fullmatch(t.strip()) and cur is not None:
                if cur.cur and not cur.cur.endswith(" "):
                    cur.cur += " "
                cur.cur += f"[{t.strip()}] "
                continue
            if cur is not None:
                cur.cur += t
        if cur is not None:
            cur.flush()
    return new, cur


# ---------------------------------------------------------------- tabelle
def table_md(rows):
    rows = [[clean(c) if c else "" for c in r] for r in rows]
    caption = []
    while rows and all(not c for c in rows[0][1:]):
        caption.append(rows.pop(0)[0])
    if not rows:
        return " ".join(caption), ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n]
    for r in rows[1:]:
        md.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
    return " ".join(caption), "\n".join(md)


# ---------------------------------------------------------------- documento
def process_pdf(path):
    doc = fitz.open(path)
    first = doc[0].get_text()
    m = re.search(r"https://doi\.org/10\.2337/dc26-S\d+", first)
    doi = m.group(0) if m else ""
    all_items = []
    for pn, pg in enumerate(doc):
        its = page_items(pg, pn)
        all_items.extend(its)

    title_lines = [it for it in all_items if it["kind"] == "title"]
    title = join_lines([l for it in title_lines for l in it["lines"]]) or path.stem
    mnum = re.match(r"(\d+)\.", title)
    chapter = mnum.group(1) if mnum else ""

    # flusso finale
    out = []            # blocchi per il full
    recs = []           # tutte le raccomandazioni
    last_rec = None
    h2 = h3 = ""
    stop = False
    prev_body = None
    for it in all_items:
        k = it["kind"]
        if stop:
            break
        if k == "title":
            continue
        if k in ("h2", "h3", "h4"):
            text = join_lines(it["lines"])
            if k == "h2" and text.lower().startswith(STOP_HEADINGS):
                stop = True
                continue
            if k == "h2":
                h2, h3 = text, ""
            elif k == "h3":
                h3 = text
            out.append(("h", {"h2": 2, "h3": 3, "h4": 4}[k], text))
            prev_body = None
        elif k == "label":
            continue
        elif k == "cap":
            out.append(("cap", join_lines(it["lines"]), it["page"] + 1))
            prev_body = None
        elif k == "body":
            text = join_lines(it["lines"])
            if (prev_body is not None and not prev_body[1].endswith(TERMINAL)
                    and not text.startswith("•") and not prev_body[1].startswith("•")
                    and (it["page"], it["col"]) != prev_body[0]):
                joined = join_lines([prev_body[1], text])
                out[-1] = ("p", joined)
                prev_body = ((it["page"], it["col"]), joined)
            else:
                out.append(("p", text))
                prev_body = ((it["page"], it["col"]), text)
        elif k == "box":
            section = " / ".join(x for x in (h2, h3) if x)
            new, last_rec = parse_box(it["lines"], last_rec, section)
            recs.extend(new)
            out.append(("box", new))
            prev_body = None
        elif k == "table":
            cap, md = table_md(it["rows"])
            out.append(("table", cap, md))
            prev_body = None
    return {"title": title, "chapter": chapter, "doi": doi, "recs": recs, "out": out}


# ---------------------------------------------------------------- scrittura
def front_matter(name, title, doi, kind):
    today = datetime.date.today().isoformat()
    return (
        "---\n"
        f"doc_id: {name}-{kind}\n"
        f"source_name: \"ADA Standards of Care in Diabetes 2026 - {title}\"\n"
        "source_org: ADA\n"
        f"source_url: {doi}\n"
        "doc_type: clinical_guideline\n"
        "audience: clinician\n"
        "jurisdiction: US/INT\n"
        "language: en\n"
        f"last_verified: {today}\n"
        "can_quote_numbers: false\n"
        "license_note: \"ADA: text/data mining e ML richiedono permesso scritto (permissions@diabetes.org)\"\n"
        "---\n\n"
    )


def check_sequence(recs, chapter):
    bases = sorted({int(r.num.split(".")[1].rstrip("abcdefghijklmnopqrstuvwxyz"))
                    for r in recs if r.num.split(".")[0] == chapter})
    if not bases:
        return [], 0
    missing = [b for b in range(1, bases[-1] + 1) if b not in bases]
    return missing, bases[-1]


def write_outputs(res, outdir, name, write_full=True):
    title, doi, recs = res["title"], res["doi"], res["recs"]
    # recs
    lines = [front_matter(name, title, doi, "recs"), f"# {title} - raccomandazioni\n"]
    last_sec = None
    for r in recs:
        if r.section != last_sec:
            lines.append(f"\n## {r.section or '(senza sezione)'}\n")
            last_sec = r.section
        lines.append(f"- **{r.num}** {r.text}")
    (outdir / f"{name}_recs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if write_full:
        full = [front_matter(name, title, doi, "full"), f"# {title}\n"]
        for blk in res["out"]:
            t = blk[0]
            if t == "h":
                full.append("\n" + "#" * blk[1] + " " + blk[2] + "\n")
            elif t == "p":
                full.append(blk[1] + "\n")
            elif t == "cap":
                full.append(f"*[{blk[1]} - vedi PDF p. {blk[2]}]*\n")
            elif t == "box" and blk[1]:
                full.append("> **Raccomandazioni**")
                for r in blk[1]:
                    full.append(f"> - **{r.num}** {r.text}")
                full.append("")
            elif t == "table":
                full.append((f"**{blk[1]}**\n\n" if blk[1] else "") + blk[2] + "\n")
        (outdir / f"{name}_full.md").write_text("\n".join(full) + "\n", encoding="utf-8")


def collect_pdfs(args):
    pdfs = []
    for a in args:
        p = pathlib.Path(a)
        if p.is_dir():
            pdfs.extend(sorted(p.glob("*.pdf")))
        elif p.exists():
            pdfs.append(p)
        else:
            print(f"[AVVISO] non trovato: {a}")
    return pdfs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="PDF o cartelle con PDF")
    ap.add_argument("--outdir", default="ada_out")
    ap.add_argument("--no-full", action="store_true", help="scrive solo i file _recs")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for pdf in collect_pdfs(args.inputs):
        try:
            res = process_pdf(pdf)
        except Exception as e:  # noqa
            print(f"[ERRORE] {pdf.name}: {e}")
            continue
        chap = res["chapter"] or pdf.stem
        name = f"ada2026-ch{chap}"
        write_outputs(res, outdir, name, write_full=not args.no_full)
        missing, mx = check_sequence(res["recs"], res["chapter"])
        print(f"[OK] {pdf.name} -> {name}: {len(res['recs'])} raccomandazioni (max n. {mx})")
        if missing:
            print(f"     ATTENZIONE: numeri mancanti nella sequenza: {res['chapter']}.{missing}")
        if not res["recs"]:
            print("     ATTENZIONE: 0 raccomandazioni: controlla il PDF")


if __name__ == "__main__":
    main()