#!/usr/bin/env python3
"""
Estrazione e pulizia del Vademecum ANIAD "Diabete: LIBERI di vincerlo -
Alleniamoci a farlo!" (ottobre 2017) per la base di conoscenza RAG.

Il PDF è impaginato in "doppie pagine" (spread 1134x567 pt) con 2 colonne per
pagina. Lo script:
  1. divide ogni spread in pagina sinistra/destra e ogni pagina in colonne;
  2. scarta rumore tipografico (citazioni decorative, numeri di pagina, punti
     elenco grafici, note a piè di pagina di marketing);
  3. ricostruisce paragrafi, titoli e elenchi, e ricuce le parole spezzate;
  4. mantiene solo le sezioni utili al RAG (configurabili in SECTIONS);
  5. inserisce le tabelle di dosaggio insulina (TABLES) gia' strutturate;
  6. scrive un file markdown con frontmatter, un'unita' per sezione.

Uso:
    python extract_aniad_vademecum.py INPUT.pdf [-o OUTPUT.md] [--all]
                                       [--debug-dir DIR]

Dipendenze: pip install pdfplumber
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pdfplumber

# --------------------------------------------------------------------------
# CONFIGURAZIONE
# --------------------------------------------------------------------------

FRONTMATTER = """---
fonte: "Diabete: LIBERI di vincerlo - Alleniamoci a farlo!"
ente: A.N.I.A.D. (Associazione Nazionale Italiana Atleti Diabetici)
tipo: vademecum divulgativo
anno: 2017 (ottobre)
lingua: italiano
uso: RAG - attività fisica e sport nel diabete
note: estratto automatico; rimosse prefazioni, storie personali, testimonianze, pubblicità FreeStyle Libre, bibliografia
avvertenza: documento del 2017; le soglie e le tabelle di dosaggio insulina sono indicative e vanno personalizzate con il diabetologo; verificare contro linee guida più recenti (ADA, SID-AMD)
---
"""

# (titolo, prima_pagina_pdf, ultima_pagina_pdf, tieni_per_default)
# Le pagine sono quelle del PDF (spread), non quelle stampate.
SECTIONS = [
    ("Introduzione e raccomandazioni generali", 6, 7, True),
    ("Il ruolo del movimento nell'evoluzione della specie umana", 8, 9, False),
    ("Diabete e attività fisica: un po' di storia", 10, 11, False),
    ("Fisiologia del metabolismo energetico in corso di attività fisica", 12, 13, True),
    ("Gestione dell'insulina, autocontrollo glicemico e carboidrati nel diabete tipo 1", 14, 14, True),
    ("Claudio Gotti: un rally per vincere il diabete", 16, 18, False),
    ("Il supporto della tecnologia in ambito di diabete e attività fisica", 19, 20, True),
    ("Il vantaggio di conoscere l'andamento glicemico senza la puntura del dito", 21, 21, True),
    ("Le testimonianze di atleti diabetici", 22, 23, False),
    ("I principali documenti di riferimento", 24, 24, False),
    ("Attività fisica e diabete gestazionale", 25, 26, True),
    ("L'ANIAD, lo sport e il CONI", 27, 28, False),
    ("Diabete e sport: aspetti medico-legali", 29, 29, True),
    ("Liberi di fare sport, anche senza pungersi (FreeStyle Libre)", 30, 31, False),
    ("Che intendiamo per... (glossario)", 32, 32, True),
]

# Pagine in cui lo spread contiene due sezioni distinte (la meta' sinistra
# appartiene a quella precedente). Qui non serve: le sezioni sono per spread.

# Sezioni di cui le tabelle vengono inserite a mano (non estraibili bene dal
# testo perchè prive di linee). Chiave: titolo di sezione.
TABLES = {
    "Gestione dell'insulina, autocontrollo glicemico e carboidrati nel diabete tipo 1": """
### Tabelle di riduzione della terapia insulinica (esercizio aerobico, 1-3 h dopo il pasto)

Punto di partenza da cui fare ulteriori aggiustamenti in base all'esperienza individuale. Valido per diabetici di tipo 1 e di tipo 2 in trattamento insulinico.

**Pazienti in terapia multiniettiva - carico aerobico (esercizio 1 h dopo il pasto)**

| Intensità | Riduzione analogo rapido pasto precedente (esercizio 30-45 min) | Riduzione analogo rapido pasto precedente (esercizio 60 min) | Riduzione analogo rapido pasto successivo (esercizio 60 min) | Riduzione analogo basale serale (esercizio 45-60 min) | Glucosio durante l'esercizio |
|---|---|---|---|---|---|
| 25% VO2max | 25% | 50% | 0% | 0% | 0 g |
| 50% VO2max | 50% | 75% | 0% | 10% | 30-60 g |
| 70-75% VO2max | 75% | 90% | 50% | 20% | 60 g |

Carico anaerobico (rispetto a esercizio aerobico al 60% VO2max): 0% di riduzione sul rapido; 10% di riduzione dell'analogo basale serale; 25 g di glucosio.

**Pazienti con microinfusore - carico aerobico (esercizio 1-3 h dopo il pasto)**

| Intensità | Riduzione bolo pasto precedente (esercizio 30-45 min) | Riduzione basale durante esercizio (durata > 30 min, anticipare di circa 1 h) | Riduzione basale serale (esercizio 45-60 min) | Glucosio durante l'esercizio |
|---|---|---|---|---|
| 25% VO2max | 25% | 0% | 0% | 0 g |
| 50% VO2max | 50% | 50% | 10-20% | 30-60 g |
| 70-75% VO2max | 75% | 100% (con eventuale bolo di compensazione prima della disconnessione; intervallo di sospensione non oltre 2 h; evitare coaguli con un piccolo bolo) | 20-30% | 60 g |

Carico anaerobico (rispetto a esercizio aerobico al 60% VO2max): 0% sul bolo; 50% sulla basale durante l'esercizio; 0% sulla basale serale; 25 g di glucosio.

**Aggiustamento della dose di insulina prima dell'esercizio nel diabete di tipo 2 (Società di Fisioterapia Belga, 2013)**

| Durata e tipo di esercizio | Glicemia pre-esercizio | Riduzione bolo insulina pre-esercizio | Glucosio extra durante l'esercizio |
|---|---|---|---|
| < 30 min, bassa intensità | < 90 mg/dl | 50% | 10-15 g |
| < 30 min, bassa intensità | > 90 mg/dl | 0% | nessuno |
| 30-60 min, intensità moderata | < 90 mg/dl | 100% | 30-45 g |
| 30-60 min, intensità moderata | 90-180 mg/dl | 50% | 15 g |
| 30-60 min, intensità moderata | > 180 mg/dl | 0% | nessuno |
| > 60 min, intensità moderata | < 90 mg/dl | 100% | 45 g/h |
| > 60 min, intensità moderata | 90-180 mg/dl | 50% | 30-45 g/h |
| > 60 min, intensità moderata | > 180 mg/dl | 50% | 15 g/h |
""",
    "Che intendiamo per... (glossario)": """
### Classificazione sport aerobici e anaerobici

| Sport aerobici (durata superiore a 5 minuti) | Sport anaerobici (durata tra 10 secondi e qualche minuto) |
|---|---|
| Jogging, corsa di fondo, sci di fondo, nuoto, ciclismo, danza, pattinaggio | Corsa (100, 200, 400 m, 110 ostacoli), calcio (scatti e accelerazioni), tennis (scatti), pallavolo (scatti e salti), basket (scatti, salti e accelerazioni), sci alpino, ciclismo su pista |
""",
}

# --------------------------------------------------------------------------
# FILTRI TIPOGRAFICI (nomi font / dimensioni emersi dall'analisi del PDF)
# --------------------------------------------------------------------------

DROP_FONT_SUBSTR = ("Berlingske", "ZapfDingbats")  # citazioni decorative, bullet grafici
BULLET_FONT = "ZapfDingbats"
HEADING_MIN_SIZE = 10.5  # titoli di sezione (20pt) e sottotitoli (11pt)
SUBHEADING_SIZES = (10.9, 11.1)
FOOTNOTE_MAX_SIZE = 8.2  # note in corsivo piccolo (marketing Libre)
PAGE_NUMBER_BAND = 30  # pt dal bordo basso dove cadono i numeri di pagina
TOP_BAND = 8


def _char_ok(obj) -> bool:
    if obj.get("object_type") != "char":
        return True
    font = obj["fontname"]
    if any(s in font for s in DROP_FONT_SUBSTR):
        return False
    return True


# --------------------------------------------------------------------------
# COSTRUZIONE RIGHE E COLONNE
# --------------------------------------------------------------------------

def words_with_style(crop):
    return crop.extract_words(
        keep_blank_chars=False,
        use_text_flow=False,
        extra_attrs=["size", "fontname"],
        x_tolerance=1.5,
        y_tolerance=2,
    )


def find_column_split(words, x_min, x_max):
    """Cerca la x che separa le due colonne: il punto di minima sovrapposizione
    di parole nella fascia centrale. Ritorna None se la pagina è a colonna
    singola o se il testo è troppo scarso."""
    if len(words) < 40:
        return None
    width = x_max - x_min
    lo, hi = int(x_min + width * 0.30), int(x_max - width * 0.30)
    best_x, best_n = None, 10**9
    for x in range(lo, hi):
        n = sum(1 for w in words if w["x0"] < x < w["x1"])
        if n < best_n or (n == best_n and best_x is not None and abs(x - (x_min + x_max) / 2) < abs(best_x - (x_min + x_max) / 2)):
            best_n, best_x = n, x
    # colonne vere: quasi nessuna parola attraversa la linea di separazione
    return best_x if best_n <= max(2, len(words) // 80) else None


def group_lines(words):
    lines, cur, cur_top = [], [], None
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if cur_top is None or abs(w["top"] - cur_top) <= 3:
            cur.append(w)
            cur_top = w["top"] if cur_top is None else (cur_top + w["top"]) / 2
        else:
            lines.append(sorted(cur, key=lambda w: w["x0"]))
            cur, cur_top = [w], w["top"]
    if cur:
        lines.append(sorted(cur, key=lambda w: w["x0"]))
    return lines


def merge_dropcap(line):
    """Capolettera (1 lettera in Avenir-Black a inizio riga): unisci alla parola seguente."""
    if len(line) > 1 and len(line[0]["text"]) == 1 and "Black" in line[0]["fontname"] and line[0]["size"] < 11:
        first = dict(line[1], text=line[0]["text"] + line[1]["text"], x0=line[0]["x0"])
        return [first] + line[2:]
    return line


def mark_term(line):
    """Voce di glossario: parole iniziali in Avenir-Black 10pt -> '**termine**:'."""
    n = 0
    while n < len(line) and "Black" in line[n]["fontname"] and 9.8 <= line[n]["size"] <= 10.2:
        n += 1
    if n == 0:
        return line, False
    term = " ".join(w["text"] for w in line[:n])
    head = dict(line[0], text=f"**{term}**:")
    return [head] + line[n:], True


def line_info(line):
    text = " ".join(w["text"] for w in line)
    text = re.sub(r"(\w) -$", r"\1-", text)  # "con -" a fine riga -> "con-"
    size = max(w["size"] for w in line)
    top = min(w["top"] for w in line)
    bottom = max(w["bottom"] for w in line)
    font = line[0]["fontname"]
    return {"text": text, "size": size, "top": top, "bottom": bottom, "font": font, "x0": line[0]["x0"]}


# --------------------------------------------------------------------------
# PULIZIA TESTO
# --------------------------------------------------------------------------

BULLET_RE = re.compile(r"^[•●▪·]\s*")


def fix_text(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def join_lines(lines_text):
    """Unisce righe dello stesso paragrafo ricucendo le sillabazioni."""
    out = ""
    for t in lines_text:
        if not out:
            out = t
        elif re.search(r"[A-Za-zÀ-ÿ]-$", out) and t and t[0].islower():
            out = out[:-1] + t  # parola spezzata: "gestio-" + "ne"
        else:
            out += " " + t
    return out


def build_blocks(lines):
    """Trasforma le righe di una colonna in blocchi (heading/par/bullet)."""
    infos = [line_info(l) for l in lines]
    blocks = []
    cur_lines, cur_kind, prev = [], None, None

    def flush():
        nonlocal cur_lines, cur_kind
        if cur_lines:
            txt = fix_text(join_lines(cur_lines))
            if txt:
                blocks.append((cur_kind or "par", txt))
        cur_lines, cur_kind = [], None

    for info, raw in zip(infos, lines):
        text = info["text"]
        size = info["size"]
        # note piccole in corsivo (marketing/bibliografia): scartate
        if size <= FOOTNOTE_MAX_SIZE and "Oblique" in info["font"]:
            continue
        has_bullet = any(BULLET_FONT in w["fontname"] for w in raw)
        if raw[0].get("is_term"):
            flush()
            cur_kind = "par"
        is_head = size >= HEADING_MIN_SIZE
        gap = (info["top"] - prev["bottom"]) if prev else 0
        new_par = prev is not None and gap > 5.5
        if is_head:
            kind = "h2" if size >= 15 else "h3"
            if cur_kind != kind:
                flush()
            cur_kind = kind
            cur_lines.append(text)
        else:
            if cur_kind in ("h2", "h3"):
                flush()
            if has_bullet or BULLET_RE.match(text):
                flush()
                cur_kind = "bullet"
                cur_lines.append(BULLET_RE.sub("", text))
            elif new_par:
                flush()
                cur_kind = "par"
                cur_lines.append(text)
            else:
                cur_kind = cur_kind or "par"
                cur_lines.append(text)
        prev = info
    flush()
    return blocks


END_PUNCT = (".", ":", ";", "?", "!", ")", '"')


DROP_BLOCK_RE = re.compile(
    r"^(Sport Aerobici|Corsa Jogging|Calcio Corsa|Sci di fondo Tennis|Nuoto Pallavolo|"
    r"Basket Ciclismo|Danza Sci alpino|Pattinaggio Ciclismo)"
)


def merge_blocks(blocks):
    """Ricuce paragrafi interrotti dal cambio colonna/pagina e sistema i capolettera."""
    fixed = [b for b in blocks if not DROP_BLOCK_RE.match(b[1])]
    out = []
    for kind, txt in fixed:
        if (kind == "par" and out and out[-1][0] == "par"
                and not txt.startswith("**")
                and not out[-1][1].rstrip().endswith(END_PUNCT)):
            prev = out[-1][1]
            out[-1] = ("par", prev[:-1] + txt if prev.endswith("-") else prev + " " + txt)
        else:
            out.append((kind, txt))
    return out


def blocks_to_md(blocks):
    out = []
    for kind, txt in blocks:
        if kind == "h2":
            out.append(f"\n## {txt.capitalize() if txt.isupper() else txt}\n")
        elif kind == "h3":
            out.append(f"\n### {txt.capitalize() if txt.isupper() else txt}\n")
        elif kind == "bullet":
            out.append(f"- {txt}")
        else:
            out.append(f"\n{txt}\n")
    return "\n".join(out)


# --------------------------------------------------------------------------
# ESTRAZIONE PAGINA
# --------------------------------------------------------------------------

def extract_half(page, x0, x1):
    crop = page.crop((x0, TOP_BAND, x1, page.height - PAGE_NUMBER_BAND)).filter(_char_ok)
    words = [w for w in words_with_style(crop) if w["size"] < 15]  # titoli 20pt: li fornisce SECTIONS
    if not words:
        return []
    # bullet grafici: ricostruisci come marcatore se il font ZapfDingbats è stato filtrato
    # (usiamo la pagina originale per rilevare dove cadevano i bullet)
    raw_crop = page.crop((x0, TOP_BAND, x1, page.height - PAGE_NUMBER_BAND))
    bullet_tops = [c["top"] for c in raw_crop.chars if BULLET_FONT in c["fontname"]]
    bullet_xs = {round(c["top"]): c["x0"] for c in raw_crop.chars if BULLET_FONT in c["fontname"]}

    x_min = min(w["x0"] for w in words)
    x_max = max(w["x1"] for w in words)
    split = find_column_split(words, x_min, x_max)
    columns = [words] if split is None else [
        [w for w in words if w["x1"] <= split + 1],
        [w for w in words if w["x0"] >= split - 1],
    ]
    blocks_all = []
    for col in columns:
        if not col:
            continue
        lines = []
        for l in group_lines(col):
            l = merge_dropcap(l)
            l, is_term = mark_term(l)
            if is_term:
                l[0] = dict(l[0], is_term=True)
            lines.append(l)
        # marca le righe che iniziano con un bullet grafico
        col_left = min(w["x0"] for w in col)
        for line in lines:
            top = min(w["top"] for w in line)
            for bt, bx in ((t, bullet_xs.get(round(t))) for t in bullet_tops):
                if abs(bt - top) < 4 and bx is not None and abs(bx - col_left) < 18:
                    line[0] = dict(line[0], fontname=line[0]["fontname"] + "+ZapfDingbatsITC")
                    break
        blocks_all.extend(build_blocks(lines))
    return blocks_all


def extract_spread(page):
    W = page.width
    if W < 900:  # pagina singola
        return extract_half(page, 0, W)
    return extract_half(page, 0, W / 2) + extract_half(page, W / 2, W)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------

POST_FIXES = [
    (r"\bdel diabete\s+tipo\s+1\b", "del diabete tipo 1"),
    (r"\s+([,.;:])", r"\1"),
    (r"\bm l\b", "ml"),
    (r"VO ?2 ?max", "VO2max"),
    (r"\n{3,}", "\n\n"),
]


def postprocess(md: str) -> str:
    for pat, rep in POST_FIXES:
        md = re.sub(pat, rep, md)
    return md.strip() + "\n"


def run(pdf_path: Path, out_path: Path, keep_all: bool, debug_dir: Path | None):
    parts = [FRONTMATTER, "\n# Vademecum ANIAD - Diabete, attività fisica e sport\n"]
    with pdfplumber.open(str(pdf_path)) as pdf:
        for title, p_from, p_to, keep in SECTIONS:
            if not (keep or keep_all):
                continue
            section_blocks = []
            for pn in range(p_from, p_to + 1):
                page = pdf.pages[pn - 1]
                blocks = extract_spread(page)
                if debug_dir:
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    (debug_dir / f"page_{pn:02d}.md").write_text(blocks_to_md(blocks), encoding="utf-8")
                section_blocks.extend(blocks)
            section_blocks = merge_blocks(section_blocks)
            md = blocks_to_md(section_blocks)
            parts.append(f"\n---\n\n## {title}\n")
            parts.append(md)
            if title in TABLES:
                parts.append(TABLES[title])
    out_path.write_text(postprocess("\n".join(parts)), encoding="utf-8")
    print(f"Scritto {out_path} ({out_path.stat().st_size} byte)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("aniad-vademecum-rag.md"))
    ap.add_argument("--all", action="store_true", help="includi anche le sezioni scartate per default")
    ap.add_argument("--debug-dir", type=Path, help="salva anche il markdown di ogni pagina")
    args = ap.parse_args()
    run(args.pdf, args.output, args.all, args.debug_dir)


if __name__ == "__main__":
    main()