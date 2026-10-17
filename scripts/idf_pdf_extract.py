#!/usr/bin/env python3
"""
Estrattore Modulare per Linee Guida Mediche (PDF) tramite PyMuPDF e Regex.

Genera tre file nella cartella di output:
  1. raccomandazioni_sintetiche_ada.md (Elenco puntato con tag [A-E])
  2. raccomandazioni_estese_rag.md     (H3, testo, forza, giustificazione in blockquote)
  3. raccomandazioni.json              (Dati strutturati)

Uso:
  uv run --with pymupdf python extract_guidelines.py linea_guida.pdf --outdir estratti
"""

import argparse
import datetime
import json
import pathlib
import re
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

# ==============================================================================
# CONFIGURAZIONE REGEX
# ==============================================================================
# 1. Identifica l'inizio di una raccomandazione (es. "Raccomandazione 1.1", "Recommendation 2", "REC:")
REC_START_RE = re.compile(r"^(?:Raccomandazione|Recommendation|REC)[\s:]*(\d+(?:\.\d+)*)?\.?\s*(.*)$", re.I)

# 2. Identifica la riga della forza/evidenza (supporta format italiani e inglesi)
STRENGTH_EVIDENCE_RE = re.compile(
    r"(?:Forza|Classe|Strength|Grade)[\s:]*(.*?)\s*(?:/|\||,|-)?\s*(?:Qualit[aà]|Livello|Prove|Level(?: of evidence)?|Evidence)[\s:]*(.*)", re.I
)

# 3. Identifica l'inizio della motivazione
JUSTIFICATION_START_RE = re.compile(r"^(?:Giustificazione|Motivazione|Rationale|Justification|Discussion)[\s:]*(.*)$", re.I)

# Parole di fine blocco o sezioni da saltare (es. bibliografia)
END_OF_RECS_RE = re.compile(r"^(?:Bibliografia|References|Indice)\s*$", re.I)


# ==============================================================================
# FUNZIONI DI UTILITA'
# ==============================================================================
def squash(s):
    """Rimuove spazi multipli e pulisce la stringa."""
    return re.sub(r"\s+", " ", s).strip()


def map_evidence_to_tag(evidence_text):
    """Mappa i livelli di evidenza testuali nei tag standard [A-E]."""
    text = evidence_text.lower()
    if "alta" in text or "livello a" in text:
        return "[A]"
    elif "moderata" in text or "livello b" in text:
        return "[B]"
    elif "bassa" in text or "livello c" in text:
        return "[C]"
    elif "molto bassa" in text or "consenso" in text or "livello e" in text:
        return "[E]"
    return "[U]"  # Unclassified/Unknown


def page_lines(page):
    """Estrae le righe della pagina escludendo intestazioni e piè di pagina numerici."""
    out = []
    lines = page.get_text("text").split("\n")
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        # Salta numeri di pagina o intestazioni ricorrenti nelle prime/ultime righe
        if re.fullmatch(r"\d{1,3}", s) and (idx < 5 or idx > len(lines) - 5):
            continue
        out.append(s)
    return out


# ==============================================================================
# MODULO DI ESTRAZIONE E PARSING
# ==============================================================================
def find_rec_range(doc):
    """Identifica l'intervallo di pagine contenente le raccomandazioni, saltando indice/copertine."""
    start_page = end_page = None

    for i, p in enumerate(doc):
        text = p.get_text("text")
        # Trova la prima pagina che contiene esplicitamente una raccomandazione formattata
        if start_page is None and re.search(r"(?:Raccomandazione|REC)\s*\d+\.\d+", text, re.I):
            # Assicuriamoci che non sia l'indice (che di solito ha molti numeri di pagina o puntini)
            if not re.search(r"\.{5,}", text):
                start_page = i

        # Condizione di uscita (es. Inizio bibliografia)
        if start_page is not None and re.search(r"^(?:Bibliografia|References)\s*$", text, re.M | re.I):
            end_page = i
            break

    if start_page is None:
        print("[AVVISO] Impossibile determinare l'inizio esatto tramite Regex. Parto dalla pagina 1.")
        start_page = 0
    if end_page is None:
        end_page = len(doc) - 1

    return start_page, end_page


def parse_recommendations(doc, start_page, end_page):
    """Macchina a stati finiti per analizzare il testo con Regex estraendo i blocchi semantici."""
    lines = []
    for i in range(start_page, end_page + 1):
        for s in page_lines(doc[i]):
            lines.append((i + 1, s))

    recs = []
    current_rec = None
    state = None  # Stati possibili: "REC_TEXT", "JUSTIFICATION"

    for pg, s in lines:
        if END_OF_RECS_RE.match(s):
            break

        # 1. Intercetta una nuova raccomandazione
        rec_match = REC_START_RE.match(s)
        if rec_match:
            if current_rec:
                recs.append(current_rec)

            rec_id = rec_match.group(1)
            rec_text = rec_match.group(2)

            current_rec = {
                "id": rec_id,
                "testo": rec_text,
                "forza": "Non specificata",
                "evidenza": "Non specificata",
                "tag_evidenza": "[U]",
                "giustificazione": "",
                "pagina": pg
            }
            state = "REC_TEXT"
            continue

        if not current_rec:
            continue

        # 2. Intercetta Forza ed Evidenza
        strength_match = STRENGTH_EVIDENCE_RE.search(s)
        if strength_match:
            current_rec["forza"] = squash(strength_match.group(1))
            evidenza_text = squash(strength_match.group(2))
            current_rec["evidenza"] = evidenza_text
            current_rec["tag_evidenza"] = map_evidence_to_tag(evidenza_text)
            continue

        # 3. Intercetta l'inizio della Giustificazione/Motivazione
        just_match = JUSTIFICATION_START_RE.match(s)
        if just_match:
            state = "JUSTIFICATION"
            current_rec["giustificazione"] += just_match.group(1) + " "
            continue

        # 4. Accodamento testo allo stato corrente
        if state == "REC_TEXT":
            current_rec["testo"] += " " + s
        elif state == "JUSTIFICATION":
            current_rec["giustificazione"] += " " + s

    # Appendi l'ultima raccomandazione
    if current_rec:
        recs.append(current_rec)

    # Pulisci i testi estratti
    for r in recs:
        r["testo"] = squash(r["testo"])
        r["giustificazione"] = squash(r["giustificazione"])

    return recs


# ==============================================================================
# MODULO DI ESPORTAZIONE FILE
# ==============================================================================
def get_front_matter(title):
    """Genera il Front Matter YAML per i file Markdown."""
    today = datetime.date.today().isoformat()
    return (
        "---\n"
        f"doc_id: {title.lower().replace(' ', '-')}\n"
        f"source_name: \"{title}\"\n"
        "doc_type: clinical_guideline\n"
        "audience: clinician\n"
        f"last_verified: {today}\n"
        "---\n\n"
    )


def write_outputs(recs, outdir, doc_title):
    outdir.mkdir(parents=True, exist_ok=True)
    fm = get_front_matter(doc_title)

    # 1. raccomandazioni_sintetiche_ada.md
    synth_lines = [fm, f"# {doc_title} - Sintesi Raccomandazioni\n"]
    for r in recs:
        synth_lines.append(f"- **{r['id']}** {r['testo']} {r['tag_evidenza']}")

    (outdir / "raccomandazioni_sintetiche_ada.md").write_text("\n".join(synth_lines) + "\n", encoding="utf-8")

    # 2. raccomandazioni_estese_rag.md
    ext_lines = [fm, f"# {doc_title} - Raccomandazioni Estese con Motivazione\n"]
    for r in recs:
        ext_lines.append(f"### Raccomandazione {r['id']}\n")
        ext_lines.append(f"**Testo:** {r['testo']}\n")
        ext_lines.append(
            f"**Forza:** {r['forza']} | **Evidenza:** {r['evidenza']} ({r['tag_evidenza']}) | **Pagina:** {r['pagina']}\n")
        if r['giustificazione']:
            ext_lines.append(f"> **Motivazione:** {r['giustificazione']}\n")
        ext_lines.append("---\n")

    (outdir / "raccomandazioni_estese_rag.md").write_text("\n".join(ext_lines) + "\n", encoding="utf-8")

    # 3. raccomandazioni.json
    (outdir / "raccomandazioni.json").write_text(
        json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", help="Percorso del file PDF da analizzare")
    ap.add_argument("--outdir", default="estratti", help="Cartella di destinazione dei file estratti")
    ap.add_argument("--title", default="Documento Linee Guida", help="Titolo del documento per il Front Matter")
    args = ap.parse_args()

    path = pathlib.Path(args.pdf)
    if not path.exists():
        sys.exit(f"[ERRORE] File non trovato: {path}")

    print(f"Apertura del PDF: {path.name}...")
    doc = fitz.open(path)

    start_pg, end_pg = find_rec_range(doc)
    print(f"Intervallo di analisi individuato: pagine {start_pg + 1} - {end_pg + 1}")

    recs = parse_recommendations(doc, start_pg, end_pg)

    if not recs:
        print("[ATTENZIONE] Nessuna raccomandazione intercettata. Verifica le Regex rispetto al formato del PDF.")
    else:
        outdir = pathlib.Path(args.outdir)
        write_outputs(recs, outdir, args.title)
        print(f"[OK] Estratte {len(recs)} raccomandazioni.")
        print(f"[OK] File salvati correttamente nella directory: {outdir}/")


if __name__ == "__main__":
    main()