#!/usr/bin/env python3
"""
Estrattore dei requisiti per patente e Diabete Mellito dal Decreto 30 novembre 2010.
Ispirato allo script ADA, ma basato su parsing testuale (Regex) poiché il decreto
non usa riquadri colorati per evidenziare i punti importanti.

Uso:
  pip install pymupdf
  python estrai_diabete_patente.py 79934_decreto-30-novembre-2010.pdf --outdir output
"""

import argparse
import pathlib
import re

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Per favore installa PyMuPDF: pip install pymupdf")
    exit(1)


def clean_text(text):
    """Pulisce il testo da a capo inutili e spazi multipli."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_diabetes_section(pdf_path):
    """Estrae l'intero testo del PDF e isola la sezione Diabete Mellito."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Trova il blocco compreso tra "DIABETE MELLITO" e "Allegato III" (Epilessia)
    match = re.search(r'DIABETE MELLITO(.*?)Allegato III', full_text, re.DOTALL)
    if not match:
        raise ValueError("Sezione 'DIABETE MELLITO' non trovata nel documento.")

    return match.group(1)


def parse_groups(diabetes_text):
    """Divide il testo in definizioni generali, Gruppo 1 e Gruppo 2."""
    # Estrai definizioni iniziali (ipoglicemia grave/ricorrente) prima di "Gruppo 1"
    intro_match = re.search(r'(.*?)Gruppo 1', diabetes_text, re.DOTALL)
    intro = clean_text(intro_match.group(1)) if intro_match else ""

    # Estrai Gruppo 1
    g1_match = re.search(r'Gruppo 1(.*?)Gruppo 2', diabetes_text, re.DOTALL)
    g1_text = g1_match.group(1) if g1_match else ""

    # Estrai Gruppo 2
    g2_match = re.search(r'Gruppo 2(.*)', diabetes_text, re.DOTALL)
    g2_text = g2_match.group(1) if g2_match else ""

    return intro, g1_text, g2_text


def extract_numbered_rules(group_text):
    """Estrae le regole numerate (es. '1. Testo...', '2. Testo...') in una lista."""
    # Cerca numeri seguiti da punto all'inizio di una frase
    # Nel testo grezzo del PDF potrebbero esserci a capo casuali, quindi usiamo un lookahead
    rules_raw = re.split(r'\s(?=\d+\.\s)', " " + group_text)
    rules = []
    for r in rules_raw:
        cleaned = clean_text(r)
        if re.match(r'^\d+\.', cleaned):
            rules.append(cleaned)
    return rules


def write_markdown(intro, g1_rules, g2_rules, out_path):
    """Scrive i risultati in formato Markdown."""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("---\n")
        f.write("titolo: Requisiti Patente - Diabete Mellito\n")
        f.write("fonte: Decreto 30 novembre 2010 (Allegato II)\n")
        f.write("---\n\n")
        f.write("# Requisiti di Idoneità alla Guida per Diabete Mellito\n\n")

        f.write("## Definizioni Chiave\n")
        f.write(f"> {intro}\n\n")

        f.write("## Gruppo 1 (Patenti A, B, B+E, ecc.)\n")
        for rule in g1_rules:
            f.write(f"* **{rule.split('.')[0]}.** {rule[rule.find('.') + 1:].strip()}\n")

        f.write("\n## Gruppo 2 (Patenti C, D, professionali, ecc.)\n")
        for rule in g2_rules:
            f.write(f"* **{rule.split('.')[0]}.** {rule[rule.find('.') + 1:].strip()}\n")


def main():
    parser = argparse.ArgumentParser(description="Estrae linee guida sul diabete dal Decreto 2010.")
    parser.add_argument("input_pdf", help="Percorso del PDF del decreto")
    parser.add_argument("--outdir", default="output_estratti", help="Cartella di destinazione")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / "diabete_patente_recs.md"

    print(f"Lettura del file: {args.input_pdf}...")
    try:
        raw_section = extract_diabetes_section(args.input_pdf)
        intro, g1_text, g2_text = parse_groups(raw_section)

        g1_rules = extract_numbered_rules(g1_text)
        g2_rules = extract_numbered_rules(g2_text)

        write_markdown(intro, g1_rules, g2_rules, out_file)
        print(f"[OK] Estrazione completata! Salvato in: {out_file}")

    except Exception as e:
        print(f"[ERRORE] Elaborazione fallita: {e}")


if __name__ == "__main__":
    main()