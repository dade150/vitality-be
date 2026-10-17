#!/usr/bin/env python3
"""
Estrattore per la Linea Guida SID-AMD "La terapia del diabete mellito di tipo 2" (ed. 2025, LG-379 ISS).

Produce, nella cartella --outdir:
  sid-amd-2025_recs.md             raccomandazioni (testo fedele + forza + qualita' delle prove NATIVE)
  sid-amd-2025_giustificazioni.md  raccomandazione + giustificazione clinica (da indicizzare con cautela)
  sid-amd-2025_recs.json           stesso contenuto, strutturato
  pico/sid-amd-2025-pico-X.Y.md    una scheda per quesito clinico: quesito, PICO, outcome, giudizi EtD (ARCHIVIO)

Cosa corregge rispetto alla v1:
  - forza/qualita' erano attaccate alla raccomandazione SBAGLIATA (nel PDF il riquadro precede il testo)
  - le raccomandazioni 4.1 e 4.2 (terapia educazionale) venivano scartate da un filtro
  - intestazioni/piedi pagina e capitoli successivi finivano dentro il testo delle raccomandazioni
  - nessuna conversione inventata in lettere A-E: si tengono forza (forte/debole) e qualita' (alta..molto bassa)
  - aggiunge gli allegati per quesito (pagine ~33-365), dove sta la parte "corposa"

Uso:
  uv run --with pymupdf python sid_amd_pdf_extract_v2.py LG-379-diabete-tipo2-ed2025.pdf --outdir extracted/sidamd
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

HDR_START = "Linea Guida della Società Italiana di Diabetologia (SID) e dell"
HDR_2 = "La terapia del diabete mellito di tipo 2"
SCOPE = ("persone con diabete di tipo 2 in eta' adulta (>18 anni, nessun limite di eta' superiore), "
         "setting ambulatoriale (non ricovero)")
TERMINAL = (".", "!", "?", ":")

SEC_RE = re.compile(
    r"^4\.([1-6])\s+(Obiettivi terapeutici|Terapia nutrizionale|Esercizio fisico|"
    r"Terapia educazionale|Terapia farmacologica|Monitoraggio glicemico)\s*$", re.I)
REC_RE = re.compile(r"^(\d\.\d(?:\.\d)?)\.?\s+(\S.*)$")
FORZA_RE = re.compile(r"Forza della raccomandazione:\s*(.*?)\.?\s*Qualità delle prove:\s*(.*?)\.?\s*$", re.I)
PICO_HEAD = re.compile(r"^(\d)\.(\d+)(?:\.(\d+))?\.?\s+([A-ZÀ-Ú][A-ZÀ-Ú0-9 ’'\-/,()]{4,})$")
PICO_LABELS = ["Popolazione", "Intervento", "Confronto", "Outcome critici", "Setting"]
ETD_TITLES = ["Effetti favorevoli", "Effetti sfavorevoli", "Grado di evidenza", "Valore",
              "Bilancio degli effetti", "Risorse necessarie", "Grado di evidenza sulle risorse necessarie",
              "Costo-efficacia", "Equità", "Accettabilità", "Fattibilità"]
ETD_NOISE = {"Giudizio", "Evidenza", "Considerazioni aggiuntive"}
NUMERIC_NOISE = re.compile(r"^[\d.,;()%\s\-+/\[\]ORIC•]+$")


# ------------------------------------------------------------------ righe pulite
def page_lines(page):
    """Righe della pagina senza intestazione, piè di pagina e numero di pagina."""
    out = []
    for idx, raw in enumerate(page.get_text().split("\n")):
        s = raw.strip()
        if not s:
            continue
        if s.startswith(HDR_START) or s == HDR_2:
            continue
        if re.fullmatch(r"\d{1,3}", s) and idx < 6:
            continue                       # numero di pagina (nelle prime righe)
        out.append(s)
    return out


def squash(s):
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ raccomandazioni
def find_rec_range(doc):
    start = end = None
    for i, p in enumerate(doc):
        t = p.get_text()
        if start is None and i > 5 and re.search(r"^4\. Raccomandazioni", t, re.M):
            start = i
        if start is not None and re.search(r"^5\. Revisione esterna", t, re.M):
            end = i
            break
    if start is None or end is None:
        raise RuntimeError("Capitolo '4. Raccomandazioni' non trovato: il PDF e' quello giusto?")
    return start, end


def parse_recs(doc):
    start, end = find_rec_range(doc)
    lines = []
    for i in range(start, end + 1):
        for s in page_lines(doc[i]):
            lines.append((i + 1, s))

    sections = {}
    cur = None
    state = None
    box = rec = None
    i = 0
    while i < len(lines):
        pg, s = lines[i]
        i += 1
        if s.startswith("5. Revisione esterna"):
            break
        if s.startswith("4. Raccomandazioni"):
            continue
        m = SEC_RE.match(s)
        if m:
            k = m.group(1)
            cur = {"k": k, "titolo": m.group(2).capitalize(), "nota": "", "boxes": [], "recs": []}
            sections[k] = cur
            state, box, rec = "NOTE", None, None
            continue
        if cur is None:
            continue

        if s.startswith("Forza della raccomandazione"):
            buf = s
            while "Qualità delle prove" not in buf and i < len(lines):
                buf += " " + lines[i][1]
                i += 1
            fm = FORZA_RE.search(squash(buf))
            box = {"forza": fm.group(1).strip() if fm else "non determinabile",
                   "qualita_prove": fm.group(2).strip() if fm else "non determinabile",
                   "giustificazione": "", "pagina": pg}
            cur["boxes"].append(box)
            state = "BOX"
            continue

        if s.startswith("Giustificazione") and box is not None:
            rest = s[len("Giustificazione"):].strip()
            if rest:
                box["giustificazione"] += rest + " "
            state = "JUST"
            continue

        rm = REC_RE.match(s)
        if rm and rm.group(1).split(".")[0] == cur["k"] and not SEC_RE.match(s):
            rec = {"id": rm.group(1), "testo": rm.group(2), "pagina": pg, "ultima": pg}
            cur["recs"].append(rec)
            state = "REC"
            continue

        if state == "NOTE":
            cur["nota"] += s + " "
        elif state == "JUST" and box is not None:
            box["giustificazione"] += s + " "
        elif state == "REC" and rec is not None:
            dest = rec.get("dest_box")
            if dest is None:
                empty = next((b for b in cur["boxes"] if not b["giustificazione"].strip()), None)
                if (pg != rec["ultima"] and rec["testo"].rstrip().endswith(TERMINAL) and empty is not None):
                    dest = rec["dest_box"] = empty     # giustificazione stampata DOPO l'elenco (nuova pagina)
            if dest is not None:
                dest["giustificazione"] += s + " "
            else:
                rec["testo"] += " " + s
                rec["ultima"] = pg

    recs, warns = [], []
    for k in sorted(sections):
        sec = sections[k]
        if len(sec["boxes"]) != len(sec["recs"]):
            warns.append(f"sezione 4.{k}: {len(sec['boxes'])} riquadri di evidenza ma {len(sec['recs'])} raccomandazioni: "
                         "gradi NON associati, controllare a mano")
        for j, r in enumerate(sec["recs"]):
            b = sec["boxes"][j] if len(sec["boxes"]) == len(sec["recs"]) else None
            recs.append({
                "id": r["id"],
                "sezione": f"4.{k} {sec['titolo']}",
                "testo": squash(r["testo"]),
                "forza": b["forza"] if b else "non associata",
                "qualita_prove": b["qualita_prove"] if b else "non associata",
                "giustificazione": squash(b["giustificazione"]) if b else "",
                "nota_sezione": squash(sec["nota"]),
                "pagina": r["pagina"],
            })
    return recs, warns


# ------------------------------------------------------------------ allegati per quesito (PICO)
def parse_annex(doc, first_page=33):
    last = first_page
    for i in range(first_page - 1, len(doc)):
        if doc[i].get_text().strip():
            last = i + 1
        elif i > first_page + 5 and not doc[i].get_text().strip():
            # prime pagine vuote consecutive = fine del testo (restano le scansioni)
            if all(not doc[j].get_text().strip() for j in range(i, min(i + 5, len(doc)))):
                break
    lines = []
    for i in range(first_page - 1, last):
        for s in page_lines(doc[i]):
            lines.append((i + 1, s))

    starts = []
    for idx, (pg, s) in enumerate(lines):
        m = PICO_HEAD.match(s)
        if not m:
            continue
        nxt = [l for _, l in lines[idx + 1: idx + 7]]
        if any(l.startswith(("Quesito", "Critici")) for l in nxt):
            title = s
            j = idx + 1
            while j < len(lines) and j < idx + 3 and lines[j][1].isupper() and not lines[j][1].startswith(("Quesito", "Critici")):
                title += " " + lines[j][1]
                j += 1
            pid = ".".join(g for g in m.groups()[:3] if g)
            starts.append((idx, pid, title))

    picos = []
    for n, (idx, pid, title) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        block = lines[idx:stop]
        picos.append(parse_pico_block(pid, title, block))
    return picos


def take_until(texts, start_pred, end_preds):
    out, on = [], False
    for t in texts:
        if not on and start_pred(t):
            on = True
            rest = re.sub(r"^[^:]*:\s*", "", t) if ":" in t else ""
            if rest:
                out.append(rest)
            continue
        if on:
            if any(p(t) for p in end_preds):
                break
            out.append(t)
    return squash(" ".join(out))


def parse_pico_block(pid, title, block):
    pages = [p for p, _ in block]
    texts = [s for _, s in block]
    quesito = take_until(texts, lambda t: t.startswith("Quesito"), [lambda t: t.startswith("Critici")])
    if not quesito:                      # alcuni quesiti non hanno l'etichetta "Quesito:"
        k0 = 1
        while k0 < len(texts) and texts[k0].isupper():
            k0 += 1
        q = []
        for t in texts[k0:]:
            if t.startswith("Critici"):
                break
            q.append(t)
        quesito = squash(" ".join(q))
    critici = take_until(texts, lambda t: t.startswith("Critici"), [lambda t: t.startswith("Non critici")])
    non_crit = take_until(texts, lambda t: t.startswith("Non critici"), [lambda t: t.startswith("Popolazione")])
    pico = {}
    for lab in PICO_LABELS:
        others = [lambda t, l=o: t.startswith(l) for o in PICO_LABELS if o != lab]
        others.append(lambda t: t in ETD_TITLES or t in ETD_NOISE or t.startswith("VALUTAZIONE"))
        v = take_until(texts, lambda t, l=lab: t.startswith(l), others)
        if lab != "Outcome critici" or v:
            pico[lab] = v
    # giudizi Evidence-to-Decision (testo grezzo, colonne unite)
    etd, cur = {}, None
    for t in texts:
        if t in ETD_TITLES:
            cur = t
            etd.setdefault(cur, [])
            continue
        if cur is None or t in ETD_NOISE or t.endswith("?"):
            continue
        if "⨁" in t or NUMERIC_NOISE.match(t):
            continue
        etd[cur].append(t)
    etd = {k: squash(" ".join(v)) for k, v in etd.items() if v}
    return {"id": pid, "titolo": re.sub(r"^\d\.\d+(?:\.\d+)?\.?\s+", "", squash(title)), "pagine": f"{min(pages)}-{max(pages)}",
            "quesito": quesito, "outcome_critici": critici, "outcome_non_critici": non_crit,
            "pico": pico, "etd": etd}


# ------------------------------------------------------------------ scrittura
def front(doc_id, title, kind, extra=""):
    today = datetime.date.today().isoformat()
    return ("---\n"
            f"doc_id: {doc_id}\n"
            f"source_name: \"SID-AMD - La terapia del diabete mellito di tipo 2 (ed. 2025) - {title}\"\n"
            "source_org: SID-AMD / SNLG-ISS\n"
            "source_file: LG-379-diabete-tipo2-ed2025.pdf\n"
            f"doc_type: {kind}\n"
            "audience: clinician\n"
            "jurisdiction: IT\n"
            "language: it\n"
            f"population: \"{SCOPE}\"\n"
            "diabetes_type: [T2]\n"
            f"last_verified: {today}\n"
            "can_quote_numbers: false\n"
            "license_note: \"verificare condizioni di riuso SNLG-ISS prima di pubblicare\"\n"
            f"{extra}---\n\n")


def write_all(recs, picos, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pico").mkdir(exist_ok=True)

    L = [front("sid-amd-2025-recs", "raccomandazioni", "clinical_guideline"),
         "# SID-AMD diabete tipo 2 (2025) - raccomandazioni\n",
         "_Forza e qualita' delle prove sono quelle della linea guida (metodo GRADE), non lettere A-E._\n"]
    sec = None
    for r in recs:
        if r["sezione"] != sec:
            L.append(f"\n## {r['sezione']}\n")
            sec = r["sezione"]
        L.append(f"- **{r['id']}** {r['testo']}  \n  _[forza: {r['forza']}; prove: {r['qualita_prove']}; p. {r['pagina']}]_")
    (outdir / "sid-amd-2025_recs.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    G = [front("sid-amd-2025-giustificazioni", "giustificazioni", "clinical_guideline_rationale"),
         "# SID-AMD diabete tipo 2 (2025) - raccomandazioni con giustificazione\n"]
    for r in recs:
        G.append(f"\n## {r['id']} - {r['sezione']}\n")
        G.append(f"**Raccomandazione:** {r['testo']}\n")
        G.append(f"**Forza:** {r['forza']} | **Qualita' delle prove:** {r['qualita_prove']} | **Pagina:** {r['pagina']}\n")
        if r["nota_sezione"]:
            G.append(f"**Nota di sezione:** {r['nota_sezione']}\n")
        if r["giustificazione"]:
            G.append(f"> **Giustificazione:** {r['giustificazione']}\n")
    (outdir / "sid-amd-2025_giustificazioni.md").write_text("\n".join(G) + "\n", encoding="utf-8")

    (outdir / "sid-amd-2025_recs.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")

    for p in picos:
        P = [front(f"sid-amd-2025-pico-{p['id']}", f"quesito {p['id']} {p['titolo']}", "guideline_evidence_archive",
                   extra=f"archive_only: true\npages: \"{p['pagine']}\"\n"),
             f"# Quesito {p['id']} - {p['titolo']}\n",
             f"**Quesito:** {p['quesito']}\n"]
        if p["outcome_critici"]:
            P.append(f"**Outcome critici (punteggio di importanza):** {p['outcome_critici']}\n")
        if p["outcome_non_critici"]:
            P.append(f"**Outcome non critici:** {p['outcome_non_critici']}\n")
        for lab, v in p["pico"].items():
            if v:
                P.append(f"- **{lab}:** {v}")
        if p["etd"]:
            P.append("\n## Giudizi del panel (testo grezzo: giudizio, evidenza e considerazioni possono essere uniti)\n")
            for k, v in p["etd"].items():
                P.append(f"### {k}\n{v}\n")
        (outdir / "pico" / f"sid-amd-2025-pico-{p['id']}.md").write_text("\n".join(P) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--outdir", default="sid_out")
    ap.add_argument("--no-pico", action="store_true", help="salta gli allegati per quesito")
    args = ap.parse_args()

    path = pathlib.Path(args.pdf)
    if not path.exists():
        sys.exit(f"[ERRORE] File non trovato: {path}")
    doc = fitz.open(path)
    recs, warns = parse_recs(doc)
    picos = [] if args.no_pico else parse_annex(doc)
    write_all(recs, picos, pathlib.Path(args.outdir))

    print(f"[OK] {len(recs)} raccomandazioni; {len(picos)} schede quesito -> {args.outdir}/")
    for w in warns:
        print("  ATTENZIONE:", w)
    undefined = [r["id"] for r in recs if "non" in r["forza"].lower() and "associata" in r["forza"].lower()]
    if undefined:
        print("  ATTENZIONE: gradi non associati per:", undefined)
    nogiust = [r["id"] for r in recs if not r["giustificazione"]]
    if nogiust:
        print("  Senza giustificazione:", nogiust)


if __name__ == "__main__":
    main()