#!/usr/bin/env python3
r"""Smoke test end-to-end del Retrieval Ibrido (Vertex Embedding EU + Supabase RPC RRF).

Esegue 15 domande specifiche e diversificate sul corpus clinico (documenti IT + EN),
mostrando per ciascuna i Top 8 chunk recuperati da `public.hybrid_search_chunks`
con punteggi semantici (`similarity`), lessicali (`fts_rank`), combinati (`rrf_score`)
e verificando automaticamente il recupero dei documenti attesi.

Uso:
  .\.venv\Scripts\python.exe scripts/smoke_rag.py
  .\.venv\Scripts\python.exe scripts/smoke_rag.py --with-llm --limit 3
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from dotenv import find_dotenv, load_dotenv
from supabase import create_client


@dataclass(frozen=True)
class BenchmarkCase:
    idx: int
    category: str
    question: str
    expected_doc_substrings: tuple[str, ...]


BENCHMARK_QUESTIONS: list[BenchmarkCase] = [
    BenchmarkCase(
        1,
        "Emergenze / Ipoglicemia (IT+EN)",
        "Ho la glicemia a 62 mg/dl e mi tremano le mani: come funziona la regola del 15 e perché devo evitare il cioccolato?",
        ("iss-ipoglicemia", "ada2026-ch6"),
    ),
    BenchmarkCase(
        2,
        "Normativa / Esenzione 013 (IT)",
        "Con il codice di esenzione 013 per il diabete mellito quali visite di controllo ed esami di laboratorio sono previsti?",
        ("esenzione-013", "faq_ministero"),
    ),
    BenchmarkCase(
        3,
        "Normativa / Patente di guida (IT)",
        "Prendo l'insulina per il diabete: ogni quanti anni devo rinnovare la patente B (Gruppo 1) e cosa cambia per la patente C (Gruppo 2)?",
        ("patente-diabete",),
    ),
    BenchmarkCase(
        4,
        "Linee Guida / Target HbA1c e Dieta (IT)",
        "Qual è il target di emoglobina glicata HbA1c raccomandato dalle linee guida italiane SID-AMD nel diabete di tipo 2 e quale dieta è preferibile?",
        ("sid-amd-2025",),
    ),
    BenchmarkCase(
        5,
        "Farmaci / Conservazione Insulina Rapida (EN)",
        "Per quante settimane posso conservare a temperatura ambiente fuori dal frigorifero una penna di insulina rapida già aperta come NovoRapid o Humalog?",
        ("rapid-acting-insulin", "travelling_with_short_acting_insulin"),
    ),
    BenchmarkCase(
        6,
        "Farmaci / Tecnica Iniezione Insulina (EN)",
        "Dove si fa l'iniezione di insulina con la penna e perché bisogna cambiare ago ad ogni iniezione e ruotare i punti per evitare la lipoipertrofia?",
        ("insulin-injection",),
    ),
    BenchmarkCase(
        7,
        "Viaggi / Aereo e Controlli Sicurezza (EN)",
        "Devo prendere l'aereo per un viaggio: dove devo mettere l'insulina e posso portare liquidi per il diabete sopra i 100 ml nel bagaglio a mano ai controlli?",
        ("cdc-t2d-travel", "travel", "rapid-acting-insulin"),
    ),
    BenchmarkCase(
        8,
        "Sport / Sedentarietà e Pause Attive (IT/EN)",
        "Passo tante ore seduto alla scrivania: ogni quanti minuti dovrei interrompere la sedentarietà con pause attive e quanti giorni consecutivi posso stare senza esercizio?",
        ("aniad-vademecum", "acsm"),
    ),
    BenchmarkCase(
        9,
        "Nutrizione / Porzioni Standard LARN (IT)",
        "Che differenza c'è secondo i LARN SINU tra porzione standard, porzione consumata e unità di vendita degli alimenti?",
        ("sinu-standard-porzioni",),
    ),
    BenchmarkCase(
        10,
        "Nutrizione / IMC e Controllo del Peso (IT)",
        "Come si calcola l'Indice di Massa Corporea (IMC) e tra quali valori si parla di normopeso, sovrappeso e obesità secondo le linee guida CREA?",
        ("crea-linee-guida",),
    ),
    BenchmarkCase(
        11,
        "Anziani / Obiettivi Glicemici, Pressione e Proteine (EN)",
        "Per un adulto anziano con diabete qual è l'obiettivo di pressione arteriosa (<130/80 mmHg) e quanti grammi di proteine per kg di peso sono consigliati per la massa magra?",
        ("ada2026-ch13",),
    ),
    BenchmarkCase(
        12,
        "Dispositivi / CGM e Time in Range (EN)",
        "Uso il sensore per il monitoraggio continuo del glucosio (CGM): qual è l'obiettivo di Time in Range (TIR) tra 70 e 180 mg/dL e quanto tempo posso stare sotto i 70 mg/dL?",
        ("ada2026-ch6", "ada2026-ch7", "continuous-glucose-monitoring"),
    ),
    BenchmarkCase(
        13,
        "Complicanze / Nefropatia e Microalbuminuria (IT/EN)",
        "Che cos'è la microalbuminuria nella nefropatia diabetica e quali farmaci e attenzioni aiutano a proteggere i reni?",
        ("sid-complicanze", "ada2026-ch11"),
    ),
    BenchmarkCase(
        14,
        "Qualità di vita / Problemi Vescicali e Sessuali (EN)",
        "Il diabete e la neuropatia diabetica possono causare problemi alla vescica come perdite urinarie, urgenza minzionale o difficoltà sessuali?",
        ("sexual-bladder-problems",),
    ),
    BenchmarkCase(
        15,
        "Salute Mentale / Diabetes Distress e Ansia (EN)",
        "Mi sento stanco, in ansia e sopraffatto dal dover gestire il diabete ogni giorno (diabetes distress): come influisce il benessere emotivo sulla cura del diabete?",
        ("mental-health", "motivational_interviewing", "ada2026-ch5"),
    ),
]


def _match_rank(chunks: list[Any], expected_substrings: tuple[str, ...]) -> int | None:
    for idx, c in enumerate(chunks, start=1):
        doc_id = (c.doc_id or "").lower()
        source = (c.source_name or "").lower()
        for sub in expected_substrings:
            if sub.lower() in doc_id or sub.lower() in source:
                return idx
    return None


def _origin_tag(c: Any) -> str:
    has_sem = c.similarity is not None
    has_fts = c.fts_rank is not None
    if has_sem and has_fts:
        return "IBRIDO (SEM+FTS)"
    if has_sem:
        return "SOLO SEMANTICO  "
    if has_fts:
        return "SOLO FTS        "
    return "N/D             "


async def run_benchmark(limit: int | None, with_llm: bool) -> int:
    load_dotenv(find_dotenv(usecwd=True))
    # Se nel .env ENVIRONMENT non è 'dev' o 'prod' (es. 'vertex'), lo forziamo a 'dev' per il test CLI
    if os.environ.get("ENVIRONMENT") not in ("dev", "prod"):
        os.environ["ENVIRONMENT"] = "dev"

    from vitality_be.core.config import get_settings
    from vitality_be.llm.embedder import get_embedder
    from vitality_be.llm.factory import get_llm
    from vitality_be.schemas.chat import ChatMessageIn
    from vitality_be.services import chat as chat_service
    from vitality_be.services.retriever import retrieve_chunks

    s = get_settings()
    supa_url = s.supabase_url
    # Per lo script CLI usiamo la service_role_key se presente (così bypassa il login JWT Flutter),
    # altrimenti la anon_key.
    supa_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or s.supabase_anon_key
    db = create_client(supa_url, supa_key)

    embedder = get_embedder()
    llm = get_llm() if with_llm else None

    cases = BENCHMARK_QUESTIONS[:limit] if limit else BENCHMARK_QUESTIONS
    print("=" * 95)
    print(
        f"BENCHMARK RETRIEVAL IBRIDO (Vertex AI '{s.embedding_model}' dim={s.embedding_dim} "
        f"+ Supabase RRF Top-{s.rag_top_k})"
    )
    print(f"Domande da eseguire: {len(cases)} | Generazione LLM attiva: {with_llm}")
    print("=" * 95)

    hits_top1 = 0
    hits_top3 = 0
    hits_top8 = 0
    both_sources_count = 0
    total_chunks_returned = 0

    try:
        for case in cases:
            print(f"\n[{case.idx:02d}/{len(cases):02d}] Categoria: {case.category}")
            print(f"Domanda: \"{case.question}\"")
            t0 = time.monotonic()
            chunks = await retrieve_chunks(case.question, user_db=db, embedder=embedder)
            dt_ms = int((time.monotonic() - t0) * 1000)

            if not chunks:
                print(f"  [!] NESSUN CHUNK RECUPERATO ({dt_ms} ms) — Verifica che la funzione SQL hybrid_search_chunks sia stata creata su Supabase!")
                continue

            total_chunks_returned += len(chunks)
            rank = _match_rank(chunks, case.expected_doc_substrings)
            if rank == 1:
                hits_top1 += 1
            if rank is not None and rank <= 3:
                hits_top3 += 1
            if rank is not None and rank <= 8:
                hits_top8 += 1

            hybrid_matches = sum(1 for c in chunks if c.similarity is not None and c.fts_rank is not None)
            both_sources_count += hybrid_matches

            rank_str = f"TROVATO in posizione #{rank}" if rank else "NON nei Top 8"
            print(f"  -> Recuperati {len(chunks)} chunk in {dt_ms} ms | Documento atteso {case.expected_doc_substrings}: {rank_str}")
            print("  " + "-" * 91)
            print(f"  {'#':<2} | {'Origine':<16} | {'RRF':<7} | {'Sim':<6} | {'FTS':<6} | {'Lang':<4} | {'doc_id #idx':<32} | Sezione")
            print("  " + "-" * 91)
            for pos, c in enumerate(chunks, start=1):
                sim_s = f"{c.similarity:.3f}" if c.similarity is not None else "  -   "
                fts_s = f"{c.fts_rank:.3f}" if c.fts_rank is not None else "  -   "
                doc_tag = f"{c.doc_id}#{c.chunk_index}"[:32]
                sec_tag = (c.heading_path or "")[:38]
                print(
                    f"  {pos:<2} | {_origin_tag(c)} | {c.rrf_score:.4f}  | {sim_s} | {fts_s} | {(c.language or '-'):<4} | {doc_tag:<32} | {sec_tag}"
                )

            if with_llm and llm is not None:
                t_llm = time.monotonic()
                answer = await chat_service.reply(
                    llm,
                    [ChatMessageIn(role="user", content=case.question)],
                    user_db=db,
                    embedder=embedder,
                )
                llm_ms = int((time.monotonic() - t_llm) * 1000)
                print(f"\n  [Risposta LLM ({llm_ms} ms)]:\n  {answer}\n")

        print("\n" + "=" * 95)
        print("RIEPILOGO METRICHE RETRIEVAL IBRIDO")
        print("=" * 95)
        n = len(cases)
        print(f"Domande testate:                {n}")
        print(f"Hit@1 (atteso al 1° posto):     {hits_top1}/{n} ({(hits_top1 / n) * 100:.1f}%)")
        print(f"Hit@3 (atteso nei primi 3):     {hits_top3}/{n} ({(hits_top3 / n) * 100:.1f}%)")
        print(f"Hit@8 (atteso nei Top 8):       {hits_top8}/{n} ({(hits_top8 / n) * 100:.1f}%)")
        print(f"Chunk con match doppio SEM+FTS: {both_sources_count} su {total_chunks_returned} chunk totali restituiti")
        print("=" * 95)
        return 0 if hits_top8 > 0 else 1
    finally:
        if hasattr(embedder, "aclose"):
            await embedder.aclose()
        if llm is not None and hasattr(llm, "aclose"):
            await llm.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test su 15 domande per il retrieval ibrido.")
    parser.add_argument("--limit", type=int, default=None, help="Limita il numero di domande da eseguire.")
    parser.add_argument("--with-llm", action="store_true", help="Genera anche la risposta con Gemini.")
    args = parser.parse_args()
    return asyncio.run(run_benchmark(args.limit, args.with_llm))


if __name__ == "__main__":
    raise SystemExit(main())
