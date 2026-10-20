
# -*- coding: utf-8 -*-
"""
Estrattore PDF ESC 2020 per RAG su sport, attività fisica e T2D.

Fonte prevista:
2020 ESC Guidelines on sports cardiology and exercise
in patients with cardiovascular disease.

Funzioni:
1. Estrazione del testo del PDF con numeri di pagina.
2. Ricerca di parole chiave.
3. Individuazione euristica delle pagine con raccomandazioni.
4. Esportazione in Markdown e JSONL.
5. Ricerca tematica per sport, diabete e attività fisica.

IMPORTANTE:
- I numeri di pagina indicano le pagine fisiche del PDF.
- L'estrazione automatica non garantisce la corretta ricostruzione
  di tabelle, colonne o box grafici.
- Le pagine individuate come raccomandazioni devono essere verificate
  sul PDF originale.
- Questo script non determina l'idoneità sportiva di una persona.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pypdf


DEFAULT_PDF = "sources/sport/2020-SPORT-E-ATTIVITA-FISICA-LG-ESC-COMPLETE.pdf"

TOPIC_KEYWORDS = {
    "T2D_DIABETE": [
        "diabetes",
        "diabetes mellitus",
        "type 2 diabetes",
        "type 1 diabetes",
        "glycaemia",
        "glycemia",
        "blood glucose",
        "hypoglycemia",
        "hypoglycaemia",
        "hyperglycemia",
        "hyperglycaemia",
        "insulin",
        "diabete",
        "glicemia",
        "ipoglicemia",
        "insulina",
    ],
    "ESERCIZIO_FISICO": [
        "physical activity",
        "physical exercise",
        "exercise training",
        "aerobic exercise",
        "resistance exercise",
        "strength training",
        "exercise intensity",
        "sedentary",
        "attività fisica",
        "esercizio fisico",
    ],
    "SPORT_CARDIOLOGIA": [
        "sports participation",
        "competitive sports",
        "recreational sports",
        "sports cardiology",
        "cardiovascular disease",
        "cardiovascular risk",
        "athlete",
        "sport agonistico",
        "cardiopatia",
    ],
    "IPERTENSIONE": [
        "hypertension",
        "blood pressure",
        "systolic blood pressure",
        "ipertensione",
        "pressione arteriosa",
    ],
    "CORONAROPATIA": [
        "coronary artery disease",
        "chronic coronary syndrome",
        "acute coronary syndrome",
        "myocardial ischemia",
        "coronaropatia",
        "ischemia miocardica",
    ],
    "CARDIOMIOPATIE": [
        "hypertrophic cardiomyopathy",
        "arrhythmogenic cardiomyopathy",
        "dilated cardiomyopathy",
        "cardiomyopathy",
        "cardiomiopatia",
    ],
}

RECOMMENDATION_PATTERN = re.compile(
    r"\brecommendations?\b|"
    r"\bclass\s*(?:I{1,3}|IV|IIa|IIb)\b|"
    r"\blevel\s*(?:of evidence\s*)?[ABC]\b|"
    r"\bshould be considered\b|"
    r"\bis recommended\b|"
    r"\bis not recommended\b",
    flags=re.IGNORECASE,
)


class ESCPDFTextExtractor:
    """Estrae e cerca testo nel PDF delle linee guida ESC."""

    def __init__(self, pdf_path: str | Path = DEFAULT_PDF) -> None:
        self.pdf_path = Path(pdf_path)
        self.reader: pypdf.PdfReader | None = None
        self.load_error: str | None = None

        if not self.pdf_path.is_file():
            self.load_error = f"File PDF non trovato: {self.pdf_path}"
            return

        try:
            self.reader = pypdf.PdfReader(str(self.pdf_path))
        except Exception as exc:
            self.load_error = f"Impossibile leggere il PDF: {exc}"

    @property
    def is_loaded(self) -> bool:
        return self.reader is not None

    @property
    def page_count(self) -> int:
        if not self.reader:
            return 0
        return len(self.reader.pages)

    def _require_reader(self) -> pypdf.PdfReader:
        if self.reader is None:
            raise RuntimeError(
                self.load_error or "Il PDF non è stato caricato."
            )
        return self.reader

    def get_page_text(self, page_index: int) -> str:
        """Restituisce il testo di una pagina, usando indice da zero."""
        reader = self._require_reader()

        if not 0 <= page_index < len(reader.pages):
            raise IndexError(f"Indice pagina non valido: {page_index}")

        return reader.pages[page_index].extract_text() or ""

    def iter_pages(self):
        """Genera record con pagina fisica e testo estratto."""
        reader = self._require_reader()

        for index, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                text = f"[ERRORE DI ESTRAZIONE: {exc}]"

            yield {
                "page": index + 1,
                "text": text,
                "characters": len(text),
                "empty": not bool(text.strip()),
            }

    @staticmethod
    def _make_snippet(text: str, keyword: str, radius: int = 220) -> str:
        """Estrae il contesto testuale attorno alla prima occorrenza."""
        match = re.search(re.escape(keyword), text, flags=re.IGNORECASE)

        if not match:
            return text[: radius * 2].strip()

        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)

        snippet = text[start:end].strip()

        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."

        return re.sub(r"\s+", " ", snippet)

    def search_keyword(
        self,
        keyword: str,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Cerca una parola o frase nel documento.

        Restituisce una lista di risultati con pagina, estratto e
        indicazione della pagina fisica del PDF.
        """
        self._require_reader()

        keyword = keyword.strip()
        if not keyword:
            raise ValueError("La parola chiave non può essere vuota.")
        if max_results < 1:
            raise ValueError("max_results deve essere almeno 1.")

        results = []

        for page_record in self.iter_pages():
            text = page_record["text"]

            if keyword.casefold() in text.casefold():
                results.append({
                    "page": page_record["page"],
                    "keyword": keyword,
                    "snippet": self._make_snippet(text, keyword),
                    "characters": page_record["characters"],
                })

                if len(results) >= max_results:
                    break

        return results

    def search_topics(
        self,
        topic: str = "T2D_DIABETE",
        max_results_per_keyword: int = 10,
    ) -> list[dict[str, Any]]:
        """Cerca tutte le parole chiave associate a un tema."""
        if topic not in TOPIC_KEYWORDS:
            available = ", ".join(TOPIC_KEYWORDS)
            raise ValueError(
                f"Tema sconosciuto: {topic}. Temi disponibili: {available}"
            )

        combined = []
        seen = set()

        for keyword in TOPIC_KEYWORDS[topic]:
            matches = self.search_keyword(
                keyword,
                max_results=max_results_per_keyword,
            )

            for match in matches:
                key = (match["page"], keyword.casefold())
                if key not in seen:
                    match["topic"] = topic
                    combined.append(match)
                    seen.add(key)

        return sorted(combined, key=lambda item: item["page"])

    def extract_recommendation_boxes(self) -> list[dict[str, Any]]:
        """
        Individua pagine potenzialmente rilevanti per le raccomandazioni.

        È un filtro euristico: NON ricostruisce necessariamente i box
        né garantisce che una raccomandazione sia stata estratta per intero.
        """
        self._require_reader()
        results = []

        for page_record in self.iter_pages():
            text = page_record["text"]

            if RECOMMENDATION_PATTERN.search(text):
                results.append({
                    "page": page_record["page"],
                    "content_preview": re.sub(
                        r"\s+", " ", text
                    ).strip()[:700],
                    "verification_required": True,
                })

        return results

    def export_markdown(self, output_path: str | Path) -> Path:
        """Esporta tutto il testo estratto, organizzato per pagina."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            file.write("# ESC 2020: testo estratto dal PDF\n\n")
            file.write(f"- File sorgente: `{self.pdf_path.name}`\n")
            file.write(f"- Pagine fisiche: {self.page_count}\n")
            file.write(
                "- Nota: controllare sul PDF originale tabelle, "
                "colonne e box grafici.\n\n"
            )

            for record in self.iter_pages():
                file.write(f"\n## Pagina PDF {record['page']}\n\n")

                if record["text"].strip():
                    file.write(record["text"].strip() + "\n")
                else:
                    file.write(
                        "_Nessun testo estratto da questa pagina._\n"
                    )

        return output_path

    def export_jsonl(
        self,
        output_path: str | Path,
        source_name: str | None = None,
    ) -> Path:
        """
        Esporta un record JSON per pagina, pronto per una pipeline RAG.

        Il JSONL contiene il testo estratto e metadati di provenienza.
        Non crea embedding e non esegue il chunking semantico.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_name = source_name or self.pdf_path.name

        with output_path.open("w", encoding="utf-8") as file:
            for record in self.iter_pages():
                payload = {
                    "source": source_name,
                    "page_pdf": record["page"],
                    "text": record["text"],
                    "characters": record["characters"],
                    "empty": record["empty"],
                    "document_type": "clinical_guideline",
                    "publisher": "European Society of Cardiology",
                    "year": 2020,
                    "topics": self._topics_for_text(record["text"]),
                    "verification_required": True,
                }

                file.write(
                    json.dumps(payload, ensure_ascii=False) + "\n"
                )

        return output_path

    @staticmethod
    def _topics_for_text(text: str) -> list[str]:
        """Assegna etichette tematiche in base alle parole presenti."""
        text_lower = text.casefold()
        topics = []

        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(word.casefold() in text_lower for word in keywords):
                topics.append(topic)

        return topics

    def export_recommendation_candidates(
        self,
        output_path: str | Path,
    ) -> Path:
        """Esporta in JSONL le pagine candidate contenenti raccomandazioni."""
        candidates = self.extract_recommendation_boxes()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            for item in candidates:
                item["source"] = self.pdf_path.name
                item["document_type"] = "recommendation_candidate"
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

        return output_path

    def quality_report(self) -> dict[str, Any]:
        """Produce statistiche semplici per verificare l'estrazione."""
        records = list(self.iter_pages())
        empty_pages = [
            record["page"] for record in records if record["empty"]
        ]

        return {
            "source": self.pdf_path.name,
            "page_count": len(records),
            "total_characters": sum(r["characters"] for r in records),
            "empty_pages": empty_pages,
            "empty_page_count": len(empty_pages),
            "pages_with_recommendation_terms": len(
                self.extract_recommendation_boxes()
            ),
            "note": (
                "Controllare manualmente le pagine vuote e un campione "
                "di tabelle e raccomandazioni."
            ),
        }


class ESCSportsClinicalEngine:
    """
    Interfaccia di ricerca tematica basata sul testo realmente estratto.

    Non contiene regole cliniche hardcoded e non assegna idoneità sportiva.
    """

    CATEGORY_TOPICS = {
        "T2D_DIABETE": ["T2D_DIABETE", "ESERCIZIO_FISICO"],
        "SOGGETTI_SANI_E_FATTORI_RISCHIO": [
            "ESERCIZIO_FISICO",
            "IPERTENSIONE",
            "T2D_DIABETE",
        ],
        "CORONAROPATIA_CCS": ["CORONAROPATIA", "SPORT_CARDIOLOGIA"],
        "CARDIOMIOPATIE": ["CARDIOMIOPATIE", "SPORT_CARDIOLOGIA"],
        "MIOCARDITE_PERICARDITE": ["SPORT_CARDIOLOGIA"],
        "ARITMIE_CANALOPATIE": ["SPORT_CARDIOLOGIA"],
    }

    def __init__(self, extractor: ESCPDFTextExtractor) -> None:
        self.extractor = extractor

    def get_categories(self) -> list[str]:
        return list(self.CATEGORY_TOPICS.keys())

    def query_rules(
        self,
        category_key: str,
        keyword: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Restituisce estratti documentali pertinenti alla categoria.

        Il nome query_rules è mantenuto per compatibilità con il vecchio
        script, ma i risultati sono estratti dal PDF, non regole decisionali.
        """
        if category_key not in self.CATEGORY_TOPICS:
            return []

        if keyword and keyword.strip():
            return self.extractor.search_keyword(
                keyword.strip(),
                max_results=max_results,
            )

        results = []
        seen_pages = set()

        for topic in self.CATEGORY_TOPICS[category_key]:
            for item in self.extractor.search_topics(
                topic,
                max_results_per_keyword=max_results,
            ):
                page = item["page"]
                if page not in seen_pages:
                    results.append(item)
                    seen_pages.add(page)

        return sorted(results, key=lambda item: item["page"])[:max_results]

    def generate_report(self, patient_data: dict[str, Any]) -> str:
        """
        Genera un report di ricerca documentale, non un giudizio clinico.

        Per minimizzare i dati personali, non è necessario inserire nome,
        data di nascita o altri identificativi del paziente.
        """
        category = patient_data.get("category", "T2D_DIABETE")
        keyword = patient_data.get("search_kw")

        matches = self.query_rules(
            category_key=category,
            keyword=keyword,
            max_results=int(patient_data.get("max_results", 10)),
        )

        lines = [
            "=" * 72,
            "REPORT DI RICERCA DOCUMENTALE: ESC SPORTS CARDIOLOGY 2020",
            "=" * 72,
            f"Categoria di ricerca: {category}",
            f"Parola chiave: {keyword or 'ricerca tematica'}",
            f"File sorgente: {self.extractor.pdf_path.name}",
            "",
            "AVVERTENZA:",
            "Questo report organizza estratti del PDF e non stabilisce",
            "diagnosi, prescrizioni, controindicazioni o idoneità sportiva.",
            "Le raccomandazioni e le classi di evidenza devono essere",
            "verificate nel documento originale e interpretate da un",
            "professionista sanitario qualificato.",
            "",
        ]

        if not matches:
            lines.append(
                "Nessun risultato trovato. Verificare il PDF, la parola "
                "chiave e la qualità del testo estratto."
            )
        else:
            lines.append(f"Risultati trovati: {len(matches)}")
            lines.append("")

            for index, item in enumerate(matches, start=1):
                lines.extend([
                    f"RISULTATO {index}",
                    f"Pagina fisica PDF: {item['page']}",
                    f"Parola chiave: {item.get('keyword', keyword or '')}",
                    f"Estratto: {item['snippet']}",
                    "-" * 72,
                ])

        return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estrae e ricerca contenuti nel PDF ESC 2020 per un RAG "
            "su sport e diabete."
        )
    )

    parser.add_argument(
        "--pdf",
        default=DEFAULT_PDF,
        help="Percorso del PDF sorgente.",
    )
    parser.add_argument(
        "--keyword",
        help="Cerca una parola o frase specifica.",
    )
    parser.add_argument(
        "--topic",
        choices=sorted(TOPIC_KEYWORDS.keys()),
        help="Esegue una ricerca tematica.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Numero massimo di risultati per ricerca.",
    )
    parser.add_argument(
        "--export-md",
        default="esc_2020_extracted.md",
        help="Percorso del Markdown integrale.",
    )
    parser.add_argument(
        "--export-jsonl",
        default="esc_2020_pages.jsonl",
        help="Percorso del JSONL per il RAG.",
    )
    parser.add_argument(
        "--export-recommendations",
        default="esc_2020_recommendation_candidates.jsonl",
        help="Percorso JSONL delle pagine candidate.",
    )
    parser.add_argument(
        "--report",
        default="esc_2020_quality_report.json",
        help="Percorso del report di qualità.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Non esporta i file, esegue solo la ricerca.",
    )

    args = parser.parse_args()

    if args.max_results < 1:
        parser.error("--max-results deve essere almeno 1.")

    extractor = ESCPDFTextExtractor(args.pdf)

    if not extractor.is_loaded:
        print(f"ERRORE: {extractor.load_error}", file=sys.stderr)
        print(
            "Controlla il percorso del PDF e verifica che il file esista.",
            file=sys.stderr,
        )
        return 1

    print(f"PDF caricato: {extractor.pdf_path}")
    print(f"Pagine fisiche: {extractor.page_count}")

    if not args.no_export:
        md_path = extractor.export_markdown(args.export_md)
        jsonl_path = extractor.export_jsonl(args.export_jsonl)
        rec_path = extractor.export_recommendation_candidates(
            args.export_recommendations
        )

        quality = extractor.quality_report()
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(quality, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Markdown: {md_path}")
        print(f"JSONL per RAG: {jsonl_path}")
        print(f"Pagine candidate: {rec_path}")
        print(f"Report qualità: {report_path}")

    if args.keyword:
        results = extractor.search_keyword(
            args.keyword,
            max_results=args.max_results,
        )
        print(f"\nRisultati per '{args.keyword}': {len(results)}")

        for item in results:
            print(f"\nPagina PDF {item['page']}:")
            print(item["snippet"])

    if args.topic:
        results = extractor.search_topics(
            args.topic,
            max_results_per_keyword=args.max_results,
        )
        print(f"\nRisultati per tema '{args.topic}': {len(results)}")

        for item in results[:args.max_results]:
            print(
                f"\nPagina PDF {item['page']} "
                f"[{item.get('keyword', '')}]:"
            )
            print(item["snippet"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
