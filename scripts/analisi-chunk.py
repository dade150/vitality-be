
import argparse
import re
import statistics as st
import sys
from pathlib import Path


def chunks_by_h2(path: Path) -> list[str]:
    """Divide un file Markdown in chunk usando le intestazioni H2."""
    text = path.read_text(encoding="utf-8-sig")

    # Rimuove il front matter YAML iniziale, se presente.
    text = re.sub(
        r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)",
        "",
        text,
        count=1,
        flags=re.S,
    )

    # Ogni intestazione ## avvia un nuovo chunk.
    chunks = re.split(r"(?m)(?=^##[ \t]+)", text)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def percentile(values: list[int], p: float) -> int:
    """Percentile nearest-rank, senza dipendenze esterne."""
    ordered = sorted(values)
    index = max(0, int((p / 100 * len(ordered) + 0.999999)) - 1)
    return ordered[index]


def format_number(value: int | float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def analyze_file(path: Path) -> list[str]:
    return chunks_by_h2(path)


def get_heading(chunk: str) -> str:
    """Restituisce la prima intestazione Markdown del chunk."""
    for line in chunk.splitlines():
        if re.match(r"^#{1,6}[ \t]+", line):
            return line.strip()
    return "(contenuto senza intestazione)"


def preview_text(text: str, limit: int = 300) -> str:
    """Compatta e limita il testo usato come contesto."""
    compact = " ".join(text.split())
    if len(compact) > limit:
        return compact[:limit] + "..."
    return compact


def preview_words(text: str, n: int = 5) -> tuple[str, str]:
    """Restituisce le prime e le ultime n parole del chunk."""
    words = text.split()
    return " ".join(words[:n]), " ".join(words[-n:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analizza dimensioni e possibili problemi dei chunk Markdown "
            "generati dalle intestazioni H2."
        )
    )

    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("curated"),
        help="Cartella da analizzare, incluse le sottocartelle (default: curated)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=3000,
        help="Soglia per segnalare chunk grandi (default: 3000)",
    )
    parser.add_argument(
        "--short-threshold",
        type=int,
        default=80,
        help="Soglia massima per segnalare chunk brevi (default: 80)",
    )
    parser.add_argument(
        "--critical-threshold",
        type=int,
        default=30,
        help="Soglia massima per chunk molto brevi (default: 30)",
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=300,
        help="Caratteri mostrati per il contesto precedente e successivo (default: 300)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Mostra le statistiche di tutti i file",
    )

    args = parser.parse_args()

    if args.threshold < 1:
        parser.error("--threshold deve essere maggiore di zero")
    if args.short_threshold < 1:
        parser.error("--short-threshold deve essere maggiore di zero")
    if args.critical_threshold < 0:
        parser.error("--critical-threshold non può essere negativo")
    if args.context_chars < 1:
        parser.error("--context-chars deve essere maggiore di zero")

    root = args.directory

    if not root.exists() or not root.is_dir():
        print(f"Errore: directory non trovata: {root}", file=sys.stderr)
        return 1

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".md"
    )

    if not files:
        print(f"Nessun file Markdown trovato in: {root.resolve()}")
        return 0

    all_lengths: list[int] = []
    file_stats = []
    oversized = []
    short_chunks = []
    errors = []
    analyzed_files = {}

    for path in files:
        try:
            chunks = analyze_file(path)
        except (OSError, UnicodeError) as exc:
            errors.append((path, str(exc)))
            continue

        relative_path = path.relative_to(root)
        analyzed_files[relative_path] = chunks
        lengths = [len(chunk) for chunk in chunks]

        if not lengths:
            file_stats.append((relative_path, 0, 0, 0, 0, 0))
            continue

        all_lengths.extend(lengths)

        file_stats.append(
            (
                relative_path,
                len(lengths),
                sum(lengths),
                int(st.median(lengths)),
                min(lengths),
                max(lengths),
            )
        )

        for index, chunk in enumerate(chunks):
            size = len(chunk)
            heading = get_heading(chunk)

            if size > args.threshold:
                oversized.append(
                    (size, relative_path, index + 1, heading)
                )

            if size < args.short_threshold:
                previous_chunk = chunks[index - 1] if index > 0 else None
                next_chunk = (
                    chunks[index + 1]
                    if index + 1 < len(chunks)
                    else None
                )

                short_chunks.append(
                    (
                        size,
                        relative_path,
                        index + 1,
                        heading,
                        chunk,
                        previous_chunk,
                        next_chunk,
                    )
                )

    print("\nANALISI CHUNK MARKDOWN")
    print("=" * 72)
    print(f"Cartella:        {root.resolve()}")
    print(f"File trovati:    {len(files)}")
    print(f"File analizzati: {len(file_stats)}")
    print(f"Errori lettura:  {len(errors)}")
    print(f"Chunk totali:    {len(all_lengths)}")
    print(f"Soglia grande:   {format_number(args.threshold)} caratteri")
    print(f"Soglia brevi:    meno di {args.short_threshold} caratteri")
    print(f"Soglia critica:  meno di {args.critical_threshold} caratteri")

    if all_lengths:
        print("\nDISTRIBUZIONE GLOBALE")
        print("-" * 72)
        print(f"Caratteri totali: {format_number(sum(all_lengths))}")
        print(f"Media:            {format_number(st.mean(all_lengths))}")
        print(f"Mediana:          {format_number(st.median(all_lengths))}")
        print(f"Minimo:           {format_number(min(all_lengths))}")
        print(f"75° percentile:   {format_number(percentile(all_lengths, 75))}")
        print(f"90° percentile:   {format_number(percentile(all_lengths, 90))}")
        print(f"95° percentile:   {format_number(percentile(all_lengths, 95))}")
        print(f"Massimo:          {format_number(max(all_lengths))}")

        large_count = sum(n > args.threshold for n in all_lengths)
        short_count = sum(n < args.short_threshold for n in all_lengths)
        critical_count = sum(
            n < args.critical_threshold for n in all_lengths
        )

        print(
            f"Chunk oltre soglia grande: {large_count} "
            f"({large_count / len(all_lengths) * 100:.1f}%)"
        )
        print(
            f"Chunk sotto soglia breve:  {short_count} "
            f"({short_count / len(all_lengths) * 100:.1f}%)"
        )
        print(
            f"Chunk sotto soglia critica: {critical_count} "
            f"({critical_count / len(all_lengths) * 100:.1f}%)"
        )

    print("\nFILE CON CHUNK PIÙ GRANDI")
    print("-" * 72)

    if oversized:
        for size, path, index, heading in sorted(
            oversized, reverse=True
        ):
            print(
                f"\n{format_number(size):>8} caratteri | "
                f"{path} | chunk {index}"
            )
            print(f"  Titolo: {heading[:100]}")

            chunk = analyzed_files[path][index - 1]
            first, last = preview_words(chunk, 5)
            print(f"  Prime 5 parole: {first}")
            print(f"  Ultime 5 parole: {last}")
    else:
        print("Nessun chunk supera la soglia.")

    print("\nANALISI DEI CHUNK BREVI")
    print("-" * 72)

    if short_chunks:
        critical_count = sum(
            size < args.critical_threshold
            for size, *_ in short_chunks
        )
        normal_short_count = len(short_chunks) - critical_count

        print(f"Chunk brevi totali: {len(short_chunks)}")
        print(f"  Molto brevi:     {critical_count}")
        print(f"  Da verificare:   {normal_short_count}")
        print(
            "\nNota: la lunghezza è un indicatore, non una prova "
            "di errore di segmentazione."
        )

        # Prima i chunk più brevi, per evidenziare quelli più sospetti.
        for (
            size,
            path,
            index,
            heading,
            chunk,
            previous_chunk,
            next_chunk,
        ) in sorted(short_chunks, key=lambda item: item[0]):

            label = (
                "CRITICO"
                if size < args.critical_threshold
                else "DA VERIFICARE"
            )

            print(
                f"\n[{label}] {size} caratteri | "
                f"{path} | chunk {index}"
            )
            print(f"  Titolo: {heading}")
            print("  Testo del chunk:")
            print("  " + chunk.replace("\n", "\n  "))

            if previous_chunk is not None:
                print("  Contesto precedente:")
                print(
                    "  "
                    + preview_text(
                        previous_chunk, args.context_chars
                    )
                )
            else:
                print("  Contesto precedente: nessuno, primo chunk")

            if next_chunk is not None:
                print("  Contesto successivo:")
                print(
                    "  "
                    + preview_text(next_chunk, args.context_chars)
                )
            else:
                print("  Contesto successivo: nessuno, ultimo chunk")
    else:
        print("Nessun chunk sotto la soglia impostata.")

    if args.show_all:
        print("\nSTATISTICHE PER FILE")
        print("-" * 100)
        print(
            f"{'File':<45} {'Chunk':>7} {'Totale':>10} "
            f"{'Mediana':>9} {'Min':>8} {'Max':>8}"
        )

        for path, count, total, median, minimum, maximum in file_stats:
            print(
                f"{str(path)[:44]:<45} {count:>7} "
                f"{format_number(total):>10} "
                f"{format_number(median):>9} "
                f"{format_number(minimum):>8} "
                f"{format_number(maximum):>8}"
            )

    if errors:
        print("\nERRORI DI LETTURA")
        print("-" * 72)
        for path, error in errors:
            print(f"{path}: {error}")

    print()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
