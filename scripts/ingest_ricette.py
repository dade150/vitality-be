import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Assicurati di aver impostato SUPABASE_URL e SUPABASE_ANON_KEY nel file .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def parse_float(val):
    if val is None or val == "" or val == "tr":
        return None
    try:
        return float(str(val).replace(",", "."))
    except (ValueError, TypeError):
        return None


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    data = []
    for line_num, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Avviso: Errore di parsing alla riga {line_num}: {e}")

    return data


def process_and_upload_ricette(json_filepath, batch_size=50):
    file_path = Path(json_filepath)
    if not file_path.exists():
        print(f"Errore: File non trovato in '{file_path.resolve()}'")
        return

    print(f"Lettura file ricette da '{file_path}'...")
    raw_data = load_json_file(file_path)

    print(f"Trovate {len(raw_data)} ricette. Mappatura dati in corso...")

    records = []
    for item in raw_data:
        food_code = str(item.get("food_code") or item.get("Codice") or "").strip()
        if not food_code:
            continue

        yield_val = parse_float(
            item.get("yield_factor") or
            item.get("Variazione Peso (Y.F.)") or
            item.get("yield")
        )

        record = {
            "food_code": food_code,
            "name": item.get("name") or item.get("Nome") or item.get("Ricetta") or "Senza nome",
            "portions": str(item.get("portions") or item.get("Porzioni") or item.get("PER UNA PIZZA") or ""),
            "yield_factor": yield_val,
            "cooking_time": str(item.get("cooking_time") or item.get("Tempo di cottura") or ""),
            "cooking_type": str(
                item.get("cooking_type") or item.get("Modalita' di cottura") or item.get("Cottura") or ""),
            "preparation": str(
                item.get("preparation") or item.get("Modalita' di preparazione") or item.get("Preparazione") or ""),
            "ingredients": item.get("ingredients") or item.get("ingredienti") or []
        }
        records.append(record)

    print(f"Inizio invio a Supabase ({len(records)} record totali)...")

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("ricette").upsert(batch, on_conflict="food_code").execute()
            print(f"Caricate {min(i + batch_size, len(records))} / {len(records)} ricette...")
        except Exception as e:
            print(f"Errore durante il caricamento del lotto {i}-{i + batch_size}: {e}")

    print("\nIngestione della tabella 'ricette' completata con successo!")


if __name__ == "__main__":
    FILE_PATH = "sources/alimentazione/crea_recipes.json"
    process_and_upload_ricette(FILE_PATH)