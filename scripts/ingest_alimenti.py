import json
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Carica le variabili d'ambiente
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Assicurati di aver impostato SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY nel file .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def parse_float(val):
    if val is None or val == "" or val == "tr":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def clean_nutrients(nutrient_list):
    if not isinstance(nutrient_list, list):
        return {}

    cleaned = {}
    for item in nutrient_list:
        if not isinstance(item, dict):
            continue

        desc = item.get("description")
        val = parse_float(item.get("value"))

        if desc and val is not None and val > 0:
            cleaned[desc] = val

    return cleaned


def load_json_file(file_path):
    """Carica dati sia da file JSON standard che da file JSON Lines (NDJSON)."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Tentativo 1: JSON standard (singolo array o oggetto)
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass

    # Tentativo 2: JSON Lines / NDJSON (un record JSON per ogni riga)
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


def process_and_upload_alimenti(json_filepath, batch_size=100):
    file_path = Path(json_filepath)
    if not file_path.exists():
        print(f"Errore: File non trovato in '{file_path.resolve()}'")
        return

    print(f"Lettura file JSON da '{file_path}'...")
    raw_data = load_json_file(file_path)

    print(f"Trovati {len(raw_data)} alimenti. Trasformazione dati in corso...")

    records = []
    for item in raw_data:
        record = {
            "food_code": str(item.get("food_code")).zfill(6),
            "name": item.get("name"),
            "english_name": item.get("english_name"),
            "category": item.get("category"),
            "portion_g": parse_float(item.get("portion")),
            "edible_part": parse_float(item.get("edible_part")),
            "macros": clean_nutrients(item.get("macro_nutrients", [])),
            "minerals": clean_nutrients(item.get("minerals", [])),
            "vitamins": clean_nutrients(item.get("vitamins", [])),
            "fatty_acids": clean_nutrients(item.get("fatty_acids", []))
        }
        records.append(record)

    print(f"Inizio invio a Supabase (lotti da {batch_size} elementi)...")
    total_records = len(records)

    for i in range(0, total_records, batch_size):
        batch = records[i:i + batch_size]
        try:
            supabase.table("alimenti").upsert(batch, on_conflict="food_code").execute()
            print(f"Caricati {min(i + batch_size, total_records)} / {total_records} alimenti...")
        except Exception as e:
            print(f"Errore durante il caricamento del lotto {i}-{i + batch_size}: {e}")

    print("\nIngestione della tabella 'alimenti' completata con successo!")


if __name__ == "__main__":
    FILE_PATH = "sources/alimentazione/crea_food_composition_tables.json"
    process_and_upload_alimenti(FILE_PATH)