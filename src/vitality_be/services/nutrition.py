"""Servizio di Stima Nutrizionale Pasti basato su Tabelle Ufficiali CREA e Gemini.

Architettura Ibrida Deterministica:
1. Scrubbing PII preventivo della descrizione pasto (GDPR Art. 5, 25).
2. Estrazione token e retrieval candidati su Supabase `public.foods` (ex `public.alimenti`).
3. Decomposizione e abbinamento `food_code` tramite Structured Output di Gemini.
4. Calcolo matematico deterministico di calorie e carboidrati in Python a partire
   dai valori nutrizionali per 100g della tabella CREA.
5. Fallback resiliente in caso di indisponibilità del provider LLM o con EchoProvider.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger
from supabase import Client

from vitality_be.llm.base import ChatMessage, LLMError, LLMProvider, LLMUnavailable
from vitality_be.schemas.nutrition import (
    LLMExtractedIngredient,
    LLMMealDecomposition,
    MealEstimateRequest,
    MealEstimateResponse,
    MealTypeEnum,
    RecognizedFoodItem,
)
# Regex difensive per rimuovere PII accidentali nel testo libero del pasto (GDPR Art. 5, 25)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CF_RE = re.compile(
    r"\b[A-Z]{6}[0-9LMNPQRSTUV]{2}[A-EHLMPR-T][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]\b",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(
    r"(?:(?:\+|00)39[\s.-]?)?(?:3\d{2}|0\d{1,3})[\s.-]?\d{3}[\s.-]?\d{3,4}\b"
)
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
_HEALTH_CARD_RE = re.compile(r"\b(?:8038000\d{13}|\d{16,20})\b")
_ADDRESS_RE = re.compile(
    r"\b(?:via|viale|piazza|corso|largo|vicolo)\s+[A-ZÀ-Úa-zà-ú']+(?:\s+[A-ZÀ-Úa-zà-ú']+){0,3}"
    r"(?:\s*,\s*|\s+(?:n\.?|civico\s+)?)\d{1,4}[A-Za-z]?\b",
    re.IGNORECASE,
)


def scrub_free_text(text: str | None) -> str:
    """Rimuove PII sensibili prima dell'invio del testo a Vertex AI."""
    if not text:
        return ""
    cleaned = str(text)
    cleaned = _EMAIL_RE.sub("[EMAIL_RIMOSSA]", cleaned)
    cleaned = _CF_RE.sub("[CF_RIMOSSO]", cleaned)
    cleaned = _IBAN_RE.sub("[IBAN_RIMOSSO]", cleaned)
    cleaned = _HEALTH_CARD_RE.sub("[TESSERA_SANITARIA_RIMOSSA]", cleaned)
    cleaned = _PHONE_RE.sub("[TELEFONO_RIMOSSO]", cleaned)
    cleaned = _ADDRESS_RE.sub("[INDIRIZZO_RIMOSSO]", cleaned)
    return cleaned

_STOPWORDS_IT = {
    "a", "ad", "al", "all", "alla", "alle", "alli", "allo", "agli", "ai",
    "da", "dal", "dalla", "dalle", "dallo", "dagli", "dai",
    "di", "del", "della", "delle", "dello", "degli", "dei", "d",
    "in", "nel", "nella", "nelle", "nello", "negli", "nei",
    "su", "sul", "sulla", "sulle", "sullo", "sugli", "sui",
    "con", "col", "colla", "coi", "per", "tra", "fra",
    "il", "lo", "la", "i", "gli", "le", "l",
    "un", "uno", "una", "un'",
    "e", "ed", "o", "od", "ma", "anche", "pure", "piu", "più", "meno",
    "senza", "circa", "quasi", "poco", "molto", "tanto",
    "g", "gr", "grammi", "etto", "etti", "hg", "kg", "chilo", "chili",
    "ml", "cl", "dl", "l", "litro", "litri", "cc",
    "cucchiaio", "cucchiai", "cucchiaino", "cucchiaini",
    "fetta", "fette", "fettina", "fettine", "pezzo", "pezzi",
    "piatto", "piatti", "porzione", "porzioni", "tazza", "tazze", "bicchiere", "bicchieri",
    "ho", "hai", "ha", "abbiamo", "mangiato", "bevuto", "preso", "consumato",
    "pranzo", "cena", "colazione", "spuntino", "merenda",
}

# Valori nutrizionali tipici di fallback per tipologia di pasto nel T2D
_MEAL_TYPE_DEFAULTS: dict[MealTypeEnum, tuple[int, int]] = {
    MealTypeEnum.COLAZIONE: (350, 45),
    MealTypeEnum.SPUNTINO_MATTINA: (150, 20),
    MealTypeEnum.PRANZO: (650, 75),
    MealTypeEnum.SPUNTINO_POMERIGGIO: (180, 25),
    MealTypeEnum.CENA: (550, 60),
    MealTypeEnum.FUORI_PASTO: (200, 25),
    MealTypeEnum.SPUNTINO: (160, 22),
}


def _parse_crea_float(val: Any, default: float = 0.0) -> float:
    """Estrae un float resiliente dai dati CREA (gestisce '80.00', 'tr', tracce, None)."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", ".")
    if not s or s.lower() in ("tr", "tracce", "nd", "-", "null", "none"):
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def extract_search_tokens(text: str) -> list[str]:
    """Estrae i token alfabetici significativi per la ricerca nel database degli alimenti."""
    cleaned = re.sub(r"[^a-zA-ZàèéìòùÀÈÉÌÒÙ\s]", " ", text)
    words = [w.lower().strip() for w in cleaned.split() if len(w.strip()) >= 3]
    filtered = [w for w in words if w not in _STOPWORDS_IT]
    # Rimuove duplicati preservando l'ordine
    seen = set()
    deduped = []
    for w in filtered:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return deduped


async def fetch_food_candidates(user_db: Client, tokens: list[str]) -> list[dict[str, Any]]:
    """Cerca alimenti candidati nella tabella `foods` (o `alimenti`) su Supabase."""
    if not tokens:
        return []

    candidates_by_code: dict[str, dict[str, Any]] = {}

    # 1. Tentativo tramite RPC dedicata search_foods per i primi token
    if hasattr(user_db, "rpc"):
        for token in tokens[:4]:
            try:
                query = user_db.rpc("search_foods", {"query_text": token, "match_limit": 10})
                resp = await query if asyncio.iscoroutine(query) else (query.execute() if hasattr(query, "execute") else query)
                rows = getattr(resp, "data", None) if resp else []
                if isinstance(rows, list):
                    for r in rows:
                        code = str(r.get("food_code") or "").strip()
                        if code and code not in candidates_by_code:
                            candidates_by_code[code] = r
            except Exception:
                # Se la RPC non è installata o fallisce, passiamo al fallback query diretta
                break

    # 2. Se la RPC non ha restituito candidati, esegue query diretta con ILIKE
    if not candidates_by_code and hasattr(user_db, "table"):
        table_name = "foods"
        cols = "food_code, name, english_name, category, portion_g, edible_part, macros"
        for token in tokens[:5]:
            try:
                query = user_db.table(table_name).select(cols).ilike("name", f"%{token}%").limit(10)
                resp = await query if asyncio.iscoroutine(query) else (query.execute() if hasattr(query, "execute") else query)
                rows = getattr(resp, "data", None) if resp else []
                if isinstance(rows, list):
                    for r in rows:
                        code = str(r.get("food_code") or "").strip()
                        if code and code not in candidates_by_code:
                            candidates_by_code[code] = r
            except Exception as exc:
                if table_name == "foods":
                    table_name = "alimenti"
                    try:
                        query = user_db.table("alimenti").select(cols).ilike("name", f"%{token}%").limit(10)
                        resp = await query if asyncio.iscoroutine(query) else (query.execute() if hasattr(query, "execute") else query)
                        rows = getattr(resp, "data", None) if resp else []
                        if isinstance(rows, list):
                            for r in rows:
                                code = str(r.get("food_code") or "").strip()
                                if code and code not in candidates_by_code:
                                    candidates_by_code[code] = r
                    except Exception:
                        pass
                else:
                    logger.debug("Query diretta alimenti fallita per token={}: {}", token, type(exc).__name__)

    return list(candidates_by_code.values())[:30]


def _build_nutrition_prompt(
    description: str,
    meal_type: MealTypeEnum,
    candidates: list[dict[str, Any]],
) -> str:
    """Costruisce il prompt istruito per Gemini Structured Output."""
    candidate_lines = []
    for c in candidates:
        code = c.get("food_code", "")
        name = c.get("name", "")
        portion = c.get("portion_g") or "80"
        macros = c.get("macros") or {}
        kcal = macros.get("energy_kcal", "n.d.")
        carbs = macros.get("available_carbohydrates", "n.d.")
        candidate_lines.append(f"- food_code: '{code}' | '{name}' | porzione: {portion}g | per 100g: {kcal} kcal, {carbs}g carboidrati")

    candidates_block = "\n".join(candidate_lines) if candidate_lines else "Nessun alimento specifico trovato in tabella."

    return (
        f"Sei un assistente nutrizionista clinico esperto in Diabete di Tipo 2.\n"
        f"Analizza la descrizione del pasto ({meal_type.value}) e scomponilo nei singoli ingredienti o cibi consumati.\n\n"
        f"REGOLE RIGOROSE:\n"
        f"1. Identifica ogni singolo cibo o condimento menzionato (es. pasta, olio, formaggio, frutta).\n"
        f"2. Se l'utente specifica la quantità esatta (es. '80g', '200g', '1 etto'), imposta quantity_g su tale valore.\n"
        f"3. Se l'utente usa misure casalinghe, convertile in grammi realistici: 1 cucchiaio olio = 10g, 1 cucchiaio parmigiano = 10g, 1 cucchiaino zucchero = 5g, 1 fetta pane = 50g, 1 mela/arancia media = 150g, 1 caffè = 30g.\n"
        f"4. Se la quantità non è specificata, usa la grammatura della porzione standard CREA (portion_g) o una porzione standard LARN coerente per {meal_type.value}.\n"
        f"5. Se un ingrediente corrisponde a uno dei candidati della lista CREA fornita sotto, seleziona OBBLIGATORIAMENTE il suo matched_food_code esatto.\n"
        f"   - Se l'utente scrive '80g di pasta/riso/farro' senza dire 'cotto', intende peso a crudo (scegli il record 'crudo'). Se scrive 'cotto' o 'un piatto cotto', scegli il record 'cotto'.\n"
        f"6. Se nessun candidato CREA corrisponde, lascia matched_food_code = null e fornisci fallback_kcal_per_100g e fallback_carbs_per_100g stimati in modo attendibile.\n"
        f"7. La clinical_note (se presente) deve essere brevissima (max 140 caratteri) e puramente educativa (es. impatto su indice glicemico o combinazione con fibre), MAI prescrittiva o farmacologica.\n\n"
        f"<candidati_crea>\n{candidates_block}\n</candidati_crea>\n\n"
        f"<pasto_utente>\n{description}\n</pasto_utente>"
    )


async def estimate_meal_nutrition(
    request: MealEstimateRequest,
    *,
    user_db: Client,
    llm: LLMProvider,
) -> MealEstimateResponse:
    """Esegue la stima nutrizionale del pasto garantendo robustezza, GDPR e accuratezza matematica."""
    # 1. Scrubbing preventivo PII (GDPR)
    clean_desc = scrub_free_text(request.description).strip()
    if not clean_desc:
        clean_desc = request.description.strip()

    # 2. Retrieval candidati su Supabase foods
    tokens = extract_search_tokens(clean_desc)
    candidates = await fetch_food_candidates(user_db, tokens)
    candidate_map = {str(c.get("food_code")): c for c in candidates if c.get("food_code")}

    # 3. Invoca Gemini con Structured Output se supportato
    decomposition: LLMMealDecomposition | None = None
    if hasattr(llm, "generate_structured"):
        prompt = _build_nutrition_prompt(clean_desc, request.meal_type, candidates)
        try:
            decomposition = await llm.generate_structured(
                [ChatMessage(role="user", content=prompt)],
                response_schema=LLMMealDecomposition,
            )
        except (LLMUnavailable, LLMError, TimeoutError, Exception) as exc:
            logger.warning("generate_structured fallito o degradato exc={}", type(exc).__name__)
            decomposition = None

    recognized_items: list[RecognizedFoodItem] = []

    # 4. Calcolo matematico deterministico se abbiamo item da Gemini
    if decomposition and decomposition.items:
        for it in decomposition.items:
            q_g = max(1.0, min(5000.0, float(it.quantity_g)))
            matched_code = str(it.matched_food_code).strip() if it.matched_food_code else None
            crea_record = candidate_map.get(matched_code) if matched_code else None

            if crea_record is not None:
                macros = crea_record.get("macros") or {}
                kcal_100g = _parse_crea_float(macros.get("energy_kcal"), default=0.0)
                carbs_100g = _parse_crea_float(macros.get("available_carbohydrates"), default=0.0)

                item_kcal = round((kcal_100g * q_g) / 100.0)
                item_carbs = round((carbs_100g * q_g) / 100.0, 1)
                item_name = str(crea_record.get("name") or it.ingredient_name).strip()

                recognized_items.append(
                    RecognizedFoodItem(
                        food_code=matched_code,
                        name=item_name,
                        quantity_g=q_g,
                        calories=item_kcal,
                        carbs_grams=item_carbs,
                        source="CREA",
                    )
                )
            else:
                fallback_kcal = it.fallback_kcal_per_100g if it.fallback_kcal_per_100g is not None else 120.0
                fallback_carbs = it.fallback_carbs_per_100g if it.fallback_carbs_per_100g is not None else 15.0

                item_kcal = round((fallback_kcal * q_g) / 100.0)
                item_carbs = round((fallback_carbs * q_g) / 100.0, 1)

                recognized_items.append(
                    RecognizedFoodItem(
                        food_code=None,
                        name=it.ingredient_name.strip(),
                        quantity_g=q_g,
                        calories=item_kcal,
                        carbs_grams=item_carbs,
                        source="AI_ESTIMATE",
                    )
                )

    # 5. Fallback Deterministico se non ci sono elementi (es. EchoProvider nei test o LLM offline)
    if not recognized_items:
        # Se abbiamo candidati CREA estratti dai token, creiamo una proposta dal migliore candidato
        if candidates:
            best_c = candidates[0]
            portion = _parse_crea_float(best_c.get("portion_g"), default=80.0)
            macros = best_c.get("macros") or {}
            kcal_100g = _parse_crea_float(macros.get("energy_kcal"), default=150.0)
            carbs_100g = _parse_crea_float(macros.get("available_carbohydrates"), default=20.0)

            item_kcal = round((kcal_100g * portion) / 100.0)
            item_carbs = round((carbs_100g * portion) / 100.0, 1)

            recognized_items.append(
                RecognizedFoodItem(
                    food_code=best_c.get("food_code"),
                    name=best_c.get("name") or clean_desc,
                    quantity_g=portion,
                    calories=item_kcal,
                    carbs_grams=item_carbs,
                    source="CREA",
                )
            )
            confidence = "MEDIUM"
            note = f"Valori calcolati sulla porzione standard CREA ({int(portion)} g) di {best_c.get('name')}."
        else:
            def_kcal, def_carbs = _MEAL_TYPE_DEFAULTS.get(request.meal_type, (400, 50))
            recognized_items.append(
                RecognizedFoodItem(
                    food_code=None,
                    name=clean_desc[:50] or request.meal_type.value.capitalize(),
                    quantity_g=150.0,
                    calories=def_kcal,
                    carbs_grams=float(def_carbs),
                    source="AI_ESTIMATE",
                )
            )
            confidence = "LOW"
            note = "Stima basata sui fabbisogni medi per questa tipologia di pasto. Puoi modificare i valori liberamente."
    else:
        # Se tutti gli item provengono da CREA, la confidenza è HIGH, altrimenti MEDIUM
        all_crea = all(it.source == "CREA" for it in recognized_items)
        confidence = "HIGH" if all_crea else (decomposition.confidence if decomposition else "MEDIUM")
        note = decomposition.clinical_note if (decomposition and decomposition.clinical_note) else None

    # Calcolo totali complessivi
    total_kcal = min(10000, max(0, sum(it.calories for it in recognized_items)))
    total_carbs = min(2000, max(0, round(sum(it.carbs_grams for it in recognized_items))))

    return MealEstimateResponse(
        total_calories=total_kcal,
        total_carbs_grams=total_carbs,
        confidence=confidence,
        items=recognized_items,
        notes=note,
    )
