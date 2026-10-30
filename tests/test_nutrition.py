import pytest
from pydantic import ValidationError

from vitality_be.llm.base import LLMUnavailable
from vitality_be.schemas.nutrition import (
    LLMExtractedIngredient,
    LLMMealDecomposition,
    MealEstimateRequest,
    MealEstimateResponse,
    MealTypeEnum,
    RecognizedFoodItem,
)
from vitality_be.services.nutrition import (
    _parse_crea_float,
    estimate_meal_nutrition,
    extract_search_tokens,
    scrub_free_text,
)


def test_parse_crea_float():
    assert _parse_crea_float(80) == 80.0
    assert _parse_crea_float("80.00") == 80.0
    assert _parse_crea_float("353") == 353.0
    assert _parse_crea_float("tr") == 0.0
    assert _parse_crea_float("tracce") == 0.0
    assert _parse_crea_float(None, default=10.0) == 10.0
    assert _parse_crea_float("invalid", default=5.0) == 5.0


def test_extract_search_tokens():
    tokens = extract_search_tokens("80g di pasta al pomodoro con parmigiano e una mela")
    assert "pasta" in tokens
    assert "pomodoro" in tokens
    assert "parmigiano" in tokens
    assert "mela" in tokens
    assert "con" not in tokens
    assert "una" not in tokens
    assert "80g" not in tokens


def test_scrub_free_text_gdpr():
    text = "Ho mangiato con Mario Rossi chiamami al 3331234567 o mario@email.it codice RSSMRA85M01H501Z"
    cleaned = scrub_free_text(text)
    assert "3331234567" not in cleaned
    assert "mario@email.it" not in cleaned
    assert "RSSMRA85M01H501Z" not in cleaned
    assert "[TELEFONO_RIMOSSO]" in cleaned
    assert "[EMAIL_RIMOSSA]" in cleaned
    assert "[CF_RIMOSSO]" in cleaned


def test_pydantic_request_validation():
    req = MealEstimateRequest(description="Pasta al sugo", meal_type=MealTypeEnum.PRANZO)
    assert req.description == "Pasta al sugo"
    assert req.meal_type == MealTypeEnum.PRANZO

    # Descrizione troppo corta
    with pytest.raises(ValidationError):
        MealEstimateRequest(description="a", meal_type=MealTypeEnum.PRANZO)

    # Tipo pasto non valido
    with pytest.raises(ValidationError):
        MealEstimateRequest(description="Pasta al sugo", meal_type="INVALID_TYPE")  # type: ignore[arg-type]


class NutritionTestLLM:
    """Mock LLM con supporto a generate_structured."""

    def __init__(self, decomposition: LLMMealDecomposition | None = None, exc: Exception | None = None):
        self.decomposition = decomposition
        self.exc = exc

    async def generate_structured(self, messages, *, system=None, response_schema=None, **kwargs):
        if self.exc:
            raise self.exc
        return self.decomposition or response_schema()


class NutritionTestUserDb:
    """Mock DB che restituisce record CREA per public.foods o public.alimenti."""

    def __init__(self, records: list[dict]):
        self.records = records

    def rpc(self, fn: str, params: dict = None):
        class Q:
            def __init__(self, data):
                self.data = data
            def execute(self):
                class R:
                    pass
                r = R()
                r.data = self.data
                return r
        return Q(self.records)

    def table(self, tbl: str):
        class Q:
            def __init__(self, data):
                self.data = data
            def select(self, *a, **k): return self
            def ilike(self, *a, **k): return self
            def limit(self, *a, **k): return self
            def execute(self):
                class R:
                    pass
                r = R()
                r.data = self.data
                return r
        return Q(self.records)


def _run_coroutine(coro):
    """Esegue una coroutine sincrona nei test senza dipendere da socket di loopback su Windows."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise RuntimeError("Coroutine did not complete synchronously")


def test_nutrition_crea_exact_calculation():
    """Verifica che un alimento abbinato a CREA calcoli calorie e carboidrati in modo deterministico."""
    crea_record = {
        "food_code": "000020",
        "name": "Farro perlato, crudo",
        "english_name": "Spelt, grain",
        "category": "Cereali e derivati",
        "portion_g": "80.00",
        "edible_part": "1.00",
        "macros": {
            "energy_kcal": 353,
            "available_carbohydrates": 69.3,
            "proteins": 14.6,
            "lipids": 2.4,
        },
    }
    db = NutritionTestUserDb([crea_record])

    llm_resp = LLMMealDecomposition(
        items=[
            LLMExtractedIngredient(
                ingredient_name="Farro perlato crudo",
                quantity_g=80.0,
                matched_food_code="000020",
            )
        ],
        confidence="HIGH",
        clinical_note="Ottima scelta di cereale a basso indice glicemico.",
    )
    llm = NutritionTestLLM(decomposition=llm_resp)

    req = MealEstimateRequest(description="80g di farro perlato", meal_type=MealTypeEnum.PRANZO)
    res = _run_coroutine(estimate_meal_nutrition(req, user_db=db, llm=llm))

    # 353 * 80 / 100 = 282.4 -> round 282
    assert res.total_calories == 282
    # 69.3 * 80 / 100 = 55.44 -> round 55.4 -> round totale 55
    assert res.total_carbs_grams == 55
    assert res.confidence == "HIGH"
    assert len(res.items) == 1
    assert res.items[0].source == "CREA"
    assert res.items[0].calories == 282
    assert res.items[0].carbs_grams == 55.4


def test_nutrition_fallback_estimate_when_no_match():
    """Verifica il calcolo quando un cibo non matcha CREA e usa fallback AI."""
    db = NutritionTestUserDb([])
    llm_resp = LLMMealDecomposition(
        items=[
            LLMExtractedIngredient(
                ingredient_name="Piatto esotico sconosciuto",
                quantity_g=200.0,
                matched_food_code=None,
                fallback_kcal_per_100g=150.0,
                fallback_carbs_per_100g=20.0,
            )
        ],
        confidence="MEDIUM",
    )
    llm = NutritionTestLLM(decomposition=llm_resp)

    req = MealEstimateRequest(description="200g di piatto esotico", meal_type=MealTypeEnum.CENA)
    res = _run_coroutine(estimate_meal_nutrition(req, user_db=db, llm=llm))

    assert res.total_calories == 300  # 150 * 200 / 100
    assert res.total_carbs_grams == 40  # 20 * 200 / 100
    assert res.items[0].source == "AI_ESTIMATE"


def test_nutrition_resilience_on_llm_error():
    """Verifica che in caso di errore LLM non ci sia eccezione ma fallback resiliente."""
    db = NutritionTestUserDb([])
    llm = NutritionTestLLM(exc=LLMUnavailable("Timeout simulato"))

    req = MealEstimateRequest(description="Pranzo normale", meal_type=MealTypeEnum.PRANZO)
    res = _run_coroutine(estimate_meal_nutrition(req, user_db=db, llm=llm))

    assert res.confidence == "LOW"
    assert res.total_calories > 0
    assert res.total_carbs_grams > 0

