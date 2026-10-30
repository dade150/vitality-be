from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class MealTypeEnum(str, Enum):
    COLAZIONE = "COLAZIONE"
    SPUNTINO_MATTINA = "SPUNTINO_MATTINA"
    PRANZO = "PRANZO"
    SPUNTINO_POMERIGGIO = "SPUNTINO_POMERIGGIO"
    CENA = "CENA"
    FUORI_PASTO = "FUORI_PASTO"
    SPUNTINO = "SPUNTINO"


class MealEstimateRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="Descrizione in testo libero del pasto",
    )
    meal_type: MealTypeEnum = Field(..., description="Tipologia di pasto")


class LLMExtractedIngredient(BaseModel):
    ingredient_name: str = Field(..., description="Nome pulito dell'alimento in italiano")
    quantity_g: float = Field(..., gt=0, le=5000, description="Grammi edibili consumati (stimati o espliciti)")
    matched_food_code: Optional[str] = Field(
        None,
        description="food_code esatto scelto dalla lista candidati CREA fornita, oppure null se nessun candidato corrisponde",
    )
    fallback_kcal_per_100g: Optional[float] = Field(
        None,
        ge=0,
        le=950,
        description="Kcal per 100g stimate solo se matched_food_code è null",
    )
    fallback_carbs_per_100g: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Carboidrati disponibili (g) per 100g stimati solo se matched_food_code è null",
    )


class LLMMealDecomposition(BaseModel):
    """Root schema con default sicuri per compatibilità con EchoProvider nei test."""

    items: list[LLMExtractedIngredient] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    clinical_note: Optional[str] = Field(
        None,
        description="Breve nota educativa opzionale per T2D (max 140 caratteri, nessun consiglio medico/farmacologico)",
    )


class RecognizedFoodItem(BaseModel):
    food_code: Optional[str] = Field(None, description="Codice CREA se abbinato")
    name: str = Field(..., description="Nome alimento riconosciuto (ufficiale CREA o estratto)")
    quantity_g: float = Field(..., description="Grammi considerati nel calcolo")
    calories: int = Field(..., ge=0, description="Kcal calcolate per questa porzione")
    carbs_grams: float = Field(..., ge=0, description="Carboidrati disponibili in grammi")
    source: Literal["CREA", "AI_ESTIMATE"] = Field(
        "CREA", description="'CREA' se calcolato da tabella public.foods, 'AI_ESTIMATE' altrimenti"
    )


class MealEstimateResponse(BaseModel):
    total_calories: int = Field(..., ge=0, le=10000, description="Calorie complessive arrotondate")
    total_carbs_grams: int = Field(..., ge=0, le=2000, description="Carboidrati complessivi arrotondati in grammi")
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(..., description="Livello di confidenza della stima")
    items: list[RecognizedFoodItem] = Field(default_factory=list)
    notes: Optional[str] = Field(None, description="Eventuale nota educativa per la gestione glicemica")
