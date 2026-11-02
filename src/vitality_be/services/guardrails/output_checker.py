"""Guardrails Livello 3: Output Sanitization Guardrail (Post-Generazione).

Verifica la risposta generata dall'assistente prima dell'invio all'utente per proteggerlo
da allucinazioni su prescrizioni posologiche o diagnosi mediche assertive.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

PRESCRIPTION_DISCLAIMER = (
    "\n\n⚠️ **Nota di sicurezza clinica:** Ricorda che come coach virtuale non posso formulare "
    "diagnosi né prescrivere o modificare le dosi dei tuoi farmaci o dell'insulina. "
    "Per qualsiasi variazione terapeutica, confrontati sempre con il tuo medico curante o diabetologo."
)

# Pattern imperativi/prescrittivi su dosaggi farmacologici specifici
# Nota: esclude esplicitamente grammi/carboidrati/zuccheri per non bloccare la Regola del 15 ("15 grammi di zucchero")
_PRESCRIPTION_PATTERNS = [
    r"\b(?:ti\s+consiglio\s+di\s+(?:prendere|assumere|iniettare|aumentare|ridurre|dimezzare|sospendere))\s+\d+(?:[.,]\d+)?\s*(?:mg|milligrammi|ui\b|unit[aà]|compress[ae]|pastigli[ae]|gocc[eia])\b",
    r"\b(?:prendi|assumi|inietta(?:ti)?|fai)\s+\d+(?:[.,]\d+)?\s*(?:mg|milligrammi|ui\b|unit[aà]|compress[ae]|pastigli[ae]|gocc[eia])\b",
    r"\b(?:aumenta(?:re)?\s+(?:la\s+dose|l['’]insulina|la\s+metformina|il\s+farmaco)\s+(?:a|di))\s+\d+(?:[.,]\d+)?\s*(?:mg|milligrammi|ui\b|unit[aà]|compress[ae]|pastigli[ae]|gocc[eia])?\b",
    r"\b(?:riduci\s+(?:la\s+dose|l['’]insulina|la\s+metformina|il\s+farmaco)\s+(?:a|di))\s+\d+(?:[.,]\d+)?\s*(?:mg|milligrammi|ui\b|unit[aà]|compress[ae]|pastigli[ae]|gocc[eia])?\b",
    r"\b(?:passa\s+a)\s+\d+(?:[.,]\d+)?\s*(?:mg|milligrammi|ui\b|unit[aà]|compress[ae]|pastigli[ae])\b",
]
_PRESCRIPTION_RE = re.compile("|".join(_PRESCRIPTION_PATTERNS), re.IGNORECASE)

# Pattern di diagnosi medica assertiva
_DIAGNOSIS_PATTERNS = [
    r"\b(?:ti\s+diagnostico|la\s+mia\s+diagnosi\s+[eè]|posso\s+diagnosticarti)\b",
    r"\b(?:sei\s+sicuramente\s+affetto\s+da|hai\s+certamente\s+sviluppato\s+(?:una\s+)?(?:nefropatia|retinopatia|neuropatia|chetoacidosi|insufficienza\s+renale))\b",
]
_DIAGNOSIS_RE = re.compile("|".join(_DIAGNOSIS_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class OutputCheckResult:
    sanitized_text: str
    triggered: bool
    violation_type: Literal["prescription", "diagnosis"] | None = None


def check_and_sanitize_output(reply_text: str) -> OutputCheckResult:
    """Verifica e corregge l'output dell'assistente se contiene prescrizioni o diagnosi asserite."""
    if not reply_text:
        return OutputCheckResult(sanitized_text=reply_text, triggered=False)

    text = reply_text.strip()
    triggered = False
    v_type: Literal["prescription", "diagnosis"] | None = None

    # Check 1: Prescrizioni di dosaggi farmacologici
    if _PRESCRIPTION_RE.search(text):
        triggered = True
        v_type = "prescription"
        # Rimuoviamo o censuriamo il frammento imperativo vietato
        text = _PRESCRIPTION_RE.sub("[indicazione posologica non consentita]", text)

    # Check 2: Diagnosi assertive
    if _DIAGNOSIS_RE.search(text):
        triggered = True
        v_type = "diagnosis" if v_type is None else v_type
        text = _DIAGNOSIS_RE.sub("[valutazione diagnostica riservata al medico]", text)

    if triggered:
        # Appende il disclaimer solo se non già presente nel testo
        if "non posso formulare diagnosi" not in text and "Nota di sicurezza clinica" not in text:
            text = text + PRESCRIPTION_DISCLAIMER

    return OutputCheckResult(
        sanitized_text=text,
        triggered=triggered,
        violation_type=v_type,
    )
