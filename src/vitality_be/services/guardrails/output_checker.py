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

# Pattern imperativi, condizionali o suggestivi su dosaggi farmacologici specifici
# Nota: esclude esplicitamente grammi/carboidrati/zuccheri per non bloccare la Regola del 15 ("15 grammi di zucchero")
_PHARMA_UNITS = r"(?:mg|milligramm[io]|mcg|microgramm[io]|ui\b|u\.i\.|unit[aà](?:\s+di\s+insulina)?|compress[ae]|pastigli[ae]|pillol[ae]|gocc[eia]|fial[ae])"
_PRESCRIPTION_PATTERNS = [
    rf"\b(?:ti\s+(?:consiglio|suggerisco|raccomando|invito)\s+(?:di|a)\s+(?:prendere|assumere|iniettar(?:ti|e)|somministrar(?:ti|e)|fare|provare(?:\s+con)?|aumentare|ridurre|dimezzare|sospendere))\s+(?:con\s+|a\s+|di\s+)?\d+(?:[.,]\d+)?\s*{_PHARMA_UNITS}\b",
    rf"\b(?:potresti|dovresti|puoi|conviene|sarebbe\s+meglio|meglio|prova\s+a|provare\s+a)\s+(?:prendere|assumere|iniettar(?:ti|e)|somministrar(?:ti|e)|fare|provare(?:\s+con)?|usare|aggiungere|aumentare(?:\s+a|\s+di)?|ridurre(?:\s+a|\s+di)?|portare(?:\s+a)?|passare\s+a)\s+(?:con\s+|a\s+|di\s+)?\d+(?:[.,]\d+)?\s*{_PHARMA_UNITS}\b",
    rf"\b(?:prendi|assumi|inietta(?:ti)?|somministra(?:ti)?|fai|aggiungi)\s+\d+(?:[.,]\d+)?\s*{_PHARMA_UNITS}\b",
    rf"\b(?:aumenta(?:re)?|riduci|ridurre|porta(?:re)?|modifica(?:re)?)\s+(?:la\s+dose|il\s+dosaggio|l['’]insulina|la\s+metformina|il\s+farmaco|la\s+terapia)\s+(?:a|di)\s+\d+(?:[.,]\d+)?\s*{_PHARMA_UNITS}?\b",
    rf"\b(?:passa(?:re)?\s+a|sali(?:re)?\s+a|scendi(?:re)?\s+a)\s+\d+(?:[.,]\d+)?\s*{_PHARMA_UNITS}\b",
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
