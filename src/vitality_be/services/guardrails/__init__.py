"""Package Guardrails Clinici e Conversazionali per diaLog2.

Architettura Defense-in-Depth a 3 Livelli:
- Livello 1: Regole e pattern deterministici per emergenze e jailbreak (rules.py)
- Livello 2: Classificatore semantico leggero In-Topic / Off-Topic (classifier.py)
- Livello 3: Validatore e sanitizzatore post-generazione (output_checker.py)
"""
from typing import Literal

from vitality_be.services.guardrails.classifier import (
    GuardrailClassification,
    classify_user_input,
)
from vitality_be.services.guardrails.output_checker import (
    PRESCRIPTION_DISCLAIMER,
    OutputCheckResult,
    check_and_sanitize_output,
)
from vitality_be.services.guardrails.rules import (
    EMERGENCY_REPLY_GENERAL,
    EMERGENCY_REPLY_HYPO,
    JAILBREAK_REPLY,
    OFF_TOPIC_REPLY,
    EmergencyMatch,
    check_emergency_input,
    check_jailbreak_input,
)

GuardrailCategory = Literal[
    "in_topic",
    "off_topic",
    "emergency",
    "jailbreak_blocked",
    "prescription_redirect",
]

__all__ = [
    "GuardrailCategory",
    "EmergencyMatch",
    "GuardrailClassification",
    "OutputCheckResult",
    "check_emergency_input",
    "check_jailbreak_input",
    "classify_user_input",
    "check_and_sanitize_output",
    "EMERGENCY_REPLY_HYPO",
    "EMERGENCY_REPLY_GENERAL",
    "JAILBREAK_REPLY",
    "OFF_TOPIC_REPLY",
    "PRESCRIPTION_DISCLAIMER",
]
