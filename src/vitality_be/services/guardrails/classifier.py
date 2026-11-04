"""Guardrails Livello 2: Classificatore Semantico (Gemini Flash con Structured Outputs).

Determina pertinenza tematica (In-Topic vs Off-Topic), rischio emergenze sfumate,
richieste di prescrizione farmaci e indicazione se l'utente sta fornendo dati da salvare nel diario.
"""
from __future__ import annotations

import re
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from vitality_be.llm.base import ChatMessage, LLMProvider
from vitality_be.llm.prompts import get_guardrails_system_prompt

_TRIVIAL_GREETING_RE = re.compile(
    r"^\s*(?:ciao|buongiorno|buonasera|salve|hey|ehi|grazie(?:\s+mille)?|ok(?:\s*,\s*va\s+bene|\s+va\s+bene)?|va\s+bene|perfetto|capito|a\s+presto)[!.,?\s]*$",
    re.IGNORECASE,
)

_GENERIC_ATTACHMENT_CAPTIONS_RE = re.compile(
    r"^\s*(?:ecco\s+(?:la\s+foto|il\s+referto|il\s+pasto|lo\s+scontrino|il\s+valore|l['’]immagine|l['’]allegato)|"
    r"foto(?:\s+pasto|\s+referto|\s+glucometro)?|referto|pasto|analizza(?:\s+l['’]allegato|\s+la\s+foto|\s+il\s+referto)?|"
    r"allegat[oa]|guarda(?:\s+quest[ao])?|questo\s+[eè]\s+il\s+(?:referto|pasto)|"
    r"ti\s+(?:allego|mando)\s+(?:la\s+foto|il\s+referto|il\s+pasto))[!.,?\s]*$",
    re.IGNORECASE,
)


class GuardrailClassification(BaseModel):
    in_topic: bool = Field(
        default=True,
        description=(
            "True se il messaggio riguarda salute, diabete, alimentazione, attività fisica, "
            "terapie, benessere emotivo, esenzioni/patente o saluti/conversazione con il coach. "
            "False se riguarda politica, calcio/sport spettacolo, programmazione, compiti scolastici, "
            "finanza, gossip o temi estranei."
        ),
    )
    intent: Literal[
        "nutrition",
        "vitals",
        "therapy",
        "activity",
        "motivation",
        "other_clinical",
        "off_topic",
    ] = Field(
        default="other_clinical",
        description="Categoria tematica prevalente del messaggio.",
    )
    is_medical_prescription_request: bool = Field(
        default=False,
        description=(
            "True se l'utente chiede esplicitamente di prescrivere un farmaco, dosaggi numerici correttivi "
            "di insulina ('quante unità devo fare?') o variazioni/sospensioni posologiche della terapia."
        ),
    )
    emergency_risk: bool = Field(
        default=False,
        description=(
            "True se l'utente descrive sintomi fisici acuti potenzialmente pericolosi per la vita "
            "(dolore/oppressione toracica, dispnea acuta, sincope/svenimento, paralisi, glicemia < 54 mg/dL)."
        ),
    )
    has_loggable_data: bool = Field(
        default=False,
        description=(
            "True SOLO se l'utente sta comunicando un proprio dato reale da registrare nel diario "
            "(glicemia, pressione, pasto consumato, attività svolta, peso, farmaco assunto, appuntamento)."
        ),
    )


async def classify_user_input(
    llm: LLMProvider,
    user_message: str,
    *,
    has_attachment: bool = False,
) -> GuardrailClassification:
    """Classifica l'input dell'utente con Fast-Path a 0 ms per allegati e saluti,

    oppure tramite chiamata strutturata a Gemini Flash.
    """
    clean_msg = (user_message or "").strip()

    # Fast-Path 1: Se c'è un allegato senza testo o con didascalia generica/vuota, fast-path a 0 ms
    if has_attachment and (not clean_msg or _GENERIC_ATTACHMENT_CAPTIONS_RE.match(clean_msg)):
        return GuardrailClassification(
            in_topic=True,
            intent="vitals",
            is_medical_prescription_request=False,
            emergency_risk=False,
            has_loggable_data=True,
        )

    # Fast-Path 2: Saluto o ringraziamento breve a zero latenza (0 ms, zero token)
    if not clean_msg or _TRIVIAL_GREETING_RE.match(clean_msg):
        return GuardrailClassification(
            in_topic=True,
            intent="motivation",
            is_medical_prescription_request=False,
            emergency_risk=False,
            has_loggable_data=has_attachment,
        )

    # Se il provider non supporta Structured Outputs o siamo in test unitario con EchoProvider
    if not hasattr(llm, "generate_structured"):
        return GuardrailClassification(
            in_topic=True,
            intent="other_clinical",
            is_medical_prescription_request=False,
            emergency_risk=False,
            has_loggable_data=True,
        )

    # Chiamata semantica a Gemini Flash (temp 0.0, prompt conciso, token adeguati anche per thinking)
    try:
        try:
            classification = await llm.generate_structured(
                [ChatMessage(role="user", content=clean_msg)],
                system=get_guardrails_system_prompt(),
                response_schema=GuardrailClassification,
                max_output_tokens=512,
            )
        except TypeError:
            classification = await llm.generate_structured(
                [ChatMessage(role="user", content=clean_msg)],
                system=get_guardrails_system_prompt(),
                response_schema=GuardrailClassification,
            )
        if not isinstance(classification, GuardrailClassification):
            return GuardrailClassification(
                in_topic=True,
                intent="other_clinical",
                is_medical_prescription_request=False,
                emergency_risk=False,
                has_loggable_data=True,
            )
        if has_attachment:
            classification.has_loggable_data = True
        return classification
    except Exception as exc:
        logger.warning("classificatore semantico guardrail fallito (degradazione graziosa) exc={} detail={}", type(exc).__name__, exc)
        # Degradazione graziosa sicura: non blocca la conversazione per un errore transitorio di rete
        return GuardrailClassification(
            in_topic=True,
            intent="other_clinical",
            is_medical_prescription_request=False,
            emergency_risk=False,
            has_loggable_data=True,
        )
