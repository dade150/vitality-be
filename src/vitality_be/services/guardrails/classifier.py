"""Guardrails Livello 2: Classificatore Semantico (Gemini Flash con Structured Outputs).

Determina pertinenza tematica (In-Topic vs Off-Topic), rischio emergenze sfumate,
richieste di prescrizione farmaci e indicazione se l'utente sta fornendo dati da salvare nel diario.
"""
from __future__ import annotations

from collections.abc import Sequence
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
    r"^\s*(?:ecco\s+(?:la\s+foto(?:\s+del\s+mio\s+(?:pasto|farmaco))?|la\s+lettura\s+del\s+mio\s+glucometro|"
    r"il\s+(?:mio\s+)?referto(?:\s+medico)?|il\s+documento/referto\s+allegato|"
    r"il\s+pasto|lo\s+scontrino|il\s+valore|l['’]immagine(?:\s+allegata)?|l['’]allegato)"
    r"(?:\.\s*puoi\s+analizzarl[oa](?:\s+dal\s+punto\s+di\s+vista\s+nutrizionale)?)?|"
    r"foto(?:\s+pasto|\s+referto|\s+glucometro)?|referto|pasto|analizza(?:\s+l['’]allegato(?:\s+fornito)?|\s+la\s+foto|\s+il\s+referto)?|"
    r"allegat[oa]|guarda(?:\s+quest[ao])?|questo\s+[eè]\s+il\s+(?:referto|pasto)|"
    r"ti\s+(?:allego|mando)\s+(?:la\s+foto|il\s+referto|il\s+pasto))[!.,?\s]*$",
    re.IGNORECASE,
)

_PDF_ACTION_VERB_RE = re.compile(
    r"\b(?:genera(?:mi|re)?|crea(?:mi|re)?|prepara(?:mi|re)?|esporta(?:mi|re)?|scarica(?:mi|re)?|"
    r"fammi|produci|stampa(?:mi|re)?|vorrei(?:\s+(?:generare|creare|preparare|esportare|scaricare|avere|il))?|"
    r"dammi|puoi\s+(?:generare|creare|preparare|esportare|fare|mandarmi|darmi)|"
    r"mi\s+(?:generi|crei|prepari|esporti|fai|mandi))\b",
    re.IGNORECASE,
)
_PDF_REPORT_NOUN_RE = re.compile(
    r"\b(?:"
    r"pdf\b|"
    r"(?:report|riepilogo|resoconto|documento)(?:\s+(?:medico|clinico|riepilogativo))?\s+"
    r"(?:in\s+pdf|pdf|per\s+il\s+(?:medico|dottore|diabetologo)|per\s+la\s+visita|da\s+portare\s+(?:al\s+medico|al\s+dottore|alla\s+visita))|"
    r"(?:report|riepilogo|resoconto)\s+(?:medico|clinico)\b"
    r")",
    re.IGNORECASE,
)
_PDF_EXCLUDE_CONTEXT_RE = re.compile(
    r"\b(?:allego|allegato|carico|caricato|ho\s+caricato|ti\s+mando\s+(?:il|la|quest[ao])|"
    r"come\s+(?:funziona|si\s+legge|leggo|leggere)|cosa\s+contiene|a\s+cosa\s+serve|"
    r"non\s+(?:voglio|vorrei|devi|serve|generare|creare|esportare|scaricare|preparare|fare))\b",
    re.IGNORECASE,
)


def is_explicit_pdf_summary_request(text: str | None) -> bool:
    """Riconosce richieste esplicite di generazione/esportazione del report clinico PDF per il medico."""
    clean = (text or "").strip()
    if not clean:
        return False
    if _PDF_EXCLUDE_CONTEXT_RE.search(clean):
        return False
    return bool(_PDF_ACTION_VERB_RE.search(clean) and _PDF_REPORT_NOUN_RE.search(clean))


def extract_requested_pdf_months(text: str | None) -> int:
    """Estrae il periodo richiesto (2, 3 o 4 mesi) dalla richiesta testuale, con default 3 mesi."""
    clean = (text or "").lower()
    if re.search(r"\b(?:2|due)\s+mes[ei]\b", clean):
        return 2
    if re.search(r"\b(?:4|quattro)\s+mes[ei]\b", clean):
        return 4
    return 3


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
    wants_pdf_summary: bool = Field(
        default=False,
        description=(
            "True SOLO se l'utente chiede esplicitamente di generare, creare, preparare, esportare "
            "o scaricare il proprio report/riepilogo clinico PDF da mostrare al medico. "
            "False se sta caricando/allegando un referto o ponendo domande informative generali."
        ),
    )


async def classify_user_input(
    llm: LLMProvider,
    user_message: str,
    *,
    has_attachment: bool = False,
    recent_history: Sequence[ChatMessage] | None = None,
) -> GuardrailClassification:
    """Classifica l'input dell'utente con Fast-Path a 0 ms per allegati e saluti,

    oppure tramite chiamata strutturata a Gemini Flash (includendo fino agli ultimi
    3 turni precedenti pseudonimizzati per preservare il contesto di richieste di follow-up).
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
            wants_pdf_summary=False,
        )

    # Fast-Path 2: Saluto o ringraziamento breve a zero latenza (0 ms, zero token)
    if not clean_msg or _TRIVIAL_GREETING_RE.match(clean_msg):
        return GuardrailClassification(
            in_topic=True,
            intent="motivation",
            is_medical_prescription_request=False,
            emergency_risk=False,
            has_loggable_data=has_attachment,
            wants_pdf_summary=False,
        )

    explicit_pdf_req = (not has_attachment) and is_explicit_pdf_summary_request(clean_msg)

    # Se il provider non supporta Structured Outputs o siamo in test unitario con EchoProvider
    if not hasattr(llm, "generate_structured"):
        return GuardrailClassification(
            in_topic=True,
            intent="other_clinical",
            is_medical_prescription_request=False,
            emergency_risk=False,
            has_loggable_data=has_attachment or (not explicit_pdf_req),
            wants_pdf_summary=explicit_pdf_req,
        )

    # Costruisce la finestra di contesto: fino agli ultimi 3 messaggi precedenti + messaggio corrente
    messages_for_classifier: list[ChatMessage] = []
    if recent_history:
        for prev in list(recent_history)[-3:]:
            prev_content = (prev.content or "").strip()
            if prev_content:
                messages_for_classifier.append(ChatMessage(role=prev.role, content=prev_content[:600]))
    messages_for_classifier.append(ChatMessage(role="user", content=clean_msg))

    # Chiamata semantica a Gemini Flash (temp 0.0, prompt conciso, token adeguati anche per thinking)
    try:
        try:
            classification = await llm.generate_structured(
                messages_for_classifier,
                system=get_guardrails_system_prompt(),
                response_schema=GuardrailClassification,
                max_output_tokens=512,
            )
        except TypeError:
            classification = await llm.generate_structured(
                messages_for_classifier,
                system=get_guardrails_system_prompt(),
                response_schema=GuardrailClassification,
            )
        if not isinstance(classification, GuardrailClassification):
            return GuardrailClassification(
                in_topic=True,
                intent="other_clinical",
                is_medical_prescription_request=False,
                emergency_risk=False,
                has_loggable_data=has_attachment or (not explicit_pdf_req),
                wants_pdf_summary=explicit_pdf_req,
            )
        if has_attachment:
            classification.has_loggable_data = True
            classification.wants_pdf_summary = False
        elif explicit_pdf_req:
            classification.wants_pdf_summary = True
        return classification
    except Exception as exc:
        logger.warning(
            "classificatore semantico guardrail fallito (degradazione graziosa) exc={}",
            type(exc).__name__,
        )
        # Degradazione graziosa sicura: non blocca la conversazione per un errore transitorio di rete
        return GuardrailClassification(
            in_topic=True,
            intent="other_clinical",
            is_medical_prescription_request=False,
            emergency_risk=False,
            has_loggable_data=has_attachment or (not explicit_pdf_req),
            wants_pdf_summary=explicit_pdf_req,
        )
