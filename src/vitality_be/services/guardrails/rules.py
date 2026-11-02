"""Guardrails Livello 1: Regole e Pattern Deterministici (Python puro, < 1 ms).

Intercetta:
1. Emergenze cliniche acute (ipoglicemie severe < 54 mg/dL, iperglicemie critiche >= 400 mg/dL,
   sintomi cardiaci/respiratori/neurologici salvavita)
2. Tentativi noti di Jailbreak / Prompt Injection
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

EmergencyTriggerKind = Literal["severe_hypoglycemia", "critical_hyperglycemia", "acute_symptoms"]

EMERGENCY_REPLY_HYPO = (
    "🚨 **ATTENZIONE: Valore di glicemia criticamente basso (< 54 mg/dL).**\n\n"
    "1. **Se ti senti confuso, debole o non riesci a deglutire in sicurezza, chiama subito il 112** "
    "o chiedi a chi è con te di farlo immediatamente.\n"
    "2. **Se sei vigile e riesci a deglutire**, applica subito la **Regola del 15**: assumi immediatamente "
    "**15 grammi di zuccheri semplici a rapido assorbimento** (es. 3 bustine di zucchero sciolte in acqua, "
    "1 cucchiaio abbondante di miele o mezzo bicchiere di succo di frutta/bibita zuccherata — "
    "*evita il cioccolato o cibi grassi perché rallentano l'assorbimento*).\n"
    "3. Siediti o sdraiati, attendi **15 minuti** e **rimisura la glicemia**: se il valore resta sotto 70 mg/dL "
    "o i sintomi peggiorano, **chiama il 112 o il tuo medico**."
)

EMERGENCY_REPLY_GENERAL = (
    "🚨 **ATTENZIONE: I sintomi o i valori che descrivi possono indicare un'emergenza medica acuta.**\n\n"
    "**Chiama immediatamente il 112** (Numero Unico di Emergenza) o fatti accompagnare subito al "
    "Pronto Soccorso più vicino. Non attendere e non modificare da solo le dosi dei farmaci o dell'insulina."
)

JAILBREAK_REPLY = (
    "Sono il tuo coach virtuale Vitality Assist e il mio compito è affiancarti in modo sicuro nell'autogestione "
    "quotidiana del diabete e dello stile di vita, nel rispetto delle linee guida cliniche. "
    "Non posso disattivare le regole di sicurezza né sostituirmi al medico. "
    "Come posso aiutarti oggi con l'alimentazione, il movimento o il monitoraggio della glicemia?"
)

OFF_TOPIC_REPLY = (
    "Mi dispiace, su questo argomento non posso aiutarti: come tuo coach Vitality Assist mi occupo esclusivamente "
    "di salute, gestione quotidiana del diabete, alimentazione, movimento e benessere. "
    "Vuoi parlare dei tuoi valori glicemici, di un pasto o delle tue abitudini quotidiane?"
)


@dataclass(frozen=True)
class EmergencyMatch:
    trigger_kind: EmergencyTriggerKind
    matched_text: str
    reply: str


# Keyword o pattern glicemici nel contesto circostante
_GLUCOSE_CONTEXT_WORDS = (
    r"glicemi[ae]|glucometr[oi]|sensor[ei]|zuccher[oi]|misurat[oa]|"
    r"valore|sces[ao]\s+a|salit[ao]\s+a|[eè]\s+(?:a|sces[ao]|salit[ao])|lettura"
)

# Unità NON glicemiche da escludere assolutamente (anni, minuti, grammi, kg, dosaggi farmaci...)
_NON_GLUCOSE_UNITS_RE = re.compile(
    r"^\s*(?:anni|ann[io]|minuti|min|secondi|sec|ore|h\b|g\b|gr\b|grammi|kg|chili|chilogrammi|"
    r"passi|km|metri|m\b|%|percento|unit[aà]|ui\b|compresse|pastiglie|bustine|euro|€)\b",
    re.IGNORECASE,
)

# Unità o contesto esplicito di milligrammi/farmaco (es. metformina 500 mg, 850 mg, 1000 mg)
_DRUG_DOSAGE_RE = re.compile(
    r"\b(?:metformina|farmaco|compressa|pastiglia|pillola|dosaggio|dose)\b",
    re.IGNORECASE,
)

# Pattern per sintomi acuti red-flag
_ACUTE_SYMPTOMS_PATTERNS = [
    # Cardiovascolari / Toracici
    r"\b(?:dolor[ei]|oppressione|senso\s+di\s+peso|fitte?|stringimento|stretta)\s+(?:(?:molto\s+)?forte\s+|acut[oa]\s+|dolorosa\s+)?al\s+(?:petto|torace|centro\s+del\s+petto)\b",
    r"\b(?:infarto|arresto\s+cardiaco|dolore\s+irradiato\s+al\s+braccio\s+sinistro)\b",
    # Respiratori
    r"\b(?:non\s+riesco\s+a\s+respirare|fatico\s+a\s+respirare|fatica\s+a\s+respirare|non\s+respiro|sto\s+soffocando|soffoco|manca\s+l['’]aria|dispnea\s+acuta)\b",
    # Neurologici / Sincope
    r"\b(?:sto\s+per\s+svenire|sono\s+svenut[oa]|perdita\s+di\s+coscienza|ho\s+perso\s+i\s+sensi|senza\s+sensi)\b",
    r"\b(?:non\s+ci\s+vedo\s+pi[uù]|vista\s+sdoppiata\s+improvvisa|cecità\s+improvvisa)\b",
    r"\b(?:met[aà]\s+(?:del\s+)?(?:viso|faccia|corpo|braccio)\s+(?:bloccata|addormentata|paralizzata|debole))\b",
    r"\b(?:non\s+riesco\s+a\s+parlare|bocca\s+storta|parole\s+confuse\s+improvvise)\b",
    # Scompenso metabolico grave (iperglicemia + vomito)
    r"\b(?:glicemia\s+(?:molto\s+)?alta.*vomit(?:o|ando|are))\b",
    r"\b(?:chetoni\s+(?:molto\s+)?alti|chetoacidosi)\b",
]

_ACUTE_SYMPTOMS_RE = re.compile("|".join(_ACUTE_SYMPTOMS_PATTERNS), re.IGNORECASE)

# Negazioni immediate da non trattare come emergenza
_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:non\s+ho|nessun|senza|escludo|mai\s+avut[oa]|non\s+sento|non\s+c['’][eè])\s+(?:alcun\s+|pi[uù]\s+)?",
    re.IGNORECASE,
)

# Pattern noti di Jailbreak / Prompt Injection
_JAILBREAK_PATTERNS = [
    r"\b(?:ignora|dimentica|disattiva|annulla|bypassa|scavalca)\s+(?:tutte\s+)?(?:le\s+)?(?:istruzioni|regole|direttive|limitazioni|guardrail|vincoli)(?:\s+precedenti|\s+di\s+sistema)?\b",
    r"\b(?:ignore|disregard|forget|bypass|override)\s+(?:all\s+)?(?:previous|prior|system|above|safety)\s+(?:instructions|rules|prompts|guidelines|constraints)\b",
    r"\b(?:fai\s+finta\s+di\s+essere|agisci\s+come|comportati\s+da|sei\s+in\s+modalit[aà])\s+(?:(?:un\s+)?(?:medico\s+vero|dottore\s+senza\s+limiti)|dan\b|developer\s+mode|jailbreak|unrestricted)\b",
    r"\b(?:stampa|mostra|rivela|ripeti|dimmi)\s+(?:il\s+tuo\s+|le\s+tue\s+|l['’]intero\s+)?(?:system\s+prompt|prompt\s+di\s+sistema|istruzioni(?:\s+di\s+sistema)?(?:\s+segrete|\s+iniziali)?)\b",
    r"\b(?:you\s+are\s+now\s+in\s+developer\s+mode|dan\s+mode|jailbroken)\b",
]
_JAILBREAK_RE = re.compile("|".join(_JAILBREAK_PATTERNS), re.IGNORECASE)


def check_emergency_input(text: str) -> EmergencyMatch | None:
    """Verifica la presenza deterministica di segnali di emergenza medica acuta."""
    if not text or len(text.strip()) < 3:
        return None

    cleaned = text.strip()

    # 1. Check Sintomi Acuti Red-Flag (con esclusione negazioni contestuali)
    sym_match = _ACUTE_SYMPTOMS_RE.search(cleaned)
    if sym_match:
        matched_str = sym_match.group(0)
        # Verifica se prima del match c'è una negazione diretta negli ultimi 25 caratteri
        start_idx = sym_match.start()
        prefix = cleaned[max(0, start_idx - 25) : start_idx]
        if not _NEGATION_PREFIX_RE.search(prefix):
            return EmergencyMatch(
                trigger_kind="acute_symptoms",
                matched_text=matched_str,
                reply=EMERGENCY_REPLY_GENERAL,
            )

    # 2. Check Valori Estremi con disambiguazione contestuale
    # Cerchiamo tutti i numeri nel testo
    for num_match in re.finditer(r"\b(\d{2,3})\b", cleaned):
        val_int = int(num_match.group(1))
        num_start = num_match.start()
        num_end = num_match.end()

        # Controlliamo cosa c'è subito dopo il numero
        post_str = cleaned[num_end : min(len(cleaned), num_end + 25)]
        if _NON_GLUCOSE_UNITS_RE.match(post_str):
            continue

        # Finestra di contesto (35 caratteri prima e dopo)
        window = cleaned[max(0, num_start - 35) : min(len(cleaned), num_end + 35)]
        has_explicit_mgdl = bool(re.search(r"mg\s*/\s*d[lL]", window, re.IGNORECASE))
        has_glucose_word = bool(re.search(_GLUCOSE_CONTEXT_WORDS, window, re.IGNORECASE))

        # A. Ipoglicemia Severa (< 54 mg/dL: tipicamente 15..53)
        if 15 <= val_int < 54:
            if has_explicit_mgdl or has_glucose_word:
                return EmergencyMatch(
                    trigger_kind="severe_hypoglycemia",
                    matched_text=num_match.group(0),
                    reply=EMERGENCY_REPLY_HYPO,
                )

        # B. Iperglicemia Critica (>= 400 mg/dL: tipicamente 400..999)
        elif 400 <= val_int <= 999:
            # Escludiamo dosaggi di farmaci (es. metformina 500 mg, compresse)
            if _DRUG_DOSAGE_RE.search(window):
                continue
            # Se è seguito da "mg" puro senza "/dl", è molto probabile un farmaco
            if re.search(r"^\s*mg\b(?!/)", post_str, re.IGNORECASE):
                continue

            if has_explicit_mgdl or has_glucose_word:
                return EmergencyMatch(
                    trigger_kind="critical_hyperglycemia",
                    matched_text=num_match.group(0),
                    reply=EMERGENCY_REPLY_GENERAL,
                )

    # 3. Check lettura glucometro "HI" o "HIGH"
    if re.search(r"\b(?:glucometro|sensore|glicemia)\b.*\b(?:HI|HIGH)\b", cleaned, re.IGNORECASE):
        return EmergencyMatch(
            trigger_kind="critical_hyperglycemia",
            matched_text="HI",
            reply=EMERGENCY_REPLY_GENERAL,
        )

    return None


def check_jailbreak_input(text: str) -> bool:
    """Verifica se il messaggio utente contiene tentativi di jailbreak o prompt injection noti."""
    if not text or len(text.strip()) < 5:
        return False
    return bool(_JAILBREAK_RE.search(text))
