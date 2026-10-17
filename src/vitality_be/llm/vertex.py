from google import genai

# Inizializzazione corretta per Vertex AI con ADC
client = genai.Client(vertexai=True, location="europe-west1")
# llm/vertex.py
"""Provider Vertex AI (Gemini) con SDK `google-genai`.

Verificato sulla versione installata (google-genai 2.30.0):
- il costruttore di `genai.Client` accetta sia `enterprise=True` sia `vertexai=True`;
  si usa `enterprise=True` (nome attuale in documentazione);
- `HttpOptions.timeout` è in MILLISECONDI;
- `ThinkingConfig` ha `thinking_level` (minimal|low|medium|high) e `thinking_budget`;
- con location 'eu' l'SDK usa l'host aiplatform.eu.rep.googleapis.com.
NON verificabile da qui (serve una chiamata reale): che il modello scelto esista su 'eu'
e che accetti `thinking_level`. Vedi scripts/smoke_vertex.py.

Dati verso Google: nessun grounding, nessun tool, nessun `cached_content`; i prompt non
contengono identificativi del paziente (li garantisce il service, che non li riceve).
"""
import asyncio
import random
import time

import httpx
from google import genai
from google.auth.exceptions import GoogleAuthError
from google.genai import errors, types
from loguru import logger

from vitality_be.llm.base import ChatMessage, LLMBlocked, LLMError, LLMUnavailable

# Safety settings ESPLICITE (non ci affidiamo ai default, che cambiano tra modelli).
# DANGEROUS_CONTENT è meno severo (solo HIGH): con soglia media un assistente sanitario
# rischia falsi positivi su farmaci/sintomi. Gli altri tre bloccano da MEDIUM in su.
SAFETY_SETTINGS = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH),
]

_BLOCKING_FINISH = {
    types.FinishReason.SAFETY,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.SPII,
    types.FinishReason.RECITATION,
}
_THINKING = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}
_RETRYABLE_CODES = {429, 503}
_SENTENCE_END = (".", "!", "?", "…", "\n")


class VertexProvider:
    def __init__(
        self,
        *,
        project: str,
        location: str,
        model: str,
        timeout_s: float,
        max_output_tokens: int,
        temperature: float,
        thinking_level: str = "low",
        client: genai.Client | None = None,  # iniettabile nei test
        retry_backoff_s: tuple[float, float] = (0.4, 1.2),
    ) -> None:
        self.model = model
        self._timeout_s = timeout_s
        self._backoff = retry_backoff_s
        # Client creato UNA volta e riusato (nessuna API key: Application Default Credentials).
        # Timeout in ms; il limite vero è la scadenza totale gestita in chat().
        self._client = client or genai.Client(
            enterprise=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        self._config_base = dict(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            safety_settings=SAFETY_SETTINGS,
            #thinking_config=types.ThinkingConfig(thinking_level=_THINKING[thinking_level]),
        )

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def chat(self, messages: list[ChatMessage], *, system: str | None = None) -> str:
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
        ]
        # Il system prompt va SOLO in system_instruction, mai concatenato all'input utente.
        config = types.GenerateContentConfig(system_instruction=system, **self._config_base)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_s
        started = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise LLMUnavailable("Timeout del provider LLM", code="timeout") from None
            try:
                async with asyncio.timeout(remaining):
                    response = await self._client.aio.models.generate_content(
                        model=self.model, contents=contents, config=config
                    )
                break
            except TimeoutError:
                raise LLMUnavailable("Timeout del provider LLM", code="timeout") from None
            except errors.APIError as exc:
                code = getattr(exc, "code", None)
                if attempt == 1 and code in _RETRYABLE_CODES:
                    pause = random.uniform(*self._backoff)  # jitter
                    if deadline - loop.time() > pause + 2:  # solo se resta tempo utile
                        logger.warning("Vertex retry code={}", code)
                        await asyncio.sleep(pause)
                        continue
                raise self._map_api_error(code) from None
            except GoogleAuthError:
                raise LLMUnavailable("Credenziali Google non disponibili", code="credentials") from None
            except httpx.TransportError:
                raise LLMUnavailable("Errore di rete verso il provider", code="network") from None

        text = self._extract_text(response)
        self._log_usage(response, time.monotonic() - started, attempt)
        return text

    @staticmethod
    def _map_api_error(code: int | None) -> LLMError:
        # 429/5xx/408 = transitori -> 503. Il resto (400/401/403/404...) è un problema
        # nostro (config, permessi, modello inesistente) -> 502, MAI 401 verso il FE.
        if code is not None and (code in (408, 429) or code >= 500):
            return LLMUnavailable("Provider LLM non disponibile", code=f"http_{code}")
        return LLMError("Errore del provider LLM", code=f"http_{code}")

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        fb = response.prompt_feedback
        if fb is not None and fb.block_reason is not None:
            raise LLMBlocked("Prompt bloccato dal provider", code=f"prompt_{fb.block_reason.name.lower()}")

        if not response.candidates:
            raise LLMError("Risposta vuota dal provider", code="empty_response")
        cand = response.candidates[0]
        reason = cand.finish_reason

        if reason in _BLOCKING_FINISH:
            raise LLMBlocked("Risposta bloccata dal provider", code=f"finish_{reason.name.lower()}")

        parts = (cand.content.parts if cand.content and cand.content.parts else [])
        # Si scartano i "thought" (ragionamento interno): non vanno mai mostrati all'utente.
        text = "".join(p.text for p in parts if p.text and not p.thought).strip()

        if reason == types.FinishReason.MAX_TOKENS:
            # Troppi token di ragionamento o risposta lunga. Un consiglio sanitario tagliato a
            # metà è pericoloso: si tiene solo fino all'ultima frase completa, altrimenti errore.
            cut = max((text.rfind(c) for c in _SENTENCE_END), default=-1)
            if cut <= 0:
                raise LLMError("Risposta troncata dal limite di token", code="max_tokens_empty")
            logger.warning("Vertex MAX_TOKENS: risposta ridotta all'ultima frase completa")
            return text[: cut + 1].strip()

        if reason not in (types.FinishReason.STOP, None) :
            raise LLMError("Risposta non valida dal provider", code=f"finish_{reason.name.lower()}")
        if not text:
            raise LLMError("Risposta vuota dal provider", code="empty_text")
        return text

    def _log_usage(self, response, duration_s: float, attempts: int) -> None:
        u = response.usage_metadata
        logger.info(
            "vertex model={} ms={} attempts={} prompt_tokens={} output_tokens={} thought_tokens={}",
            self.model, int(duration_s * 1000), attempts,
            getattr(u, "prompt_token_count", None),
            getattr(u, "candidates_token_count", None),
            getattr(u, "thoughts_token_count", None),
        )
