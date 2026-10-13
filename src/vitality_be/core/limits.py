"""Limiti anti-abuso (minaccia: denial of wallet).

LIMITE NOTO: lo stato è in memoria, PER PROCESSO. Con N worker/istanze il limite
effettivo è N volte quello configurato e si azzera a ogni riavvio. Va bene con un
solo worker (es. Render free/starter). Percorso verso Redis: sostituire
`SlidingWindowRateLimiter.check` con uno script Lua su sorted set (ZADD + ZREMRANGEBYSCORE
+ ZCARD, chiave `rl:{user_id}`) e il semaforo con un contatore con TTL; l'interfaccia
pubblica (check / slot) resta la stessa.
"""
import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from functools import lru_cache

from vitality_be.core.config import get_settings


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


class GateBusy(Exception):
    """Coda di attesa piena: troppe richieste LLM in corso."""


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_s: int, max_keys: int = 10_000):
        self._limit, self._window, self._max_keys = limit, window_s, max_keys
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> None:
        now = time.monotonic()
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self._window:
            q.popleft()
        if len(q) >= self._limit:
            raise RateLimitExceeded(max(1, int(self._window - (now - q[0])) + 1))
        q.append(now)
        if len(self._hits) > self._max_keys:  # evita crescita illimitata della memoria
            self._prune(now)

    def _prune(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self._window]:
            del self._hits[k]


class ConcurrencyGate:
    """Semaforo globale con timeout di attesa: oltre il tetto si risponde 503, non si accoda all'infinito."""

    def __init__(self, max_concurrency: int, queue_timeout_s: float):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._timeout = queue_timeout_s

    @asynccontextmanager
    async def slot(self):
        try:
            await asyncio.wait_for(self._sem.acquire(), self._timeout)
        except TimeoutError:
            raise GateBusy from None
        try:
            yield
        finally:
            self._sem.release()


@lru_cache
def get_rate_limiter() -> SlidingWindowRateLimiter:
    s = get_settings()
    return SlidingWindowRateLimiter(s.rate_limit_requests, s.rate_limit_window_s)


@lru_cache
def get_gate() -> ConcurrencyGate:
    s = get_settings()
    return ConcurrencyGate(s.llm_max_concurrency, s.llm_queue_timeout_s)
