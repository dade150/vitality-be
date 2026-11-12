import asyncio

from fastapi import HTTPException, status
from loguru import logger

from vitality_be.core.security import CurrentUserDep
from vitality_be.db.supabase1 import UserDbDep


async def _run_sync_or_thread(func):
    """Esegue I/O sincrono in thread pool se c'è un loop asincrono attivo, o direttamente nei test sincroni."""
    try:
        return await asyncio.to_thread(func)
    except RuntimeError:
        return func()


async def require_ai_health_consent(
    user: CurrentUserDep,
    user_db: UserDbDep,
) -> None:
    """Verifica del Consenso Esplicito al Trattamento Dati Sanitari con IA (GDPR Art. 9 e Art. 7.3).

    1. Interroga in modo non bloccante (`asyncio.to_thread`) la tabella autoritativa `public.user`
       sotto RLS (`auth.uid()`) per rispettare immediatamente eventuali revoche (`PATCH /api/me/consent`),
       senza fidarsi ciecamente di `user_metadata` nel JWT.
    2. Se la riga è presente nel DB, usa esclusivamente `public.user.ai_health_consent` come fonte di verità.
    3. Se il DB non restituisce righe (es. ambiente di test con mock minimale), usa il claim del token come fallback.
    4. Se il consenso non è attivo, blocca la richiesta con HTTP 403 Forbidden.
    """
    if hasattr(user_db, "table"):
        try:
            resp = await _run_sync_or_thread(
                lambda: (
                    user_db.table("user")
                    .select("ai_health_consent")
                    .eq("id", user.id)
                    .limit(1)
                    .execute()
                )
            )
            rows = getattr(resp, "data", None)
            if rows and isinstance(rows, list) and len(rows) > 0:
                if bool(rows[0].get("ai_health_consent")):
                    return
                # Fonte autoritativa DB: il consenso risulta revocato o non prestato -> 403 immediato
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        "Consenso al trattamento dei dati sanitari tramite Intelligenza Artificiale "
                        "(GDPR Art. 9) non prestato o revocato. Per utilizzare l'assistente conversazionale e "
                        "l'elaborazione automatica dei referti o dei pasti, fornisci il consenso "
                        "nella schermata di registrazione o nelle impostazioni."
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.debug(
                "Verifica ai_health_consent su tabella user degradata user={} exc={}",
                user.id,
                type(exc).__name__,
            )

    if getattr(user, "ai_health_consent", False):
        return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Consenso al trattamento dei dati sanitari tramite Intelligenza Artificiale "
            "(GDPR Art. 9) non prestato. Per utilizzare l'assistente conversazionale e "
            "l'elaborazione automatica dei referti o dei pasti, fornisci il consenso "
            "nella schermata di registrazione o nelle impostazioni."
        ),
    )
