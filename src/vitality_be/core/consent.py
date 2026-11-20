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
    """Verifica fail-closed del Consenso Esplicito al Trattamento Dati Sanitari con IA (GDPR Art. 9 e Art. 7.3).

    Interroga sempre la tabella autoritativa `public.user` sotto RLS (`auth.uid()`):
    - NON usa mai `user_metadata` nel JWT come fallback (campo modificabile lato client).
    - In caso di errore di comunicazione con il DB, blocca fail-closed con HTTP 503.
    - Se la riga è assente o il consenso AI/privacy risulta non prestato o revocato, blocca con HTTP 403.
    """
    if not hasattr(user_db, "table"):
        logger.error("Verifica ai_health_consent impossibile: client DB senza supporto table user={}", user.id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servizio di verifica del consenso temporaneamente non disponibile. Riprova tra poco.",
        )

    try:
        resp = await _run_sync_or_thread(
            lambda: (
                user_db.table("user")
                .select("ai_health_consent, consenso_privacy")
                .eq("id", user.id)
                .limit(1)
                .execute()
            )
        )
    except Exception as exc:
        logger.warning(
            "Verifica ai_health_consent su tabella user fallita (fail-closed) user={} exc={}",
            user.id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servizio di verifica del consenso temporaneamente non disponibile. Riprova tra poco.",
        ) from None

    rows = getattr(resp, "data", None)
    if rows and isinstance(rows, list) and len(rows) > 0:
        row = rows[0] if isinstance(rows[0], dict) else {}
        ai_ok = bool(row.get("ai_health_consent"))
        privacy_ok = row.get("consenso_privacy", True) is not False
        if ai_ok and privacy_ok:
            return

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Consenso al trattamento dei dati sanitari tramite Intelligenza Artificiale "
            "(GDPR Art. 9) non prestato o revocato. Per utilizzare l'assistente conversazionale e "
            "l'elaborazione automatica dei referti o dei pasti, fornisci il consenso "
            "nella schermata di registrazione o nelle impostazioni."
        ),
    )
