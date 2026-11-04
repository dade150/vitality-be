from fastapi import HTTPException, status
from loguru import logger

from vitality_be.core.security import CurrentUserDep
from vitality_be.db.supabase1 import UserDbDep


async def require_ai_health_consent(
    user: CurrentUserDep,
    user_db: UserDbDep,
) -> None:
    """Verifica del Consenso Esplicito al Trattamento Dati Sanitari con IA (GDPR Art. 9).

    1. Controlla prima il claim `ai_health_consent` nei metadati del token dell'utente.
    2. Se assente o non abilitato nel token, interroga la tabella `public.user` sotto RLS auth.uid().
    3. Se il consenso non è stato prestato, blocca la richiesta con HTTP 403 Forbidden.
    """
    if getattr(user, "ai_health_consent", False):
        return

    # Fallback su tabella DB se il token è precedente all'aggiornamento dei metadati
    if hasattr(user_db, "table"):
        try:
            resp = (
                user_db.table("user")
                .select("ai_health_consent")
                .eq("id", user.id)
                .limit(1)
                .execute()
            )
            rows = getattr(resp, "data", None)
            if rows and isinstance(rows, list) and len(rows) > 0:
                if bool(rows[0].get("ai_health_consent")):
                    return
        except Exception as exc:
            logger.debug(
                "Verifica ai_health_consent su tabella user degradata user={} exc={}",
                user.id,
                type(exc).__name__,
            )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Consenso al trattamento dei dati sanitari tramite Intelligenza Artificiale "
            "(GDPR Art. 9) non prestato. Per utilizzare l'assistente conversazionale e "
            "l'elaborazione automatica dei referti o dei pasti, fornisci il consenso "
            "nella schermata di registrazione o nelle impostazioni."
        ),
    )
