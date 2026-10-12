from dataclasses import dataclass, field
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from supabase import AuthApiError, Client, create_client

from vitality_be.core.config import get_settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: str | None
    # serve solo a costruire il client RLS; repr=False così non finisce nei log
    token: str = field(repr=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


@lru_cache
def get_auth_client() -> Client:
    """Client condiviso, SOLO per auth.get_user(). Non impostargli mai un token."""
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_anon_key)


def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> CurrentUser:
    if creds is None:
        raise _unauthorized("Token mancante")
    try:
        # Supabase verifica firma + scadenza + sessione non revocata
        resp = get_auth_client().auth.get_user(jwt=creds.credentials)
    except AuthApiError:
        raise _unauthorized("Token non valido o scaduto")
    except Exception:
        logger.exception("Supabase Auth non raggiungibile")
        raise HTTPException(503, "Servizio di autenticazione non disponibile")

    if resp is None or resp.user is None:
        raise _unauthorized("Token non valido o scaduto")
    return CurrentUser(id=resp.user.id, email=resp.user.email, token=creds.credentials)


CurrentUserDep = Annotated[CurrentUser, Depends(get_current_user)]