from typing import Annotated
from fastapi import Depends
from supabase import Client, create_client

from vitality_be.core.config import get_settings
from vitality_be.core.security import CurrentUserDep


def get_user_db(user: CurrentUserDep) -> Client:
    """Client NUOVO per richiesta, con il token dell'utente: le query passano dall'RLS."""
    s = get_settings()
    client = create_client(s.supabase_url, s.supabase_anon_key)
    client.postgrest.auth(user.token)
    return client


UserDbDep = Annotated[Client, Depends(get_user_db)]