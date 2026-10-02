# core/security.py
import os
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client

security = HTTPBearer()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

def get_db_client(credentials: HTTPAuthorizationCredentials = Security(security)) -> Client:
    token = credentials.credentials
    try:
        # Inizializza il client e inietta il token dell'utente per le RLS
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        supabase.postgrest.auth(token)
        return supabase
    except Exception as e:
        raise HTTPException(status_code=401, detail="Token non valido o scaduto")