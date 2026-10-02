# main.py
from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from loguru import logger
from core.security import get_db_client
from supabase import Client

app = FastAPI()

# ControllerAdvice equivalente
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Fallita richiesta {request.method} {request.url} | Errore: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Si è verificato un errore interno. Riprova più tardi."},
    )

# Esempio di endpoint protetto che usa il client autenticato e i log
@app.get("/api/test")
def test_endpoint(supabase: Client = Depends(get_db_client)):
    logger.info("Chiamata all'endpoint di test effettuata")
    # Il backend esegue la query con i permessi dell'utente chiamante
    response = supabase.table("tua_tabella").select("*").execute()
    return {"data": response.data}