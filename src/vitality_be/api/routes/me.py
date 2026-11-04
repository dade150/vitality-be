# api/routes/me.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger
from pydantic import BaseModel

from vitality_be.core.limits import RateLimitExceeded, get_rate_limiter
from vitality_be.core.security import CurrentUserDep
from vitality_be.db.supabase1 import UserDbDep

router = APIRouter(tags=["me"])


def enforce_rate_limit(user: CurrentUserDep) -> None:
    try:
        get_rate_limiter().check(user.id)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Troppe richieste. Riprova tra poco.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from None


class MeResponse(BaseModel):
    id: str
    email: str | None


async def _run_sync_or_thread(fn):
    res = fn()
    if inspect.isawaitable(res):
        return await res
    return res


async def _safe_select(user_db: Any, table_name: str, columns: str = "*") -> list[dict[str, Any]]:
    if not hasattr(user_db, "table"):
        return []
    try:
        resp = await _run_sync_or_thread(
            lambda: user_db.table(table_name).select(columns).execute()
        )
        data = getattr(resp, "data", None)
        return list(data) if isinstance(data, list) else []
    except Exception as exc:
        logger.debug("Export select degradata table={} exc={}", table_name, type(exc).__name__)
        return []


@router.get("/me", response_model=MeResponse, dependencies=[Depends(enforce_rate_limit)])
def me(user: CurrentUserDep):
    return MeResponse(id=user.id, email=user.email)  # response_model: il token non può uscire


@router.get(
    "/me/export",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(enforce_rate_limit)],
)
async def export_my_personal_data(
    user: CurrentUserDep,
    user_db: UserDbDep,
) -> Response:
    """Esercizio del Diritto alla Portabilità dei Dati (GDPR Art. 20).

    Esporta in formato strutturato e leggibile da dispositivo automatico (JSON)
    tutti i dati anagrafici, di profilo, clinici, nutrizionali, terapeutici,
    referti e conversazioni associati all'utente autenticato (isolato via RLS auth.uid()).
    """
    user_rows = await _safe_select(user_db, "user", "*")
    profile_rows = await _safe_select(
        user_db, "profile", "id, peso, altezza, sesso, note_anamnestiche, created_at"
    )
    settings_rows = await _safe_select(user_db, "user_settings", "*")
    diabetes_rows = await _safe_select(
        user_db,
        "diabetes_data",
        "id, metric_type, metric_value, measured_at, source_report_id, created_at",
    )
    meals_rows = await _safe_select(
        user_db,
        "meals",
        "id, meal_type, description, carbs_grams, calories, is_estimated, meal_at, created_at",
    )
    activities_rows = await _safe_select(
        user_db,
        "physical_activities",
        "id, activity_type, duration_minutes, intensity, steps, notes, activity_at, created_at",
    )
    therapies_rows = await _safe_select(
        user_db,
        "therapies",
        "id, therapy_type, drug_name, dosage, frequency, scheduled_time, notes, is_active, created_at",
    )
    therapy_logs_rows = await _safe_select(
        user_db,
        "therapy_logs",
        "id, therapy_id, taken_at, status, created_at",
    )
    appointments_rows = await _safe_select(
        user_db,
        "appointments",
        "id, title, doctor_name, appointment_at, notes, created_at",
    )
    reports_rows = await _safe_select(
        user_db,
        "patient_reports",
        "id, source_name, source_path, report_type, clean_summary, content, extracted_params, report_at, created_at",
    )
    sessions_rows = await _safe_select(
        user_db,
        "chat_session",
        "id, title, created_at",
    )
    messages_rows = await _safe_select(
        user_db,
        "chat_message",
        "id, session_id, role, content, created_at",
    )
    logs_rows = await _safe_select(
        user_db,
        "app_logs",
        "id, level, message, route, created_at",
    )

    export_bundle = {
        "metadata": {
            "application": "diaLog2",
            "gdpr_reference": "Art. 20 Regolamento (UE) 2016/679 (Diritto alla portabilità dei dati)",
            "schema_version": "1.1",
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "user_id": user.id,
            "email": user.email,
        },
        "account_and_consents": user_rows[0] if user_rows else {"id": user.id, "email": user.email},
        "clinical_profile": profile_rows[0] if profile_rows else None,
        "preferences": settings_rows[0] if settings_rows else None,
        "health_diary": {
            "diabetes_and_clinical_measurements": diabetes_rows,
            "meals": meals_rows,
            "physical_activities": activities_rows,
            "therapies": therapies_rows,
            "therapy_adherence_logs": therapy_logs_rows,
            "medical_appointments": appointments_rows,
            "medical_reports": reports_rows,
        },
        "conversational_history": {
            "sessions": sessions_rows,
            "messages": messages_rows,
        },
        "application_logs": logs_rows,
    }

    logger.info("Export portabilità dati GDPR Art. 20 generato user={}", user.id)
    json_bytes = json.dumps(export_bundle, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    filename = f"diaLog2_Dati_GDPR_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"

    return Response(
        content=json_bytes,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.delete(
    "/me",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(enforce_rate_limit)],
)
async def delete_my_account_and_data(
    user: CurrentUserDep,
    user_db: UserDbDep,
) -> Response:
    """Esercizio del Diritto all'Oblio / Cancellazione (GDPR Art. 17).

    Elimina definitivamente tutti i dati sanitari, il diario, i referti, la cronologia chat,
    i file nello storage e l'account dell'utente autenticato.
    """
    # 1. Cancellazione preventiva dei file archiviati nello Storage privato via API Supabase (B1)
    if hasattr(user_db, "storage"):
        try:
            bucket = user_db.storage.from_("patient-reports")
            files_resp = await _run_sync_or_thread(lambda: bucket.list(user.id))
            if files_resp and isinstance(files_resp, list):
                paths_to_remove = [
                    f"{user.id}/{item['name']}"
                    for item in files_resp
                    if isinstance(item, dict) and item.get("name")
                ]
                if paths_to_remove:
                    await _run_sync_or_thread(lambda: bucket.remove(paths_to_remove))
                    logger.info("Rimossi {} file da patient-reports per user={}", len(paths_to_remove), user.id)
        except Exception as st_exc:
            logger.warning("Cancellazione file Storage ignorata/fallita user={} exc={}", user.id, type(st_exc).__name__)

    # 2. Eliminazione definitiva transazionale via RPC (elimina dati + auth.users)
    rpc_ok = False
    if hasattr(user_db, "rpc"):
        try:
            await _run_sync_or_thread(lambda: user_db.rpc("delete_own_account", {}).execute())
            rpc_ok = True
            logger.info("Account e dati sanitari eliminati tramite RPC delete_own_account user={}", user.id)
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        except Exception as rpc_exc:
            logger.warning(
                "RPC delete_own_account fallita, avvio pulizia tabellare di sicurezza user={} exc={}",
                user.id,
                type(rpc_exc).__name__,
            )

    # 3. Fallback tabellare per dati sanitari se l'RPC non era disponibile o è fallita
    if hasattr(user_db, "table"):
        # Cancella prima i messaggi associati alle chat session dell'utente (B2)
        try:
            s_resp = await _run_sync_or_thread(
                lambda: user_db.table("chat_session").select("id").eq("user_id", user.id).execute()
            )
            s_data = getattr(s_resp, "data", None)
            if s_data and isinstance(s_data, list):
                for s in s_data:
                    if isinstance(s, dict) and s.get("id"):
                        try:
                            await _run_sync_or_thread(
                                lambda sid=s["id"]: user_db.table("chat_message").delete().eq("session_id", sid).execute()
                            )
                        except Exception:
                            pass
        except Exception as cm_exc:
            logger.debug("Delete fallback chat_message ignorata exc={}", type(cm_exc).__name__)

        tables_by_user_id = (
            "chat_session",
            "patient_report_chunks",
            "patient_reports",
            "therapy_logs",
            "therapies",
            "diabetes_data",
            "meals",
            "physical_activities",
            "appointments",
            "app_logs",
            "user_settings",
            "profile",
        )
        for tbl in tables_by_user_id:
            try:
                await _run_sync_or_thread(
                    lambda t=tbl: user_db.table(t).delete().eq("user_id", user.id).execute()
                )
            except Exception as exc:
                logger.debug("Delete fallback table={} ignorata exc={}", tbl, type(exc).__name__)

        try:
            await _run_sync_or_thread(
                lambda: user_db.table("user").delete().eq("id", user.id).execute()
            )
        except Exception as exc:
            logger.debug("Delete fallback table=user ignorata exc={}", type(exc).__name__)

    logger.warning("Pulizia tabellare completata ma identità auth non eliminata per user={}", user.id)
    if not rpc_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impossibile completare l'eliminazione definitiva dell'account (auth.users). Riprova.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
