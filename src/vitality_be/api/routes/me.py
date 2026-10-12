# api/routes/me.py
from fastapi import APIRouter
from pydantic import BaseModel
from src.vitality_be.core.security import CurrentUserDep

router = APIRouter(tags=["me"])

class MeResponse(BaseModel):
    id: str
    email: str | None

@router.get("/me", response_model=MeResponse)
def me(user: CurrentUserDep):
    return MeResponse(id=user.id, email=user.email)  # response_model: il token non può uscire
