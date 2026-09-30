from fastapi import APIRouter, HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel

from auth.jwt import create_access_token
from helpers.config import get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(data: LoginRequest):
    settings = get_settings()
    if not settings.ADMIN_PASSWORD_HASH:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin password not configured",
        )
    if data.username != settings.ADMIN_USERNAME:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    try:
        ok = pwd_context.verify(data.password, settings.ADMIN_PASSWORD_HASH)
    except Exception:
        ok = False
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token({"sub": data.username})
    return {"access_token": access_token, "token_type": "bearer"}
