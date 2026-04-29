from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from .auth_service import register_user, authenticate_user
from .auth_utils import create_access_token
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm
from app.models import LoginLog


router = APIRouter(prefix="/auth", tags=["Authentication"])


class AuthRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
def register(data: AuthRequest, db: Session = Depends(get_db)):
    user = register_user(db, data.email, data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return {"message": "User registered successfully"}


@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({"sub": user.email})

    log = LoginLog(user_id=user.id)
    db.add(log)
    db.commit()

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }