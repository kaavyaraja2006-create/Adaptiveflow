"""
Minimal auth stub.

This is intentionally NOT real security \u2014 no password hashing, no
sessions, no tokens. It exists so the app has a real User entity and a
real "who am I" concept wired through the UI (per the spec's role list:
Admin / Project Manager / Product Manager / Developer / Viewer), ready to
be swapped for real authentication (hashed passwords, JWT/session
cookies, per-request auth checks) before this is ever deployed anywhere
with real user data.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from .. import models
from ..database import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class UserOut(BaseModel):
    id: int
    name: str
    email: str | None
    role: str

    class Config:
        from_attributes = True


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()


@router.post("/login/{user_id}", response_model=UserOut)
def login_as(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).get(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user
