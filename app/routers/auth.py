"""Authentication routes: Register, Login, Logout, Me."""

import uuid
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import jwt
import bcrypt

from app.config import settings
from app.database import get_db, User
from app.schemas import RegisterBody, LoginBody, AuthResponse
from app.dependencies import get_current_user
from app.limiter import limiter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Auth"])

def _create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRY_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


@router.post("/register", status_code=201, response_model=AuthResponse)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def register(request: Request, body: RegisterBody, db: AsyncSession = Depends(get_db)):
    # Check if user exists
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalars().first():
        raise HTTPException(409, detail={"error": "conflict", "detail": "Email already registered"})
    
    # Hash password with bcrypt
    hashed_pw = bcrypt.hashpw(body.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    user = User(
        id=str(uuid.uuid4()),
        email=body.email,
        name=body.name,
        password_hash=hashed_pw,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    token = _create_access_token({"sub": user.id, "email": user.email, "name": user.name})
    return AuthResponse(
        message="Registered", 
        access_token=token, 
        user={"name": user.name, "email": user.email}
    )


@router.post("/login", response_model=AuthResponse)
@limiter.limit(settings.AUTH_RATE_LIMIT)
async def login(request: Request, body: LoginBody, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalars().first()
    
    if not user or not bcrypt.checkpw(body.password.encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(401, detail={"error": "unauthorized", "detail": "Invalid email or password"})
    
    token = _create_access_token({"sub": user.id, "email": user.email, "name": user.name})
    return AuthResponse(
        message="Login successful", 
        access_token=token, 
        user={"name": user.name, "email": user.email}
    )


@router.post("/logout")
async def logout():
    # With JWT, logout is handled client-side by deleting the token.
    # This endpoint exists for API contract consistency.
    return {"message": "Logged out"}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"user": {"name": current_user["name"], "email": current_user["email"]}}