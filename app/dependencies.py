"""Shared FastAPI dependencies."""

import logging

from fastapi import Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from jose import jwt, JWTError

from app.config import settings
from app.database import get_db

logger = logging.getLogger(__name__)


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Validate JWT from the Authorization header.
    Returns {"user_id": ..., "email": ..., "name": ...}
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "not_authenticated", "detail": "Missing or invalid Authorization header"},
        )

    token = auth_header[7:]
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail={"error": "invalid_token", "detail": "Token missing subject"},
            )
        return {
            "user_id": user_id,
            "email": payload.get("email", ""),
            "name": payload.get("name", ""),
        }
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail={"error": "invalid_token", "detail": "Invalid or expired token"},
        )