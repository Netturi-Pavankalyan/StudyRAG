"""Pydantic request / response models with validation."""

import re
from pydantic import BaseModel, field_validator

from app.config import settings


# ── Auth ───────────────────────────────────────────────────────

class RegisterBody(BaseModel):
    email: str
    password: str
    name: str

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not v or "@" not in v:
            raise ValueError("Valid email required")
        return v

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required")
        return v

    @field_validator("password", mode="before")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < settings.PASSWORD_MIN_LENGTH:
            raise ValueError(f"Password must be at least {settings.PASSWORD_MIN_LENGTH} characters")
        if settings.PASSWORD_REQUIRE_UPPERCASE and not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if settings.PASSWORD_REQUIRE_LOWERCASE and not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if settings.PASSWORD_REQUIRE_DIGIT and not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginBody(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    message: str
    access_token: str
    token_type: str = "bearer"
    user: dict


# ── Chat ───────────────────────────────────────────────────────

class ChatBody(BaseModel):
    question: str
    doc_id: str | None = None
    session_id: str | None = None
    messages: list[dict] = []


# ── Questions ──────────────────────────────────────────────────

class GenerateQuestionsBody(BaseModel):
    doc_id: str
    count: int = 10
    offset: int = 0
    async_mode: bool = False


# ── Export ─────────────────────────────────────────────────────

class ExportPDFBody(BaseModel):
    questions: list[dict]
    doc_name: str = "Study Material"


# ── Standardised error ────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    retry_after: float | None = None