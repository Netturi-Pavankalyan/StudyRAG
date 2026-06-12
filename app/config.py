"""Centralised settings — every magic number lives here."""

from pydantic_settings import BaseSettings
from typing import Set


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────────
    APP_NAME: str = "StudyRAG — Smart Study Assistant"
    DEBUG: bool = False

    # ── Security ───────────────────────────────────────────────
    SECRET_KEY: str  # REQUIRED — app won't start without it
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_HOURS: int = 24
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # ── Database ───────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///studyrag.db"

    # ── Uploads ────────────────────────────────────────────────
    UPLOAD_FOLDER: str = "uploads"
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: Set[str] = {"pdf", "txt", "doc", "docx", "rtf", "odt"}

    # ── Groq ───────────────────────────────────────────────────
    GROQ_API_KEY: str  # REQUIRED
    GROQ_MODEL_PRIMARY: str = "llama-3.1-8b-instant"
    GROQ_MODEL_FALLBACK: str = "gemma2-9b-it"
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── Context / truncation limits ────────────────────────────
    DOC_CONTEXT_CHARS: int = 2000
    CHAT_CONTEXT_CHARS: int = 2000
    PDF_MAX_CHARS: int = 6000
    CHAT_MESSAGE_MAX_CHARS: int = 2000
    CHAT_HISTORY_MAX_CHARS: int = 8000

    # ── Cache ──────────────────────────────────────────────────
    PDF_CACHE_MAXSIZE: int = 128
    CANCELLED_TASKS_TTL: int = 300  # seconds

    # ── Rate limiting ──────────────────────────────────────────
    AUTH_RATE_LIMIT: str = "5/minute"

    # ── Password policy ────────────────────────────────────────
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()