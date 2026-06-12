"""Main FastAPI application setup with middleware, lifespan, and routers."""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import init_db
from app.limiter import limiter

# ── Routers ────────────────────────────────────────────────────
from app.routers import auth, documents, chat, questions, health, export

# ── Logging Setup (Upgrade #13) ───────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Lifespan Handler (replaces deprecated @app.on_event) ──────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Execute startup tasks and yield control to the app."""
    logger.info("🚀 Starting StudyRAG API...")
    
    # Ensure upload folder exists
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize SQLite Database
    await init_db()
    logger.info("📦 Database initialized")
    
    yield
    
    logger.info("🛑 Shutting down StudyRAG API...")

# ── App Initialization ─────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
)

# ── Rate Limiting Middleware ───────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS Middleware (Upgrade #3: Configurable origins) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # No more "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API Routers ────────────────────────────────────────
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(questions.router)
app.include_router(export.router)

# ── Frontend Static Files (Upgrade #21: Mounted LAST) ─────────
# Mounting "/" last ensures API routes like /api/auth/login 
# are matched BEFORE the static file server tries to find a file.
static_path = Path("static")
if static_path.exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    logger.info("📁 Static frontend mounted at /")
else:
    logger.warning("📁 'static' directory not found. Frontend not mounted.")