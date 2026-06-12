"""Groq API wrapper with fallback, retry logic, and async support."""

import re
import logging
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)

# ── Custom Error ───────────────────────────────────────────────

class RateLimitError(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after

# ── Helpers ────────────────────────────────────────────────────

def _parse_retry_after(err_str: str) -> float | None:
    m = re.search(r"(\d+)m(\d+(?:\.\d+)?)s", err_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
    if m:
        return float(m.group(1))
    return None

def _is_rate_limit_error(err_str: str) -> bool:
    return (
        "429" in err_str or "413" in err_str or
        "tokens per minute" in err_str or
        "rate_limit_exceeded" in err_str or
        "tokens per day" in err_str
    )

# ── Lazy Groq Client ──────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client

# ── Sync Chat (used by background threads) ────────────────────

def groq_chat_sync(system: str, messages: list[dict], max_tokens: int = 2048) -> str:
    full_messages = [{"role": "system", "content": system}] + messages
    client = _get_client()
    
    for model in (settings.GROQ_MODEL_PRIMARY, settings.GROQ_MODEL_FALLBACK):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=max_tokens,
                messages=full_messages, temperature=0.7
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if _is_rate_limit_error(err_str):
                if model == settings.GROQ_MODEL_FALLBACK:
                    retry = _parse_retry_after(err_str)
                    raise RateLimitError(err_str, retry_after=retry or 60)
                continue
            raise

# ── Async Chat (used by FastAPI routes) ───────────────────────

async def groq_chat(system: str, messages: list[dict], max_tokens: int = 2048) -> str:
    """Runs sync Groq call in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(groq_chat_sync, system, messages, max_tokens)