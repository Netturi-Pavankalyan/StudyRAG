"""LRU cache for extracted PDF text & TTL set for cancelled tasks."""

import time
from collections import OrderedDict
from typing import Any

from app.config import settings


class LRUCache:
    """Size-bounded LRU cache (thread-safe under GIL)."""

    def __init__(self, maxsize: int = settings.PDF_CACHE_MAXSIZE) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return default

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def delete(self, key: str) -> None:
        self._cache.pop(key, None)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class TTLSet:
    """Auto-expiring set — entries removed after *ttl* seconds."""

    def __init__(self, ttl: int = settings.CANCELLED_TASKS_TTL) -> None:
        self._store: dict[str, float] = {}
        self._ttl = ttl

    def add(self, key: str) -> None:
        self._store[key] = time.time()

    def __contains__(self, key: str) -> bool:
        if key in self._store and (time.time() - self._store[key]) < self._ttl:
            return True
        self._store.pop(key, None)
        return False

    def discard(self, key: str) -> None:
        self._store.pop(key, None)

    def cleanup(self) -> None:
        now = time.time()
        expired = [k for k, t in self._store.items() if now - t >= self._ttl]
        for k in expired:
            del self._store[k]


# ── Global singletons ──────────────────────────────────────────

pdf_text_cache = LRUCache()
cancelled_tasks = TTLSet()