"""
Tiny in-memory TTL cache shared by the tools.

Originally only the Stock Data tool cached responses; this generalizes that same
dict-with-timestamp approach so the web-page fetch and YouTube transcript tools
can reuse it. Deliberately dependency-free and process-local (no Redis, no
persistence) — it just spares an agent loop from re-fetching the same URL or
transcript repeatedly within a task.

Semantics match the original Stock cache: a ``ttl_seconds <= 0`` disables caching
entirely. ``max_entries`` is new — it bounds memory by evicting the oldest entry
once the cache is full (``0`` means unbounded, preserving the old behavior).

Concurrency: like the code it replaces, this is lock-free. Each ``get``/``set`` is
a sequence of individual dict operations (no iteration), which is safe under
CPython's GIL for our usage — async callers run single-threaded, and the Stock
tool's threaded callers leave eviction disabled (``max_entries=0``). The worst
case is a redundant fetch on a cache miss race, never corruption.
"""

import time
from collections import OrderedDict
from typing import Any, Optional


class TTLCache:
    """A minimal time-to-live cache keyed by string."""

    def __init__(self, ttl_seconds: float, max_entries: int = 0) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value, or None if absent, expired, or caching is off."""
        if self.ttl <= 0:
            return None
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > self.ttl:
            self._data.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with the current timestamp (no-op when caching is off)."""
        if self.ttl <= 0:
            return
        self._data[key] = (time.time(), value)
        if self.max_entries > 0:
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # evict the oldest entry

    def clear(self) -> None:
        self._data.clear()
