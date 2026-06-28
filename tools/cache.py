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

Concurrency: callers include async tool handlers and the Stock tool's worker
threads, so accesses are guarded with a small re-entrant lock. The cache is tiny
and process-local; holding the lock only around dict bookkeeping keeps this
simple without affecting network-bound tool latency.
"""

import time
from collections import OrderedDict
from threading import RLock
from typing import Any


class TTLCache:
    """A minimal time-to-live cache keyed by string."""

    def __init__(self, ttl_seconds: float, max_entries: int = 0) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = RLock()

    def get(self, key: str) -> Any | None:
        """Return the cached value, or None if absent, expired, or caching is off.

        A successful read refreshes recency so bounded caches evict the least
        recently used entry, not merely the oldest inserted one.
        """
        if self.ttl <= 0:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.time() - ts > self.ttl:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Store a value with the current timestamp (no-op when caching is off)."""
        if self.ttl <= 0:
            return
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            if self.max_entries > 0:
                while len(self._data) > self.max_entries:
                    self._data.popitem(last=False)  # evict least-recently used

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
