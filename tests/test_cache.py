"""Tests for tools/cache.py — the process-local TTL cache."""

import tools.cache as cache_mod
from tools.cache import TTLCache


def test_set_get_roundtrip():
    c = TTLCache(ttl_seconds=60)
    c.set("k", {"v": 1})
    assert c.get("k") == {"v": 1}


def test_missing_key_returns_none():
    c = TTLCache(ttl_seconds=60)
    assert c.get("nope") is None


def test_ttl_zero_disables_cache():
    c = TTLCache(ttl_seconds=0)
    c.set("k", "v")
    assert c.get("k") is None
    # And the value was never stored.
    assert "k" not in c._data


def test_negative_ttl_disables_cache():
    c = TTLCache(ttl_seconds=-5)
    c.set("k", "v")
    assert c.get("k") is None


def test_expiry(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now[0])
    c = TTLCache(ttl_seconds=10)
    c.set("k", "v")
    assert c.get("k") == "v"
    now[0] += 9  # still within TTL
    assert c.get("k") == "v"
    now[0] += 2  # now 11s elapsed, past the 10s TTL
    assert c.get("k") is None
    # Expired entry is evicted on read.
    assert "k" not in c._data


def test_set_prunes_expired_entries(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: now[0])
    c = TTLCache(ttl_seconds=10, max_entries=0)
    c.set("old", "value")
    now[0] += 11
    c.set("new", "value")
    assert "old" not in c._data
    assert c.get("new") == "value"


def test_max_entries_evicts_oldest():
    c = TTLCache(ttl_seconds=60, max_entries=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # should evict "a" (oldest)
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_max_entries_refreshes_on_read():
    c = TTLCache(ttl_seconds=60, max_entries=2)
    c.set("a", 1)
    c.set("b", 2)
    assert c.get("a") == 1
    c.set("c", 3)  # "b" is now least recently used.
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_max_entries_zero_is_unbounded():
    c = TTLCache(ttl_seconds=60, max_entries=0)
    for i in range(100):
        c.set(str(i), i)
    assert len(c._data) == 100
    assert c.get("0") == 0


def test_clear():
    c = TTLCache(ttl_seconds=60)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert c.get("a") is None
    assert len(c._data) == 0
