"""Tests for tools/wolfram_alpha.py — the query_wolfram_alpha tool."""

import httpx
import pytest
from fastmcp.exceptions import ToolError

import tools.wolfram_alpha as wolfram
from conftest import run


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    from tools.cache import TTLCache
    monkeypatch.setattr(wolfram, "_result_cache", TTLCache(3600))


def _set_appid(monkeypatch, value="TESTID"):
    monkeypatch.setattr(wolfram.cfg, "app_id", value)


def test_not_configured_raises(monkeypatch, tool_fns):
    _set_appid(monkeypatch, "")
    fn = tool_fns["query_wolfram_alpha"]
    with pytest.raises(ToolError) as exc:
        run(fn(query="2+2"))
    assert "AppID" in str(exc.value)


def test_empty_query_raises(monkeypatch, tool_fns):
    _set_appid(monkeypatch)
    fn = tool_fns["query_wolfram_alpha"]
    with pytest.raises(ToolError):
        run(fn(query="   "))


def test_invalid_units_raises(monkeypatch, tool_fns):
    _set_appid(monkeypatch)
    fn = tool_fns["query_wolfram_alpha"]
    with pytest.raises(ToolError):
        run(fn(query="speed of light", units="imperial"))


def test_success_returns_body(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text="4"))
    fn = tool_fns["query_wolfram_alpha"]
    assert run(fn(query="2+2")) == "4"


def test_501_raises_with_suggestions(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(501, text="Did you mean: foo?"))
    fn = tool_fns["query_wolfram_alpha"]
    with pytest.raises(ToolError) as exc:
        run(fn(query="asdfqwer"))
    assert "could not interpret" in str(exc.value).lower()
    assert "foo" in str(exc.value)


def test_403_raises(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(403, text="invalid appid"))
    fn = tool_fns["query_wolfram_alpha"]
    with pytest.raises(ToolError) as exc:
        run(fn(query="2+2"))
    assert "403" in str(exc.value)


def test_400_raises(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(400, text="bad request"))
    fn = tool_fns["query_wolfram_alpha"]
    with pytest.raises(ToolError):
        run(fn(query="2+2"))


def test_empty_body_raises(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text="   "))
    fn = tool_fns["query_wolfram_alpha"]
    with pytest.raises(ToolError):
        run(fn(query="2+2"))


def test_result_is_cached(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, text="42")

    patch_httpx(handler)
    fn = tool_fns["query_wolfram_alpha"]
    assert run(fn(query="meaning of life")) == "42"
    assert run(fn(query="meaning of life")) == "42"
    assert calls["n"] == 1  # second call served from cache
