"""Tests for tools/wolfram_alpha.py — the query_wolfram_alpha tool."""

import json

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


def test_success_returns_json(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text="4"))
    fn = tool_fns["query_wolfram_alpha"]
    out = json.loads(run(fn(query="2+2")))
    assert out["query"] == "2+2"
    # A bare, sectionless answer is preserved under data.
    assert out["data"] == {"answer": "4"}


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
    first = run(fn(query="meaning of life"))
    second = run(fn(query="meaning of life"))
    assert json.loads(first)["data"] == {"answer": "42"}
    assert first == second  # identical serialized payload
    assert calls["n"] == 1  # second call served from cache


# --------------------------- response structuring ---------------------------

# A trimmed but real-shaped LLM API body: echoed query, an image-only section, a
# data table, and the trailing permalink.
_FRANCE_BODY = """Query:
"France population"

Input interpretation:
France | population

Result:
66.4 million people (2023 estimate)

Recent population history:
image: https://example.com/PNG_x.png

Wolfram|Alpha website result for "France population":
https://www.wolframalpha.com/input?i=France+population"""

# An ambiguous query: Wolfram picks an interpretation and lists alternatives.
_MERCURY_ASSUMPTION = """Query:
"mercury"

Assumption:
Assuming "mercury" is a chemical element
To use as a planet set assumption=*C.mercury-_*Planet-
To use as a word set assumption=*C.mercury-_*Word-

Input interpretation:
mercury (chemical element)"""


def test_structuring_drops_boilerplate_and_lifts_url(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text=_FRANCE_BODY))
    out = json.loads(run(tool_fns["query_wolfram_alpha"](query="France population")))
    # Echoed query and the image-only section are gone; the answer/data remain.
    assert "Query" not in out["data"]
    assert "Recent population history" not in out["data"]
    assert out["data"]["Result"] == "66.4 million people (2023 estimate)"
    # The permalink is lifted into its own field, not left in data.
    assert out["url"] == "https://www.wolframalpha.com/input?i=France+population"
    assert "assumptions" not in out  # unambiguous query


def test_structuring_extracts_assumptions(monkeypatch, patch_httpx, tool_fns):
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text=_MERCURY_ASSUMPTION))
    out = json.loads(run(tool_fns["query_wolfram_alpha"](query="mercury")))
    assert out["assumptions"]["used"] == "a chemical element"
    alts = out["assumptions"]["alternatives"]
    assert {"description": "a planet", "assumption": "*C.mercury-_*Planet-"} in alts
    # The verbose "To use as …" lines don't leak into the answer data.
    assert all("set assumption=" not in v for v in out["data"].values())


# An Assumption block with a chosen interpretation but NO "To use as …"
# alternatives — Wolfram emits this even for unambiguous queries.
_NO_ALTERNATIVES_BODY = """Query:
"2 + 2"

Assumption:
Assuming "2 + 2" is a sum

Result:
4"""


def test_assumptions_omitted_when_no_alternatives(monkeypatch, patch_httpx, tool_fns):
    # A used-only assumption block (no alternatives to retry with) is non-actionable
    # bloat, so the field must be omitted entirely for unambiguous queries.
    _set_appid(monkeypatch)
    patch_httpx(lambda req: httpx.Response(200, text=_NO_ALTERNATIVES_BODY))
    out = json.loads(run(tool_fns["query_wolfram_alpha"](query="2 + 2")))
    assert "assumptions" not in out
    assert out["data"]["Result"] == "4"
