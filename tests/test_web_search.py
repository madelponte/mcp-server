"""Tests for tools/web_search.py — count clamping, SearXNG query, enrichment."""

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

import tools.web_search as ws
from tools.web_search import _clamp_count, _searxng_query, _enrich_result
from conftest import run


# --------------------------- _clamp_count ---------------------------

def test_clamp_count_none_uses_default():
    assert _clamp_count(None, 5, minimum=0, default=3) == 3


def test_clamp_count_none_no_default_uses_max():
    assert _clamp_count(None, 5, minimum=1) == 5


def test_clamp_count_within_range():
    assert _clamp_count(2, 5, minimum=1) == 2


def test_clamp_count_above_max():
    assert _clamp_count(100, 5, minimum=1) == 5


def test_clamp_count_below_minimum():
    assert _clamp_count(-1, 5, minimum=0) == 0
    assert _clamp_count(0, 5, minimum=1) == 1


def test_clamp_count_invalid_uses_max():
    assert _clamp_count("abc", 5, minimum=1) == 5


# --------------------------- _searxng_query (async, mocked) ---------------------------

def test_searxng_query_parses_results(patch_httpx):
    payload = {"results": [
        {"url": "https://a.com", "title": "A", "content": " snippet a ", "publishedDate": "2023-01-01"},
        {"url": "https://b.com", "title": "B", "content": "snippet b"},
    ]}
    patch_httpx(lambda req: httpx.Response(200, json=payload))
    out = run(_searxng_query(
        "http://searxng:8080", "test",
        num_results=5, categories="general", language="en",
        time_range="", safe_search=0, timeout=5, verify_ssl=True, user_agent="t",
    ))
    assert len(out) == 2
    assert out[0]["url"] == "https://a.com"
    assert out[0]["snippet"] == "snippet a"  # stripped
    assert out[0]["published_date"] == "2023-01-01"
    assert "published_date" not in out[1]


def test_searxng_query_respects_num_results(patch_httpx):
    payload = {"results": [{"url": f"https://{i}.com", "title": str(i), "content": ""} for i in range(10)]}
    patch_httpx(lambda req: httpx.Response(200, json=payload))
    out = run(_searxng_query(
        "http://searxng:8080", "test",
        num_results=3, categories="general", language="en",
        time_range="", safe_search=0, timeout=5, verify_ssl=True, user_agent="t",
    ))
    assert len(out) == 3


def test_searxng_query_403_raises_runtime_error(patch_httpx):
    patch_httpx(lambda req: httpx.Response(403, text="forbidden"))
    with pytest.raises(RuntimeError) as exc:
        run(_searxng_query(
            "http://searxng:8080", "test",
            num_results=5, categories="general", language="en",
            time_range="", safe_search=0, timeout=5, verify_ssl=True, user_agent="t",
        ))
    assert "json" in str(exc.value).lower()


def test_searxng_query_empty_results(patch_httpx):
    patch_httpx(lambda req: httpx.Response(200, json={}))
    out = run(_searxng_query(
        "http://searxng:8080", "test",
        num_results=5, categories="general", language="en",
        time_range="", safe_search=0, timeout=5, verify_ssl=True, user_agent="t",
    ))
    assert out == []


# --------------------------- _enrich_result (async, mocked) ---------------------------

def test_enrich_result_none_url():
    assert run(_enrich_result(None)) is None


def test_enrich_result_html(monkeypatch):
    async def fake_fetch(url):
        return {"content_type": "text/html",
                "text": "<title>My Page</title><meta name=description content='D'><h1>H</h1>"}

    monkeypatch.setattr(ws, "_cached_resilient_fetch", fake_fetch)
    out = run(_enrich_result("https://e.com"))
    assert out["title"] == "My Page"
    assert out["description"] == "D"


def test_enrich_result_tika_document(monkeypatch):
    async def fake_fetch(url):
        return {"content_type": "application/pdf", "text": None}

    monkeypatch.setattr(ws, "_cached_resilient_fetch", fake_fetch)
    out = run(_enrich_result("https://e.com/file.pdf"))
    assert out == {"title": None, "description": None, "headings": [], "toc": None}


def test_enrich_result_no_text_returns_none(monkeypatch):
    async def fake_fetch(url):
        return {"content_type": "text/html", "text": ""}

    monkeypatch.setattr(ws, "_cached_resilient_fetch", fake_fetch)
    assert run(_enrich_result("https://e.com")) is None


def test_enrich_result_fetch_error_returns_error_dict(monkeypatch):
    async def fake_fetch(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(ws, "_cached_resilient_fetch", fake_fetch)
    out = run(_enrich_result("https://e.com"))
    assert "error" in out


# --------------------------- search_web tool ---------------------------

def test_search_web_empty_query_raises(tool_fns):
    fn = tool_fns["search_web"]
    with pytest.raises(ToolError):
        run(fn(query="  "))


def test_search_web_invalid_time_range_raises(tool_fns):
    fn = tool_fns["search_web"]
    with pytest.raises(ToolError) as exc:
        run(fn(query="test", time_range="fortnight"))
    assert "time_range" in str(exc.value)


def test_search_web_happy_path_no_enrich(monkeypatch, tool_fns):
    async def fake_query(**kwargs):
        return [{"url": "https://a.com", "title": "A", "snippet": "s"}]

    monkeypatch.setattr(ws, "_searxng_query", fake_query)
    fn = tool_fns["search_web"]
    out = json.loads(run(fn(query="test", enrich_results=0)))
    assert out["query"] == "test"
    assert out["results"][0]["url"] == "https://a.com"
    # No enrichment requested -> no page metadata fields.
    assert "page_title" not in out["results"][0]


def test_search_web_enriches_top_results(monkeypatch, tool_fns):
    # Ensure enrichment is enabled regardless of the configured cap in .env.
    monkeypatch.setattr(ws.cfg, "max_enrich_results", 5)

    async def fake_query(**kwargs):
        return [{"url": "https://a.com", "title": "A", "snippet": "s"}]

    async def fake_enrich(url):
        return {"title": "Enriched", "description": "desc", "headings": [{"level": 1, "text": "H"}], "toc": None}

    monkeypatch.setattr(ws, "_searxng_query", fake_query)
    monkeypatch.setattr(ws, "_enrich_result", fake_enrich)
    fn = tool_fns["search_web"]
    out = json.loads(run(fn(query="test", enrich_results=1)))
    assert out["results"][0]["page_title"] == "Enriched"
    assert out["results"][0]["page_description"] == "desc"


def test_search_web_no_results(monkeypatch, tool_fns):
    async def fake_query(**kwargs):
        return []

    monkeypatch.setattr(ws, "_searxng_query", fake_query)
    fn = tool_fns["search_web"]
    out = json.loads(run(fn(query="nothing")))
    assert out["results"] == []


def test_search_web_query_failure_raises_toolerror(monkeypatch, tool_fns):
    async def fake_query(**kwargs):
        raise RuntimeError("searxng down")

    monkeypatch.setattr(ws, "_searxng_query", fake_query)
    fn = tool_fns["search_web"]
    with pytest.raises(ToolError):
        run(fn(query="test"))
