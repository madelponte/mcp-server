"""Tests for tools/web_search.py — count clamping, SearXNG query, enrichment."""

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

import tools.web_search as ws
from tools.web_search import (
    _category_includes_videos,
    _clamp_count,
    _enrich_result,
    _is_youtube_result_url,
    _SearXNGRequestQueue,
    _searxng_query,
    _youtube_engine_query,
)
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


def test_category_includes_videos():
    assert _category_includes_videos("general,videos") is True
    assert _category_includes_videos("Videos") is True
    assert _category_includes_videos("general,images") is False


def test_youtube_engine_query():
    assert _youtube_engine_query("python talks") == "!yt python talks"
    assert _youtube_engine_query("!yt python talks") == "!yt python talks"
    assert _youtube_engine_query("!youtube python talks") == "!youtube python talks"


def test_is_youtube_result_url():
    assert _is_youtube_result_url("https://www.youtube.com/watch?v=abc12345678") is True
    assert _is_youtube_result_url("https://youtu.be/abc12345678") is True
    assert _is_youtube_result_url("https://notyoutube.com/watch?v=abc12345678") is False
    assert _is_youtube_result_url("https://peer.tube/w/abc") is False


# --------------------------- SearXNG request queue ---------------------------

def test_searxng_request_queue_delays_concurrent_calls_after_completion(monkeypatch):
    """Queued requests leave the configured quiet period after the prior
    response completes, not merely between request start times."""
    import anyio as _anyio

    clock = [1000.0]
    starts = []
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(ws.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ws.anyio, "sleep", fake_sleep)
    queue = _SearXNGRequestQueue()

    async def worker():
        async with queue.request_slot(1.0):
            starts.append(clock[0])
            # Simulate time spent awaiting the SearXNG response.
            clock[0] += 0.25

    async def main():
        async with _anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(worker)

    run(main())

    assert sleeps == [1.0, 1.0]
    assert starts == [1000.0, 1001.25, 1002.5]


def test_searxng_request_queue_prevents_overlapping_requests():
    import anyio as _anyio

    queue = _SearXNGRequestQueue()
    active = 0
    max_active = 0

    async def worker():
        nonlocal active, max_active
        async with queue.request_slot(0.001):
            active += 1
            max_active = max(max_active, active)
            await _anyio.sleep(0.005)
            active -= 1

    async def main():
        async with _anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(worker)

    run(main())
    assert max_active == 1


def test_searxng_request_queue_is_disabled_when_delay_is_zero(monkeypatch):
    slept = []
    monkeypatch.setattr(ws.anyio, "sleep", lambda seconds: slept.append(seconds))
    queue = _SearXNGRequestQueue()

    async def main():
        async with queue.request_slot(0):
            pass

    run(main())
    assert slept == []


# --------------------------- _searxng_query (async, mocked) ---------------------------

def test_searxng_query_parses_results(monkeypatch, patch_httpx):
    queued_delays = []

    class RecordingQueue:
        def request_slot(self, delay_seconds):
            queued_delays.append(delay_seconds)

            class Slot:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, *args):
                    return None

            return Slot()

    monkeypatch.setattr(ws, "_searxng_request_queue", RecordingQueue())
    payload = {"results": [
        {"url": "https://a.com", "title": "A", "content": " snippet a ", "publishedDate": "2023-01-01"},
        {"url": "https://b.com", "title": "B", "content": "snippet b"},
    ]}
    patch_httpx(lambda req: httpx.Response(200, json=payload))
    out = run(_searxng_query(
        "http://searxng:8080", "test",
        num_results=5, categories="general", language="en",
        time_range="", safe_search=0, timeout=5, verify_ssl=True, user_agent="t",
        request_delay_seconds=1.5,
    ))
    assert queued_delays == [1.5]
    assert len(out) == 2
    assert out[0]["url"] == "https://a.com"
    assert out[0]["snippet"] == "snippet a"  # stripped
    assert out[0]["published_date"] == "2023-01-01"
    assert "published_date" not in out[1]


@pytest.mark.parametrize(
    "operator_query",
    [
        pytest.param("site:example.com python", id="site"),
        pytest.param('"exact phrase" python', id="exact-phrase"),
        pytest.param("python -exclude", id="exclude"),
        pytest.param("python OR rust", id="or"),
        pytest.param("python filetype:pdf", id="filetype"),
        pytest.param("intitle:python guide", id="intitle"),
        pytest.param("inurl:docs python", id="inurl"),
    ],
)
def test_searxng_query_preserves_common_operators(patch_httpx, operator_query):
    seen_queries = []

    def handler(request):
        seen_queries.append(request.url.params["q"])
        return httpx.Response(200, json={"results": []})

    patch_httpx(handler)
    run(_searxng_query(
        "http://searxng:8080", operator_query,
        num_results=5, categories="general", language="en",
        time_range="", safe_search=0, timeout=5, verify_ssl=True, user_agent="t",
    ))
    assert seen_queries == [operator_query]


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


def test_searxng_query_reuses_client_within_event_loop(monkeypatch):
    created = 0

    def handler(request):
        return httpx.Response(200, json={"results": []})

    class CountingClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            nonlocal created
            created += 1
            kwargs.pop("verify", None)
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", CountingClient)
    ws._searxng_clients.clear()

    async def scenario():
        for _ in range(2):
            await _searxng_query(
                "http://searxng:8080", "test",
                num_results=5, categories="general", language="en",
                time_range="", safe_search=0, timeout=5,
                verify_ssl=True, user_agent="t",
            )

    run(scenario())
    assert created == 1


# --------------------------- _enrich_result (async, mocked) ---------------------------

def test_enrich_result_none_url():
    assert run(_enrich_result(None)) is None


def test_enrich_result_html(monkeypatch):
    async def fake_fetch(url):
        return {"content_type": "text/html",
                "text": "<title>My Page</title><meta name=description content='D'><h1>H</h1>"}

    monkeypatch.setattr(ws, "_enrich_fetch", fake_fetch)
    out = run(_enrich_result("https://e.com"))
    assert out["title"] == "My Page"
    assert out["description"] == "D"


def test_enrich_result_tika_document(monkeypatch):
    async def fake_fetch(url):
        return {"content_type": "application/pdf", "text": None}

    monkeypatch.setattr(ws, "_enrich_fetch", fake_fetch)
    out = run(_enrich_result("https://e.com/file.pdf"))
    assert out == {"title": None, "description": None, "headings": [], "toc": None}


def test_enrich_result_no_text_returns_none(monkeypatch):
    async def fake_fetch(url):
        return {"content_type": "text/html", "text": ""}

    monkeypatch.setattr(ws, "_enrich_fetch", fake_fetch)
    assert run(_enrich_result("https://e.com")) is None


def test_enrich_result_fetch_error_returns_error_dict(monkeypatch):
    async def fake_fetch(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(ws, "_enrich_fetch", fake_fetch)
    out = run(_enrich_result("https://e.com"))
    assert "error" in out


def test_enrich_result_fetch_none_returns_none(monkeypatch):
    # A page too big to enrich (or a blocked redirect) yields None, not metadata.
    async def fake_fetch(url):
        return None

    monkeypatch.setattr(ws, "_enrich_fetch", fake_fetch)
    assert run(_enrich_result("https://e.com")) is None


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
    seen = {}

    async def fake_query(**kwargs):
        seen.update(kwargs)
        return [{"url": "https://a.com", "title": "A", "snippet": "s"}]

    monkeypatch.setattr(ws, "_searxng_query", fake_query)
    fn = tool_fns["search_web"]
    out = json.loads(run(fn(query="test", enrich_results=0)))
    assert out["query"] == "test"
    assert seen["request_delay_seconds"] == ws.cfg.searxng_request_delay_seconds
    assert out["results"][0]["url"] == "https://a.com"
    # No enrichment requested -> no page metadata fields.
    assert "page_title" not in out["results"][0]


def test_search_web_videos_category_restricts_query_to_youtube(monkeypatch, tool_fns):
    seen = {}

    async def fake_query(**kwargs):
        seen.update(kwargs)
        return [
            {"url": "https://www.youtube.com/watch?v=abc12345678", "title": "A", "snippet": "s"},
            {"url": "https://peer.tube/w/abc", "title": "B", "snippet": "s"},
        ]

    monkeypatch.setattr(ws, "_searxng_query", fake_query)
    fn = tool_fns["search_web"]
    out = json.loads(run(fn(query="python talks", category="videos", enrich_results=0)))
    assert seen["query"] == "!yt python talks"
    assert out["query"] == "python talks"
    assert out["category"] == "videos"
    assert [r["url"] for r in out["results"]] == ["https://www.youtube.com/watch?v=abc12345678"]


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


def test_search_web_enrich_error_surfaces_as_page_meta_error(monkeypatch, tool_fns):
    # `_enrich_result` reports a fetch failure as {"error": ...}; the tool should
    # surface it as page_meta_error rather than emitting null page_title/description.
    monkeypatch.setattr(ws.cfg, "max_enrich_results", 5)

    async def fake_query(**kwargs):
        return [{"url": "https://a.com", "title": "A", "snippet": "s"}]

    async def fake_enrich(url):
        return {"error": "boom"}

    monkeypatch.setattr(ws, "_searxng_query", fake_query)
    monkeypatch.setattr(ws, "_enrich_result", fake_enrich)
    fn = tool_fns["search_web"]
    out = json.loads(run(fn(query="test", enrich_results=1)))
    assert out["results"][0]["page_meta_error"] == "boom"
    assert "page_title" not in out["results"][0]
    assert "page_description" not in out["results"][0]


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
