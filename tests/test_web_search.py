"""Tests for Brave-backed tools/web_search.py and optional enrichment."""

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

import tools.web_search as ws
from tools.web_search import (
    _BraveRequestQueue,
    _brave_query,
    _clamp_count,
    _enrich_result,
    _resolve_country,
    _resolve_optional_choice,
    _resolve_search_lang,
    _resolve_time_range,
)
from conftest import run


@pytest.fixture(autouse=True)
def _pin_brave_defaults(monkeypatch):
    """Keep provider tests independent of the developer's .env."""
    monkeypatch.setattr(ws.cfg, "brave_api_key", "brave-test-key")
    monkeypatch.setattr(
        ws.cfg,
        "brave_api_url",
        "https://api.search.brave.com/res/v1/llm/context",
    )
    monkeypatch.setattr(ws.cfg, "brave_country", "US")
    monkeypatch.setattr(ws.cfg, "brave_search_lang", "en")
    monkeypatch.setattr(ws.cfg, "brave_freshness", "")
    monkeypatch.setattr(ws.cfg, "brave_safesearch", "")
    monkeypatch.setattr(ws.cfg, "brave_context_threshold_mode", "")


# --------------------------- pure validation ---------------------------


def test_clamp_count_uses_default_or_max_and_clamps():
    assert _clamp_count(None, 5, minimum=0, default=3) == 3
    assert _clamp_count(None, 5, minimum=1) == 5
    assert _clamp_count(2, 5, minimum=1) == 2
    assert _clamp_count(100, 5, minimum=1) == 5
    assert _clamp_count(-1, 5, minimum=0) == 0
    assert _clamp_count(0, 5, minimum=1) == 1
    assert _clamp_count("abc", 5, minimum=1) == 5


@pytest.mark.parametrize(
    ("value", "model_value", "freshness"),
    [
        (None, "all", ""),
        ("all", "all", ""),
        ("day", "day", "pd"),
        ("week", "week", "pw"),
        ("pm", "month", "pm"),
        ("year", "year", "py"),
        ("2024-01-01to2024-02-29", "2024-01-01to2024-02-29", "2024-01-01to2024-02-29"),
    ],
)
def test_resolve_time_range(value, model_value, freshness):
    assert _resolve_time_range(value, "") == (model_value, freshness)


@pytest.mark.parametrize(
    "value",
    ["fortnight", "2024-02-30to2024-03-01", "2024-03-01to2024-02-01"],
)
def test_resolve_time_range_rejects_invalid_values(value):
    with pytest.raises(ToolError):
        _resolve_time_range(value, "")


def test_country_and_language_validation():
    assert _resolve_country(None, "us") == "US"
    assert _resolve_country("gb", "US") == "GB"
    assert _resolve_search_lang(None, "EN") == "en"
    assert _resolve_search_lang("zh-hans", "en") == "zh-hans"
    with pytest.raises(ToolError):
        _resolve_country("USA", "US")
    with pytest.raises(ToolError):
        _resolve_country("ALL", "US")
    with pytest.raises(ToolError):
        _resolve_search_lang("e", "en")


def test_optional_choice_uses_brave_default_when_blank():
    assert _resolve_optional_choice(None, "", {"strict"}, "mode") == ""
    assert _resolve_optional_choice("auto", "strict", {"strict"}, "mode") == ""
    assert _resolve_optional_choice(None, "strict", {"strict"}, "mode") == "strict"
    with pytest.raises(ToolError):
        _resolve_optional_choice("loose", "", {"strict"}, "mode")


# --------------------------- Brave request queue ---------------------------


def test_brave_request_queue_spaces_concurrent_calls_after_completion(monkeypatch):
    import anyio as _anyio

    clock = [1000.0]
    starts = []
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(ws.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ws.anyio, "sleep", fake_sleep)
    queue = _BraveRequestQueue()

    async def worker():
        async with queue.request_slot(1.0):
            starts.append(clock[0])
            clock[0] += 0.25

    async def main():
        async with _anyio.create_task_group() as task_group:
            for _ in range(3):
                task_group.start_soon(worker)

    run(main())
    assert sleeps == [1.0, 1.0]
    assert starts == [1000.0, 1001.25, 1002.5]


def test_brave_request_queue_zero_delay_allows_overlap():
    import anyio as _anyio

    queue = _BraveRequestQueue()
    active = 0
    max_active = 0

    async def worker():
        nonlocal active, max_active
        async with queue.request_slot(0):
            active += 1
            max_active = max(max_active, active)
            await _anyio.sleep(0.005)
            active -= 1

    async def main():
        async with _anyio.create_task_group() as task_group:
            for _ in range(3):
                task_group.start_soon(worker)

    run(main())
    assert max_active == 3


# --------------------------- Brave API query ---------------------------


def _brave_response():
    return {
        "grounding": {
            "generic": [
                {
                    "url": "https://a.example/page",
                    "title": "A title",
                    "snippets": ["First excerpt", '{"table":[1,2]}'],
                },
                {
                    "url": "https://b.example/page",
                    "title": "B title",
                    "snippets": ["Second excerpt"],
                },
            ],
            "map": [],
        },
        "sources": {
            "https://a.example/page": {
                "title": "Source A",
                "hostname": "a.example",
                "age": [
                    "Wednesday, January 15, 2025",
                    "2025-01-15",
                    "392 days ago",
                    "2025-01-15T13:45:02Z",
                ],
                "description": "Page description",
                "site_name": "Example A",
            }
        },
    }


def _query_kwargs(**overrides):
    values = {
        "api_url": "https://api.search.brave.com/res/v1/llm/context",
        "api_key": "secret-brave-key",
        "query": "test query",
        "num_results": 2,
        "country": "US",
        "search_lang": "en",
        "freshness": "pw",
        "safesearch": "moderate",
        "context_threshold_mode": "balanced",
        "max_tokens": 4096,
        "search_count": 20,
        "max_tokens_per_url": 2048,
        "timeout": 30,
        "verify_ssl": True,
        "user_agent": "test-agent",
        "request_delay_seconds": 0,
        "max_retries": 0,
        "retry_backoff_seconds": 1,
    }
    values.update(overrides)
    return values


def test_brave_query_posts_native_context_parameters_and_parses_sources(patch_httpx):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["token"] = request.headers["X-Subscription-Token"]
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_brave_response())

    patch_httpx(handler)
    out = run(_brave_query(**_query_kwargs()))

    assert seen["url"] == "https://api.search.brave.com/res/v1/llm/context"
    assert seen["token"] == "secret-brave-key"
    assert seen["payload"] == {
        "q": "test query",
        "country": "US",
        "search_lang": "en",
        "count": 20,
        "maximum_number_of_urls": 2,
        "maximum_number_of_tokens": 4096,
        "maximum_number_of_tokens_per_url": 2048,
        "enable_source_metadata": True,
        "freshness": "pw",
        "safesearch": "moderate",
        "context_threshold_mode": "balanced",
    }
    assert out[0] == {
        "url": "https://a.example/page",
        "title": "A title",
        "snippets": ["First excerpt", '{"table":[1,2]}'],
        "published_date": "2025-01-15T13:45:02Z",
        "description": "Page description",
        "site_name": "Example A",
    }


@pytest.mark.parametrize(
    "operator_query",
    [
        "site:example.com python",
        '"exact phrase" python',
        "python -exclude",
        "python OR rust",
        "python filetype:pdf",
        "intitle:python guide",
        "inbody:asyncio python",
        "python lang:en loc:us",
    ],
)
def test_brave_query_preserves_operators(patch_httpx, operator_query):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["q"])
        return httpx.Response(200, json={"grounding": {"generic": []}, "sources": {}})

    patch_httpx(handler)
    run(_brave_query(**_query_kwargs(query=operator_query)))
    assert seen == [operator_query]


def test_brave_query_omits_optional_filters_and_bounds_per_url_tokens(patch_httpx):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"grounding": {"generic": []}, "sources": {}})

    patch_httpx(handler)
    run(
        _brave_query(
            **_query_kwargs(
                freshness="",
                safesearch="",
                context_threshold_mode="",
                max_tokens=1024,
                max_tokens_per_url=4096,
            )
        )
    )
    assert "freshness" not in seen
    assert "safesearch" not in seen
    assert "context_threshold_mode" not in seen
    assert seen["maximum_number_of_tokens_per_url"] == 1024


def test_brave_query_limits_and_deduplicates_results(patch_httpx):
    payload = _brave_response()
    payload["grounding"]["generic"].insert(1, payload["grounding"]["generic"][0])
    patch_httpx(lambda request: httpx.Response(200, json=payload))
    out = run(_brave_query(**_query_kwargs(num_results=1)))
    assert [item["url"] for item in out] == ["https://a.example/page"]


def test_brave_query_retries_429_using_exhausted_rate_window_reset(
    monkeypatch, patch_httpx
):
    calls = 0
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={
                    "X-RateLimit-Remaining": "0, 1000",
                    "X-RateLimit-Reset": "1.25, 1419704",
                },
                json={"error": {"detail": "burst limit"}},
            )
        return httpx.Response(200, json=_brave_response())

    monkeypatch.setattr(ws.anyio, "sleep", fake_sleep)
    patch_httpx(handler)
    out = run(_brave_query(**_query_kwargs(max_retries=1)))
    assert calls == 2
    assert sleeps == [1.25]
    assert out[0]["url"] == "https://a.example/page"


def test_brave_query_honors_retry_after_and_exponential_backoff(
    monkeypatch, patch_httpx
):
    calls = 0
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, headers={"Retry-After": "1.5"}, text="busy")
        if calls == 2:
            return httpx.Response(504, text="timeout")
        return httpx.Response(200, json=_brave_response())

    monkeypatch.setattr(ws.anyio, "sleep", fake_sleep)
    patch_httpx(handler)
    run(
        _brave_query(
            **_query_kwargs(
                max_retries=2,
                retry_backoff_seconds=0.5,
            )
        )
    )
    assert calls == 3
    assert sleeps == [1.5, 1.0]


def test_brave_query_retries_transient_transport_error(monkeypatch, patch_httpx):
    calls = 0
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary connect failure", request=request)
        return httpx.Response(200, json=_brave_response())

    monkeypatch.setattr(ws.anyio, "sleep", fake_sleep)
    patch_httpx(handler)
    run(
        _brave_query(
            **_query_kwargs(max_retries=1, retry_backoff_seconds=0.25)
        )
    )
    assert calls == 2
    assert sleeps == [0.25]


def test_brave_query_logs_remaining_rate_limit_at_debug(caplog, patch_httpx):
    patch_httpx(
        lambda request: httpx.Response(
            200,
            headers={
                "X-RateLimit-Remaining": "0, 999",
                "X-RateLimit-Limit": "1, 15000",
                "X-RateLimit-Reset": "1, 1000",
            },
            json=_brave_response(),
        )
    )
    with caplog.at_level("DEBUG", logger=ws.__name__):
        run(_brave_query(**_query_kwargs()))
    assert "remaining=0, 999" in caplog.text
    assert "limit=1, 15000" in caplog.text


def test_brave_query_surfaces_api_error(patch_httpx):
    patch_httpx(
        lambda request: httpx.Response(
            429,
            json={"type": "ErrorResponse", "error": {"detail": "rate limit exceeded"}},
        )
    )
    with pytest.raises(RuntimeError) as exc:
        run(_brave_query(**_query_kwargs()))
    assert "HTTP 429" in str(exc.value)
    assert "rate limit exceeded" in str(exc.value)


def test_brave_query_non_json_error_keeps_status_and_body_excerpt(patch_httpx):
    patch_httpx(
        lambda request: httpx.Response(
            502,
            text="<html><body>upstream CDN unavailable</body></html>",
        )
    )
    with pytest.raises(RuntimeError) as exc:
        run(_brave_query(**_query_kwargs()))
    message = str(exc.value)
    assert "HTTP 502" in message
    assert "upstream CDN unavailable" in message
    assert "non-JSON" in message


def test_brave_query_rejects_malformed_success(patch_httpx):
    patch_httpx(lambda request: httpx.Response(200, json={"sources": {}}))
    with pytest.raises(RuntimeError, match="grounding"):
        run(_brave_query(**_query_kwargs()))


def test_brave_query_reuses_client_within_event_loop(monkeypatch):
    created = 0

    def handler(request):
        return httpx.Response(200, json={"grounding": {"generic": []}, "sources": {}})

    class CountingClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            nonlocal created
            created += 1
            kwargs.pop("verify", None)
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", CountingClient)
    ws._brave_clients.clear()

    async def scenario():
        await _brave_query(**_query_kwargs())
        await _brave_query(**_query_kwargs(query="another query"))

    run(scenario())
    assert created == 1


# --------------------------- enrichment ---------------------------


def test_enrich_result_none_url():
    assert run(_enrich_result(None)) is None


def test_enrich_result_html(monkeypatch):
    async def fake_fetch(url):
        return {
            "content_type": "text/html",
            "text": "<title>My Page</title><meta name=description content='D'><h1>H</h1>",
        }

    monkeypatch.setattr(ws, "_enrich_fetch", fake_fetch)
    out = run(_enrich_result("https://e.com"))
    assert out["title"] == "My Page"
    assert out["description"] == "D"


def test_enrich_result_document_and_failures(monkeypatch):
    async def document(url):
        return {"content_type": "application/pdf", "text": None}

    monkeypatch.setattr(ws, "_enrich_fetch", document)
    assert run(_enrich_result("https://e.com/file.pdf")) == {
        "title": None,
        "description": None,
        "headings": [],
        "toc": None,
    }

    async def failure(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(ws, "_enrich_fetch", failure)
    assert "error" in run(_enrich_result("https://e.com"))


# --------------------------- search_web tool ---------------------------


def test_search_web_empty_or_oversized_query_raises(tool_fns):
    fn = tool_fns["search_web"]
    with pytest.raises(ToolError):
        run(fn(query="  "))
    with pytest.raises(ToolError, match="400-character"):
        run(fn(query="x" * 401))
    with pytest.raises(ToolError, match="50-word"):
        run(fn(query=" ".join(["x"] * 51)))


def test_search_web_requires_brave_key(monkeypatch, tool_fns):
    monkeypatch.setattr(ws.cfg, "brave_api_key", "")
    with pytest.raises(ToolError, match="WEB_SEARCH_BRAVE_API_KEY"):
        run(tool_fns["search_web"](query="test"))


def test_search_web_invalid_native_options_raise(tool_fns):
    fn = tool_fns["search_web"]
    with pytest.raises(ToolError, match="time_range"):
        run(fn(query="test", time_range="fortnight"))
    with pytest.raises(ToolError, match="country"):
        run(fn(query="test", country="USA"))
    with pytest.raises(ToolError, match="safesearch"):
        run(fn(query="test", safesearch="maximum"))
    with pytest.raises(ToolError, match="context_threshold_mode"):
        run(fn(query="test", context_threshold_mode="maximum"))


def test_search_web_happy_path_maps_options_and_caps(monkeypatch, tool_fns):
    seen = {}

    async def fake_query(**kwargs):
        seen.update(kwargs)
        return [{"url": "https://a.com", "title": "A", "snippets": ["excerpt"]}]

    monkeypatch.setattr(ws, "_brave_query", fake_query)
    out = json.loads(
        run(
            tool_fns["search_web"](
                query="test",
                time_range="week",
                country="ca",
                search_lang="fr",
                safesearch="strict",
                context_threshold_mode="balanced",
                num_results=999,
                max_tokens=999999,
                enrich_results=0,
            )
        )
    )
    assert out["provider"] == "brave_llm_context"
    assert out["time_range"] == "week"
    assert out["country"] == "CA"
    assert out["search_lang"] == "fr"
    assert out["safesearch"] == "strict"
    assert out["context_threshold_mode"] == "balanced"
    assert seen["freshness"] == "pw"
    assert seen["num_results"] == ws.cfg.max_num_results
    assert seen["max_tokens"] == ws.cfg.max_context_tokens
    assert seen["request_delay_seconds"] == ws.cfg.brave_request_delay_seconds
    assert seen["max_retries"] == ws.cfg.brave_max_retries
    assert seen["retry_backoff_seconds"] == ws.cfg.brave_retry_backoff_seconds
    assert out["results"][0]["snippets"] == ["excerpt"]


def test_search_web_enriches_top_results(monkeypatch, tool_fns):
    monkeypatch.setattr(ws.cfg, "max_enrich_results", 5)

    async def fake_query(**kwargs):
        return [{"url": "https://a.com", "title": "A", "snippets": ["s"]}]

    async def fake_enrich(url):
        return {
            "title": "Enriched",
            "description": "desc",
            "headings": [{"level": 1, "text": "H"}],
            "toc": "toc",
        }

    monkeypatch.setattr(ws, "_brave_query", fake_query)
    monkeypatch.setattr(ws, "_enrich_result", fake_enrich)
    out = json.loads(run(tool_fns["search_web"](query="test", enrich_results=1)))
    result = out["results"][0]
    assert result["page_title"] == "Enriched"
    assert result["page_description"] == "desc"
    assert result["page_headings"][0]["text"] == "H"
    assert result["page_toc"] == "toc"


def test_search_web_enrich_error_surfaces_as_page_meta_error(monkeypatch, tool_fns):
    async def fake_query(**kwargs):
        return [{"url": "https://a.com", "title": "A", "snippets": ["s"]}]

    async def fake_enrich(url):
        return {"error": "boom"}

    monkeypatch.setattr(ws, "_brave_query", fake_query)
    monkeypatch.setattr(ws, "_enrich_result", fake_enrich)
    out = json.loads(run(tool_fns["search_web"](query="test", enrich_results=1)))
    assert out["results"][0]["page_meta_error"] == "boom"


def test_search_web_valid_empty_result_is_not_error(monkeypatch, tool_fns):
    async def fake_query(**kwargs):
        return []

    monkeypatch.setattr(ws, "_brave_query", fake_query)
    out = json.loads(run(tool_fns["search_web"](query="nothing")))
    assert out["provider"] == "brave_llm_context"
    assert out["results"] == []


def test_search_web_failure_raises_and_redacts_key(monkeypatch, tool_fns):
    async def fake_query(**kwargs):
        raise RuntimeError("rejected brave-test-key")

    monkeypatch.setattr(ws, "_brave_query", fake_query)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["search_web"](query="test"))
    assert "brave-test-key" not in str(exc.value)
    assert "REDACTED" in str(exc.value)
