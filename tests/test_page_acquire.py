"""Tests for browser-first page acquisition and provider orchestration."""

import asyncio

import pytest

from conftest import run
from tools.cache import TTLCache
from tools import page_acquire as pa


def _artifact(text, *, via="flaresolverr", status=200, ctype="text/html"):
    return {
        "url": "https://example.com",
        "status": status,
        "content_type": ctype,
        "text": text,
        "bytes": None,
        "via": via,
        "blocked_detected": False,
    }


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    pa._host_failures.clear()
    pa._open_circuits.clear()
    pa._acquire_inflight.clear()
    monkeypatch.setattr(pa, "_page_cache", TTLCache(0, 8))
    monkeypatch.setattr(pa.cfg, "flaresolverr_url", "http://flaresolverr:8191")
    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "")
    monkeypatch.setattr(pa.cfg, "firecrawl_hedge_enabled", False)
    monkeypatch.setattr(pa.cfg, "classifier_api_url", "")
    monkeypatch.setattr(pa.cfg, "classifier_model", "")
    monkeypatch.setattr(pa.cfg, "circuit_breaker_enabled", True)

    async def allowed(url):
        return None

    monkeypatch.setattr(pa, "_assert_url_allowed", allowed)


def test_json_stays_on_direct_path(monkeypatch):
    calls = []

    async def direct(*args, **kwargs):
        calls.append("direct")
        return 200, {}, b'{"ok":true}', "application/json", False

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", lambda url: pytest.fail("browser used"))
    out = run(pa.acquire_page("https://example.com/api"))
    assert calls == ["direct"]
    assert out["via"] == "direct"
    assert out["text"] == '{"ok":true}'


def test_document_url_returning_html_is_not_sent_to_browser_or_tika(monkeypatch):
    async def direct(*args, **kwargs):
        return 200, {}, b"<html><body>Download denied</body></html>", "text/html", False

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", lambda url: pytest.fail("browser used"))
    with pytest.raises(pa.PageAcquisitionError, match="Document URL returned HTML"):
        run(pa.acquire_page("https://example.com/report.pdf"))


def test_html_probe_routes_to_flaresolverr_first(monkeypatch):
    calls = []

    async def direct(*args, **kwargs):
        calls.append("probe")
        return 200, {}, b"", "text/html", True

    async def browser(url):
        calls.append("flaresolverr")
        return _artifact("<main><h1>Loaded</h1><p>Browser-rendered content is available.</p></main>")

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    out = run(pa.acquire_page("https://example.com/page"))
    assert calls == ["probe", "flaresolverr"]
    assert out["via"] == "flaresolverr"


def test_blocked_browser_render_falls_back_to_firecrawl(monkeypatch):
    calls = []

    async def direct(*args, **kwargs):
        return 200, {}, b"", "text/html", True

    async def browser(url):
        calls.append("flaresolverr")
        return _artifact("<form id='human-verification'>Verify that you are human</form>")

    async def firecrawl(url):
        calls.append("firecrawl")
        return _artifact(
            "<main><h1>Recovered</h1><p>The complete requested page is available now.</p></main>",
            via="firecrawl",
        )

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "_render_with_firecrawl", firecrawl)
    out = run(pa.acquire_page("https://example.com/page"))
    assert calls == ["flaresolverr", "firecrawl"]
    assert out["via"] == "firecrawl"


def test_sparse_page_is_accepted_when_classifier_is_not_configured(monkeypatch):
    async def direct(*args, **kwargs):
        return 200, {}, b"", "text/html", True

    async def browser(url):
        return _artifact("<body>Service operational</body>")

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    out = run(pa.acquire_page("https://example.com/status"))
    assert out["via"] == "flaresolverr"
    assert out["assessment"]["verdict"] == "uncertain"


def test_hedged_firecrawl_starts_before_slow_failed_browser_finishes(monkeypatch):
    calls = []

    async def direct(*args, **kwargs):
        return 200, {}, b"", "text/html", True

    async def browser(url):
        calls.append("fs-start")
        await asyncio.sleep(0.03)
        calls.append("fs-end")
        return _artifact("<body></body>")

    async def firecrawl(url):
        calls.append("fc")
        return _artifact("<main><p>Recovered by the hedged request.</p></main>", via="firecrawl")

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa.cfg, "firecrawl_hedge_enabled", True)
    monkeypatch.setattr(pa.cfg, "firecrawl_hedge_delay_seconds", 0.001)
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "_render_with_firecrawl", firecrawl)
    out = run(pa.acquire_page("https://example.com/page"))
    assert "fc" in calls
    assert "fs-end" not in calls  # the slower browser request was cancelled
    assert out["via"] == "firecrawl"


def test_concurrent_requests_share_one_acquisition(monkeypatch):
    calls = 0

    async def direct(*args, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return 200, {}, b'{"ok":true}', "application/json", False

    async def scenario():
        return await asyncio.gather(
            pa.acquire_page("https://example.com/api"),
            pa.acquire_page("https://example.com/api"),
        )

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    results = run(scenario())
    assert calls == 1
    assert [result["text"] for result in results] == ['{"ok":true}'] * 2


def test_host_circuit_skips_browser_after_distinct_url_failures(monkeypatch):
    browser_calls = []

    async def direct(*args, **kwargs):
        return 200, {}, b"", "text/html", True

    async def browser(url):
        browser_calls.append(url)
        return _artifact("<form id='human-verification'>Verify that you are human</form>")

    async def firecrawl(url):
        return _artifact("<main><p>Recovered page content.</p></main>", via="firecrawl")

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa.cfg, "circuit_breaker_failure_threshold", 2)
    monkeypatch.setattr(pa.cfg, "circuit_breaker_window_seconds", 60)
    monkeypatch.setattr(pa.cfg, "circuit_breaker_ttl_seconds", 60)
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "_render_with_firecrawl", firecrawl)

    run(pa.acquire_page("https://www.example.com/a"))
    run(pa.acquire_page("https://example.com/b"))
    run(pa.acquire_page("https://example.com/c"))
    assert browser_calls == ["https://www.example.com/a", "https://example.com/b"]
