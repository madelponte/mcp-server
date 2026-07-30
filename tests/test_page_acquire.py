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


def test_blocked_document_can_use_firecrawl_document_parser(monkeypatch):
    async def direct(*args, **kwargs):
        return 403, {}, b"<html><body>Challenge</body></html>", "text/html", False

    async def document_firecrawl(url):
        result = _artifact("Recovered PDF text", via="firecrawl", ctype="text/markdown")
        result["resource_kind"] = "document_text"
        return result

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_document_with_firecrawl", document_firecrawl)
    out = run(pa.acquire_page("https://example.com/report.pdf"))
    assert out["resource_kind"] == "document_text"
    assert out["text"] == "Recovered PDF text"


def test_blocked_document_rejects_firecrawl_challenge_text(monkeypatch):
    async def direct(*args, **kwargs):
        return 403, {}, b"<html><body>Challenge</body></html>", "text/html", False

    async def document_firecrawl(url):
        result = _artifact(
            "Access denied. Verify that you are human.",
            via="firecrawl",
            ctype="text/markdown",
        )
        result["resource_kind"] = "document_text"
        return result

    async def blocked_assessment(url, status, html):
        return pa.PageAssessment(
            pa.PageVerdict.BLOCKED,
            0.96,
            "challenge_page",
            {"challenge_language": True},
        )

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_document_with_firecrawl", document_firecrawl)
    monkeypatch.setattr(pa, "assess_page", blocked_assessment)
    with pytest.raises(pa.PageAcquisitionError, match="document recovery was blocked"):
        run(pa.acquire_page("https://example.com/report.pdf"))


def test_document_url_returning_html_is_not_sent_to_browser_or_tika(monkeypatch):
    async def direct(*args, **kwargs):
        return 200, {}, b"<html><body>Download denied</body></html>", "text/html", False

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", lambda url: pytest.fail("browser used"))
    with pytest.raises(pa.PageAcquisitionError, match="Document URL returned HTML"):
        run(pa.acquire_page("https://example.com/report.pdf"))


def test_probe_timeout_continues_to_browser_and_firecrawl(monkeypatch):
    calls = []

    async def direct(*args, **kwargs):
        calls.append("probe")
        raise TimeoutError()

    async def browser(url):
        calls.append("flaresolverr")
        return _artifact("<form id='human-verification'>Verify that you are human</form>")

    async def firecrawl(url):
        calls.append("firecrawl")
        return _artifact("<main><p>Recovered product price is available.</p></main>", via="firecrawl")

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "_render_with_firecrawl", firecrawl)
    out = run(pa.acquire_page("https://example.com/product/123"))
    assert calls == ["probe", "flaresolverr", "firecrawl"]
    assert out["via"] == "firecrawl"


def test_known_direct_resource_timeout_does_not_launch_browser(monkeypatch):
    async def direct(*args, **kwargs):
        raise TimeoutError()

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", lambda url: pytest.fail("browser used"))
    with pytest.raises(pa.PageAcquisitionError, match="TimeoutError"):
        run(pa.acquire_page("https://example.com/data.json"))


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


@pytest.mark.parametrize("status", [404, 410, 451])
def test_missing_html_status_is_authoritative(monkeypatch, status):
    calls = []

    async def direct(*args, **kwargs):
        calls.append("probe")
        return status, {}, b"", "text/html", True

    async def browser(url):
        calls.append("flaresolverr")
        return _artifact("<main><h1>Branded missing page</h1></main>")

    async def firecrawl(url):
        calls.append("firecrawl")
        return _artifact(
            "<main><h1>Branded missing page</h1></main>",
            via="firecrawl",
        )

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "_render_with_firecrawl", firecrawl)
    out = run(pa.acquire_page("https://example.com/missing"))
    assert calls == ["probe"]
    assert out["status"] == status
    assert out["via"] == "direct"
    assert out["text"] == ""


def test_blocked_html_status_still_uses_browser_recovery(monkeypatch):
    calls = []

    async def direct(*args, **kwargs):
        calls.append("probe")
        return 403, {}, b"", "text/html", True

    async def browser(url):
        calls.append("flaresolverr")
        return _artifact("<main><p>Recovered browser content.</p></main>")

    async def accepted(url, status, html):
        return pa.PageAssessment(
            pa.PageVerdict.ACCEPT,
            0.96,
            "substantive_content",
            {"main_content": True},
        )

    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "assess_page", accepted)
    out = run(pa.acquire_page("https://example.com/protected"))
    assert calls == ["probe", "flaresolverr"]
    assert out["status"] == 200
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


def test_sparse_page_is_accepted_when_classifier_fails(monkeypatch):
    async def direct(*args, **kwargs):
        return 200, {}, b"", "text/html", True

    async def browser(url):
        return _artifact("<body>Service operational</body>")

    async def classifier_failed(url, status, html):
        return pa.PageAssessment(
            pa.PageVerdict.UNCERTAIN,
            0.5,
            "sparse_content",
            {"word_count": 2},
            source="deterministic",
        )

    monkeypatch.setattr(pa.cfg, "classifier_api_url", "http://classifier/v1")
    monkeypatch.setattr(pa.cfg, "classifier_model", "small-4b")
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "assess_page", classifier_failed)
    out = run(pa.acquire_page("https://example.com/status"))
    assert out["via"] == "flaresolverr"
    assert out["assessment"]["source"] == "deterministic"


def test_slow_browser_is_bounded_before_sequential_firecrawl(monkeypatch):
    calls = []

    async def direct(*args, **kwargs):
        return 200, {}, b"", "text/html", True

    async def browser(url):
        calls.append("fs-start")
        await asyncio.sleep(1)
        return _artifact("late")

    async def firecrawl(url):
        calls.append("firecrawl")
        return _artifact("<main><p>Recovered after browser timeout.</p></main>", via="firecrawl")

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa.cfg, "flaresolverr_attempt_timeout_seconds", 0.01)
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "_render_with_firecrawl", firecrawl)
    out = run(pa.acquire_page("https://example.com/product"))
    assert calls == ["fs-start", "firecrawl"]
    assert out["via"] == "firecrawl"


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


def test_http_errors_do_not_open_host_browser_circuit(monkeypatch):
    browser_calls = []

    async def direct(*args, **kwargs):
        return 200, {}, b"", "text/html", True

    async def browser(url):
        browser_calls.append(url)
        if url.endswith(("/a", "/b")):
            return _artifact("<h1>Not Found</h1>", status=404)
        return _artifact("<main><p>The valid page is available.</p></main>")

    async def firecrawl(url):
        return _artifact(
            "<main><p>Fallback result for the unavailable URL.</p></main>",
            via="firecrawl",
        )

    async def assess(url, status, html):
        if status == 404:
            return pa.PageAssessment(
                pa.PageVerdict.UNUSABLE,
                0.98,
                "http_404",
                {"status": 404},
            )
        return pa.PageAssessment(
            pa.PageVerdict.ACCEPT,
            0.96,
            "substantive_content",
            {"main_content": True},
        )

    monkeypatch.setattr(pa.cfg, "firecrawl_api_key", "fc-test")
    monkeypatch.setattr(pa.cfg, "circuit_breaker_failure_threshold", 2)
    monkeypatch.setattr(pa.cfg, "circuit_breaker_window_seconds", 60)
    monkeypatch.setattr(pa.cfg, "circuit_breaker_ttl_seconds", 60)
    monkeypatch.setattr(pa, "_direct_resource_fetch", direct)
    monkeypatch.setattr(pa, "_render_with_flaresolverr", browser)
    monkeypatch.setattr(pa, "_render_with_firecrawl", firecrawl)
    monkeypatch.setattr(pa, "assess_page", assess)

    run(pa.acquire_page("https://example.com/a"))
    run(pa.acquire_page("https://example.com/b"))
    run(pa.acquire_page("https://example.com/c"))
    assert browser_calls == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
