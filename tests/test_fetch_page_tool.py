"""Integration tests for the fetch_page tool's _fetch_one logic.

The acquisition layer (`_acquire_page`), the YouTube transcript helper, and
Tika extraction are monkeypatched so the routing/formatting logic is exercised
offline. FlareSolverr and Firecrawl fallbacks are disabled unless a test needs them.
"""

import json
from base64 import b64encode

import pytest
from fastmcp.exceptions import ToolError

import tools.fetch_page as fp
import tools.page_acquire as pa
from conftest import run


@pytest.fixture(autouse=True)
def disable_fallbacks_and_truncation(monkeypatch):
    monkeypatch.setattr(fp.cfg, "flaresolverr_url", "")
    monkeypatch.setattr(fp.cfg, "firecrawl_api_key", "")
    monkeypatch.setattr(fp.cfg, "markdown", True)
    monkeypatch.setattr(fp.cfg, "max_page_chars", 100000)
    monkeypatch.setattr(fp.cfg, "reddit_client_id", "")
    monkeypatch.setattr(fp.cfg, "reddit_client_secret", "")
    monkeypatch.setattr(fp.cfg, "reddit_user_agent", "")
    monkeypatch.setattr(fp.cfg, "reddit_request_delay_seconds", 0)
    monkeypatch.setattr(fp.cfg, "reddit_rate_limit_retry_seconds", 0)
    monkeypatch.setattr(fp, "_reddit_access_token", None)
    monkeypatch.setattr(fp, "_reddit_access_token_expires_at", 0.0)
    monkeypatch.setattr(fp, "_reddit_access_token_credentials", None)
    # Brave and the Arctic Shift archive are opt-in per test (the archive would
    # otherwise reach the real network); the RSS retirement latch and archive
    # cache start clean so no test inherits another's state.
    monkeypatch.setattr(fp.cfg, "brave_api_key", "")
    monkeypatch.setattr(fp.cfg, "reddit_search_fallback", True)
    monkeypatch.setattr(fp.cfg, "reddit_archive_api_url", "")
    monkeypatch.setattr(fp, "_reddit_rss_retired_until", 0.0)
    monkeypatch.setattr(fp, "_reddit_archive_cache", fp.TTLCache(0, 8))
    monkeypatch.setattr(fp, "_ARCHIVE_TIMEOUT_RETRY_SECONDS", 0)


@pytest.fixture(autouse=True)
def inline_fetch_page_thread_offloads(monkeypatch):
    """Run fetch_page CPU offload targets inline for deterministic unit tests.

    These tests patch the network layer and assert routing/formatting behavior;
    they do not need real worker threads. The local Python 3.13 test environment
    can hang during thread cleanup, so keep this file focused on fetch_page logic.
    """
    async def run_sync(func, *args, **kwargs):
        kwargs.pop("abandon_on_cancel", None)
        kwargs.pop("cancellable", None)
        kwargs.pop("limiter", None)
        return func(*args, **kwargs)

    async def to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(fp.anyio.to_thread, "run_sync", run_sync)
    monkeypatch.setattr(fp.asyncio, "to_thread", to_thread)


def _fetched(text=None, *, status=200, content_type="text/html",
             body=None, via="direct", blocked=False):
    return {
        "url": "https://example.com",
        "status": status,
        "content_type": content_type,
        "text": text,
        "bytes": body,
        "via": via,
        "blocked_detected": blocked,
    }


def _patch_fetch(monkeypatch, result):
    async def fake_fetch(url):
        return result if not callable(result) else result(url)

    monkeypatch.setattr(fp, "_acquire_page", fake_fetch)


# --------------------------- input validation ---------------------------

def test_invalid_url_raises(tool_fns):
    with pytest.raises(ToolError):
        run(tool_fns["fetch_page"](url="ftp://example.com/x"))


def test_missing_url_raises(tool_fns):
    with pytest.raises(ToolError):
        run(tool_fns["fetch_page"](url="   "))


def test_invalid_mode_raises(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text="<p>x</p>"))
    with pytest.raises(ToolError):
        run(tool_fns["fetch_page"](url="https://example.com", mode="weird"))


# --------------------------- text / markdown ---------------------------

def test_text_mode_markdown(monkeypatch, tool_fns):
    html = "<html><head><title>My Title</title></head><body><article><h1>Heading</h1><p>Body text here.</p></article></body></html>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com")))
    assert out["format"] == "markdown"
    assert out["title"] == "My Title"
    assert "# Heading {#cite-heading}" in out["content"]
    assert "Body text here." in out["content"]


def test_text_mode_offset_echoes_continuation_anchor(monkeypatch, tool_fns):
    html = (
        "<html><body><article><h1>First</h1><p>" + "alpha " * 40 + "</p>"
        "<h2>Second</h2><p>" + "beta " * 40 + "</p></article></body></html>"
    )
    _patch_fetch(monkeypatch, _fetched(text=html))
    full = json.loads(run(tool_fns["fetch_page"](url="https://example.com")))["content"]
    offset = full.index("{#cite-second}") + len("{#cite-second}") + 5

    monkeypatch.setattr(fp.cfg, "max_page_chars", 20)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://example.com", offset=offset))
    )
    # The continuation names the nearest heading anchor before the offset, so a
    # model paging mid-page still knows which section it is reading.
    assert out["offset"] == offset
    assert out["continuation_anchor"] == "cite-second"
    assert out["truncated"] is True
    assert out["next_offset"] == offset + 20


def test_text_mode_marks_prominent_image_in_place(monkeypatch, tool_fns):
    html = (
        '<article><p>Before image.</p><img src="/chart.png" '
        'alt="Revenue rose each quarter"><p>After image.</p></article>'
    )
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/report")))
    marker = "[Image at this location: Revenue rose each quarter]"
    assert marker in out["content"]
    assert out["content"].index("Before image.") < out["content"].index(marker)
    assert out["content"].index(marker) < out["content"].index("After image.")


def test_firecrawl_metadata_title_overrides_body_widget_title(monkeypatch, tool_fns):
    fetched = _fetched(
        text=(
            "<html><body><div><title>reCAPTCHA</title></div>"
            "<main><h1>ASUS Zenbook A14</h1><p>Recovered product details.</p></main>"
            "</body></html>"
        ),
        via="firecrawl",
    )
    fetched["title"] = "ASUS Zenbook A14 - Best Buy"
    _patch_fetch(monkeypatch, fetched)

    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/product")))

    assert out["via"] == "firecrawl"
    assert out["title"] == "ASUS Zenbook A14 - Best Buy"
    assert out["content"].startswith("ASUS Zenbook A14 - Best Buy")
    assert "Recovered product details." in out["content"]


# --------------------------- structured mode ---------------------------

def test_structured_mode(monkeypatch, tool_fns):
    html = '<html><head><title>T</title><meta name="description" content="D"></head><body><h1>H1</h1></body></html>'
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com", mode="structured")))
    assert out["format"] == "structured"
    assert out["content"]["title"] == "T"
    assert out["content"]["description"] == "D"


def test_structured_mode_includes_image_description_and_placeholder_note(monkeypatch, tool_fns):
    html = '<main><img src="hero.jpg" alt="Snow-covered mountain"></main>'
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://example.com", mode="structured"))
    )
    image = out["content"]["images"][0]
    assert image["replaces_image"] is True
    assert image["description"] == "Snow-covered mountain"
    assert "not visual analysis" in out["content"]["image_note"]


def test_structured_mode_prefers_firecrawl_metadata_title(monkeypatch, tool_fns):
    fetched = _fetched(
        text=(
            "<html><body><div><title>reCAPTCHA</title></div>"
            "<main><h1>ASUS Zenbook A14</h1></main></body></html>"
        ),
        via="firecrawl",
    )
    fetched["title"] = "ASUS Zenbook A14 - Best Buy"
    _patch_fetch(monkeypatch, fetched)

    out = json.loads(
        run(tool_fns["fetch_page"](url="https://example.com/product", mode="structured"))
    )

    assert out["content"]["title"] == "ASUS Zenbook A14 - Best Buy"


def test_structured_mode_with_section_scopes_headings(monkeypatch, tool_fns):
    html = (
        "<body><h1>Top</h1>"
        "<h2>Alpha</h2><p>a</p>"
        "<h2>Beta</h2><p>b</p><h3>Beta-1</h3><p>b1</p>"
        "<h2>Gamma</h2><p>g</p></body>"
    )
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://example.com", mode="structured", section="Beta"))
    )
    assert out["format"] == "structured"
    # Only the Beta subtree, not the whole page's headings.
    assert out["content"]["section"] == "Beta"
    assert [h["text"] for h in out["content"]["headings"]] == ["Beta", "Beta-1"]
    assert "Alpha" not in (out["content"].get("toc") or [])


def test_structured_mode_section_not_found_raises(monkeypatch, tool_fns):
    html = "<body><h2>Alpha</h2><p>a</p></body>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com", mode="structured", section="Nope"))
    assert "Available headings" in str(exc.value)


# --------------------------- section extraction ---------------------------

SECTION_HTML = (
    "<body><h2>Intro</h2><p>intro text</p>"
    "<h2>Details</h2><p>the detail paragraph</p>"
    "<h2>End</h2><p>ending</p></body>"
)


def test_section_extraction(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text=SECTION_HTML))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com", section="Details")))
    assert out["format"] == "section"
    assert out["matched_heading"] == "Details"
    assert out["anchor"] == "cite-details"
    assert out["content"].startswith("# Details {#cite-details}")
    assert "the detail paragraph" in out["content"]
    assert "ending" not in out["content"]


def test_section_exposes_source_native_citation_url(monkeypatch, tool_fns):
    html = '<h2 id="methods">Methods</h2><p>Method details.</p><h2>End</h2>'
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(
        run(
            tool_fns["fetch_page"](
                url="https://example.com/paper", section="Methods"
            )
        )
    )
    assert out["anchor"] == "methods"
    assert out["source_fragment"] == "methods"
    assert out["citation_url"] == "https://example.com/paper#methods"
    assert out["content"].startswith("# Methods {#methods}")


def test_section_not_found_raises_with_available(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text=SECTION_HTML))
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com", section="Nonexistent"))
    assert "Available headings" in str(exc.value)


# --------------------------- query filtering ---------------------------

def test_query_filters_matching_passages(monkeypatch, tool_fns):
    html = "<body><article><p>The cat sat.</p><p>The dog ran fast.</p><p>Birds fly.</p></article></body>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com", query="dog")))
    assert out["query"] == "dog"
    assert out["match_count"] >= 1
    assert "dog ran fast" in out["content"]
    assert out["line_numbering"] == "1-based nonblank lines in searched content"
    assert out["match_metadata"][0]["match_lines"] == [2]
    assert out["match_metadata"][0]["match_quality"] == "literal_substring"
    assert "note" not in out


def test_query_result_keeps_nearest_heading_anchor(monkeypatch, tool_fns):
    html = (
        '<article><h2 id="results">Results</h2>'
        '<p>Context one.</p><p>Context two.</p><p>The target value is 42.</p>'
        '</article>'
    )
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(
        run(
            tool_fns["fetch_page"](
                url="https://example.com/paper",
                query="target",
                context_lines=0,
                include_match_toc=True,
            )
        )
    )
    assert "## Results {#results}" in out["content"]
    assert "The target value is 42" in out["content"]
    assert out["context_lines"] == 0
    assert out["match_metadata"][0]["heading"] == {
        "level": 2,
        "text": "Results",
        "anchor": "results",
        "source_fragment": "results",
        "citation_url": "https://example.com/paper#results",
        "line": 1,
    }
    assert out["matching_toc"] == [{
        "label": "Results",
        "anchor": "results",
        "citation_url": "https://example.com/paper#results",
        "match_count": 1,
        "windows": [1],
    }]


def test_query_controls_max_match_windows(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.cfg, "max_query_matches", 3)
    monkeypatch.setattr(fp.cfg, "max_query_context_lines", 4)
    html = "<article>" + "".join(
        f"<p>{text}</p>"
        for text in (
            "target one", "a", "b", "target two", "c", "d", "target three"
        )
    ) + "</article>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(
        run(
            tool_fns["fetch_page"](
                url="https://example.com",
                query="target",
                max_matches=1,
                context_lines=0,
            )
        )
    )
    assert out["sections"] == 1
    assert len(out["match_metadata"]) == 1
    assert out["match_count"] == 3
    assert "showing the first 1" in out["note"]


def test_query_no_match_returns_empty_result(monkeypatch, tool_fns):
    html = "<body><article><p>Only this paragraph.</p></article></body>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://example.com", query="zebra"))
    )
    assert out["query"] == "zebra"
    assert out["match_count"] == 0
    assert out["sections"] == 0
    assert out["match_metadata"] == []
    assert "Only this paragraph" not in out.get("content", "")
    assert "empty result" in out["note"]
    assert "not a fetch failure" in out["note"]
    assert "matching query 'zebra'" in out["note"]


def test_query_timeout_surfaces_as_tool_error(monkeypatch, tool_fns):
    monkeypatch.setattr(fp, "_QUERY_MATCH_BUDGET_SECONDS", 0.01)
    html = f"<body><article><p>{'a' * 10_000}!</p></article></body>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    with pytest.raises(ToolError) as exc:
        run(
            tool_fns["fetch_page"](
                url="https://example.com", query=r"(a|aa)+$"
            )
        )
    assert "took too long" in str(exc.value)


# --------------------------- document (Tika) ---------------------------

def test_document_routed_to_tika(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(content_type="application/pdf", body=b"%PDF-1.4..."))
    monkeypatch.setattr(fp.cfg, "tika_ocr_strategy", "no_ocr")
    monkeypatch.setattr(fp.cfg, "tika_ocr_retry", True)

    def extract(data, url, **kw):
        assert kw["ocr_strategy"] == "no_ocr"
        assert kw["ocr_retry"] is True
        return "# Extracted PDF\n\n- Text"

    monkeypatch.setattr(fp, "_tika_extract", extract)
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/report.pdf")))
    assert out["format"] == "document_text"
    assert out["content_format"] == "markdown"
    assert out["content"] == "# Extracted PDF\n\n- Text"


def test_document_query_matching_toc_uses_line_ranges(monkeypatch, tool_fns):
    _patch_fetch(
        monkeypatch,
        _fetched(content_type="application/pdf", body=b"%PDF-1.4..."),
    )
    monkeypatch.setattr(
        fp, "_tika_extract", lambda data, url, **kw: "intro\nnotes\ntarget result\nend"
    )
    out = json.loads(
        run(
            tool_fns["fetch_page"](
                url="https://example.com/report.pdf",
                query="target",
                context_lines=0,
                include_match_toc=True,
            )
        )
    )
    assert out["match_metadata"][0]["match_lines"] == [3]
    assert out["matching_toc"] == [{
        "label": "Lines 3-3",
        "match_count": 1,
        "windows": [1],
    }]


def test_firecrawl_recovered_document_text_skips_tika(monkeypatch, tool_fns):
    fetched = _fetched(text="Recovered protected PDF content.", via="firecrawl")
    fetched["resource_kind"] = "document_text"
    fetched["content_type"] = "text/markdown"
    _patch_fetch(monkeypatch, fetched)
    monkeypatch.setattr(
        fp, "_tika_extract", lambda *a, **k: pytest.fail("Tika should not run")
    )
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/report.pdf")))
    assert out["format"] == "document_text"
    assert out["via"] == "firecrawl"
    assert out["content"] == "Recovered protected PDF content."


def test_document_no_content_raises(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(content_type="application/pdf", body=None, text=None))
    with pytest.raises(ToolError):
        run(tool_fns["fetch_page"](url="https://example.com/report.pdf"))


def test_document_extraction_failure_raises(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(content_type="application/pdf", body=b"%PDF-1.4..."))

    def fail_extract(*args, **kwargs):
        raise RuntimeError("Tika unavailable")

    monkeypatch.setattr(fp, "_tika_extract", fail_extract)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com/report.pdf"))
    assert "document extraction failed" in str(exc.value).lower()


def test_error_html_at_document_url_is_not_extracted(monkeypatch, tool_fns):
    """A `.pdf` URL that returns an HTTP error with an HTML body (e.g. a bot wall
    that slipped past block detection) must raise, not be Tika-extracted into
    bogus document_text."""
    _patch_fetch(
        monkeypatch,
        _fetched(
            status=403,
            content_type="text/html",
            text="Enable JavaScript and cookies to continue",
            body=b"Enable JavaScript and cookies to continue",
        ),
    )
    # Tika should never be reached; make it loud if it is.
    monkeypatch.setattr(
        fp, "_tika_extract", lambda *a, **k: pytest.fail("Tika ran on an error page")
    )
    with pytest.raises(ToolError):
        run(tool_fns["fetch_page"](url="https://example.com/report.pdf"))


def test_mislabeled_pdf_sniffed_to_tika(monkeypatch, tool_fns):
    """A PDF served as octet-stream from an extensionless URL is routed by magic bytes."""
    _patch_fetch(
        monkeypatch,
        _fetched(content_type="application/octet-stream", body=b"%PDF-1.5\n%binary", text=None),
    )
    monkeypatch.setattr(fp, "_tika_extract", lambda data, url, **kw: "Sniffed PDF text.")
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/download?id=42")))
    assert out["format"] == "document_text"
    assert out["content"] == "Sniffed PDF text."


# --------------------------- direct-resource HTTP errors ---------------------------

def test_direct_resource_http_error_raises(monkeypatch, tool_fns):
    _patch_fetch(
        monkeypatch,
        _fetched(text='{"error":"not found"}', content_type="application/json", status=404),
    )
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com/api"))
    assert "HTTP 404" in str(exc.value)


# --------------------------- Reddit / JSON ---------------------------

def test_reddit_oauth_json_compacted(monkeypatch, tool_fns, patch_httpx):
    reddit_json = json.dumps([
        {"data": {"children": [{"data": {"title": "Post Title", "author": "u", "selftext": "hi"}}]}},
        {"data": {"children": [{"kind": "t1", "data": {"author": "c", "score": 3, "body": "a comment"}}]}},
    ])
    monkeypatch.setattr(fp.cfg, "reddit_client_id", "client-id")
    monkeypatch.setattr(fp.cfg, "reddit_client_secret", "client-secret")
    monkeypatch.setattr(fp.cfg, "reddit_user_agent", "linux:mcp-server:1.0 (by /u/tester)")
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/api/v1/access_token":
            return fp.httpx.Response(
                200,
                json={"access_token": "token", "expires_in": 3600},
                request=request,
            )
        return fp.httpx.Response(
            200,
            text=reddit_json,
            headers={"content-type": "application/json"},
            request=request,
        )

    patch_httpx(handler)
    out = json.loads(run(tool_fns["fetch_page"](url="https://www.reddit.com/r/x/comments/abc/title")))
    assert out["format"] == "json"
    assert out["via"] == "reddit_oauth"
    content = json.loads(out["content"])
    assert content["post"]["title"] == "Post Title"
    assert content["comments"][0]["body"] == "a comment"
    assert requests[0].headers["authorization"].startswith("Basic ")
    assert requests[1].headers["authorization"] == "Bearer token"
    assert requests[1].url.host == "oauth.reddit.com"
    assert requests[1].url.params["raw_json"] == "1"


def test_reddit_without_oauth_uses_rss(monkeypatch, tool_fns):
    calls = []
    monkeypatch.setattr(fp.server_settings, "debug", False)
    rss = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <author><name>/u/poster</name></author><content type="html">&lt;div class="md"&gt;&lt;p&gt;RSS body&lt;/p&gt;&lt;/div&gt;</content>
      <id>t3_abc</id><link href="https://reddit.com/comments/abc"/><updated>2026-01-01T00:00:00Z</updated><title>RSS title</title>
    </entry></feed>"""

    async def acquire(url):
        calls.append(url)
        return _fetched(text=rss, content_type="application/atom+xml")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert calls == ["https://www.reddit.com/r/python/comments/abc/title/.rss"]
    assert json.loads(out["content"])["post"]["title"] == "RSS title"
    assert "RSS feed" in out["note"]
    assert "fallback trace" not in out["note"]


def test_reddit_debug_note_traces_and_redacts_fallbacks(monkeypatch, tool_fns):
    secret = "super-secret-reddit-value"
    monkeypatch.setattr(fp.server_settings, "debug", True)
    monkeypatch.setattr(fp.cfg, "reddit_client_id", "client-id")
    monkeypatch.setattr(fp.cfg, "reddit_client_secret", secret)
    monkeypatch.setattr(
        fp.cfg, "reddit_user_agent", "linux:mcp-server:1.0 (by /u/tester)"
    )
    rss = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <author><name>/u/poster</name></author><content type="html">&lt;div class="md"&gt;&lt;p&gt;RSS body&lt;/p&gt;&lt;/div&gt;</content>
      <id>t3_abc</id><link href="https://reddit.com/comments/abc"/><updated>2026-01-01T00:00:00Z</updated><title>RSS title</title>
    </entry></feed>"""

    async def oauth_failure(url):
        raise RuntimeError(f"credential {secret} was rejected")

    async def acquire(url):
        return _fetched(text=rss, content_type="application/atom+xml")

    monkeypatch.setattr(fp, "_fetch_reddit_oauth", oauth_failure)
    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert "Reddit fallback trace:" in out["note"]
    assert "OAuth JSON failed (credential REDACTED was rejected)" in out["note"]
    assert "RSS succeeded" in out["note"]
    assert secret not in out["note"]


def _oembed_fetched(title="Oembed title", author="poster"):
    return _fetched(
        text=json.dumps({"title": title, "author_name": author, "provider_name": "reddit"}),
        content_type="application/json",
    )


def _enable_brave(monkeypatch, results):
    """Configure Brave and replace the LLM Context call with canned results."""
    monkeypatch.setattr(fp.cfg, "brave_api_key", "brave-key")
    monkeypatch.setattr(fp.cfg, "brave_api_url", "https://brave.test/llm/context")
    queries = []

    async def brave_query(**kwargs):
        queries.append(kwargs)
        return results

    monkeypatch.setattr(fp, "_brave_query", brave_query)
    return queries


def test_reddit_failures_use_oembed_last(monkeypatch, tool_fns):
    calls = []

    async def acquire(url):
        calls.append(url)
        if "/oembed?" not in url:
            raise RuntimeError("blocked")
        return _fetched(
            text=json.dumps({"title": "Fallback title", "provider_name": "reddit"}),
            content_type="application/json",
        )

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert calls[0].endswith("/.rss")
    assert "/oembed?" in calls[1]
    assert len(calls) == 2
    assert json.loads(out["content"])["title"] == "Fallback title"
    assert "without the post body or comments" in out["note"]


def test_reddit_rss_failure_uses_brave_excerpts(monkeypatch, tool_fns):
    calls = []

    async def acquire(url):
        calls.append(url)
        if url.endswith(".rss"):
            raise RuntimeError("RSS blocked")
        return _oembed_fetched(title="Exact post title")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    queries = _enable_brave(monkeypatch, [
        {
            "url": "https://www.reddit.com/r/python/comments/zzz/other/",
            "title": "Unrelated thread",
            "snippets": ["unrelated text"],
        },
        {
            "url": "https://www.reddit.com/r/python/comments/abc/title/",
            "title": "r/python on Reddit: Exact post title",
            "snippets": ["post body excerpt", "## commenter\ncomment excerpt"],
            "page_date": "2026-09-30T17:37:11Z",
        },
        {
            "url": "https://www.reddit.com/r/python/comments/ABC/title/def456/",
            "title": "comment permalink",
            "snippets": ["post body excerpt", "deep comment excerpt"],
        },
    ])
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert calls[0].endswith("/.rss")
    assert "/oembed?" in calls[1]
    assert queries[0]["query"] == "site:reddit.com r/python Exact post title"
    assert queries[0]["freshness"] == ""
    assert out["url"] == "https://www.reddit.com/r/python/comments/abc/title/"
    assert out["via"] == "brave_llm_context"
    content = json.loads(out["content"])
    assert content["post"] == {
        "title": "Exact post title",
        "author": "poster",
        "subreddit": "python",
        "permalink": "https://www.reddit.com/r/python/comments/abc/title",
    }
    # Only same-submission results, deduplicated, in result order.
    assert content["excerpts"] == [
        "post body excerpt",
        "## commenter\ncomment excerpt",
        "deep comment excerpt",
    ]
    assert content["complete"] is False
    assert content["page_date"] == "2026-09-30T17:37:11Z"
    assert "search-index excerpts" in out["note"]


def test_reddit_brave_uses_slug_when_oembed_fails(monkeypatch, tool_fns):
    async def acquire(url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    queries = _enable_brave(monkeypatch, [{
        "url": "https://www.reddit.com/r/python/comments/abc/my_post_title/",
        "title": "My post title",
        "snippets": ["excerpt"],
    }])
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/my_post_title/"
    )))
    assert queries[0]["query"] == "site:reddit.com r/python my post title"
    content = json.loads(out["content"])
    assert content["post"]["title"] == "My post title"
    assert "author" not in content["post"]


def test_reddit_brave_without_matching_post_returns_oembed(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.server_settings, "debug", True)

    async def acquire(url):
        if "/oembed?" in url:
            return _oembed_fetched()
        raise RuntimeError("blocked")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    _enable_brave(monkeypatch, [{
        "url": "https://www.reddit.com/r/python/comments/zzz/similar_title/",
        "title": "Similar",
        "snippets": ["wrong thread"],
    }])
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert json.loads(out["content"])["title"] == "Oembed title"
    assert "without the post body or comments" in out["note"]
    assert "Brave excerpts failed (search index returned no excerpts for post abc)" in out["note"]
    assert "brave-key" not in out["note"]


def test_reddit_search_fallback_can_be_disabled(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.server_settings, "debug", True)

    async def acquire(url):
        if "/oembed?" in url:
            return _oembed_fetched()
        raise RuntimeError("blocked")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    queries = _enable_brave(monkeypatch, [])
    monkeypatch.setattr(fp.cfg, "reddit_search_fallback", False)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert queries == []
    assert json.loads(out["content"])["title"] == "Oembed title"
    assert "reddit_search_fallback disabled" in out["note"]


def test_reddit_skips_rss_while_retired_latch_is_open(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.server_settings, "debug", True)
    fp._mark_reddit_rss_retired()
    calls = []

    async def acquire(url):
        calls.append(url)
        return _oembed_fetched()

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    _enable_brave(monkeypatch, [{
        "url": "https://www.reddit.com/r/python/comments/abc/title/",
        "title": "t",
        "snippets": ["excerpt"],
    }])
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert not any(url.endswith(".rss") for url in calls)
    assert "RSS skipped (Reddit recently stopped serving feeds)" in out["note"]
    assert json.loads(out["content"])["excerpts"] == ["excerpt"]


def test_reddit_listing_without_sources_points_to_search(monkeypatch, tool_fns):
    fp._mark_reddit_rss_retired()
    calls = []

    async def acquire(url):
        calls.append(url)
        raise AssertionError("listing fallbacks should not fetch anything")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/"))
    assert calls == []
    assert "site:reddit.com/r/<subreddit>" in str(exc.value)


def test_reddit_post_fails_when_every_source_fails(monkeypatch, tool_fns):
    async def acquire(url):
        raise RuntimeError("blocked")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    message = str(exc.value)
    assert "RSS failed (blocked)" in message
    assert "oEmbed failed (blocked)" in message
    assert "Brave Search not configured" in message


def test_reddit_developer_docs_use_normal_page_fetch(monkeypatch, tool_fns):
    calls = []

    async def acquire(url):
        calls.append(url)
        return _fetched(text="<html><head><title>Docs</title></head><body><h1>Migrate</h1><p>Guide text.</p></body></html>")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://developers.reddit.com/docs/guides/migrate/public-api"
    )))
    assert calls == ["https://developers.reddit.com/docs/guides/migrate/public-api"]
    assert out["format"] != "json"
    assert "Guide text." in out["content"]


def test_reddit_rss_retries_once_after_rate_limit(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.cfg, "reddit_rate_limit_retry_seconds", 3)
    calls = []
    sleeps = []
    rss = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>t3_abc</id><title>RSS title</title>
    </entry></feed>"""

    async def acquire(url):
        calls.append(url)
        if len(calls) == 1:
            return _fetched(status=429, text="Too Many Requests")
        return _fetched(text=rss, content_type="application/atom+xml")

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    monkeypatch.setattr(fp.anyio, "sleep", sleep)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert len(calls) == 2
    assert sleeps == [3]
    assert json.loads(out["content"])["post"]["title"] == "RSS title"
    assert "HTTP 429" in out["note"]
    assert "automatic retry succeeded" in out["note"]


def test_reddit_oembed_reports_rate_limit_without_debug(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.server_settings, "debug", False)

    async def acquire(url):
        if url.endswith("/.rss"):
            return _fetched(status=429, text="Too Many Requests")
        if "/oembed?" in url:
            return _fetched(
                text=json.dumps({"title": "Fallback title", "provider_name": "reddit"}),
                content_type="application/json",
            )
        raise RuntimeError("blocked")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert "HTTP 429" in out["note"]
    assert "Retry later" in out["note"]


# --------------------------- Reddit / Arctic Shift archive ---------------------------

_ARCHIVE = "https://archive.test/api"


def _archive_post(**overrides):
    post = {
        "id": "abc", "title": "Archived title", "author": "poster",
        "subreddit": "python", "score": 12, "num_comments": 2,
        "permalink": "/r/python/comments/abc/title/",
        "url": "https://www.reddit.com/r/python/comments/abc/title/",
        "selftext": "Archived body", "created_utc": 1700000000,
        "retrieved_on": 1700000020, "_meta": {"retrieved_2nd_on": 1700100000},
        "removed_by_category": None,
    }
    post.update(overrides)
    return post


def _archive_tree(n=2, more=0):
    children = [
        {"kind": "t1", "data": {"author": f"c{i}", "score": i, "body": f"comment {i}"}}
        for i in range(n)
    ]
    if more:
        children.append({"kind": "more", "data": {"count": more, "children": ["x"] * more}})
    return children


def _enable_archive(monkeypatch, patch_httpx, routes, *, forbid_acquire=True):
    """Point the archive at a mock; routes maps API path -> response or callable."""
    monkeypatch.setattr(fp.cfg, "reddit_archive_api_url", _ARCHIVE)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "archive.test"
        route = routes[request.url.path.removeprefix("/api")]
        if callable(route):
            return route(request)
        return fp.httpx.Response(200, json=route, request=request)

    patch_httpx(handler)
    if forbid_acquire:
        async def acquire(url):
            raise AssertionError(f"unexpected Reddit/RSS/oEmbed fetch: {url}")

        monkeypatch.setattr(fp, "_acquire_page", acquire)
    return requests


def test_reddit_archive_post_hit(monkeypatch, tool_fns, patch_httpx):
    monkeypatch.setattr(fp.server_settings, "debug", False)
    monkeypatch.setattr(fp.cfg, "reddit_archive_max_comments", 123)
    queries = _enable_brave(monkeypatch, [])
    requests = _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post()]},
        "/comments/tree": {"data": _archive_tree(2)},
    })
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title/?context=3"
    )))
    assert queries == []
    params = {r.url.path: dict(r.url.params) for r in requests}
    assert params["/api/posts/ids"] == {"ids": "abc"}  # no fields= whitelist
    assert params["/api/comments/tree"] == {"link_id": "t3_abc", "limit": "123"}
    assert out["format"] == "json"
    assert out["via"] == "arctic_shift"
    assert out["url"].startswith(_ARCHIVE + "/posts/ids?")
    content = json.loads(out["content"])
    assert content["post"]["title"] == "Archived title"
    assert content["post"]["permalink"] == "https://www.reddit.com/r/python/comments/abc/title/"
    assert "removed_by_category" not in content["post"]
    assert content["archived_at"] == "2023-11-16T02:00:00Z"
    assert [c["body"] for c in content["comments"]] == ["comment 0", "comment 1"]
    assert content["comments_incomplete"] is False
    assert "archived_at" in out["note"]
    assert "Arctic Shift" in out["note"]


def test_reddit_archive_credentials_are_not_in_provenance_or_partial_errors(
    monkeypatch, tool_fns, patch_httpx
):
    monkeypatch.setattr(fp.server_settings, "debug", False)
    requests = _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post()]},
        "/comments/tree": {
            "error": "Bad password mock:p@ss or mock%3Ap%40ss",
        },
    })
    monkeypatch.setattr(
        fp.cfg, "reddit_archive_api_url", "https://archive-user:mock%3Ap%40ss@archive.test/api"
    )
    raw = run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title/"
    ))
    out = json.loads(raw)
    assert out["url"] == _ARCHIVE + "/posts/ids?ids=abc"
    assert len(requests) == 2
    for request in requests:
        assert not request.url.userinfo
        assert request.headers["authorization"] == (
            "Basic " + b64encode(b"archive-user:mock:p@ss").decode("ascii")
        )
    assert "archive-user" not in raw
    assert "mock:p@ss" not in raw
    assert "mock%3Ap%40ss" not in raw
    assert "Bad password REDACTED or REDACTED" in json.loads(out["content"])["comments_unavailable"]


def test_reddit_archive_credentials_are_redacted_from_tool_error(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/search": {"error": "Could not authenticate with archive-user:private-pass"},
    })
    monkeypatch.setattr(
        fp.cfg, "reddit_archive_api_url", "https://archive-user:private-pass@archive.test/api"
    )
    with pytest.raises(ToolError, match="Could not authenticate with REDACTED") as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/search/?q=asyncio"))
    assert "private-pass" not in str(exc.value)
    assert "archive-user" not in str(exc.value)


def test_reddit_archive_more_nodes_mark_incomplete(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post(num_comments=5)]},
        "/comments/tree": {"data": _archive_tree(2, more=3)},
    })
    content = json.loads(json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))["content"])
    assert content["more_children"] == 3
    assert content["comments_incomplete"] is True
    assert content["comments_reported"] == 5


def test_reddit_archive_fresh_snapshot_counts_are_not_trusted(monkeypatch, tool_fns, patch_httpx):
    # First-ingest snapshot: no _meta yet and num_comments still 0.
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post(num_comments=0, _meta=None)]},
        "/comments/tree": {"data": _archive_tree(1)},
    })
    content = json.loads(json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))["content"])
    assert content["archived_at"] == "2023-11-14T22:13:40Z"  # retrieved_on
    assert "comments_reported" not in content
    assert content["comments_incomplete"] is None


def test_reddit_archive_removed_post_is_served_and_flagged(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post(removed_by_category="moderator")]},
        "/comments/tree": {"data": []},
    })
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))
    content = json.loads(out["content"])
    assert content["post"]["selftext"] == "Archived body"
    assert content["post"]["removed_by_category"] == "moderator"
    assert "removed_by_category: moderator" in out["note"]


def test_reddit_archive_copy_without_content_reports_deleted(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post(title="[deleted]", selftext="[removed]")]},
        "/comments/tree": {"data": []},
    })
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    assert "deleted or removed" in str(exc.value)


def test_reddit_archive_miss_falls_through_to_rss(monkeypatch, tool_fns, patch_httpx):
    monkeypatch.setattr(fp.server_settings, "debug", True)
    rss = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>t3_abc</id><title>RSS title</title><content type="html">body</content>
    </entry></feed>"""
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": []},
        "/comments/tree": {"data": []},
    }, forbid_acquire=False)
    calls = []

    async def acquire(url):
        calls.append(url)
        return _fetched(text=rss, content_type="application/atom+xml")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))
    assert calls == ["https://www.reddit.com/r/python/comments/abc/title/.rss"]
    assert json.loads(out["content"])["post"]["title"] == "RSS title"
    assert "Archive failed (archive has no copy of post abc yet)" in out["note"]
    assert "RSS succeeded" in out["note"]


@pytest.mark.parametrize("failure, detail", [
    (lambda r: fp.httpx.Response(429, headers={"x-ratelimit-reset": "42"}, request=r),
     "HTTP 429, rate limit resets in 42s"),
    (lambda r: fp.httpx.Response(500, request=r), "HTTP 500"),
    (lambda r: fp.httpx.Response(
        200, json={"data": None, "error": "Timeout. Maybe slow down a bit"}, request=r),
     "archive error: Timeout. Maybe slow down a bit"),
])
def test_reddit_archive_failure_falls_through_to_brave(
    monkeypatch, tool_fns, patch_httpx, failure, detail
):
    monkeypatch.setattr(fp.server_settings, "debug", True)
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": failure,
        "/comments/tree": {"data": []},
    }, forbid_acquire=False)

    async def acquire(url):
        if url.endswith(".rss"):
            return _fetched(status=429, text="", content_type="")
        return _oembed_fetched()

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    _enable_brave(monkeypatch, [{
        "url": "https://www.reddit.com/r/python/comments/abc/title/",
        "title": "t", "snippets": ["excerpt"],
    }])
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))
    assert out["via"] == "brave_llm_context"
    assert f"Archive failed ({detail})" in out["note"]
    assert "Retry later" in out["note"]
    # A 429 is the per-minute budget, not a retirement.
    assert fp._reddit_rss_available()


def test_reddit_archive_timeout_is_retried_once(monkeypatch, tool_fns, patch_httpx):
    attempts = []

    def posts(request):
        attempts.append(request)
        if len(attempts) == 1:
            return fp.httpx.Response(
                422, json={"data": None, "error": "Timeout. Maybe slow down a bit"},
                request=request,
            )
        return fp.httpx.Response(200, json={"data": [_archive_post()]}, request=request)

    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": posts,
        "/comments/tree": {"data": []},
    })
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))
    assert len(attempts) == 2
    assert out["via"] == "arctic_shift"


def test_reddit_archive_error_body_on_http_error_is_reported(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/search": lambda r: fp.httpx.Response(
            400, json={"data": None, "error": "'x' is not a valid field"}, request=r),
    }, forbid_acquire=False)

    async def acquire(url):
        return _fetched(status=429, text="", content_type="")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/"))
    assert "HTTP 400, archive error: 'x' is not a valid field" in str(exc.value)


def test_reddit_archive_comment_tree_failure_keeps_post(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post()]},
        "/comments/tree": lambda r: fp.httpx.Response(502, request=r),
    })
    content = json.loads(json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))["content"])
    assert content["post"]["selftext"] == "Archived body"
    assert content["comments"] == []
    assert content["comments_incomplete"] is True
    assert "HTTP 502" in content["comments_unavailable"]


@pytest.mark.parametrize("invalid", [{}, {"data": None}, {"data": {"unexpected": "shape"}}])
def test_reddit_malformed_archive_tree_is_unavailable_and_not_cached(
    monkeypatch, tool_fns, patch_httpx, invalid
):
    monkeypatch.setattr(fp, "_reddit_archive_cache", fp.TTLCache(300, 8))
    trees = iter([invalid, {"data": []}])
    requests = _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post(num_comments=0)]},
        "/comments/tree": lambda r: fp.httpx.Response(200, json=next(trees), request=r),
    })
    url = "https://www.reddit.com/r/python/comments/abc/title"
    first = json.loads(json.loads(run(tool_fns["fetch_page"](url=url)))["content"])
    assert first["post"]["selftext"] == "Archived body"
    assert first["comments"] == []
    assert first["comments_incomplete"] is True
    assert "expected an array" in first["comments_unavailable"]

    second = json.loads(json.loads(run(tool_fns["fetch_page"](url=url)))["content"])
    assert second["comments"] == []
    assert second["comments_incomplete"] is False
    assert "comments_unavailable" not in second
    assert sum(r.url.path == "/api/comments/tree" for r in requests) == 2


def test_reddit_archive_download_cap(monkeypatch, tool_fns, patch_httpx):
    monkeypatch.setattr(fp.server_settings, "debug", True)
    monkeypatch.setattr(fp.cfg, "max_download_bytes", 64)
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/ids": {"data": [_archive_post(selftext="x" * 500)]},
        "/comments/tree": {"data": []},
    }, forbid_acquire=False)

    async def acquire(url):
        if "/oembed?" in url:
            return _oembed_fetched()
        raise RuntimeError("blocked")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))
    assert json.loads(out["content"])["title"] == "Oembed title"
    assert "Archive failed (Archive response exceeds the 64-byte" in out["note"]


def test_reddit_archive_disabled_is_traced(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.server_settings, "debug", True)

    async def acquire(url):
        if "/oembed?" in url:
            return _oembed_fetched()
        raise RuntimeError("blocked")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/comments/abc/title"
    )))
    assert "Archive skipped (not configured)" in out["note"]


def test_reddit_archive_subreddit_listing(monkeypatch, tool_fns, patch_httpx):
    monkeypatch.setattr(fp.cfg, "reddit_archive_max_listing", 10)
    requests = _enable_archive(monkeypatch, patch_httpx, {
        "/posts/search": {"data": [
            {"id": "p1", "title": "Older", "subreddit": "python", "created_utc": 100},
            {"id": "p2", "title": "Newer", "subreddit": "python", "created_utc": 200,
             "selftext": "body", "url": "https://example.com/a"},
        ]},
    })
    out = json.loads(run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/top/")))
    assert dict(requests[0].url.params)["subreddit"] == "python"
    content = json.loads(out["content"])
    assert content["listing"] == "r/python"
    assert content["sort"] == "newest_first"
    assert [i["title"] for i in content["items"]] == ["Newer", "Older"]
    assert content["items"][0]["permalink"] == "https://www.reddit.com/r/python/comments/p2/"
    assert content["items"][0]["url"] == "https://example.com/a"
    assert "type" not in content["items"][0]
    assert "newest first" in out["note"]
    assert "hot/top ranking is not available" in out["note"]


def test_reddit_archive_subreddit_search_listing(monkeypatch, tool_fns, patch_httpx):
    requests = _enable_archive(monkeypatch, patch_httpx, {
        "/posts/search": {"data": [{"id": "p1", "title": "asyncio tips", "subreddit": "python"}]},
    })
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/search/?q=asyncio&restrict_sr=1"
    )))
    assert dict(requests[0].url.params)["title"] == "asyncio"
    assert "post titles only" in out["note"]


def test_reddit_archive_search_with_no_matches_succeeds(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {"/posts/search": {"data": []}})
    out = json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/r/python/search/?q=nomatch"
    )))
    assert out["via"] == "arctic_shift"
    content = json.loads(out["content"])
    assert content["match"] == "title"
    assert content["items"] == []
    assert content["items_returned"] == 0


def test_reddit_archive_search_with_invalid_data_fails(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {"/posts/search": {"data": None}})
    with pytest.raises(ToolError, match="expected an array in archive listing data"):
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/search/?q=asyncio"))


@pytest.mark.parametrize("suffix", ["comments", "comments/"])
def test_reddit_archive_user_comments_view(monkeypatch, tool_fns, patch_httpx, suffix):
    requests = _enable_archive(monkeypatch, patch_httpx, {
        "/comments/search": {"data": [
            {"id": "c1", "body": "Comment", "subreddit": "s", "link_id": "t3_p9",
             "created_utc": 200},
        ]},
    })
    out = json.loads(run(tool_fns["fetch_page"](
        url=f"https://www.reddit.com/user/someone/{suffix}"
    )))
    assert [r.url.path for r in requests] == ["/api/comments/search"]
    assert requests[0].url.params["author"] == "someone"
    assert json.loads(out["content"])["items"][0]["text"] == "Comment"


def test_reddit_archive_user_listing_merges_posts_and_comments(monkeypatch, tool_fns, patch_httpx):
    monkeypatch.setattr(fp.cfg, "reddit_archive_max_listing", 2)
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/search": {"data": [
            {"id": "p1", "title": "Post", "subreddit": "s", "created_utc": 300},
            {"id": "p0", "title": "Old post", "subreddit": "s", "created_utc": 50},
        ]},
        "/comments/search": {"data": [
            {"id": "c1", "body": "Comment", "subreddit": "s", "link_id": "t3_p9",
             "created_utc": 200},
        ]},
    })
    content = json.loads(json.loads(run(tool_fns["fetch_page"](
        url="https://www.reddit.com/user/someone/"
    )))["content"])
    assert content["listing"] == "u/someone"
    assert [i["type"] for i in content["items"]] == ["post", "comment"]
    assert content["items_returned"] == 2


def test_reddit_listing_archive_failure_uses_rss(monkeypatch, tool_fns, patch_httpx):
    monkeypatch.setattr(fp.server_settings, "debug", True)
    rss = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>t3_p1</id><title>From RSS</title></entry></feed>"""
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/search": lambda r: fp.httpx.Response(503, request=r),
    }, forbid_acquire=False)

    async def acquire(url):
        assert url == "https://www.reddit.com/r/python.rss"
        return _fetched(text=rss, content_type="application/atom+xml")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/")))
    assert json.loads(out["content"])["items"][0]["title"] == "From RSS"
    assert "Archive failed (HTTP 503)" in out["note"]


def test_reddit_listing_all_sources_fail_points_to_search(monkeypatch, tool_fns, patch_httpx):
    _enable_archive(monkeypatch, patch_httpx, {
        "/posts/search": lambda r: fp.httpx.Response(503, request=r),
    }, forbid_acquire=False)

    async def acquire(url):
        return _fetched(status=429, text="", content_type="")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/"))
    message = str(exc.value)
    assert "Archive failed (HTTP 503)" in message
    assert "RSS failed (HTTP 429)" in message
    assert "site:reddit.com/r/<subreddit>" in message


def test_reddit_search_listing_never_requests_search_rss(monkeypatch, tool_fns):
    async def acquire(url):
        raise AssertionError("search feeds are refused by Reddit and must not be fetched")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/search/?q=x"))
    assert "RSS skipped (Reddit refuses anonymous search feeds)" in str(exc.value)


# --------------------------- Reddit / RSS gate and deleted posts ---------------------------

@pytest.mark.parametrize("response", [
    {"status": 410, "text": "gone", "content_type": "text/plain"},
    {"status": 200, "text": "<!doctype html><html>login</html>", "content_type": "text/plain"},
])
def test_reddit_rss_definitive_failure_opens_latch(monkeypatch, tool_fns, response):
    calls = []

    async def acquire(url):
        calls.append(url)
        if url.endswith(".rss"):
            return _fetched(**response)
        return _oembed_fetched()

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    url = "https://www.reddit.com/r/python/comments/abc/title"
    run(tool_fns["fetch_page"](url=url))
    assert not fp._reddit_rss_available()
    run(tool_fns["fetch_page"](url=url))
    assert sum(u.endswith(".rss") for u in calls) == 1


@pytest.mark.parametrize("status,available", [(200, False), (410, False), (429, True), (503, True)])
def test_reddit_rss_html_latch_uses_http_status(
    monkeypatch, tool_fns, patch_httpx, status, available
):
    async def allow(url):
        assert url == "https://www.reddit.com/r/python.rss"

    def handler(request):
        assert str(request.url) == "https://www.reddit.com/r/python.rss"
        return fp.httpx.Response(status, html="<html>Not an Atom feed</html>", request=request)

    monkeypatch.setattr(pa, "_assert_url_allowed", allow)
    patch_httpx(handler)
    detail = f"HTTP {status}" if status >= 400 else "response was not a feed"
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/"))
    assert f"RSS failed ({detail})" in str(exc.value)
    assert fp._reddit_rss_available() is available


def test_reddit_rss_atom_404_does_not_open_latch(monkeypatch, tool_fns):
    async def acquire(url):
        if url.endswith(".rss"):
            return _fetched(status=404, text="<?xml version='1.0'?><feed></feed>",
                            content_type="application/atom+xml")
        return _oembed_fetched()

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    assert fp._reddit_rss_available()


def test_reddit_rss_deleted_post_reports_deleted(monkeypatch, tool_fns):
    rss = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>t3_abc</id><title>[deleted]</title><content type="html">[removed] &lt;br/&gt;</content>
    </entry></feed>"""
    calls = []

    async def acquire(url):
        calls.append(url)
        return _fetched(text=rss, content_type="application/atom+xml")

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    assert "deleted or removed" in str(exc.value)
    assert len(calls) == 1  # no oEmbed/Brave after a definitive answer


@pytest.mark.parametrize("oembed", [
    _fetched(status=404, text="not found", content_type="text/plain"),
    _fetched(text=json.dumps({"title": "[deleted]", "author_name": ""}),
             content_type="application/json"),
])
def test_reddit_oembed_deleted_post_reports_deleted(monkeypatch, tool_fns, oembed):
    queries = _enable_brave(monkeypatch, [])

    async def acquire(url):
        if url.endswith(".rss"):
            return _fetched(status=429, text="", content_type="")
        return oembed

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    assert "deleted" in str(exc.value)
    assert queries == []


def test_json_content_returned_as_json(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text='{"key": "value"}', content_type="application/json"))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/api")))
    assert out["format"] == "json"
    assert json.loads(out["content"]) == {"key": "value"}


# --------------------------- blocked / contentless ---------------------------

def test_contentless_accepted_artifact_still_raises(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text="<html><body>; ;</body></html>"))
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com"))
    assert "no extractable text" in str(exc.value)


def test_standalone_raster_image_returns_explicit_placeholder(monkeypatch, tool_fns):
    fetched = _fetched(
        content_type="image/png", body=b"\x89PNG\r\n\x1a\nbinary", text=None
    )
    fetched["resource_kind"] = "image"
    _patch_fetch(monkeypatch, fetched)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://example.com/photo.png"))
    )
    assert out["format"] == "image"
    assert out["media_type"] == "image/png"
    assert out["filename"] == "photo.png"
    assert "[Image at this location:" in out["content"]
    assert "no alt text, caption, or embedded description" in out["content"]
    assert "in place of the image" in out["note"]
    assert "not visual analysis" in out["note"]


def test_standalone_svg_returns_embedded_description(monkeypatch, tool_fns):
    svg = "<svg><title>Network diagram</title><desc>Three connected nodes</desc></svg>"
    fetched = _fetched(content_type="image/svg+xml", text=svg, body=None)
    fetched["resource_kind"] = "image"
    _patch_fetch(monkeypatch, fetched)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://example.com/network.svg"))
    )
    assert out["format"] == "image"
    assert out["description"] == "Network diagram — Three connected nodes"
    assert out["description_source"] == "embedded SVG title+description"
    assert out["content"] == (
        "[Image at this location: Network diagram — Three connected nodes]"
    )


def test_unsupported_binary_media_raises_actionable_error(monkeypatch, tool_fns):
    _patch_fetch(
        monkeypatch,
        _fetched(content_type="audio/mpeg", body=b"ID3binary", text=None),
    )
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com/audio.mp3"))
    msg = str(exc.value)
    assert "cannot extract that media type" in msg
    assert "audio/mpeg" in msg
    assert "direct download" in msg


# --------------------------- YouTube routing ---------------------------

def test_youtube_url_returns_transcript(monkeypatch, tool_fns):
    async def fake_transcript(url, force_timestamps=False):
        return "Transcript for YouTube video abc\n---\nhello world"

    monkeypatch.setattr(fp, "fetch_transcript", fake_transcript)
    out = json.loads(run(tool_fns["fetch_page"](url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")))
    assert out["format"] == "youtube_transcript"
    assert "hello world" in out["content"]


def test_youtube_query_matching_toc_uses_timestamps(monkeypatch, tool_fns):
    async def fake_transcript(url, force_timestamps=False):
        assert force_timestamps is True
        return (
            "Transcript for YouTube video abc\n---\n"
            "[0:10] introduction\n[1:25] target phrase\n[2:00] ending"
        )

    monkeypatch.setattr(fp, "fetch_transcript", fake_transcript)
    out = json.loads(
        run(
            tool_fns["fetch_page"](
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                query="target",
                context_lines=0,
                include_match_toc=True,
            )
        )
    )
    assert out["match_metadata"][0]["match_lines"] == [2]
    assert out["matching_toc"] == [{
        "label": "Transcript at 1:25",
        "timestamp": "1:25",
        "match_count": 1,
        "windows": [1],
    }]


def test_youtube_transcript_uses_truncation(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.cfg, "max_page_chars", 10)

    async def fake_transcript(url, force_timestamps=False):
        return "0123456789ABCDEFGHIJ"

    monkeypatch.setattr(fp, "fetch_transcript", fake_transcript)
    out = json.loads(run(tool_fns["fetch_page"](url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")))
    assert out["format"] == "youtube_transcript"
    assert out["content"].startswith("0123456789")
    assert "truncated at 10 chars" in out["content"]
    assert out["truncated"] is True
    assert out["next_offset"] == 10
    assert "query=" in out["note"]


def test_youtube_transcript_respects_offset(monkeypatch, tool_fns):
    monkeypatch.setattr(fp.cfg, "max_page_chars", 10)

    async def fake_transcript(url, force_timestamps=False):
        return "0123456789ABCDEFGHIJ"

    monkeypatch.setattr(fp, "fetch_transcript", fake_transcript)
    out = json.loads(
        run(
            tool_fns["fetch_page"](
                url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                offset=10,
            )
        )
    )
    assert out["format"] == "youtube_transcript"
    assert out["content"] == "ABCDEFGHIJ"
    assert out["offset"] == 10
    assert "truncated" not in out
