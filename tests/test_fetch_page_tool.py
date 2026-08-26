"""Integration tests for the fetch_page tool's _fetch_one logic.

The acquisition layer (`_acquire_page`), the YouTube transcript helper, and
Tika extraction are monkeypatched so the routing/formatting logic is exercised
offline. FlareSolverr and Firecrawl fallbacks are disabled unless a test needs them.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

import tools.fetch_page as fp
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
    monkeypatch.setattr(fp, "_reddit_access_token", None)
    monkeypatch.setattr(fp, "_reddit_access_token_expires_at", 0.0)
    monkeypatch.setattr(fp, "_reddit_access_token_credentials", None)


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


def test_query_no_match_raises(monkeypatch, tool_fns):
    html = "<body><article><p>Only this paragraph.</p></article></body>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com", query="zebra"))
    assert "matching query" in str(exc.value)


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
    monkeypatch.setattr(fp, "_tika_extract", lambda data, url, **kw: "Extracted PDF text.")
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/report.pdf")))
    assert out["format"] == "document_text"
    assert out["content"] == "Extracted PDF text."


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


def test_reddit_rss_failure_uses_old_html(monkeypatch, tool_fns):
    calls = []
    old_html = """<html><body><div class="thing link" data-fullname="t3_abc"
      data-author="poster" data-subreddit="python" data-score="5" data-comments-count="0">
      <a class="title">Old title</a><div class="entry"><div class="usertext-body"><div class="md"><p>Old body</p></div></div></div>
    </div></body></html>"""

    async def acquire(url):
        calls.append(url)
        if url.endswith(".rss"):
            raise RuntimeError("RSS blocked")
        return _fetched(text=old_html)

    monkeypatch.setattr(fp, "_acquire_page", acquire)
    out = json.loads(
        run(tool_fns["fetch_page"](url="https://www.reddit.com/r/python/comments/abc/title"))
    )
    assert calls[0].endswith("/.rss")
    assert calls[1].startswith("https://old.reddit.com/")
    assert json.loads(out["content"])["post"]["title"] == "Old title"
    assert "old.reddit" in out["note"]


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
    assert calls[1].startswith("https://old.reddit.com/")
    assert "/oembed?" in calls[2]
    assert json.loads(out["content"])["title"] == "Fallback title"
    assert "without comments" in out["note"]


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
    assert "configure Reddit OAuth" in out["note"]


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
