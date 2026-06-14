"""Integration tests for the fetch_page tool's _fetch_one logic.

The network layer (`_cached_resilient_fetch`), the YouTube transcript helper, and
Tika extraction are monkeypatched so the routing/formatting logic is exercised
offline. FlareSolverr and Wayback fallbacks are disabled unless a test needs them.
"""

import json

import pytest
from fastmcp.exceptions import ToolError

import tools.fetch_page as fp
from conftest import run


@pytest.fixture(autouse=True)
def disable_fallbacks_and_truncation(monkeypatch):
    monkeypatch.setattr(fp.cfg, "flaresolverr_url", "")
    monkeypatch.setattr(fp.cfg, "wayback_fallback", False)
    monkeypatch.setattr(fp.cfg, "markdown", True)
    monkeypatch.setattr(fp.cfg, "max_page_chars", 100000)


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

    monkeypatch.setattr(fp, "_cached_resilient_fetch", fake_fetch)


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
    assert "# Heading" in out["content"]
    assert "Body text here." in out["content"]


# --------------------------- structured mode ---------------------------

def test_structured_mode(monkeypatch, tool_fns):
    html = '<html><head><title>T</title><meta name="description" content="D"></head><body><h1>H1</h1></body></html>'
    _patch_fetch(monkeypatch, _fetched(text=html))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com", mode="structured")))
    assert out["format"] == "structured"
    assert out["content"]["title"] == "T"
    assert out["content"]["description"] == "D"


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
    assert "the detail paragraph" in out["content"]
    assert "ending" not in out["content"]


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


def test_query_no_match_raises(monkeypatch, tool_fns):
    html = "<body><article><p>Only this paragraph.</p></article></body>"
    _patch_fetch(monkeypatch, _fetched(text=html))
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com", query="zebra"))
    assert "matching query" in str(exc.value)


# --------------------------- document (Tika) ---------------------------

def test_document_routed_to_tika(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(content_type="application/pdf", body=b"%PDF-1.4..."))
    monkeypatch.setattr(fp, "_tika_extract", lambda data, url, **kw: "Extracted PDF text.")
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/report.pdf")))
    assert out["format"] == "document_text"
    assert out["content"] == "Extracted PDF text."


def test_document_no_content_raises(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(content_type="application/pdf", body=None, text=None))
    with pytest.raises(ToolError):
        run(tool_fns["fetch_page"](url="https://example.com/report.pdf"))


# --------------------------- Reddit / JSON ---------------------------

def test_reddit_url_compacted(monkeypatch, tool_fns):
    reddit_json = json.dumps([
        {"data": {"children": [{"data": {"title": "Post Title", "author": "u", "selftext": "hi"}}]}},
        {"data": {"children": [{"kind": "t1", "data": {"author": "c", "score": 3, "body": "a comment"}}]}},
    ])
    _patch_fetch(monkeypatch, _fetched(text=reddit_json, content_type="application/json"))
    out = json.loads(run(tool_fns["fetch_page"](url="https://www.reddit.com/r/x/comments/abc/title")))
    assert out["format"] == "json"
    content = json.loads(out["content"])
    assert content["post"]["title"] == "Post Title"
    assert content["comments"][0]["body"] == "a comment"


def test_json_content_returned_as_json(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text='{"key": "value"}', content_type="application/json"))
    out = json.loads(run(tool_fns["fetch_page"](url="https://example.com/api")))
    assert out["format"] == "json"
    assert json.loads(out["content"]) == {"key": "value"}


# --------------------------- blocked / contentless ---------------------------

def test_blocked_without_fallback_raises(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text="challenge", status=403, blocked=True))
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com"))
    assert "wall" in str(exc.value).lower() or "could not be retrieved" in str(exc.value)


def test_contentless_without_fallback_raises(monkeypatch, tool_fns):
    _patch_fetch(monkeypatch, _fetched(text="<html><body>; ;</body></html>"))
    with pytest.raises(ToolError) as exc:
        run(tool_fns["fetch_page"](url="https://example.com"))
    assert "client-side" in str(exc.value) or "no extractable text" in str(exc.value)


# --------------------------- YouTube routing ---------------------------

def test_youtube_url_returns_transcript(monkeypatch, tool_fns):
    async def fake_transcript(url, force_timestamps=False):
        return "Transcript for YouTube video abc\n---\nhello world"

    monkeypatch.setattr(fp, "fetch_transcript", fake_transcript)
    out = json.loads(run(tool_fns["fetch_page"](url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")))
    assert out["format"] == "youtube_transcript"
    assert "hello world" in out["content"]


# --------------------------- batch (multiple URLs) ---------------------------

def test_multiple_urls_returns_results_list(monkeypatch, tool_fns):
    def by_url(url):
        return _fetched(text=f"<body><article><p>content of {url}</p></article></body>")

    _patch_fetch(monkeypatch, by_url)
    out = json.loads(run(tool_fns["fetch_page"](url=["https://a.com", "https://b.com"])))
    assert "results" in out
    assert len(out["results"]) == 2


def test_multiple_urls_partial_failure(monkeypatch, tool_fns):
    def by_url(url):
        if "bad" in url:
            raise RuntimeError("boom")
        return _fetched(text="<body><article><p>ok</p></article></body>")

    async def fake_fetch(url):
        return by_url(url)

    monkeypatch.setattr(fp, "_cached_resilient_fetch", fake_fetch)
    out = json.loads(run(tool_fns["fetch_page"](url=["https://good.com", "https://bad.com"])))
    errors = [r for r in out["results"] if "error" in r]
    assert len(errors) == 1


def test_json_string_list_is_decoded(monkeypatch, tool_fns):
    def by_url(url):
        return _fetched(text="<body><article><p>ok</p></article></body>")

    _patch_fetch(monkeypatch, by_url)
    out = json.loads(run(tool_fns["fetch_page"](url='["https://a.com", "https://b.com"]')))
    assert "results" in out
    assert len(out["results"]) == 2


def test_urls_beyond_cap_are_skipped(monkeypatch, tool_fns):
    monkeypatch.setattr(fp, "MAX_FETCH_URLS", 1)

    def by_url(url):
        return _fetched(text="<body><article><p>ok</p></article></body>")

    _patch_fetch(monkeypatch, by_url)
    out = json.loads(run(tool_fns["fetch_page"](url=["https://a.com", "https://b.com"])))
    assert any("Skipped" in r.get("error", "") for r in out["results"])
    assert "note" in out


def test_all_urls_fail_raises(monkeypatch, tool_fns):
    async def fake_fetch(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(fp, "_cached_resilient_fetch", fake_fetch)
    with pytest.raises(ToolError):
        run(tool_fns["fetch_page"](url=["https://a.com", "https://b.com"]))
