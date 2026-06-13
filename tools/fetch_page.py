"""
Page-fetching MCP tool.

Exposes `fetch_page`: reads one or more web page URLs (returning markdown or a
structured metadata summary) or, when handed a YouTube video URL, that video's
transcript. PDF/Office/OpenDocument/RTF/EPUB documents are routed to Apache Tika
and returned as text; Reddit links are read through the JSON API. An optional
`query` does server-side extractive filtering (only the passages matching a
keyword/regex), `section` extracts a single heading, and `offset` pages through
content too long to return at once.

The HTTP fetching this sits on lives in `tools/web_fetch.py`; the HTML→content
extraction in `tools/web_extract.py`; the YouTube transcript logic in
`tools/youtube_transcript.py`. The companion `search_web` tool is in
`tools/web_search.py`.
"""

import asyncio
import json
import logging
import re
import time
from functools import partial
from typing import Any
from urllib.parse import urlparse, urlunparse

import anyio
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import web_search_settings as cfg
from .serialize import to_json, log_call, log_result, debug_enabled
from .youtube_transcript import is_youtube_video_url, fetch_transcript
from .web_fetch import (
    _cached_resilient_fetch,
    _fetch_from_wayback,
    _is_tika_document,
    _render_with_flaresolverr,
    _tika_extract,
)
from .web_extract import (
    _find_section,
    _markdown_from_html,
    _page_title,
    _plain_text_from_html,
    _structured_from_html,
    _trim_flagged,
)

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake the failure for
# page data. A valid-but-empty result is NOT a failure and is returned as normal
# JSON. See the README "Error handling" section.

# Hard cap on how many URLs a single fetch_page call will fetch when passed a
# list. A context-budget guard: more than a few full pages in one response blows
# a small model's window. Kept as a literal (not a config valve) so the exact
# number can be stated in the tool docstring the model reads — keep the "3" in
# that docstring in sync if you change this.
MAX_FETCH_URLS = 3


# ---------------------------------------------------------------------------
# Content paging / truncation hints
#
# Hints appended to a fetch_page result whose content had to be truncated to fit
# cfg.max_page_chars. `_OFFSET_HINT` always applies: `offset=` is the guaranteed
# way to read the rest of *any* truncated content (it works on JSON and on a
# headingless document, which `query=`/`section=` can't narrow). `_NARROW_HINT`
# is added only for formats those two params can target, so the model can jump
# straight to what it needs instead of paging.
# ---------------------------------------------------------------------------

_OFFSET_HINT = (
    "Content was truncated to fit the size limit; the rest was dropped. To read "
    "the next chunk, retry with `offset=<next_offset>` (the value echoed below)."
)
_NARROW_HINT = (
    " To jump straight to the relevant part instead, retry with `query=` (a "
    "keyword/phrase or regex — returns only the matching passages) or `section=` "
    "(a heading — returns only that section)."
)


def _join_note(existing: str | None, extra: str) -> str:
    """Append `extra` to an existing note (space-separated), or return it alone."""
    return f"{existing} {extra}" if existing else extra


def _set_content(
    payload: dict, content: str, *, hint: bool = True, offset: int = 0
) -> None:
    """Slice `content` from `offset`, trim to the page-size cap, store on `payload`.

    `offset` (a character position, default 0) lets the model page through
    content too long to return at once: the rest of a truncated response is
    reachable by re-fetching with ``offset=`` set to the ``next_offset`` echoed
    here. When trimming actually dropped text, set ``payload["truncated"] = True``
    and ``payload["next_offset"]`` to where the next chunk begins, and fold the
    offset hint into ``payload["note"]``. When `hint`, also add the
    `query=`/`section=` narrowing hint — those can't narrow some formats (e.g.
    JSON), so `hint` is False there, but `offset=` still works.
    """
    total = len(content)
    start = offset if offset > 0 else 0
    if start and start >= total:
        # The model paged past the end — return empty content and say so rather
        # than silently looking like a successful empty fetch.
        payload["content"] = ""
        payload["note"] = _join_note(
            payload.get("note"),
            f"offset {start} is at or past the end of the {total}-char content.",
        )
        return

    body = content[start:] if start else content
    trimmed, truncated = _trim_flagged(body, cfg.max_page_chars)
    payload["content"] = trimmed
    if start:
        payload["offset"] = start
    if truncated:
        payload["truncated"] = True
        payload["next_offset"] = start + cfg.max_page_chars
        payload["note"] = _join_note(payload.get("note"), _OFFSET_HINT)
        if hint:
            payload["note"] = payload["note"] + _NARROW_HINT


# ---------------------------------------------------------------------------
# Query (lexical / regex) filtering
#
# `fetch_page`'s optional `query` does server-side extractive filtering: rather
# than returning a whole page (or transcript), it returns only the segments that
# lexically match the query, each with a little surrounding context. This keeps a
# long page or video transcript from flooding the model's context window when it
# only needs the parts about one topic.
# ---------------------------------------------------------------------------

# ReDoS guard. The model's `query` is compiled as arbitrary regex and run over
# every segment of a page, so a pathological pattern (catastrophic backtracking)
# against a long page could pin the CPU and — since the server is single-process
# — stall every other in-flight tool call. Two bounds keep that in check:
#   * patterns that are over-long or contain a classically dangerous shape are
#     refused as regex and matched as a literal substring instead (see
#     `_compile_query`), and
#   * the per-call scan is given a wall-clock budget (see `_extract_matches`).
# The matching itself is also offloaded to a worker thread (`_query_payload`),
# consistent with the sync-in-async convention.
_MAX_REGEX_QUERY_CHARS = 200

# Classic catastrophic-backtracking shape: a group that itself contains a
# quantifier and is then immediately quantified — (a+)+, (.*)+, (x* )*. We refuse
# to compile these as regex and fall back to a literal search rather than risk an
# exponential match.
_NESTED_QUANTIFIER_RE = re.compile(r"\([^()]*[+*][^()]*\)\s*[+*{]")

# Wall-clock budget for a single query's scan over a page's segments. A bound on
# total matching work even when no individual pattern is pathological (e.g. a
# very long page); on hitting it we return the matches found so far.
_QUERY_MATCH_BUDGET_SECONDS = 2.0


def _compile_query(query: str) -> re.Pattern:
    """Compile the model's `query` into a case-insensitive pattern.

    The query may be a plain keyword/phrase or a regular expression. It is matched
    as a literal substring (via ``re.escape``) when it is not valid regex (e.g.
    unbalanced brackets in `cost (usd)`), or when it trips the ReDoS guard — too
    long, or containing a nested-quantifier shape prone to catastrophic
    backtracking — so a naive query still works while a dangerous one can't stall
    the server.
    """
    if len(query) <= _MAX_REGEX_QUERY_CHARS and not _NESTED_QUANTIFIER_RE.search(query):
        try:
            return re.compile(query, re.IGNORECASE)
        except re.error:
            pass
    return re.compile(re.escape(query), re.IGNORECASE)


def _segment_text(text: str) -> list[str]:
    """Split readable text (or a transcript) into the units query matching works on.

    A segment is one line of `_plain_text_from_html` output (≈ a paragraph) or
    one transcript caption line (which carries its [timestamp] when timestamps
    are enabled). Blank lines are dropped.
    """
    return [s for s in (ln.strip() for ln in text.split("\n")) if s]


def _extract_matches(
    text: str, query: str, *, context: int, max_windows: int
) -> tuple[list[str], int, int]:
    """Find segments matching `query` and return them with surrounding context.

    Returns ``(windows, total_matches, total_windows)``. Each window is the
    matching segment(s) plus ``context`` segments on either side; windows that
    overlap or sit adjacent are merged so a run of nearby matches reads as one
    block. ``windows`` holds at most ``max_windows`` of them (the formatted
    output), while ``total_windows`` counts how many existed before that cap and
    ``total_matches`` counts every matching segment. ``windows`` is empty when
    nothing matched.
    """
    segments = _segment_text(text)
    if not segments:
        return [], 0, 0

    pattern = _compile_query(query)
    # Scan under a wall-clock budget: if matching is taking too long (a long page,
    # or a query that slipped past the compile-time guard), stop and use what we
    # have rather than block the single-process server. The literal/nested-
    # quantifier guard in `_compile_query` prevents a single segment from blowing
    # up; this bounds the total across many segments.
    deadline = time.monotonic() + _QUERY_MATCH_BUDGET_SECONDS
    match_idxs = []
    for i, seg in enumerate(segments):
        if pattern.search(seg):
            match_idxs.append(i)
        if time.monotonic() > deadline:
            break
    if not match_idxs:
        return [], 0, 0

    # Expand each match into a [start, end] window, merging into the previous one
    # when they touch (gap of at most one segment) so adjacent matches don't
    # produce a string of tiny, overlapping fragments.
    bounds: list[list[int]] = []
    for idx in match_idxs:
        start = max(0, idx - context)
        end = min(len(segments) - 1, idx + context)
        if bounds and start <= bounds[-1][1] + 1:
            bounds[-1][1] = max(bounds[-1][1], end)
        else:
            bounds.append([start, end])

    windows = ["\n".join(segments[s : e + 1]) for s, e in bounds[:max_windows]]
    return windows, len(match_idxs), len(bounds)


def _format_match_windows(windows: list[str]) -> str:
    """Join match windows, separating distinct ones with a labeled marker.

    A single window is returned as-is; multiple windows each get a
    ``───── match i of N ─────`` header so the model can tell the pieces come
    from different parts of the page.
    """
    if len(windows) == 1:
        return windows[0]
    total = len(windows)
    return "\n\n".join(
        f"───── match {i} of {total} ─────\n{w}" for i, w in enumerate(windows, 1)
    )


async def _query_payload(text: str, query: str, url: str, *, kind: str) -> dict:
    """Filter `text` to the segments matching `query`; raise if nothing matches.

    Returns fields to merge into a fetch_page result: the formatted ``content``
    (match windows joined by labeled separators), the total ``match_count``, the
    number of ``sections`` (windows) shown, and a ``note`` when some windows were
    dropped to stay within the cap. ``kind`` names what was searched (e.g.
    "content", "transcript segment") for the not-found error.

    The regex matching runs in a worker thread (sync-in-async convention) so a
    slow scan can't block the event loop along with every other in-flight call.
    """
    windows, match_count, total_windows = await anyio.to_thread.run_sync(
        partial(
            _extract_matches,
            text,
            query,
            context=cfg.query_context_segments,
            max_windows=cfg.max_query_matches,
        )
    )
    if not windows:
        raise ToolError(
            f"No {kind} matching query {query!r} found on {url}. "
            "Try a simpler keyword or a different spelling, loosen the regex, or "
            "omit `query` to retrieve the full content."
        )
    note = None
    if total_windows > len(windows):
        note = (
            f"{total_windows} matching sections found; showing the first "
            f"{len(windows)}. Refine `query` to narrow the results."
        )
    return {
        "query": query,
        "match_count": match_count,
        "sections": len(windows),
        "content": _format_match_windows(windows),
        "note": note,
    }


# ---------------------------------------------------------------------------
# Reddit handling
#
# Reddit blocks HTML scraping, so a reddit link is rewritten to its .json
# endpoint before fetching and the verbose response is compacted to just the
# post + comment fields the model needs.
# ---------------------------------------------------------------------------

def _normalize_reddit_url(url: str) -> str:
    """Reddit blocks HTML scraping; force the .json endpoint for reddit links."""
    try:
        p = urlparse(url)
    except Exception:
        return url
    # Match reddit.com and its subdomains (www./old./np. …) only. A bare
    # `endswith("reddit.com")` on the netloc would also catch look-alike domains
    # like `notreddit.com` (and miss `reddit.com:443`, since the port is part of
    # the netloc), so test the parsed hostname for an exact or dotted-suffix match.
    host = (p.hostname or "").lower()
    if host != "reddit.com" and not host.endswith(".reddit.com"):
        return url
    path = p.path or "/"
    if path.endswith("/"):
        path = path[:-1]
    if not path.endswith(".json"):
        path = path + ".json"
    return urlunparse((p.scheme or "https", "www.reddit.com", path, "", p.query, ""))


def _compact_reddit_json(data: Any) -> Any:
    """Compact Reddit's verbose post+comments JSON to just what the model needs."""
    try:
        if isinstance(data, list) and len(data) == 2:
            post_listing, comments_listing = data

            def child_of(listing):
                return (listing or {}).get("data", {}).get("children", []) or []

            post = None
            kids = child_of(post_listing)
            if kids:
                pd = kids[0].get("data", {})
                post = {
                    "title": pd.get("title"),
                    "author": pd.get("author"),
                    "subreddit": pd.get("subreddit"),
                    "score": pd.get("score"),
                    "num_comments": pd.get("num_comments"),
                    "permalink": pd.get("permalink"),
                    "url": pd.get("url"),
                    "selftext": pd.get("selftext"),
                    "created_utc": pd.get("created_utc"),
                }

            comments = []

            def walk(node, depth=0):
                if not isinstance(node, dict):
                    return
                kind = node.get("kind")
                d = node.get("data") or {}
                if kind == "t1":
                    comments.append(
                        {
                            "author": d.get("author"),
                            "score": d.get("score"),
                            "depth": depth,
                            "body": d.get("body"),
                        }
                    )
                    replies = d.get("replies")
                    if isinstance(replies, dict):
                        for c in (replies.get("data") or {}).get("children", []) or []:
                            walk(c, depth + 1)

            for c in child_of(comments_listing):
                walk(c, 0)

            return {"post": post, "comments": comments}
    except Exception:
        pass
    return data


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _provenance(
    url: str,
    fetch_url: str,
    status: int | None,
    ctype: str | None,
    via: str | None,
) -> dict:
    """Diagnostic fetch fields (original_url / status / content_type / via) to
    splice into a fetch_page payload.

    On the happy path — HTTP 200, a direct fetch, and an un-rewritten URL — these
    just burn the model's context window, so they're omitted. They reappear only
    when something the model should know about deviated from that path (a non-200
    status, a FlareSolverr fallback, a rewritten URL), or when MCP_DEBUG forces
    the full picture for troubleshooting.
    """
    debug = debug_enabled()
    out: dict = {}
    if debug or fetch_url != url:
        out["original_url"] = url
    error = status is not None and status != 200
    if debug or error:
        out["status"] = status
        if ctype is not None:
            out["content_type"] = ctype
    if via is not None and (debug or via != "direct"):
        out["via"] = via
    return out


# ---------------------------------------------------------------------------
# Single-URL fetch
# ---------------------------------------------------------------------------

def _render_text_body(html: str, base_url: str) -> tuple[str, str]:
    """Render page HTML to ``(body, format)`` for text mode.

    `format` is "markdown" or "text" per the WEB_SEARCH_MARKDOWN valve. Shared by
    the initial extraction and the FlareSolverr re-render retry in `_fetch_one`.
    """
    if cfg.markdown:
        return _markdown_from_html(html, base_url), "markdown"
    return _plain_text_from_html(html), "text"


# A run of two or more letters — i.e. an actual word. A rendered body without
# even one has no readable text and is an empty shell, even when it isn't
# strictly whitespace: a JS page that only partly rendered (a direct fetch of a
# client-side app, or a FlareSolverr snapshot taken before a lazy XHR resolved)
# can leave stray punctuation like "; ;" that defeats a bare `.strip()` test.
_WORD_RE = re.compile(r"[^\W\d_]{2,}")


def _is_contentless(body: str) -> bool:
    """True if `body` has no readable text (not one word of 2+ letters)."""
    return not _WORD_RE.search(body)


def _format_wayback_date(ts: str) -> str:
    """Format a 14-digit Wayback timestamp (YYYYMMDDhhmmss) as 'YYYY-MM-DD'."""
    if len(ts) >= 8 and ts[:8].isdigit():
        return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
    return ts or "an unknown date"


def _strip_wayback_chrome(html: str) -> str:
    """Remove the Wayback Machine's injected toolbar/banner from a replay DOM.

    Rendering a Wayback *replay* URL yields the page plus archive.org's own
    navigation chrome (``#wm-ipp-base`` etc.). The extractor prefers an
    ``<article>``/``<main>`` and usually ignores the chrome anyway, but strip it
    so it can't pollute content on pages whose body falls back to ``<body>``.
    """
    soup = BeautifulSoup(html, "lxml")
    for el_id in ("wm-ipp-base", "wm-ipp", "donato"):
        el = soup.find(id=el_id)
        if el:
            el.decompose()
    return str(soup)


async def _wayback_content(fetch_url: str) -> dict | None:
    """Recover readable text for `fetch_url` from the Wayback Machine, or None.

    Tries the archived page's static HTML first (cheap; works when the snapshot
    captured server-rendered text). If that snapshot is *itself* a JavaScript
    shell — as a client-side-rendered SPA's archived main document is — and
    FlareSolverr is configured, it renders the Wayback *replay* URL in a real
    browser: Wayback serves the page's archived sub-resources/XHRs to that
    browser, so the JS-built body can materialize from the archive (and without
    the live site's API latency that defeated the live render). Returns
    ``{text, plain, format, status, content_type, meta}`` on success, else None.
    """
    archived = await _fetch_from_wayback(fetch_url)
    if not archived or not archived.get("text"):
        return None

    text = archived["text"]
    status = archived.get("status")
    ctype = (archived.get("content_type") or "text/html").lower()
    try:
        plain, fmt = _render_text_body(text, fetch_url)
    except Exception:
        plain, fmt = "", "markdown"

    # Archived main document is a JS shell — render the replay in a real browser.
    if _is_contentless(plain) and cfg.flaresolverr_url and archived.get("wayback_url"):
        try:
            rr = await _render_with_flaresolverr(archived["wayback_url"])
        except Exception:
            rr = None
        if rr and rr.get("text"):
            stripped = _strip_wayback_chrome(rr["text"])
            try:
                p2, f2 = _render_text_body(stripped, fetch_url)
            except Exception:
                p2, f2 = "", fmt
            if not _is_contentless(p2):
                text, plain, fmt = stripped, p2, f2
                status = rr.get("status", status)
                ctype = (rr.get("content_type") or ctype).lower()

    if _is_contentless(plain):
        return None
    ts = archived.get("wayback_timestamp", "")
    return {
        "text": text,
        "plain": plain,
        "format": fmt,
        "status": status,
        "content_type": ctype,
        "meta": {"timestamp": ts, "date": _format_wayback_date(ts), "url": archived.get("wayback_url")},
    }


async def _fetch_one(
    url: str,
    mode: str = "text",
    section: str | None = None,
    query: str | None = None,
    offset: int = 0,
) -> dict:
    """Fetch a single URL and return its result payload dict.

    Shared by fetch_page for both the one-URL and list-of-URLs paths. Raises
    ToolError on a genuine failure (caught per-URL in the batch path).
    `offset` resumes truncated content at that character position.
    """
    if not url or not isinstance(url, str):
        raise ToolError("Missing url.")
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        raise ToolError(f"Invalid URL: {url}")

    query = (query or "").strip() or None

    # A YouTube video URL has no useful scrapeable page content — the actual
    # content is the spoken transcript. Detect it and return the transcript
    # directly (via the YouTube helper) instead of fetching the watch page,
    # so the model needs only this one tool for both web pages and videos.
    if is_youtube_video_url(url):
        # Force timestamps on when filtering so each matched caption line
        # carries the [M:SS] marker the caller is usually after.
        transcript = await fetch_transcript(url, force_timestamps=bool(query))
        payload = {
            "url": url,
            **_provenance(url, url, 200, None, None),
            "format": "youtube_transcript",
            "content": transcript,
        }
        if query:
            # Keep the metadata header (video id/language/source) for context
            # and filter only the transcript body that follows the "---" rule.
            header, sep, body = transcript.partition("\n---\n")
            qres = await _query_payload(
                body or transcript, query, url, kind="transcript segment"
            )
            filtered = f"{header}{sep}{qres.pop('content')}" if sep else qres.pop("content")
            payload.update(qres)
            payload["content"] = filtered
        return payload

    mode = (mode or "text").lower().strip()
    if mode not in ("text", "structured"):
        raise ToolError(f"Invalid mode '{mode}'. Use 'text' or 'structured'.")

    section = (section or "").strip() or None

    fetch_url = _normalize_reddit_url(url)
    reddit_rewritten = fetch_url != url

    try:
        fetched = await _cached_resilient_fetch(fetch_url)
    except Exception as e:
        raise ToolError(f"Fetch failed for {fetch_url}: {e}")

    status = fetched["status"]
    ctype = (fetched.get("content_type") or "").lower()
    via = fetched.get("via")

    # Bot wall we couldn't get past: a challenge was detected in the response
    # we're holding. `blocked_detected` is set whether that response came
    # direct or back from FlareSolverr — FlareSolverr returning a page is not
    # the same as a bypass, since an interactive wall (PerimeterX "Press &
    # Hold", a CAPTCHA) renders as an ordinary page it can't solve. Returning
    # the challenge would dress a failure up as data (the very thing the error
    # convention guards against), so raise instead — telling the model the
    # page is protected, and the operator what (if anything) is left to try.
    if fetched.get("blocked_detected"):
        if via == "flaresolverr":
            detail = (
                "the FlareSolverr fallback rendered it in a real browser but "
                "the page is still an interactive challenge (e.g. a "
                "'Press & Hold' / CAPTCHA) it cannot solve"
            )
        elif fetched.get("flaresolverr_error"):
            detail = (
                "the FlareSolverr fallback could not fetch it "
                f"({fetched['flaresolverr_error']})"
            )
        elif cfg.flaresolverr_url:
            detail = "the FlareSolverr fallback did not resolve it"
        else:
            detail = (
                "no FlareSolverr fallback is configured "
                "(set WEB_SEARCH_FLARESOLVERR_URL to enable it)"
            )
        raise ToolError(
            f"{fetch_url} is behind a bot/CAPTCHA wall (HTTP {status}) and "
            f"{detail}. The page content could not be retrieved."
        )

    # Document handling: PDF, Office, OpenDocument, RTF, EPUB, etc. are
    # routed to Apache Tika and returned as plain text, regardless of the
    # requested mode.
    if _is_tika_document(ctype, fetch_url):
        body = fetched.get("bytes")
        if not body and fetched.get("text"):
            body = fetched["text"].encode("utf-8", errors="replace")
        if not body:
            raise ToolError(f"Document returned no content (url={fetch_url}, status={status}).")
        extracted = await asyncio.to_thread(
            _tika_extract,
            body,
            cfg.tika_url,
            timeout=cfg.tika_timeout_seconds,
            ocr_strategy=cfg.tika_ocr_strategy,
        )
        doc_payload = {
            "url": fetch_url,
            **_provenance(url, fetch_url, status, ctype or "application/octet-stream", via),
            "format": "document_text",
        }
        if query:
            qres = await _query_payload(extracted, query, fetch_url, kind="content")
            doc_payload.update(qres)
            _set_content(doc_payload, qres["content"], offset=offset)
        else:
            _set_content(doc_payload, extracted, offset=offset)
        return doc_payload

    text = fetched.get("text") or ""

    # Reddit / JSON responses
    if reddit_rewritten or "json" in ctype:
        try:
            parsed = json.loads(text)
            compact = _compact_reddit_json(parsed) if reddit_rewritten else parsed
            json_payload = {
                "url": fetch_url,
                **_provenance(url, fetch_url, status, ctype or "application/json", via),
                "format": "json",
            }
            # query=/section= can't narrow JSON, so flag truncation without
            # the retry hint that points at them — but offset= still pages it.
            _set_content(json_payload, to_json(compact), hint=False, offset=offset)
            return json_payload
        except Exception:
            pass

    # HTML / text
    # `query` is a content search, so it overrides "structured" (which would
    # return only metadata); structured mode applies only without a query.
    if mode == "structured" and not query:
        try:
            structured = _structured_from_html(text, fetch_url)
        except Exception as e:
            raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")
        if structured.get("headings"):
            structured["headings"] = structured["headings"][: cfg.max_enrich_headings]
        return {
            "url": fetch_url,
            **_provenance(url, fetch_url, status, ctype, via),
            "format": "structured",
            "content": structured,
        }

    # mode == "text"
    try:
        full_soup = BeautifulSoup(text, "lxml")
    except Exception as e:
        raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")

    soup_title = _page_title(full_soup)

    if section:
        section_data = _find_section(full_soup, section)
        if section_data is None:
            available = [
                h.get_text(" ", strip=True)
                for h in full_soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
            ]
            available = [a for a in (a.strip() for a in available) if a][: cfg.max_enrich_headings]
            available_str = "; ".join(available) if available else "(none found)"
            raise ToolError(
                f"Section '{section}' not found on {fetch_url}. "
                f"Available headings: {available_str}. "
                "Retry with one of those headings, or omit `section` to fetch "
                "the whole page."
            )

        section_payload = {
            "url": fetch_url,
            **_provenance(url, fetch_url, status, ctype, via),
            "format": "section",
            "title": soup_title,
            "matched_heading": section_data["matched_heading"],
            "level": section_data["level"],
            "next_heading": section_data["next_heading"],
        }
        # When `query` is also given, search within the matched section.
        if query:
            qres = await _query_payload(
                section_data["text"], query, fetch_url, kind="content in that section"
            )
            section_payload.update(qres)
            section_body = f"# {section_data['matched_heading']}\n\n{qres['content']}".strip()
        else:
            section_body = f"# {section_data['matched_heading']}\n\n{section_data['text']}".strip()
        _set_content(section_payload, section_body, offset=offset)
        return section_payload

    try:
        plain, text_format = _render_text_body(text, fetch_url)
    except Exception as e:
        raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")

    # Empty-shell handling. A 200 whose HTML renders to no text is the signature
    # of a client-side-rendered (JavaScript) page — e.g. MSN, many news SPAs —
    # whose article body is loaded by an XHR after page load, so the static HTML
    # this tool fetched is an empty shell. It isn't a bot wall (no challenge for
    # _is_blocked_response to catch). When FlareSolverr is configured and we
    # fetched directly, retry through it: it renders in a real browser that runs
    # the page's JS, which often materializes the body. Re-render from whatever it
    # returns. (Checked before the query/title paths so an empty page reports as
    # empty rather than as "query matched nothing".)
    tried_render = False
    if _is_contentless(plain) and via != "flaresolverr" and cfg.flaresolverr_url:
        tried_render = True
        try:
            rendered = await _render_with_flaresolverr(fetch_url)
        except Exception:
            rendered = None
        if rendered and rendered.get("text") and not rendered.get("blocked_detected"):
            text = rendered["text"]
            status = rendered["status"]
            ctype = (rendered.get("content_type") or ctype).lower()
            via = rendered.get("via")
            soup_title = _page_title(BeautifulSoup(text, "lxml"))
            try:
                plain, text_format = _render_text_body(text, fetch_url)
            except Exception as e:
                raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")

    # Last resort: the live page (even rendered) still has no readable text. Try
    # the Wayback Machine — a snapshot may have captured content the live SPA
    # hides behind JS, or the page may since have changed/disappeared. Archived
    # content is potentially stale, so it's used only after the live attempts
    # failed, only when it actually has content, and is clearly flagged below
    # (via="archive.org" + archived_snapshot + a staleness note).
    wayback_meta = None
    tried_wayback = False
    if _is_contentless(plain) and cfg.wayback_fallback:
        tried_wayback = True
        wb = await _wayback_content(fetch_url)
        if wb:
            text = wb["text"]
            plain, text_format = wb["plain"], wb["format"]
            status = wb["status"] or status
            ctype = (wb["content_type"] or ctype).lower()
            via = "archive.org"
            soup_title = _page_title(BeautifulSoup(text, "lxml"))
            wayback_meta = wb["meta"]

    # Still no readable text after every fallback (or none was possible):
    # returning the bare <title> or stray render noise ("; ;") as `content` would
    # look like a real-but-empty article, so signal the retrieval failure instead,
    # per the error convention. The recovered title (if any) is surfaced as a
    # breadcrumb so the model at least knows what the page was about.
    if _is_contentless(plain):
        tried = [
            name
            for name, used in (
                ("the FlareSolverr browser fallback", tried_render),
                ("the Wayback Machine archive", tried_wayback),
            )
            if used
        ]
        if tried:
            hint = (
                f"Even {' and '.join(tried)} produced no readable text, so the "
                "content is loaded by a script that wasn't captured or no usable "
                "archive exists. Try another source."
            )
        else:
            hint = (
                "Enable the FlareSolverr (WEB_SEARCH_FLARESOLVERR_URL) or Wayback "
                "(WEB_SEARCH_WAYBACK_FALLBACK) fallbacks to recover JavaScript-"
                "rendered or archived pages, or fetch the article from another source."
            )
        titled = f" (page title: {soup_title!r})" if soup_title else ""
        raise ToolError(
            f"{fetch_url} returned HTTP {status} but no extractable text content — "
            f"the page renders its content client-side with JavaScript.{titled} {hint}"
        )

    # An undetected/bypassed block can't reach here: a detected wall on the
    # direct response already raised above, and a FlareSolverr success has
    # via="flaresolverr". So no block note is needed on this path.
    text_payload = {
        "url": fetch_url,
        **_provenance(url, fetch_url, status, ctype, via),
        "format": text_format,
        "title": soup_title,
    }

    if query:
        qres = await _query_payload(plain, query, fetch_url, kind="content")
        content = qres.pop("content")
        if soup_title:
            content = f"{soup_title}\n\n{content}"
        text_payload.update(qres)
        _set_content(text_payload, content, offset=offset)
    else:
        if soup_title:
            plain = f"{soup_title}\n\n{plain}"
        _set_content(text_payload, plain, offset=offset)

    # Flag archived content so the model treats it as potentially stale, not live.
    # Done after content is set so it survives the query path's `update()` and is
    # merged onto (not clobbered by) any truncation note from `_set_content`.
    if wayback_meta:
        text_payload["archived_snapshot"] = wayback_meta
        text_payload["note"] = _join_note(
            text_payload.get("note"),
            f"The live page had no readable content; this is an archived snapshot "
            f"from {wayback_meta['date']} via the Wayback Machine and may be out of date.",
        )

    return text_payload


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def fetch_page(
        url: str | list[str],
        mode: str = "text",
        section: str | None = None,
        query: str | None = None,
        offset: int | None = None,
    ) -> str:
        """Fetch one or more web page URLs or YouTube video transcripts.

        Pass several URLs as a list (up to 3) to read them;
        URLs past the first 3 are skipped. YouTube auto-detects: returns transcript
        (use query= to find topics with [M:SS] timestamps). mode="text"
        (default): page as markdown — headings/lists/tables/links kept, link
        URLs absolute (fetchable). mode="structured": metadata (title,
        description, headings, JSON-LD). section= extracts one heading's
        content. query= extracts matching passages only (keyword/regex,
        case-insensitive). offset= (char position) reads the next chunk of
        content that was truncated — pass the next_offset value from a truncated
        result to continue reading; works for any format incl. JSON/long docs.
        mode/section/query/offset apply to every URL. Use when mcp_search_web
        result needs deeper reading or to read documents
        (PDF/Word/Excel/RTF/EPUB).

        :param url: One http/https URL, or a list of them.
        :param mode: "text" or "structured".
        :param section: Optional heading text to extract.
        :param query: Optional keyword/regex to filter content.
        :param offset: Character offset to resume reading truncated content from.
        :return: For one URL: JSON {url,format,provenance?,content,query?,
            match_count?,sections?,truncated?,offset?,next_offset?,note?} (format:
            "youtube_transcript"|"markdown"|"text"|"structured"|"section"|
            "document_text"|"json"). For a list: JSON {results:[<that object|
            {url,error}>, ...],
            note?}, one entry per URL in order.
        """
        log_call(
            log, "fetch_page", url=url, mode=mode, section=section, query=query,
            offset=offset,
        )

        # Accept a single URL (string) or several (list). A bare string returns
        # one page object unchanged; a list returns {"results": [...]} so a
        # small model can read several pages in one call instead of chaining
        # fetches and risking a derail.
        single_input = isinstance(url, str)
        raw = [url] if single_input else url if isinstance(url, list) else None
        if raw is None:
            raise ToolError("Missing url.")

        # Clean: stringify-guard, strip, drop blanks, de-dupe (preserve order).
        urls: list[str] = []
        seen: set[str] = set()
        for u in raw:
            if not isinstance(u, str):
                continue
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
        if not urls:
            raise ToolError("Missing url.")

        # Context-budget cap: clamp how many URLs we fetch in one call. Extra
        # URLs are reported as skipped rather than silently dropped.
        total = len(urls)
        skipped = urls[MAX_FETCH_URLS:]
        urls = urls[:MAX_FETCH_URLS]

        # Normalize offset once: 1-based char position into the content; anything
        # below 1 (or unset) means "from the start".
        resolved_offset = offset if isinstance(offset, int) and offset > 0 else 0

        if single_input and not skipped:
            payload = await _fetch_one(urls[0], mode, section, query, resolved_offset)
            return log_result(log, "fetch_page", to_json(payload))

        # Multiple URLs: fetch concurrently, capturing each URL's failure so one
        # bad URL doesn't sink the batch (partial-success contract).
        settled = await asyncio.gather(
            *(_fetch_one(u, mode, section, query, resolved_offset) for u in urls),
            return_exceptions=True,
        )
        results: list[dict] = []
        failures = 0
        for u, res in zip(urls, settled):
            if isinstance(res, Exception):
                failures += 1
                results.append({"url": u, "error": str(res)})
            else:
                results.append(res)
        for u in skipped:
            results.append(
                {"url": u, "error": f"Skipped: exceeded the {MAX_FETCH_URLS}-URL per-call limit."}
            )

        # Only a total failure is a ToolError; any success is a normal result.
        if failures == len(urls):
            joined = "; ".join(f"{r['url']}: {r['error']}" for r in results if "error" in r)
            raise ToolError(f"All fetches failed: {joined}")

        batch: dict = {"results": results}
        if skipped:
            batch["note"] = (
                f"Fetched the first {MAX_FETCH_URLS} of {total} URLs; "
                f"{len(skipped)} skipped (per-call limit)."
            )
        return log_result(log, "fetch_page", to_json(batch))
