"""
Page-fetching MCP tool.

Exposes `fetch_page`: reads one or more web page URLs (returning markdown or a
structured metadata summary) or, when handed a YouTube video URL, that video's
transcript. PDF/Office/OpenDocument/RTF/EPUB documents are normally routed to
Apache Tika (with Firecrawl text recovery for HTML-blocked document URLs); Reddit
uses authenticated OAuth JSON when configured, then tries RSS, old.reddit HTML,
and finally oEmbed metadata for individual posts.
An optional
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
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import anyio
import httpx
import regex as safe_regex
from bs4 import BeautifulSoup
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from config import web_search_settings as cfg, server_settings
from .serialize import to_json, log_call, log_result, debug_enabled, redact_secrets
from .youtube_transcript import is_youtube_video_url, fetch_transcript
from .web_fetch import (
    DownloadTooLargeError,
    _decode_body,
    _is_tika_document,
    _sniff_document_bytes,
    _tika_extract,
)
from .page_acquire import acquire_page as _acquire_page
from .web_extract import (
    _find_section,
    _markdown_from_soup,
    _page_title,
    _plain_text_from_soup,
    _structured_from_html,
    _structured_section_from_html,
    _trim_flagged,
)

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake the failure for
# page data. A valid-but-empty result is NOT a failure and is returned as normal
# JSON. See the README "Error handling" section.

# The model-facing tool description. Built at registration time so the sibling
# tool reference (search_web) carries the client's tool-name prefix — the same
# prefix the model sees on that tool (see ServerSettings.tool_prefix). The
# return-shape part is a plain string so its JSON braces don't need escaping; the
# prefix is spliced in with `+` rather than an f-string for the same reason.
def _fetch_page_desc(prefix: str) -> str:
    return (
        "Read one web page, document, or YouTube video transcript.\n\n"
        "BASIC USE — this is all most calls need: pass one URL to get that page as "
        "markdown (headings, lists, tables, and links are kept, links as absolute "
        "URLs you can fetch). A YouTube video URL is detected automatically and "
        "returns the video's transcript. The parameters below are optional "
        "refinements — omit them all to get the whole page.\n\n"
        "PAGE OUTLINE / SECTION WORKFLOW:\n"
        "If a " + prefix + "search_web result already has page_headings/page_toc "
        "from enrich_results, you can use one of those headings with section= to read only "
        "that part. If the result was not enriched, or the user supplied a URL "
        "directly, you can call this tool with mode=\"structured\" to get the page's "
        "headings/toc without fetching the full text; then call it again with "
        "section=<heading> (and the default text mode) to read only that section. "
        "Do not fetch the outline first when you already have it.\n\n"
        "OPTIONAL PARAMETERS (leave unset unless you need them):\n"
        "• query — a keyword/phrase or regex (case-insensitive). Returns only the "
        "passages that match, instead of the full page. On a YouTube transcript it "
        "returns the matching lines tagged with [M:SS] timestamps.\n"
        "• section — a heading's text. Returns only the content under that "
        "heading (in structured mode, only that section's sub-headings/toc).\n"
        '• mode — "text" (the default) returns the page content; "structured" '
        "returns the page outline and metadata (title, description, headings, toc, "
        "JSON-LD).\n"
        "• offset — for CONTINUING a long page only. If a result comes back marked "
        '"truncated", call again with offset set to the "next_offset" value from '
        "that result to read the next chunk. Do NOT set it on a first fetch. It "
        "works for every format, including JSON and long documents.\n\n"
        "Reads ONE URL per call — to read several pages, call this tool once per "
        "URL. Use it to read an " + prefix + "search_web result in depth, or to "
        "read documents (PDF/Word/Excel/RTF/EPUB).\n\n"
        "Returns JSON {url,format,provenance?,content,query?,match_count?,sections?,"
        "truncated?,offset?,next_offset?,content_length?,note?} (format: "
        "\"youtube_transcript\"|"
        '"markdown"|"text"|"structured"|"section"|"document_text"|"json").'
    )


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
    here. When trimming actually dropped text, set ``payload["truncated"] = True``,
    ``payload["next_offset"]`` to where the next chunk begins, and
    ``payload["content_length"]`` to the full content's size in characters; also fold
    the offset hint into ``payload["note"]``. When `hint`, additionally add the
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
        payload["content_length"] = total
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
# against a long page could pin the CPU. Three bounds keep that in check:
#   * patterns that are over-long or contain a classically dangerous shape are
#     refused as regex and matched as a literal substring instead (see
#     `_compile_query`), and
#   * every `regex` search receives the remaining per-call wall-clock budget, so
#     even one pathological segment is interrupted inside the engine, and
#   * exhausting that budget raises rather than returning incomplete results.
# The matching itself is also offloaded to a worker thread (`_query_payload`),
# consistent with the sync-in-async convention.
_MAX_REGEX_QUERY_CHARS = 200

# Classic catastrophic-backtracking shape: a group that itself contains a
# quantifier and is then immediately quantified — (a+)+, (.*)+, (x* )*. We refuse
# to compile these as regex and fall back to a literal search rather than risk an
# exponential match.
_NESTED_QUANTIFIER_RE = re.compile(r"\([^()]*[+*][^()]*\)\s*[+*{]")

# Wall-clock budget for a single query's complete scan over a page's segments.
# Each search receives only the remaining time, so the bound covers one
# catastrophic match as well as cumulative work across a long page.
_QUERY_MATCH_BUDGET_SECONDS = 2.0


class QueryMatchTimeoutError(RuntimeError):
    """A model-supplied query exhausted its total regex matching budget."""


def _compile_query(query: str):
    """Compile the model's `query` into a case-insensitive pattern.

    The query may be a plain keyword/phrase or a regular expression. It is matched
    as a literal substring (via ``regex.escape``) when it is not valid regex (e.g.
    unbalanced brackets in `cost (usd)`), or when it trips the ReDoS guard — too
    long, or containing a nested-quantifier shape prone to catastrophic
    backtracking — so a naive query still works while a dangerous one can't stall
    the server.
    """
    if len(query) <= _MAX_REGEX_QUERY_CHARS and not _NESTED_QUANTIFIER_RE.search(query):
        try:
            return safe_regex.compile(query, safe_regex.IGNORECASE)
        except safe_regex.error:
            pass
    return safe_regex.compile(safe_regex.escape(query), safe_regex.IGNORECASE)


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
    # Give every search only the time remaining in the shared budget. Unlike a
    # deadline check after `re.search`, `regex` can interrupt catastrophic
    # backtracking within a single segment. Never return partial scan results.
    deadline = time.monotonic() + _QUERY_MATCH_BUDGET_SECONDS
    match_idxs = []
    for i, seg in enumerate(segments):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise QueryMatchTimeoutError(
                f"Query matching exceeded {_QUERY_MATCH_BUDGET_SECONDS:g} seconds."
            )
        try:
            if pattern.search(seg, timeout=remaining):
                match_idxs.append(i)
        except TimeoutError as exc:
            raise QueryMatchTimeoutError(
                f"Query matching exceeded {_QUERY_MATCH_BUDGET_SECONDS:g} seconds."
            ) from exc
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
    try:
        windows, match_count, total_windows = await anyio.to_thread.run_sync(
            partial(
                _extract_matches,
                text,
                query,
                context=cfg.query_context_segments,
                max_windows=cfg.max_query_matches,
            )
        )
    except QueryMatchTimeoutError:
        raise ToolError(
            f"Query regex {query!r} took too long to evaluate on {url}. "
            "Use a simpler regex or a literal keyword/phrase."
        )
    if not windows:
        raise ToolError(
            f"No {kind} matching query {query!r} found on {url}. "
            "Try a simpler keyword or a different spelling, loosen the regex, or "
            "omit `query` to retrieve the full content."
        )
    payload = {
        "query": query,
        "match_count": match_count,
        "sections": len(windows),
        "content": _format_match_windows(windows),
    }
    if total_windows > len(windows):
        payload["note"] = (
            f"{total_windows} matching sections found; showing the first "
            f"{len(windows)}. Refine `query` to narrow the results."
        )
    return payload


# ---------------------------------------------------------------------------
# Reddit handling
#
# Authenticated JSON is the durable first choice. Anonymous hosted-provider
# traffic is commonly blocked, so the remaining representations are best-effort:
# RSS, targeted old.reddit HTML, and (for posts) official oEmbed metadata.
# ---------------------------------------------------------------------------

_REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_reddit_access_token: str | None = None
_reddit_access_token_expires_at = 0.0
_reddit_access_token_credentials: tuple[str, str] | None = None


def _is_reddit_url(url: str) -> bool:
    """Whether a URL targets reddit.com or one of its real subdomains."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    # Match reddit.com and its subdomains (www./old./np. …) only. A bare
    # `endswith("reddit.com")` on the netloc would also catch look-alike domains
    # like `notreddit.com` (and miss `reddit.com:443`, since the port is part of
    # the netloc), so test the parsed hostname for an exact or dotted-suffix match.
    host = (p.hostname or "").lower()
    return host == "reddit.com" or host.endswith(".reddit.com")


def _normalize_reddit_url(url: str) -> str:
    """Return the legacy unauthenticated JSON representation of a Reddit URL."""
    if not _is_reddit_url(url):
        return url
    p = urlparse(url)
    path = p.path or "/"
    if path.endswith("/"):
        path = path[:-1]
    if not path.endswith(".json"):
        path = path + ".json"
    return urlunparse((p.scheme or "https", "www.reddit.com", path, "", p.query, ""))


def _reddit_clean_path(url: str) -> str:
    """Strip a Reddit representation suffix and return its canonical path."""
    path = urlparse(url).path or "/"
    for suffix in (".json", ".rss"):
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    return path


def _is_reddit_post_url(url: str) -> bool:
    return "/comments/" in _reddit_clean_path(url).lower()


def _reddit_oauth_url(url: str) -> str:
    """Build the OAuth Data API URL corresponding to a public Reddit URL."""
    parsed = urlparse(url)
    path = _reddit_clean_path(url).rstrip("/") or "/"
    params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    params["raw_json"] = "1"
    if "/comments/" in path.lower():
        params.setdefault("limit", "500")
    return urlunparse(("https", "oauth.reddit.com", path, "", urlencode(params), ""))


def _reddit_rss_url(url: str) -> str:
    """Build a Reddit Atom feed URL for a post or listing."""
    parsed = urlparse(url)
    path = _reddit_clean_path(url).rstrip("/")
    path += "/.rss" if _is_reddit_post_url(url) else ".rss"
    return urlunparse(("https", "www.reddit.com", path or "/.rss", "", parsed.query, ""))


def _reddit_old_url(url: str) -> str:
    """Build the old.reddit HTML URL for a post or listing."""
    parsed = urlparse(url)
    return urlunparse(
        ("https", "old.reddit.com", _reddit_clean_path(url), "", parsed.query, "")
    )


def _reddit_oembed_url(url: str) -> str:
    """Metadata-only fallback when Reddit's post/comments JSON is blocked."""
    p = urlparse(url)
    path = _reddit_clean_path(url)
    post_url = urlunparse(
        (p.scheme or "https", "www.reddit.com", path, "", p.query, "")
    )
    return "https://www.reddit.com/oembed?" + urlencode({"url": post_url})


def _reddit_fragment_text(fragment: str, base_url: str) -> str:
    """Render Reddit's HTML-in-XML/body fragment without surrounding chrome."""
    soup = BeautifulSoup(fragment or "", "lxml")
    root = soup.select_one(".md") or soup.body or soup
    if cfg.markdown:
        return _markdown_from_soup(root, base_url).strip()
    return _plain_text_from_soup(root).strip()


def _compact_reddit_rss(xml: str, url: str) -> dict | None:
    """Convert a Reddit Atom feed into the compact Reddit result shape."""
    try:
        soup = BeautifulSoup(xml, "xml")
        entries = soup.find_all("entry")
        if not entries:
            return None

        parsed_entries = []
        for entry in entries:
            id_tag = entry.find("id")
            author_tag = entry.find("author")
            name_tag = author_tag.find("name") if author_tag else None
            author = name_tag.get_text(strip=True) if name_tag else None
            if author and author.startswith("/u/"):
                author = author[3:]
            title_tag = entry.find("title")
            content_tag = entry.find("content")
            link_tag = entry.find("link", href=True)
            updated_tag = entry.find("updated")
            parsed_entries.append(
                {
                    "id": id_tag.get_text(strip=True) if id_tag else None,
                    "title": title_tag.get_text(" ", strip=True) if title_tag else None,
                    "author": author,
                    "url": link_tag.get("href") if link_tag else None,
                    "body": _reddit_fragment_text(
                        content_tag.get_text() if content_tag else "", url
                    ),
                    "updated": updated_tag.get_text(strip=True) if updated_tag else None,
                }
            )

        # A post feed begins with the submission (t3), followed by comments (t1).
        if (parsed_entries[0].get("id") or "").startswith("t3_"):
            first = parsed_entries[0]
            post = {
                "title": first["title"],
                "author": first["author"],
                "permalink": first["url"],
                "selftext": first["body"],
                "created": first["updated"],
            }
            comments = [
                {
                    "id": item["id"],
                    "author": item["author"],
                    "body": item["body"],
                    "permalink": item["url"],
                    "created": item["updated"],
                }
                for item in parsed_entries[1:]
                if (item.get("id") or "").startswith("t1_")
            ]
            return {
                "post": post,
                "comments": comments,
                "comments_returned": len(comments),
                "comments_incomplete": True,
            }

        return {"items": parsed_entries, "items_returned": len(parsed_entries)}
    except Exception:
        return None


def _old_reddit_entry_body(node, url: str) -> str:
    entry = node.find("div", class_="entry", recursive=False)
    body = entry.select_one(".usertext-body") if entry else None
    return _reddit_fragment_text(str(body), url) if body else ""


def _compact_reddit_html(html: str, url: str) -> dict | None:
    """Extract posts/comments from old.reddit HTML without page chrome."""
    try:
        soup = BeautifulSoup(html, "lxml")
        links = soup.select("div.thing.link[data-fullname^='t3_']")
        comments = []
        for node in soup.select("div.thing.comment[data-fullname^='t1_']"):
            author_node = node.select_one(".author")
            score_node = node.select_one(".score.unvoted, .score.likes, .score.dislikes")
            depth = sum(
                1
                for parent in node.parents
                if getattr(parent, "get", lambda *_: None)("data-type") == "comment"
            )
            comments.append(
                {
                    "id": node.get("data-fullname"),
                    "author": (
                        author_node.get_text(" ", strip=True)
                        if author_node
                        else node.get("data-author")
                    ),
                    "score": score_node.get_text(" ", strip=True) if score_node else None,
                    "depth": depth,
                    "body": _old_reddit_entry_body(node, url),
                    "permalink": node.get("data-permalink"),
                }
            )

        if _is_reddit_post_url(url) and links:
            node = links[0]
            title_node = node.select_one("a.title")
            reported = node.get("data-comments-count")
            try:
                reported = int(reported) if reported is not None else None
            except (TypeError, ValueError):
                reported = None
            post = {
                "title": title_node.get_text(" ", strip=True) if title_node else None,
                "author": node.get("data-author"),
                "subreddit": node.get("data-subreddit"),
                "score": (
                    int(node.get("data-score"))
                    if str(node.get("data-score", "")).isdigit()
                    else None
                ),
                "num_comments": reported,
                "permalink": node.get("data-permalink"),
                "url": node.get("data-url"),
                "selftext": _old_reddit_entry_body(node, url),
            }
            result = {
                "post": post,
                "comments": comments,
                "comments_returned": len(comments),
                "more_comment_placeholders": len(soup.select(".morecomments")),
            }
            if reported is not None:
                result["comments_reported"] = reported
                result["comments_incomplete"] = len(comments) < reported
            return result

        items = []
        for node in links:
            title_node = node.select_one("a.title")
            items.append(
                {
                    "id": node.get("data-fullname"),
                    "title": title_node.get_text(" ", strip=True) if title_node else None,
                    "author": node.get("data-author"),
                    "subreddit": node.get("data-subreddit"),
                    "score": node.get("data-score"),
                    "num_comments": node.get("data-comments-count"),
                    "permalink": node.get("data-permalink"),
                    "url": node.get("data-url"),
                    "selftext": _old_reddit_entry_body(node, url),
                }
            )
        return {"items": items, "items_returned": len(items)} if items else None
    except Exception:
        return None


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
            more_children = 0

            def walk(node, depth=0):
                nonlocal more_children
                if not isinstance(node, dict):
                    return
                kind = node.get("kind")
                d = node.get("data") or {}
                if kind == "more":
                    more_children += len(d.get("children") or [])
                    return
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

            result = {
                "post": post,
                "comments": comments,
                "comments_returned": len(comments),
            }
            if post and post.get("num_comments") is not None:
                result["comments_reported"] = post["num_comments"]
                result["comments_incomplete"] = len(comments) < post["num_comments"]
            if more_children:
                result["more_children"] = more_children
            return result
    except Exception:
        pass
    return data


async def _reddit_token() -> str:
    """Return a cached application-only Reddit OAuth token."""
    global _reddit_access_token
    global _reddit_access_token_expires_at
    global _reddit_access_token_credentials

    client_id = cfg.reddit_client_id.strip()
    client_secret = cfg.reddit_client_secret.strip()
    user_agent = cfg.reddit_user_agent.strip()
    missing = [
        name
        for name, value in (
            ("WEB_SEARCH_REDDIT_CLIENT_ID", client_id),
            ("WEB_SEARCH_REDDIT_CLIENT_SECRET", client_secret),
            ("WEB_SEARCH_REDDIT_USER_AGENT", user_agent),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Incomplete Reddit OAuth configuration: " + ", ".join(missing)
        )

    credentials = (client_id, client_secret)
    now = time.monotonic()
    if (
        _reddit_access_token
        and _reddit_access_token_credentials == credentials
        and now < _reddit_access_token_expires_at - 30
    ):
        return _reddit_access_token

    async with httpx.AsyncClient(verify=cfg.verify_ssl) as client:
        try:
            response = await client.post(
                _REDDIT_TOKEN_URL,
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                timeout=cfg.http_timeout_seconds,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Reddit OAuth network error: {exc.__class__.__name__} - {str(exc)[:200]}"
            ) from exc
    if response.status_code >= 400:
        # Distinguish auth failures from other server errors
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Reddit OAuth authentication failed (HTTP {response.status_code}). "
                "Check that client_id and client_secret are correct and the app is approved."
            )
        raise RuntimeError(
            f"Reddit OAuth token endpoint returned HTTP {response.status_code}"
        )
    try:
        data = response.json()
        token = str(data["access_token"]).strip()
        expires_in = max(60, int(data.get("expires_in", 3600)))
    except Exception as exc:
        raise RuntimeError("Reddit OAuth token endpoint returned invalid JSON") from exc
    if not token:
        raise RuntimeError("Reddit OAuth token endpoint returned an empty access token")

    _reddit_access_token = token
    _reddit_access_token_expires_at = time.monotonic() + expires_in
    _reddit_access_token_credentials = credentials
    return token


async def _fetch_reddit_oauth(url: str) -> dict:
    """Fetch one Reddit JSON representation using application-only OAuth."""
    token = await _reddit_token()
    fetch_url = _reddit_oauth_url(url)
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": cfg.reddit_user_agent.strip(),
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(verify=cfg.verify_ssl) as client:
        async with client.stream(
            "GET", fetch_url, headers=headers, timeout=cfg.http_timeout_seconds
        ) as response:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if cfg.max_download_bytes and total > cfg.max_download_bytes:
                    raise DownloadTooLargeError(
                        f"Reddit OAuth response exceeds the {cfg.max_download_bytes}-byte "
                        "WEB_SEARCH_MAX_DOWNLOAD_BYTES cap."
                    )
                chunks.append(chunk)
            status = response.status_code
            ctype = response.headers.get("content-type", "application/json")
    if status >= 400:
        raise RuntimeError(f"Reddit OAuth JSON returned HTTP {status}")
    body = b"".join(chunks)
    return {
        "url": fetch_url,
        "status": status,
        "content_type": ctype,
        "text": _decode_body(body, ctype),
        "bytes": None,
        "via": "reddit_oauth",
        "blocked_detected": False,
    }


def _reddit_oauth_requested() -> bool:
    return any(
        value.strip()
        for value in (
            cfg.reddit_client_id,
            cfg.reddit_client_secret,
            cfg.reddit_user_agent,
        )
    )


def _reddit_failure_trace(name: str, exc: BaseException) -> str:
    """Bound one provider failure so a debug note cannot dominate the result."""
    detail = " ".join((str(exc).strip() or type(exc).__name__).split())
    if len(detail) > 500:
        detail = detail[:497] + "..."
    return f"{name} failed ({detail})"


async def _acquire_reddit(url: str) -> tuple[dict, Any, str, list[str]]:
    """Try OAuth JSON -> RSS -> old.reddit HTML -> oEmbed."""
    errors: list[str] = []
    trace: list[str] = []

    if _reddit_oauth_requested():
        try:
            fetched = await _fetch_reddit_oauth(url)
            parsed = json.loads(fetched.get("text") or "")
            trace.append("OAuth JSON succeeded")
            return fetched, _compact_reddit_json(parsed), "oauth", trace
        except Exception as exc:
            failure = _reddit_failure_trace("OAuth JSON", exc)
            errors.append(failure)
            trace.append(failure)
    else:
        trace.append("OAuth JSON skipped (not configured)")

    rss_url = _reddit_rss_url(url)
    try:
        fetched = await _acquire_page(rss_url)
        if fetched.get("status", 0) >= 400:
            raise RuntimeError(f"HTTP {fetched['status']}")
        compact = await anyio.to_thread.run_sync(
            _compact_reddit_rss, fetched.get("text") or "", url
        )
        if compact is None:
            raise RuntimeError("feed contained no Reddit entries")
        trace.append("RSS succeeded")
        return fetched, compact, "rss", trace
    except Exception as exc:
        failure = _reddit_failure_trace("RSS", exc)
        errors.append(failure)
        trace.append(failure)

    old_url = _reddit_old_url(url)
    try:
        fetched = await _acquire_page(old_url)
        if fetched.get("status", 0) >= 400:
            raise RuntimeError(f"HTTP {fetched['status']}")
        compact = await anyio.to_thread.run_sync(
            _compact_reddit_html, fetched.get("text") or "", url
        )
        if compact is None:
            raise RuntimeError("page contained no extractable Reddit entries")
        trace.append("old.reddit HTML succeeded")
        return fetched, compact, "old_html", trace
    except Exception as exc:
        failure = _reddit_failure_trace("old.reddit HTML", exc)
        errors.append(failure)
        trace.append(failure)

    if _is_reddit_post_url(url):
        oembed_url = _reddit_oembed_url(url)
        try:
            fetched = await _acquire_page(oembed_url)
            if fetched.get("status", 0) >= 400:
                raise RuntimeError(f"HTTP {fetched['status']}")
            trace.append("oEmbed succeeded")
            return fetched, json.loads(fetched.get("text") or ""), "oembed", trace
        except Exception as exc:
            failure = _reddit_failure_trace("oEmbed", exc)
            errors.append(failure)
            trace.append(failure)

    raise RuntimeError("; ".join(errors) or "no Reddit acquisition method succeeded")


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

# `_render_text_mode` and `_parse_section` parse the page HTML and run
# markdownify — both CPU-bound — so callers offload them to a worker thread
# (`anyio.to_thread.run_sync`), per the sync-in-async convention: on the
# single-process server an inline parse of a large page would stall every other
# in-flight tool call. Each parses the HTML exactly once and pulls the title from
# the same soup, rather than the old title-parse-then-render-parse double pass.

def _render_text_mode(html: str, base_url: str) -> tuple[str | None, str, str]:
    """Parse page HTML once and return ``(title, body, format)`` for text mode.

    The title is read before the markdown/plain render mutates the soup, so the
    document is parsed a single time. CPU-bound — offload via the caller.
    """
    soup = BeautifulSoup(html, "lxml")
    title = _page_title(soup)
    if cfg.markdown:
        return title, _markdown_from_soup(soup, base_url), "markdown"
    return title, _plain_text_from_soup(soup), "text"


def _parse_section(html: str, section: str, base_url: str = "") -> tuple[str | None, dict | None, list[str]]:
    """Parse page HTML once for section mode.

    Returns ``(title, section_data, available_headings)``. ``section_data`` is the
    `_find_section` result (or None when the heading wasn't found, in which case
    ``available_headings`` lists the page's headings for the error message).
    `base_url` is threaded to `_find_section` so the section's reference/citation
    links resolve to followable URLs. CPU-bound — offload via the caller.
    """
    soup = BeautifulSoup(html, "lxml")
    title = _page_title(soup)
    section_data = _find_section(soup, section, base_url)
    if section_data is not None:
        return title, section_data, []
    available = [
        h.get_text(" ", strip=True)
        for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    ]
    available = [a for a in (a.strip() for a in available) if a][: cfg.max_enrich_headings]
    return title, None, available


# A run of two or more letters — i.e. an actual word. A rendered body without
# even one has no readable text and is an empty shell, even when it isn't
# strictly whitespace: a JS page that only partly rendered (a direct fetch of a
# client-side app, or a FlareSolverr snapshot taken before a lazy XHR resolved)
# can leave stray punctuation like "; ;" that defeats a bare `.strip()` test.
_WORD_RE = re.compile(r"[^\W\d_]{2,}")


def _is_contentless(body: str) -> bool:
    """True if `body` has no readable text (not one word of 2+ letters)."""
    return not _WORD_RE.search(body)


def _unsupported_media_error(
    fetch_url: str, status: int | None, ctype: str, size: int | None
) -> ToolError:
    """Actionable ToolError for fetched bytes this tool cannot extract."""
    details = []
    if status is not None:
        details.append(f"HTTP {status}")
    details.append(f"Content-Type {ctype!r}" if ctype else "no Content-Type header")
    if size is not None:
        details.append(f"{size} bytes")
    return ToolError(
        f"{fetch_url} was fetched successfully ({', '.join(details)}), but this "
        "tool cannot extract that media type. fetch_page can read HTML/text/JSON, "
        "YouTube transcripts, and document files such as PDF, Word, Excel, "
        "PowerPoint, OpenDocument, RTF, and EPUB via Tika. Use a browser/direct "
        "download for this file, or fetch a text/HTML/document version of the "
        "same content."
    )


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
            _set_content(payload, filtered, offset=offset, hint=False)
        else:
            _set_content(payload, transcript, offset=offset, hint=False)
        if payload.get("truncated"):
            payload["note"] = _join_note(
                payload.get("note"),
                "To jump straight to relevant transcript lines instead, retry with `query=`.",
            )
        return payload

    mode = (mode or "text").lower().strip()
    if mode not in ("text", "structured"):
        raise ToolError(f"Invalid mode '{mode}'. Use 'text' or 'structured'.")

    section = (section or "").strip() or None

    reddit_request = _is_reddit_url(url)
    if reddit_request:
        try:
            fetched, compact, reddit_source, reddit_trace = await _acquire_reddit(url)
        except Exception as exc:
            backend_passwords = tuple(
                urlparse(endpoint).password or ""
                for endpoint in (
                    cfg.flaresolverr_url,
                    cfg.firecrawl_api_url,
                    cfg.tika_url,
                    cfg.classifier_api_url,
                )
                if endpoint
            )
            detail = redact_secrets(
                exc,
                cfg.reddit_client_secret,
                cfg.firecrawl_api_key,
                cfg.classifier_api_key,
                *backend_passwords,
            )
            raise ToolError(f"Fetch failed for {url}: {detail}")

        source_url = {
            "oauth": _reddit_oauth_url(url),
            "rss": _reddit_rss_url(url),
            "old_html": _reddit_old_url(url),
            "oembed": _reddit_oembed_url(url),
        }[reddit_source]
        status = fetched.get("status")
        ctype = (fetched.get("content_type") or "").lower()
        payload = {
            "url": source_url,
            **_provenance(url, source_url, status, ctype, fetched.get("via")),
            "format": "json",
        }
        _set_content(payload, to_json(compact), hint=False, offset=offset)
        if reddit_source == "rss":
            payload["note"] = _join_note(
                payload.get("note"),
                "Returned Reddit's RSS feed; it may contain only an initial "
                "snapshot of the comments.",
            )
        elif reddit_source == "old_html":
            payload["note"] = _join_note(
                payload.get("note"),
                "Reddit OAuth JSON and RSS were unavailable; returned targeted "
                "old.reddit HTML extraction, which may omit unloaded comments.",
            )
        elif reddit_source == "oembed":
            payload["note"] = _join_note(
                payload.get("note"),
                "Reddit OAuth JSON, RSS, and old.reddit HTML were unavailable; "
                "returning official oEmbed post metadata without comments.",
            )
        if debug_enabled():
            backend_passwords = tuple(
                urlparse(endpoint).password or ""
                for endpoint in (
                    cfg.flaresolverr_url,
                    cfg.firecrawl_api_url,
                    cfg.tika_url,
                    cfg.classifier_api_url,
                )
                if endpoint
            )
            trace_note = redact_secrets(
                "Reddit fallback trace: " + "; ".join(reddit_trace) + ".",
                cfg.reddit_client_secret,
                cfg.firecrawl_api_key,
                cfg.classifier_api_key,
                *backend_passwords,
            )
            payload["note"] = _join_note(payload.get("note"), trace_note)
        return payload

    fetch_url = url
    fetched: dict | None = None
    primary_exc: Exception | None = None
    try:
        # The acquisition module owns the complete browser-first policy: direct
        # resource probe, FlareSolverr, quality assessment, optional classifier,
        # circuit breaker, hedge, and Firecrawl fallback.
        fetched = await _acquire_page(fetch_url)
    except Exception as exc:
        primary_exc = exc

    if primary_exc is not None:
        backend_passwords = tuple(
            urlparse(endpoint).password or ""
            for endpoint in (
                cfg.flaresolverr_url,
                cfg.firecrawl_api_url,
                cfg.tika_url,
                cfg.classifier_api_url,
            )
            if endpoint
        )
        detail = redact_secrets(
            primary_exc,
            cfg.firecrawl_api_key,
            cfg.classifier_api_key,
            *backend_passwords,
        )
        detail = detail.strip() or type(primary_exc).__name__
        raise ToolError(f"Fetch failed for {fetch_url}: {detail}")

    if fetched is None:  # Defensive invariant for type checkers and future edits.
        raise ToolError(f"Fetch failed for {fetch_url}: no acquisition result")

    status = fetched["status"]
    ctype = (fetched.get("content_type") or "").lower()
    via = fetched.get("via")

    # Direct non-HTML resources do not go through browser recovery. Never return
    # an HTTP error payload as if it were the requested JSON/document/text file.
    if status is not None and status >= 400:
        raise ToolError(
            f"{fetch_url} returned HTTP {status}; page content was not returned."
        )

    # Firecrawl's `html` format is cleaned page HTML and may omit the document
    # head. A body-level widget can then contain the first remaining <title>
    # element (Best Buy's embedded reCAPTCHA does this), which BeautifulSoup's
    # broad `soup.title` lookup would mistake for the page title. Firecrawl
    # provides the canonical page title separately in response metadata, so
    # retain it as the authoritative title for this fetch source.
    firecrawl_title = (
        fetched.get("title").strip()
        if via == "firecrawl"
        and isinstance(fetched.get("title"), str)
        and fetched["title"].strip()
        else None
    )

    # Document handling: PDF, Office, OpenDocument, RTF, EPUB, etc. are
    # routed to Apache Tika and returned as plain text, regardless of the
    # requested mode. Detected by content-type/extension or — for a document
    # served with a generic/wrong content-type and no telling extension — by a
    # magic-byte sniff of the raw bytes.
    if (
        fetched.get("resource_kind") in ("document", "document_text")
        or _is_tika_document(ctype, fetch_url)
        or _sniff_document_bytes(fetched.get("bytes"))
    ):
        if fetched.get("resource_kind") == "document_text":
            extracted = fetched.get("text") or ""
            if not extracted:
                raise ToolError(
                    f"Document recovery returned no text (url={fetch_url}, status={status})."
                )
        else:
            body = fetched.get("bytes")
            if not body and fetched.get("text"):
                body = fetched["text"].encode("utf-8", errors="replace")
            if not body:
                raise ToolError(f"Document returned no content (url={fetch_url}, status={status}).")
            try:
                extracted = await asyncio.to_thread(
                    _tika_extract,
                    body,
                    cfg.tika_url,
                    timeout=cfg.tika_timeout_seconds,
                    ocr_strategy=cfg.tika_ocr_strategy,
                    max_output_bytes=cfg.max_download_bytes,
                )
            except Exception as exc:
                tika_password = urlparse(cfg.tika_url).password or ""
                detail = redact_secrets(exc, tika_password)
                raise ToolError(f"Document extraction failed for {fetch_url}: {detail}")
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

    if fetched.get("bytes") is not None and fetched.get("text") is None:
        raise _unsupported_media_error(
            fetch_url, status, ctype, len(fetched.get("bytes") or b"")
        )

    text = fetched.get("text") or ""

    # JSON responses
    if "json" in ctype:
        try:
            parsed = json.loads(text)
            compact = parsed
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
    try:
        soup_title, plain, text_format = await anyio.to_thread.run_sync(
            _render_text_mode, text, fetch_url
        )
        soup_title = firecrawl_title or soup_title
    except Exception as e:
        raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")

    # Acquisition already rejected empty/block/error renders. Keep a final
    # extraction-level invariant in case a parser removes all nominally visible
    # content. Structured mode may legitimately consist only of metadata.
    if _is_contentless(plain) and not (mode == "structured" and not query):
        raise ToolError(
            f"{fetch_url} produced no extractable text after page acquisition."
        )

    if mode == "structured" and not query:
        # With a `section`, scope the metadata to that heading's subtree so the
        # headings/toc describe just that section, not the whole page.
        if section:
            try:
                structured, available = await anyio.to_thread.run_sync(
                    _structured_section_from_html, text, fetch_url, section
                )
            except Exception as e:
                raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")
            if structured is None:
                available = available[: cfg.max_enrich_headings]
                available_str = "; ".join(available) if available else "(none found)"
                raise ToolError(
                    f"Section '{section}' not found on {fetch_url}. "
                    f"Available headings: {available_str}. "
                    "Retry with one of those headings, or omit `section` to fetch "
                    "the whole page."
                )
        else:
            try:
                structured = await anyio.to_thread.run_sync(
                    _structured_from_html, text, fetch_url
                )
            except Exception as e:
                raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")
        if firecrawl_title:
            structured["title"] = firecrawl_title
        # Cap headings and the (heading-derived) toc together so they stay aligned.
        if structured.get("headings"):
            structured["headings"] = structured["headings"][: cfg.max_enrich_headings]
        if structured.get("toc"):
            structured["toc"] = structured["toc"][: cfg.max_enrich_headings]
        payload = {
            "url": fetch_url,
            **_provenance(url, fetch_url, status, ctype, via),
            "format": "structured",
            "content": structured,
        }
        return payload

    # mode == "text"
    if section:
        try:
            soup_title, section_data, available = await anyio.to_thread.run_sync(
                _parse_section, text, section, fetch_url
            )
            soup_title = firecrawl_title or soup_title
        except Exception as e:
            raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")
        if section_data is None:
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

    return text_payload


def register(mcp: FastMCP) -> None:
    @mcp.tool(description=_fetch_page_desc(server_settings.tool_prefix))
    async def fetch_page(
        url: str,
        mode: str = "text",
        section: str | None = None,
        query: str | None = None,
        offset: int | None = None,
    ) -> str:
        """Fetch one web page / YouTube transcript. The model-facing guidance
        lives in the @mcp.tool(description=...) above.

        :param url: One http/https URL to read.
        :param mode: "text" (default; returns the page content) or "structured"
            (returns the page outline and metadata, including headings/toc). Omit
            for "text".
        :param section: Optional. A heading's text to return only that section.
            Omit to get the whole page.
        :param query: Optional. A keyword/phrase or regex; returns only the
            matching passages. Omit to get the whole page.
        :param offset: Optional, for continuing a truncated result only. Set it
            to the "next_offset" value from a previous response to read the next
            chunk. Leave unset on a first fetch.
        """
        log_call(
            log, "fetch_page", url=url, mode=mode, section=section, query=query,
            offset=offset,
        )

        # One URL per call. Reading several pages is done by calling the tool
        # again rather than batching, which small models handled unreliably.
        # `_fetch_one` validates the URL (scheme/blank) and raises ToolError.
        resolved_offset = offset if isinstance(offset, int) and offset > 0 else 0
        payload = await _fetch_one(url, mode, section, query, resolved_offset)
        return log_result(log, "fetch_page", to_json(payload))
