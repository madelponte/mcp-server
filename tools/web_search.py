"""
Agentic Web Search MCP tool.

Exposes `search_web` and `fetch_page`, backed by a self-hosted SearXNG instance
with an optional FlareSolverr fallback for Cloudflare-protected pages, Reddit
JSON handling, and Apache Tika document extraction (PDF, Office, OpenDocument,
RTF, EPUB). Translated from the Open WebUI tool; status/citation event emitters
were removed.
"""

import asyncio
import ipaddress
import json
import logging
import re
import socket
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import web_search_settings as cfg
from .cache import TTLCache
from .serialize import to_json, log_call, log_result, debug_enabled
from .youtube_transcript import is_youtube_video_url, fetch_transcript

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake the failure for
# search/page data. A valid-but-empty result (e.g. a search with zero hits) is
# NOT a failure and is still returned as normal JSON. See the README
# "Error handling" section.

# SearXNG time_range values the API accepts; "" means no time restriction.
# We also accept a few friendly synonyms for "no restriction" from the model.
SEARXNG_TIME_RANGES = {"", "day", "week", "month", "year"}
_TIME_RANGE_NO_RESTRICTION = {"", "all", "any", "none", "anytime"}

# Process-wide cache of fetched pages, keyed by URL. Shared by fetch_page and
# search_web's result enrichment so a repeated fetch within a task skips the
# network round-trip. Fetch settings all come from static config, so the URL
# alone is a sufficient key. See the README "Caching" section.
_page_cache = TTLCache(cfg.cache_ttl_seconds, cfg.cache_max_entries)

# Hard cap on how many URLs a single fetch_page call will fetch when passed a
# list. A context-budget guard: more than a few full pages in one response blows
# a small model's window. Kept as a literal (not a config valve) so the exact
# number can be stated in the tool docstring the model reads — keep the "3" in
# that docstring in sync if you change this.
MAX_FETCH_URLS = 3

# ---------------------------------------------------------------------------
# Cloudflare detection
# ---------------------------------------------------------------------------

CLOUDFLARE_STATUS_CODES = {403, 503, 520, 521, 522, 523, 524, 525, 526, 527}
CLOUDFLARE_MARKERS = (
    "cf-ray",
    "cf-chl",
    "just a moment",
    "attention required",
    "cf-browser-verification",
    "cf_chl_opt",
    "challenge-platform",
    "please enable cookies",
    "/cdn-cgi/challenge-platform",
)


def _is_cloudflare_block(status: int, text: str, headers: dict) -> bool:
    """Best-effort detection that a response is a Cloudflare/CAPTCHA wall."""
    hdr_lower = {k.lower(): str(v).lower() for k, v in (headers or {}).items()}
    server = hdr_lower.get("server", "")
    if "cloudflare" in server and status in CLOUDFLARE_STATUS_CODES:
        return True
    if status in CLOUDFLARE_STATUS_CODES:
        t = (text or "")[:8000].lower()
        if any(m in t for m in CLOUDFLARE_MARKERS):
            return True
    if status == 200 and text:
        t = text[:4000].lower()
        hits = sum(1 for m in CLOUDFLARE_MARKERS if m in t)
        if hits >= 2:
            return True
    return False


def _normalize_reddit_url(url: str) -> str:
    """Reddit blocks HTML scraping; force the .json endpoint for reddit links."""
    try:
        p = urlparse(url)
    except Exception:
        return url
    host = (p.netloc or "").lower()
    if not host.endswith("reddit.com"):
        return url
    host = "www.reddit.com"
    path = p.path or "/"
    if path.endswith("/"):
        path = path[:-1]
    if not path.endswith(".json"):
        path = path + ".json"
    return urlunparse((p.scheme or "https", host, path, "", p.query, ""))


def _trim_flagged(text: str, limit: int) -> tuple[str, bool]:
    """Trim `text` to `limit` chars, also reporting whether anything was dropped.

    Returns ``(text, truncated)``. ``limit <= 0`` disables trimming. The marker
    appended on truncation keeps the old visible cue in the text itself; the
    boolean lets callers also flag it structurally on the payload.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + f"\n\n[... truncated at {limit} chars ...]", True


def _trim(text: str, limit: int) -> str:
    return _trim_flagged(text, limit)[0]


# Retry hint appended to a fetch_page result whose content had to be truncated to
# fit cfg.max_page_chars. It points the model at the two ways to pull just the
# part it needs instead of silently losing the rest or re-fetching the whole page.
_TRUNCATION_HINT = (
    "Content was truncated to fit the size limit; the rest was dropped. To get "
    "more, retry with `query=` (a keyword/phrase or regex — returns only the "
    "matching passages) or `section=` (a heading — returns only that section)."
)


def _set_content(payload: dict, content: str, *, hint: bool = True) -> None:
    """Trim `content` to the page-size cap and store it on `payload`.

    When trimming actually dropped text, set ``payload["truncated"] = True`` and
    (when `hint`) fold the retry hint into ``payload["note"]`` so the model
    targets the rest with `query=`/`section=` rather than losing it silently.
    `hint` is False for formats those params can't narrow (e.g. JSON).
    """
    trimmed, truncated = _trim_flagged(content, cfg.max_page_chars)
    payload["content"] = trimmed
    if truncated:
        payload["truncated"] = True
        if hint:
            existing = payload.get("note")
            payload["note"] = f"{existing} {_TRUNCATION_HINT}" if existing else _TRUNCATION_HINT


def _clamp_count(
    requested: int | None,
    maximum: int,
    *,
    minimum: int,
    default: int | None = None,
) -> int:
    """Resolve a model-requested count against its configured maximum.

    ``None`` (the model didn't ask) yields ``default`` when one is configured,
    otherwise ``maximum`` (the old fixed-amount behavior). Either way the result
    is clamped to ``[minimum, maximum]`` so the model can dial the amount down
    but never request more than the cap — the guard that keeps an oversized
    response from overwhelming its context window. ``minimum`` is 1 for the
    result count and 0 for enrichment (0 meaningfully disables it).
    """
    if requested is None:
        if default is None:
            return maximum
        requested = default
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        return maximum
    if requested < minimum:
        return minimum
    return min(requested, maximum)


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def _extract_jsonld(soup: BeautifulSoup) -> list:
    """Pull JSON-LD structured data blocks from <script type=application/ld+json>."""
    out = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            out.extend(parsed)
        else:
            out.append(parsed)
    return out


def _headings_outline(soup: BeautifulSoup, max_items: int = 40) -> list[dict]:
    """Build a lightweight 'table of contents' from heading tags."""
    outline = []
    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = " ".join(h.get_text(" ", strip=True).split())
        if not text:
            continue
        outline.append({"level": int(h.name[1]), "text": text})
        if len(outline) >= max_items:
            break
    return outline


def _toc_from_jsonld(jsonld: list) -> list[str] | None:
    """Extract useful 'table of contents'-like info from JSON-LD when possible."""
    toc: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            t = obj.get("@type")
            if t in ("Recipe", "HowTo") or (isinstance(t, list) and ("Recipe" in t or "HowTo" in t)):
                steps = obj.get("recipeInstructions") or obj.get("step") or []
                if isinstance(steps, list):
                    for s in steps:
                        if isinstance(s, str):
                            toc.append(s.strip())
                        elif isinstance(s, dict):
                            name = s.get("name") or s.get("text") or ""
                            if name:
                                toc.append(str(name).strip())
            if isinstance(t, str) and t.endswith("Article"):
                hl = obj.get("headline")
                if hl and hl not in toc:
                    toc.append(str(hl).strip())
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(jsonld)
    return toc or None


def _page_title(soup: BeautifulSoup) -> str | None:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    return None


def _page_description(soup: BeautifulSoup) -> str | None:
    for sel in [
        ("meta", {"name": "description"}),
        ("meta", {"property": "og:description"}),
        ("meta", {"name": "twitter:description"}),
    ]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None


def _plain_text_from_html(html: str) -> str:
    """Strip scripts/styles/nav and return readable text."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    text = root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _norm_heading(s: str) -> str:
    """Normalize a heading for fuzzy comparison."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("¶#§*•·.:;-—–_ ")
    return s


def _find_section(soup: BeautifulSoup, section: str) -> dict | None:
    """Locate a heading matching `section` and return text up to the next equal/higher heading."""
    if not section:
        return None

    target = _norm_heading(section)
    if not target:
        return None

    for t in soup(["script", "style", "noscript", "template", "iframe", "svg"]):
        t.decompose()

    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not headings:
        return None

    matched = None
    for h in headings:
        if _norm_heading(h.get_text(" ", strip=True)) == target:
            matched = h
            break
    if matched is None:
        for h in headings:
            ht = _norm_heading(h.get_text(" ", strip=True))
            if target in ht or ht in target:
                if ht and len(ht) >= 3:
                    matched = h
                    break
    if matched is None:
        return None

    matched_level = int(matched.name[1])
    matched_text = " ".join(matched.get_text(" ", strip=True).split())

    pieces: list[str] = []
    next_heading_text: str | None = None

    for el in matched.find_all_next():
        if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            try:
                lvl = int(el.name[1])
            except ValueError:
                lvl = 99
            if lvl <= matched_level:
                next_heading_text = " ".join(el.get_text(" ", strip=True).split())
                break
            sub = " ".join(el.get_text(" ", strip=True).split())
            if sub:
                pieces.append(f"\n## {sub}\n")
            continue
        if el.name in ("p", "li", "pre", "code", "blockquote", "td", "th", "dd", "dt", "figcaption"):
            txt = el.get_text(" ", strip=True)
            if txt:
                pieces.append(txt)

    if not pieces:
        collected: list[str] = []
        for el in matched.find_all_next(string=False):
            if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                try:
                    lvl = int(el.name[1])
                except ValueError:
                    lvl = 99
                if lvl <= matched_level:
                    if next_heading_text is None:
                        next_heading_text = " ".join(el.get_text(" ", strip=True).split())
                    break
            txt = el.get_text(" ", strip=True) if hasattr(el, "get_text") else ""
            if txt and txt not in collected:
                collected.append(txt)
        body_text = "\n\n".join(collected)
    else:
        body_text = "\n\n".join(pieces)

    body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()

    return {
        "matched_heading": matched_text,
        "level": matched_level,
        "text": body_text,
        "next_heading": next_heading_text,
    }


# ---------------------------------------------------------------------------
# Query (lexical / regex) filtering
#
# `fetch_page`'s optional `query` does server-side extractive filtering: rather
# than returning a whole page (or transcript), it returns only the segments that
# lexically match the query, each with a little surrounding context. This keeps a
# long page or video transcript from flooding the model's context window when it
# only needs the parts about one topic.
# ---------------------------------------------------------------------------

def _compile_query(query: str) -> re.Pattern:
    """Compile the model's `query` into a case-insensitive pattern.

    The query may be a plain keyword/phrase or a regular expression. If it isn't
    valid regex (e.g. unbalanced brackets in `cost (usd)`), fall back to matching
    it as a literal substring so a naive query still works.
    """
    try:
        return re.compile(query, re.IGNORECASE)
    except re.error:
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
    match_idxs = [i for i, seg in enumerate(segments) if pattern.search(seg)]
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


def _query_payload(text: str, query: str, url: str, *, kind: str) -> dict:
    """Filter `text` to the segments matching `query`; raise if nothing matches.

    Returns fields to merge into a fetch_page result: the formatted ``content``
    (match windows joined by labeled separators), the total ``match_count``, the
    number of ``sections`` (windows) shown, and a ``note`` when some windows were
    dropped to stay within the cap. ``kind`` names what was searched (e.g.
    "content", "transcript segment") for the not-found error.
    """
    windows, match_count, total_windows = _extract_matches(
        text,
        query,
        context=cfg.query_context_segments,
        max_windows=cfg.max_query_matches,
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


def _structured_from_html(html: str, url: str) -> dict:
    """Return a structured representation of the page."""
    soup = BeautifulSoup(html, "lxml")
    jsonld = _extract_jsonld(soup)
    return {
        "url": url,
        "title": _page_title(soup),
        "description": _page_description(soup),
        "headings": _headings_outline(soup),
        "jsonld": jsonld if jsonld else None,
        "toc": _toc_from_jsonld(jsonld),
    }


# Document types Apache Tika can extract that are NOT served as text/html.
# Tika auto-detects the format from the bytes, so we route any of these to it
# rather than treating the response as HTML/text.
TIKA_DOCUMENT_CTYPES = (
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument",  # docx/xlsx/pptx (prefix match)
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.oasis.opendocument",  # odt/ods/odp (prefix match)
    "application/rtf",
    "text/rtf",
    "application/epub+zip",
)

TIKA_DOCUMENT_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
    ".epub",
)


def _is_tika_document(ctype: str, url: str) -> bool:
    """True if the response looks like a binary document Tika should extract."""
    ctype = (ctype or "").lower()
    if any(ctype.startswith(c) for c in TIKA_DOCUMENT_CTYPES):
        return True
    # Fallback on the URL path when the server sends a generic content-type
    # (e.g. application/octet-stream) but the extension is telling.
    path = (urlparse(url).path or "").lower()
    return path.endswith(TIKA_DOCUMENT_EXTENSIONS)


def _tika_extract(
    data: bytes,
    tika_url: str,
    *,
    timeout: float = 90.0,
    ocr_strategy: str = "no_ocr",
) -> str:
    """Extract plain text from a document byte stream via Apache Tika.

    No Content-Type is sent: Tika auto-detects the format from the bytes, so
    this handles PDF, Office (doc/docx/xls/xlsx/ppt/pptx), OpenDocument, RTF,
    EPUB, etc. with one path.

    `ocr_strategy` maps to Tika's X-Tika-PDFOcrStrategy header. The default
    "no_ocr" extracts only embedded text, which is fast and avoids OCR of
    image-heavy PDFs blowing past the timeout. Set it to "auto" or
    "ocr_and_text_extraction" if you actually need scanned-image OCR.
    """
    headers = {"Accept": "text/plain"}
    if ocr_strategy:
        headers["X-Tika-PDFOcrStrategy"] = ocr_strategy
    try:
        resp = httpx.put(
            f"{tika_url.rstrip('/')}/tika",
            content=data,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.text.strip()
        return text if text else "[Document contained no extractable text]"
    except Exception as e:
        return f"[Document extraction failed: {e}]"


# ---------------------------------------------------------------------------
# Fetching (with FlareSolverr fallback)
# ---------------------------------------------------------------------------

# SSRF guard. The model's URLs can come from search results or page content it
# just read, so an attacker can use indirect prompt injection to steer a fetch at
# internal targets — cloud metadata (169.254.169.254), localhost, or LAN hosts.
# We resolve the host and refuse any non-publicly-routable address, on the
# initial URL AND every redirect hop (follow_redirects can otherwise 302 a public
# URL into a private one). Cap redirects while we follow them by hand. Operators
# can opt specific hosts/IPs/CIDRs back in via WEB_SEARCH_SSRF_ALLOWLIST.
MAX_REDIRECTS = 20


class SSRFError(RuntimeError):
    """A fetch target's host resolved to a non-public (blocked) address."""


@lru_cache(maxsize=8)
def _parse_allowlist(raw: str) -> tuple[frozenset[str], tuple[Any, ...]]:
    """Parse WEB_SEARCH_SSRF_ALLOWLIST into (hostnames, ip-networks).

    Entries are comma/whitespace-separated; each is either an IP/CIDR (matched
    against resolved addresses) or a hostname (matched verbatim against the URL
    host, case-insensitive). Cached since the raw string is a fixed config value.
    """
    hosts: set[str] = set()
    nets: list = []
    for item in re.split(r"[,\s]+", raw.strip()):
        if not item:
            continue
        try:
            nets.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            hosts.add(item.lower())
    return frozenset(hosts), tuple(nets)


def _ip_of(addr: str):
    """Parse a resolved address into an ip_address, unwrapping IPv4-mapped IPv6
    (e.g. ::ffff:169.254.169.254) so the v4 rules apply. None if unparseable."""
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])  # drop any IPv6 scope id
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip


def _addr_is_blocked(addr: str, allowed_nets: tuple = ()) -> bool:
    """True if `addr` is not a globally-routable public IP (loopback, private,
    link-local, reserved, CGNAT, …) and not covered by an allowlisted network —
    or unparseable, in which case refuse."""
    ip = _ip_of(addr)
    if ip is None:
        return True
    if ip.is_global:
        return False
    return not any(ip.version == n.version and ip in n for n in allowed_nets)


async def _assert_url_allowed(url: str) -> None:
    """Raise SSRFError unless `url` is http(s) and its host is allowed: either
    explicitly allowlisted, or resolving only to public addresses. Resolution is
    offloaded so the event loop isn't blocked."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFError(f"Refusing non-http(s) URL: {url!r}")
    host = parsed.hostname or ""
    if not host:
        raise SSRFError(f"Refusing URL with no host: {url!r}")
    allowed_hosts, allowed_nets = _parse_allowlist(cfg.ssrf_allowlist)
    if host.lower() in allowed_hosts:
        return
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except socket.gaierror as e:
        raise SSRFError(f"Could not resolve host {host!r}: {e}")
    blocked = sorted(
        {info[4][0] for info in infos if _addr_is_blocked(info[4][0], allowed_nets)}
    )
    if blocked:
        raise SSRFError(
            f"Refusing to fetch {host!r}: resolves to non-public address(es) "
            f"{', '.join(blocked)} (allowlist via WEB_SEARCH_SSRF_ALLOWLIST)"
        )


async def _httpx_fetch(
    url: str, timeout: float, user_agent: str, verify_ssl: bool
) -> tuple[int, dict, bytes, str]:
    """Direct fetch via httpx. Returns (status, headers, body_bytes, content_type).

    Redirects are followed by hand (not httpx's follow_redirects) so each hop's
    target passes the SSRF guard before we connect to it. The caller validates
    the initial URL; here we re-validate every redirect destination.
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "application/json;q=0.9,application/pdf;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(
        follow_redirects=False, timeout=timeout, verify=verify_ssl, headers=headers
    ) as client:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            resp = await client.get(current)
            location = resp.headers.get("location")
            if resp.is_redirect and location:
                current = str(resp.url.join(location))
                await _assert_url_allowed(current)
                continue
            ctype = resp.headers.get("content-type", "")
            return resp.status_code, dict(resp.headers), resp.content, ctype
    raise RuntimeError(f"Exceeded {MAX_REDIRECTS} redirects fetching {url!r}")


async def _flaresolverr_fetch(
    url: str, flaresolverr_url: str, max_timeout_ms: int, http_timeout: float
) -> tuple[int, dict, str]:
    """Use FlareSolverr to fetch a Cloudflare-protected page."""
    endpoint = flaresolverr_url.rstrip("/") + "/v1"
    payload = {"cmd": "request.get", "url": url, "maxTimeout": max_timeout_ms}
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        resp = await client.post(
            endpoint, json=payload, headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
    if data.get("status") != "ok":
        msg = data.get("message", "unknown FlareSolverr error")
        raise RuntimeError(f"FlareSolverr failed: {msg}")
    sol = data.get("solution") or {}
    status = int(sol.get("status") or 0)
    hdrs = sol.get("headers") or {}
    body = sol.get("response") or ""
    return status, hdrs, body


async def _resilient_fetch(
    url: str,
    *,
    timeout: float,
    user_agent: str,
    verify_ssl: bool,
    flaresolverr_url: str | None,
    flaresolverr_timeout_ms: int,
) -> dict:
    """Try direct httpx; on a Cloudflare wall, retry through FlareSolverr if configured."""
    # SSRF guard the initial URL before EITHER fetch path. FlareSolverr is the
    # more dangerous one (it renders in real Chrome and runs JS), so the check
    # must gate it too — hence here, not inside _httpx_fetch.
    await _assert_url_allowed(url)
    try:
        status, headers, body, ctype = await _httpx_fetch(
            url, timeout=timeout, user_agent=user_agent, verify_ssl=verify_ssl
        )
    except SSRFError:
        # A redirect hop resolved to a blocked host — refuse outright rather than
        # handing the same URL to FlareSolverr, which would follow it in-browser.
        raise
    except Exception as e:
        if flaresolverr_url:
            try:
                fs_status, _, fs_html = await _flaresolverr_fetch(
                    url,
                    flaresolverr_url=flaresolverr_url,
                    max_timeout_ms=flaresolverr_timeout_ms,
                    http_timeout=max(timeout, flaresolverr_timeout_ms / 1000 + 10),
                )
                return {
                    "url": url,
                    "status": fs_status,
                    "content_type": "text/html",
                    "text": fs_html,
                    "bytes": None,
                    "via": "flaresolverr",
                    "blocked_detected": True,
                }
            except Exception as fe:
                raise RuntimeError(
                    f"Both direct and FlareSolverr fetches failed: {e!r} / {fe!r}"
                )
        raise

    # Office/OpenDocument content-types contain the substring "xml" (e.g.
    # application/vnd.openxmlformats-officedocument...), so the loose checks
    # below would mis-classify them as text and corrupt the bytes via UTF-8
    # decode. Documents we hand to Tika must stay binary.
    is_textlike = (
        not _is_tika_document(ctype, url)
        and (
            ctype.startswith("text/")
            or "json" in ctype
            or "xml" in ctype
            or "html" in ctype
        )
    )

    if not is_textlike:
        return {
            "url": url,
            "status": status,
            "content_type": ctype,
            "text": None,
            "bytes": body,
            "via": "direct",
            "blocked_detected": False,
        }

    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        text = body.decode("latin-1", errors="replace")

    blocked = _is_cloudflare_block(status, text, headers)
    if blocked and flaresolverr_url:
        try:
            fs_status, _, fs_html = await _flaresolverr_fetch(
                url,
                flaresolverr_url=flaresolverr_url,
                max_timeout_ms=flaresolverr_timeout_ms,
                http_timeout=max(timeout, flaresolverr_timeout_ms / 1000 + 10),
            )
            return {
                "url": url,
                "status": fs_status,
                "content_type": "text/html",
                "text": fs_html,
                "bytes": None,
                "via": "flaresolverr",
                "blocked_detected": True,
            }
        except Exception as fe:
            return {
                "url": url,
                "status": status,
                "content_type": ctype,
                "text": text,
                "bytes": None,
                "via": "direct",
                "blocked_detected": True,
                "flaresolverr_error": str(fe),
            }

    return {
        "url": url,
        "status": status,
        "content_type": ctype,
        "text": text,
        "bytes": None,
        "via": "direct",
        "blocked_detected": blocked,
    }


# ---------------------------------------------------------------------------
# SearXNG query
# ---------------------------------------------------------------------------

async def _searxng_query(
    base_url: str,
    query: str,
    *,
    num_results: int,
    categories: str,
    language: str,
    time_range: str,
    safe_search: int,
    timeout: float,
    verify_ssl: bool,
    user_agent: str,
) -> list[dict]:
    """Run a SearXNG JSON query and return [{url, title, snippet, engine}]."""
    params = {"q": query, "format": "json", "safesearch": str(safe_search)}
    if categories:
        params["categories"] = categories
    if language:
        params["language"] = language
    if time_range:
        params["time_range"] = time_range

    url = base_url.rstrip("/") + "/search"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=timeout, verify=verify_ssl, headers=headers) as client:
        resp = await client.get(url, params=params)
        if resp.status_code == 403:
            raise RuntimeError(
                "SearXNG returned 403. Make sure `search.formats` in its settings.yml "
                "includes `json`."
            )
        resp.raise_for_status()
        data = resp.json()

    items = data.get("results") or []
    out: list[dict] = []
    for r in items[:num_results]:
        item = {
            "url": r.get("url"),
            "title": r.get("title"),
            "snippet": (r.get("content") or "").strip(),
        }
        # News/dated sources populate a publish date; general-web engines omit
        # it. SearXNG exposes it inconsistently as either "publishedDate" or
        # "pubdate" depending on the engine, so surface whichever is present.
        published = r.get("publishedDate") or r.get("pubdate")
        if published:
            item["published_date"] = published
        out.append(item)
    return out


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
# Tool registration
# ---------------------------------------------------------------------------

async def _cached_resilient_fetch(url: str) -> dict:
    """``_resilient_fetch`` with a process-wide TTL cache keyed by URL.

    Caching the raw fetch (rather than the formatted tool output) means a
    re-fetch of the same URL — common in agent loops, and shared between
    fetch_page and search_web enrichment — skips the network round-trip, while
    each caller still formats the cached result for its own mode/section needs.
    Only successful fetches are cached; a failure propagates and is not stored.
    """
    cached = _page_cache.get(url)
    if cached is not None:
        return cached
    fetched = await _resilient_fetch(
        url,
        timeout=cfg.http_timeout_seconds,
        user_agent=cfg.user_agent,
        verify_ssl=cfg.verify_ssl,
        flaresolverr_url=cfg.flaresolverr_url or None,
        flaresolverr_timeout_ms=cfg.flaresolverr_timeout_ms,
    )
    _page_cache.set(url, fetched)
    return fetched


async def _enrich_result(url: str | None) -> dict | None:
    """Fetch a URL just enough to extract structured metadata."""
    if not url:
        return None
    try:
        fetched = await _cached_resilient_fetch(url)
    except Exception as e:
        return {"error": str(e)}

    ctype = (fetched.get("content_type") or "").lower()
    if _is_tika_document(ctype, url):
        return {"title": None, "description": None, "headings": [], "toc": None}
    text = fetched.get("text")
    if not text:
        return None
    return _structured_from_html(text, url)


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


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def search_web(
        query: str,
        time_range: str | None = None,
        category: str | None = None,
        num_results: int | None = None,
        enrich_results: int | None = None,
    ) -> str:
        """Search the web. Use for unknown facts, current events, or verification.

        Results include url/title/snippet + optional page metadata (headings,
        description) for top results. Then use mcp_fetch_page to read full content.

        Query: short keywords only (not sentences). time_range: "day"/"week"/
        "month"/"year"/"all". category: "general"|"news"|"science"|"it"|"social
        media"|"videos"|"images"|"music"|"files"|"map" (comma-separate).
        num_results/enrich_results: max counts (capped; enrich fetches metadata).

        :param query: Keywords.
        :param time_range: Recency filter.
        :param category: Category (comma-separate).
        :param num_results: Max results (capped).
        :param enrich_results: Top N to enrich with metadata (capped).
        :return: JSON {query, time_range, category, results:[{url,title,snippet,
            published_date?,page_title?,page_description?,page_headings?,page_toc?}]}
        """
        log_call(
            log,
            "search_web",
            query=query,
            time_range=time_range,
            category=category,
            num_results=num_results,
            enrich_results=enrich_results,
        )
        query = (query or "").strip()
        if not query:
            raise ToolError("Empty query.")

        # Resolve optional overrides, falling back to the configured env valves.
        if time_range is None:
            resolved_time_range = cfg.searxng_time_range
        else:
            tr = time_range.strip().lower()
            if tr in _TIME_RANGE_NO_RESTRICTION:
                resolved_time_range = ""
            elif tr in SEARXNG_TIME_RANGES:
                resolved_time_range = tr
            else:
                raise ToolError(
                    f"Invalid time_range {time_range!r}. Use one of: day, week, "
                    "month, year, or all (no restriction)."
                )

        if category is None:
            resolved_categories = cfg.searxng_categories
        else:
            resolved_categories = category.strip() or cfg.searxng_categories

        try:
            results = await _searxng_query(
                base_url=cfg.searxng_url,
                query=query,
                num_results=_clamp_count(num_results, cfg.max_num_results, minimum=1),
                categories=resolved_categories,
                language=cfg.searxng_language,
                time_range=resolved_time_range,
                safe_search=cfg.searxng_safesearch,
                timeout=cfg.http_timeout_seconds,
                verify_ssl=cfg.verify_ssl,
                user_agent=cfg.user_agent,
            )
        except Exception as e:
            raise ToolError(f"SearXNG query failed for {query!r}: {e}")

        applied = {"time_range": resolved_time_range or "all", "category": resolved_categories}

        if not results:
            return log_result(
                log, "search_web", to_json({"query": query, **applied, "results": []})
            )

        for r in results:
            if r.get("snippet"):
                r["snippet"] = _trim(r["snippet"], cfg.max_snippet_chars)

        enrich_n = min(
            _clamp_count(
                enrich_results,
                cfg.max_enrich_results,
                minimum=0,
                default=cfg.default_enrich_results,
            ),
            len(results),
        )
        if enrich_n > 0:
            tasks = [_enrich_result(r.get("url")) for r in results[:enrich_n]]
            enriched = await asyncio.gather(*tasks, return_exceptions=True)
            for i, data in enumerate(enriched):
                if isinstance(data, Exception):
                    results[i]["page_meta_error"] = str(data)
                    continue
                if not data:
                    continue
                headings = (data.get("headings") or [])[: cfg.max_enrich_headings]
                results[i]["page_title"] = data.get("title")
                results[i]["page_description"] = data.get("description")
                if headings:
                    results[i]["page_headings"] = headings
                if data.get("toc"):
                    results[i]["page_toc"] = data["toc"][:20]

        return log_result(
            log,
            "search_web",
            to_json({"query": query, **applied, "results": results}),
        )

    @mcp.tool()
    async def fetch_page(
        url: str | list[str],
        mode: str = "text",
        section: str | None = None,
        query: str | None = None,
    ) -> str:
        """Fetch one or more web page URLs or YouTube video transcripts.

        Pass several URLs as a list (up to 3) to read them; 
        URLs past the first 3 are skipped. YouTube auto-detects: returns transcript
        (use query= to find topics with [M:SS] timestamps). mode="text"
        (default): readable text/docs. mode="structured": metadata (title,
        description, headings, JSON-LD). section= extracts one heading's
        content. query= extracts matching passages only (keyword/regex,
        case-insensitive). mode/section/query apply to every URL. Use when
        mcp_search_web result needs deeper reading or to read documents
        (PDF/Word/Excel/RTF/EPUB).

        :param url: One http/https URL, or a list of them.
        :param mode: "text" or "structured".
        :param section: Optional heading text to extract.
        :param query: Optional keyword/regex to filter content.
        :return: For one URL: JSON {url,format,provenance?,content,query?,
            match_count?,sections?,truncated?,note?} (format:
            "youtube_transcript"|"text"|"structured"|"section"|"document_text"|
            "json"). For a list: JSON {results:[<that object|{url,error}>, ...],
            note?}, one entry per URL in order.
        """
        log_call(log, "fetch_page", url=url, mode=mode, section=section, query=query)

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

        if single_input and not skipped:
            payload = await _fetch_one(urls[0], mode, section, query)
            return log_result(log, "fetch_page", to_json(payload))

        # Multiple URLs: fetch concurrently, capturing each URL's failure so one
        # bad URL doesn't sink the batch (partial-success contract).
        settled = await asyncio.gather(
            *(_fetch_one(u, mode, section, query) for u in urls),
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

    async def _fetch_one(
        url: str,
        mode: str = "text",
        section: str | None = None,
        query: str | None = None,
    ) -> dict:
        """Fetch a single URL and return its result payload dict.

        Shared by fetch_page for both the one-URL and list-of-URLs paths. Raises
        ToolError on a genuine failure (caught per-URL in the batch path).
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
                qres = _query_payload(
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
                qres = _query_payload(extracted, query, fetch_url, kind="content")
                doc_payload.update(qres)
                _set_content(doc_payload, qres["content"])
            else:
                _set_content(doc_payload, extracted)
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
                # the retry hint that points at them.
                _set_content(json_payload, to_json(compact), hint=False)
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
                qres = _query_payload(
                    section_data["text"], query, fetch_url, kind="content in that section"
                )
                section_payload.update(qres)
                section_body = f"# {section_data['matched_heading']}\n\n{qres['content']}".strip()
            else:
                section_body = f"# {section_data['matched_heading']}\n\n{section_data['text']}".strip()
            _set_content(section_payload, section_body)
            return section_payload

        try:
            plain = _plain_text_from_html(text)
        except Exception as e:
            raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")

        if fetched.get("blocked_detected") and via == "direct":
            note = "NOTE: page appeared to be Cloudflare-blocked and FlareSolverr fallback did not succeed."
        else:
            note = None

        text_payload = {
            "url": fetch_url,
            **_provenance(url, fetch_url, status, ctype, via),
            "format": "text",
            "title": soup_title,
        }

        if query:
            qres = _query_payload(plain, query, fetch_url, kind="content")
            content = qres.pop("content")
            if soup_title:
                content = f"{soup_title}\n\n{content}"
            text_payload.update(qres)
            # A query-filter note takes precedence over the Cloudflare note, but
            # keep the block warning if there was no query note to report.
            text_payload["note"] = text_payload.get("note") or note
            _set_content(text_payload, content)
        else:
            if soup_title:
                plain = f"{soup_title}\n\n{plain}"
            text_payload["note"] = note
            _set_content(text_payload, plain)

        return text_payload
