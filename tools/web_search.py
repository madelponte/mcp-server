"""
Agentic Web Search MCP tool.

Exposes `search_web` and `fetch_page`, backed by a self-hosted SearXNG instance
with an optional FlareSolverr fallback for Cloudflare-protected pages, Reddit
JSON handling, and Apache Tika document extraction (PDF, Office, OpenDocument,
RTF, EPUB). Translated from the Open WebUI tool; status/citation event emitters
were removed.
"""

import asyncio
import json
import re
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import web_search_settings as cfg
from .cache import TTLCache

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


def _trim(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[... truncated at {limit} chars ...]"


def _clamp_count(requested: Optional[int], maximum: int) -> int:
    """Resolve a model-requested count against its configured maximum.

    ``None`` (the model didn't ask) yields ``maximum``, preserving the old
    fixed-amount behavior. Otherwise the value is clamped to ``[1, maximum]`` so
    the model can dial the amount down but never request more than the cap — the
    guard that keeps an oversized response from overwhelming its context window.
    """
    if requested is None:
        return maximum
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        return maximum
    if requested < 1:
        return 1
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


def _toc_from_jsonld(jsonld: list) -> Optional[list[str]]:
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


def _page_title(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    return None


def _page_description(soup: BeautifulSoup) -> Optional[str]:
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

async def _httpx_fetch(
    url: str, timeout: float, user_agent: str, verify_ssl: bool
) -> tuple[int, dict, bytes, str]:
    """Direct fetch via httpx. Returns (status, headers, body_bytes, content_type)."""
    headers = {
        "User-Agent": user_agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "application/json;q=0.9,application/pdf;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=timeout, verify=verify_ssl, headers=headers
    ) as client:
        resp = await client.get(url)
        ctype = resp.headers.get("content-type", "")
        return resp.status_code, dict(resp.headers), resp.content, ctype


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
    flaresolverr_url: Optional[str],
    flaresolverr_timeout_ms: int,
) -> dict:
    """Try direct httpx; on a Cloudflare wall, retry through FlareSolverr if configured."""
    try:
        status, headers, body, ctype = await _httpx_fetch(
            url, timeout=timeout, user_agent=user_agent, verify_ssl=verify_ssl
        )
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
    """Run a SearXNG JSON query and return [{url, title, snippet}]."""
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
        out.append(
            {
                "url": r.get("url"),
                "title": r.get("title"),
                "snippet": (r.get("content") or "").strip(),
            }
        )
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
    each caller still formats the cached result for its own mode needs.
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


async def _enrich_result(url: Optional[str]) -> Optional[dict]:
    """Fetch a URL just enough to extract its page title and description."""
    if not url:
        return None
    try:
        fetched = await _cached_resilient_fetch(url)
    except Exception as e:
        return {"error": str(e)}

    ctype = (fetched.get("content_type") or "").lower()
    if _is_tika_document(ctype, url):
        return {"title": None, "description": None}
    text = fetched.get("text")
    if not text:
        return None
    soup = BeautifulSoup(text, "lxml")
    return {"title": _page_title(soup), "description": _page_description(soup)}


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def search_web(
        query: str,
        time_range: str | None = None,
        category: str | None = None,
        num_results: int | None = None,
    ) -> str:
        """
        Search the web and return a ranked list of results.

        Use this when you don't already know the answer, the question concerns
        current events, or you need to verify a fact. Craft a focused query
        (a few keywords) — do NOT just echo the user's whole prompt. If the
        first search isn't useful, you may call this again with a refined query.

        Each result includes: url, title, snippet, and the target page's own
        page_title and page_description, so you can decide which links are worth
        reading in full. To actually read a link, pass its url to fetch_page.

        :param query: A concise search query (keywords, not a full sentence).
        :param time_range: Optional recency filter. One of "day", "week",
            "month", or "year" to restrict results to that window (use "day"
            for "today"/"latest" news). Pass "all" (or omit) for no time
            restriction. Defaults to the server's configured value.
        :param category: Optional SearXNG category to search in, e.g. "general"
            (default), "news", "science", "it", "social media", "videos",
            "images", "music", "files", or "map". Use "news" for current-events
            reporting. Comma-separate to combine categories. Defaults to the
            server's configured value.
        :param num_results: How many search results to return. Request fewer for
            a focused lookup, or omit to use the server default. Value is capped by
            the server.
        :return: JSON string of results.
        """
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
                num_results=_clamp_count(num_results, cfg.max_num_results),
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
            return json.dumps({"query": query, **applied, "results": []})

        for r in results:
            if r.get("snippet"):
                r["snippet"] = _trim(r["snippet"], cfg.max_snippet_chars)

        # Fetch each result's page just enough to attach its own title and
        # description, so the model can judge relevance before reading in full.
        tasks = [_enrich_result(r.get("url")) for r in results]
        enriched = await asyncio.gather(*tasks, return_exceptions=True)
        for i, data in enumerate(enriched):
            if isinstance(data, Exception):
                results[i]["page_meta_error"] = str(data)
                continue
            if not data:
                continue
            if data.get("error"):
                results[i]["page_meta_error"] = data["error"]
                continue
            results[i]["page_title"] = data.get("title")
            results[i]["page_description"] = data.get("description")

        return json.dumps(
            {"query": query, **applied, "results": results},
            ensure_ascii=False,
            indent=2,
        )

    @mcp.tool()
    async def fetch_page(url: str, mode: str = "text") -> str:
        """
        Fetch the contents of a web page (or a URL returned by search_web).

        Choose the mode that fits your need:
        - "text":       plain readable text of the page. Best for reading an
                        article or extracting facts. Also used automatically for
                        document links (PDF, Word, Excel, PowerPoint, OpenDocument,
                        RTF, EPUB), which are extracted via Apache Tika.
        - "structured": metadata only — title, description, heading outline,
                        and JSON-LD structured data (schema.org Recipe, HowTo,
                        Article, etc.).

        :param url: Absolute URL to fetch (http/https).
        :param mode: "text" or "structured".
        :return: JSON string with the result.
        """
        if not url or not isinstance(url, str):
            raise ToolError("Missing url.")
        url = url.strip()
        if not re.match(r"^https?://", url, re.I):
            raise ToolError(f"Invalid URL: {url}")

        mode = (mode or "text").lower().strip()
        if mode not in ("text", "structured"):
            raise ToolError(f"Invalid mode '{mode}'. Use 'text' or 'structured'.")

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
            extracted = _trim(extracted, cfg.max_page_chars)
            return json.dumps(
                {
                    "url": fetch_url,
                    "original_url": url,
                    "status": status,
                    "content_type": ctype or "application/octet-stream",
                    "via": via,
                    "format": "document_text",
                    "content": extracted,
                },
                ensure_ascii=False,
            )

        text = fetched.get("text") or ""

        # Reddit / JSON responses
        if reddit_rewritten or "json" in ctype:
            try:
                parsed = json.loads(text)
                compact = _compact_reddit_json(parsed) if reddit_rewritten else parsed
                dumped = json.dumps(compact, ensure_ascii=False, indent=2)
                dumped = _trim(dumped, cfg.max_page_chars)
                return json.dumps(
                    {
                        "url": fetch_url,
                        "original_url": url,
                        "status": status,
                        "content_type": ctype or "application/json",
                        "via": via,
                        "format": "json",
                        "content": dumped,
                    },
                    ensure_ascii=False,
                )
            except Exception:
                pass

        # HTML / text
        if mode == "structured":
            try:
                structured = _structured_from_html(text, fetch_url)
            except Exception as e:
                raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")
            if structured.get("headings"):
                structured["headings"] = structured["headings"][: cfg.max_enrich_headings]
            return json.dumps(
                {
                    "url": fetch_url,
                    "original_url": url,
                    "status": status,
                    "content_type": ctype,
                    "via": via,
                    "format": "structured",
                    "content": structured,
                },
                ensure_ascii=False,
            )

        # mode == "text"
        try:
            full_soup = BeautifulSoup(text, "lxml")
        except Exception as e:
            raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")

        soup_title = _page_title(full_soup)

        try:
            plain = _plain_text_from_html(text)
        except Exception as e:
            raise ToolError(f"Failed to parse HTML for {fetch_url}: {e}")

        if soup_title:
            plain = f"{soup_title}\n\n{plain}"
        plain = _trim(plain, cfg.max_page_chars)

        if fetched.get("blocked_detected") and via == "direct":
            note = "NOTE: page appeared to be Cloudflare-blocked and FlareSolverr fallback did not succeed."
        else:
            note = None

        return json.dumps(
            {
                "url": fetch_url,
                "original_url": url,
                "status": status,
                "content_type": ctype,
                "via": via,
                "format": "text",
                "title": soup_title,
                "content": plain,
                "note": note,
            },
            ensure_ascii=False,
        )
