"""
Agentic Web Search MCP tool.

Exposes `search_web`, backed by a self-hosted SearXNG instance. Results are
url/title/snippet, optionally enriched with structured page metadata (title,
description, heading outline) for the top hits so the model can decide what to
read. The companion `fetch_page` tool (in `tools/fetch_page.py`) reads the full
content of a result. The HTTP fetching both tools share lives in
`tools/web_fetch.py`; the HTML extraction in `tools/web_extract.py`.

Translated from the Open WebUI tool; status/citation event emitters were removed.
"""

import asyncio
import logging
from typing import Annotated
from urllib.parse import urlparse

import anyio
import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from config import web_search_settings as cfg, server_settings
from .serialize import to_json, log_call, log_result
from .web_fetch import _enrich_fetch, _is_tika_document
from .web_extract import _structured_from_html, _trim

log = logging.getLogger(__name__)

# SearXNG is queried repeatedly during agent loops. Keep one async client per
# event loop / verify setting so calls reuse keep-alive connections without
# binding a client to the wrong loop in tests or alternate server runners. The
# client class is part of the key so tests that monkeypatch httpx.AsyncClient get
# a fresh mock-backed client.
_searxng_clients: dict[tuple[int, bool, int], httpx.AsyncClient] = {}


def _searxng_client(verify_ssl: bool) -> httpx.AsyncClient:
    loop_id = id(asyncio.get_running_loop())
    key = (loop_id, verify_ssl, id(httpx.AsyncClient))
    client = _searxng_clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(verify=verify_ssl)
        _searxng_clients[key] = client
    return client


# The model-facing tool description. Built at registration time so the sibling
# tool reference (fetch_page) carries the client's tool-name prefix — the same
# prefix the model sees on that tool (see ServerSettings.tool_prefix). The
# return-shape part is a plain string so its JSON braces don't need escaping; the
# prefix is spliced in with `+` rather than an f-string for the same reason.
def _search_web_desc(prefix: str) -> str:
    return (
        "Search the web. Use for unknown facts, current events, or verification.\n\n"
        "Results include url/title/snippet + optional page metadata (headings, "
        "description) for top results. Then use " + prefix + "fetch_page to read "
        "full content.\n\n"
        "Query: short keywords only (not sentences). Search operators are "
        "supported and sharpen results — use them when they fit: site:domain.com "
        '(restrict to one site), "exact phrase" (quoted), -word (exclude a term), '
        "term OR term (alternatives), filetype:pdf (a file type). time_range: "
        '"day"/"week"/"month"/"year"/"all". category: "general"|"news"|"science"|'
        '"it"|"social media"|"videos"|"map" '
        "(comma-separate).\n"
        "num_results/enrich_results: max counts (see per-arg caps; enrich fetches "
        "metadata). page: result page (1-based, default 1) — set page=2 to get the "
        "next batch of results for the SAME query when the first page wasn't "
        "useful, instead of reformulating.\n\n"
        "Returns JSON {query, time_range, category, page, results:[{url,title,"
        "snippet,published_date?,page_title?,page_description?,page_headings?,"
        "page_toc?}]}"
    )

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake the failure for
# search data. A valid-but-empty result (e.g. a search with zero hits) is NOT a
# failure and is still returned as normal JSON. See the README "Error handling"
# section.

# SearXNG time_range values the API accepts; "" means no time restriction.
# We also accept a few friendly synonyms for "no restriction" from the model.
SEARXNG_TIME_RANGES = {"", "day", "week", "month", "year"}
_TIME_RANGE_NO_RESTRICTION = {"", "all", "any", "none", "anytime"}
_YOUTUBE_ENGINE_SELECTOR = "!yt"
_YOUTUBE_ENGINE_ALIASES = ("!yt", "!youtube")
_YOUTUBE_HOSTS = ("youtube.com", "youtube-nocookie.com", "youtu.be")


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


def _category_includes_videos(categories: str) -> bool:
    """True when a comma-separated SearXNG category list includes videos."""
    return any(part.strip().lower() == "videos" for part in categories.split(","))


def _youtube_engine_query(query: str) -> str:
    """Restrict a video search to SearXNG's YouTube engine."""
    lowered = query.strip().lower()
    if any(
        lowered == alias or lowered.startswith(f"{alias} ")
        for alias in _YOUTUBE_ENGINE_ALIASES
    ):
        return query
    return f"{_YOUTUBE_ENGINE_SELECTOR} {query}"


def _is_youtube_result_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    host = host.lower()
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in _YOUTUBE_HOSTS)


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
    page: int = 1,
) -> list[dict]:
    """Run a SearXNG JSON query and return [{url, title, snippet, engine}]."""
    params = {"q": query, "format": "json", "safesearch": str(safe_search)}
    if categories:
        params["categories"] = categories
    if language:
        params["language"] = language
    if time_range:
        params["time_range"] = time_range
    if page and page > 1:
        params["pageno"] = str(page)

    url = base_url.rstrip("/") + "/search"
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    client = _searxng_client(verify_ssl)
    resp = await client.get(url, params=params, headers=headers, timeout=timeout)
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


async def _enrich_result(url: str | None) -> dict | None:
    """Fetch a URL just enough to extract structured metadata."""
    if not url:
        return None
    try:
        fetched = await _enrich_fetch(url)
    except Exception as e:
        return {"error": str(e)}

    if not fetched:
        return None
    ctype = (fetched.get("content_type") or "").lower()
    if _is_tika_document(ctype, url):
        return {"title": None, "description": None, "headings": [], "toc": None}
    text = fetched.get("text")
    if not text:
        return None
    # lxml parsing is CPU-bound; offload it so enriching several results doesn't
    # serialize on (and block) the event loop.
    return await anyio.to_thread.run_sync(_structured_from_html, text, url)


def register(mcp: FastMCP) -> None:
    @mcp.tool(description=_search_web_desc(server_settings.tool_prefix))
    async def search_web(
        query: str,
        time_range: str | None = None,
        category: str | None = None,
        num_results: Annotated[
            int | None,
            Field(
                description=f"Max results to return, up to {cfg.max_num_results} "
                "(larger is clamped); omit for the max."
            ),
        ] = None,
        enrich_results: Annotated[
            int | None,
            Field(
                description=f"Top N results to enrich with page metadata, up to "
                f"{cfg.max_enrich_results} (larger is clamped); default "
                f"{cfg.default_enrich_results}, 0 disables enrichment."
            ),
        ] = None,
        page: int | None = None,
    ) -> str:
        """Search the web. The model-facing guidance lives in the
        @mcp.tool(description=...) above.

        :param query: Keywords (operators supported: site:, "phrase", -exclude,
            OR, filetype:).
        :param time_range: Recency filter.
        :param category: Category (comma-separate).
        :param page: Result page number (1-based; default 1).
        """
        log_call(
            log,
            "search_web",
            query=query,
            time_range=time_range,
            category=category,
            num_results=num_results,
            enrich_results=enrich_results,
            page=page,
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

        # Page is 1-based; anything below 1 (or unset) means the first page.
        resolved_page = page if isinstance(page, int) and page > 1 else 1

        try:
            search_query = (
                _youtube_engine_query(query)
                if _category_includes_videos(resolved_categories)
                else query
            )
            results = await _searxng_query(
                base_url=cfg.searxng_url,
                query=search_query,
                num_results=_clamp_count(num_results, cfg.max_num_results, minimum=1),
                categories=resolved_categories,
                language=cfg.searxng_language,
                time_range=resolved_time_range,
                safe_search=cfg.searxng_safesearch,
                timeout=cfg.http_timeout_seconds,
                verify_ssl=cfg.verify_ssl,
                user_agent=cfg.user_agent,
                page=resolved_page,
            )
        except Exception as e:
            raise ToolError(f"SearXNG query failed for {query!r}: {e}")

        applied = {
            "time_range": resolved_time_range or "all",
            "category": resolved_categories,
            "page": resolved_page,
        }

        if not results:
            return log_result(
                log, "search_web", to_json({"query": query, **applied, "results": []})
            )

        if _category_includes_videos(resolved_categories):
            results = [r for r in results if _is_youtube_result_url(r.get("url"))]

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
                # `_enrich_result` reports a fetch failure as {"error": ...} rather
                # than raising, so surface it as page_meta_error instead of letting
                # it fall through (which would set page_title/description to null and
                # drop the reason).
                if data.get("error"):
                    results[i]["page_meta_error"] = data["error"]
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
