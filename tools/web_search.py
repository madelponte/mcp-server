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

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import web_search_settings as cfg
from .serialize import to_json, log_call, log_result
from .web_fetch import _cached_resilient_fetch, _is_tika_document
from .web_extract import _structured_from_html, _trim

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake the failure for
# search data. A valid-but-empty result (e.g. a search with zero hits) is NOT a
# failure and is still returned as normal JSON. See the README "Error handling"
# section.

# SearXNG time_range values the API accepts; "" means no time restriction.
# We also accept a few friendly synonyms for "no restriction" from the model.
SEARXNG_TIME_RANGES = {"", "day", "week", "month", "year"}
_TIME_RANGE_NO_RESTRICTION = {"", "all", "any", "none", "anytime"}


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


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def search_web(
        query: str,
        time_range: str | None = None,
        category: str | None = None,
        num_results: int | None = None,
        enrich_results: int | None = None,
        page: int | None = None,
    ) -> str:
        """Search the web. Use for unknown facts, current events, or verification.

        Results include url/title/snippet + optional page metadata (headings,
        description) for top results. Then use mcp_fetch_page to read full content.

        Query: short keywords only (not sentences). time_range: "day"/"week"/
        "month"/"year"/"all". category: "general"|"news"|"science"|"it"|"social
        media"|"videos"|"images"|"music"|"files"|"map" (comma-separate).
        num_results/enrich_results: max counts (capped; enrich fetches metadata).
        page: result page (1-based, default 1) — set page=2 to get the next batch
        of results for the SAME query when the first page wasn't useful, instead
        of reformulating.

        :param query: Keywords.
        :param time_range: Recency filter.
        :param category: Category (comma-separate).
        :param num_results: Max results (capped).
        :param enrich_results: Top N to enrich with metadata (capped).
        :param page: Result page number (1-based; default 1).
        :return: JSON {query, time_range, category, page, results:[{url,title,
            snippet,published_date?,page_title?,page_description?,page_headings?,
            page_toc?}]}
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
