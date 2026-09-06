"""
Agentic Web Search MCP tool.

Exposes `search_web`, backed by Brave's LLM Context API. Brave returns
relevance-ranked page excerpts (including text, tables, code, and serialized
structured data) and source metadata ready for model consumption. Use the
companion `fetch_page` tool in structured mode for page outlines.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import re
import time
from typing import Annotated

import anyio
import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from config import web_search_settings as cfg, server_settings
from .serialize import to_json, log_call, log_result, redact_secrets
from .tool_annotations import READ_ONLY_EXTERNAL_TOOL

log = logging.getLogger(__name__)

# Keep one Brave client per event loop / TLS setting so repeated agent searches
# reuse keep-alive connections without binding a client to the wrong loop in
# tests or alternate server runners. The client class is part of the key so a
# monkeypatched httpx.AsyncClient gets a fresh mock-backed client.
_brave_clients: dict[tuple[int, bool, int], httpx.AsyncClient] = {}


class _BraveRequestQueue:
    """Serialize Brave requests and leave a quiet period after each call."""

    def __init__(self) -> None:
        self._lock = anyio.Lock()
        self._next_request_at = 0.0

    @asynccontextmanager
    async def request_slot(self, delay_seconds: float):
        """Yield an exclusive request slot after the configured spacing.

        A zero delay disables serialization for higher-throughput Brave plans.
        The slot covers retries as well as the initial request, preventing a
        second concurrent tool call from creating another retrying burst.
        """
        if delay_seconds <= 0:
            yield
            return

        async with self._lock:
            wait = self._next_request_at - time.monotonic()
            if wait > 0:
                await anyio.sleep(wait)
            try:
                yield
            finally:
                self._next_request_at = time.monotonic() + delay_seconds


_brave_request_queue = _BraveRequestQueue()

_TIME_RANGE_NO_RESTRICTION = {"", "all", "any", "none", "anytime"}
_TIME_RANGE_TO_FRESHNESS = {
    "day": "pd",
    "week": "pw",
    "month": "pm",
    "year": "py",
}
_BRAVE_SAFESEARCH = {"off", "moderate", "strict"}
_BRAVE_THRESHOLD_MODES = {"strict", "balanced", "lenient", "disabled"}
_DATE_RANGE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s*to\s*(\d{4}-\d{2}-\d{2})$")


def _brave_client(verify_ssl: bool) -> httpx.AsyncClient:
    loop_id = id(asyncio.get_running_loop())
    key = (loop_id, verify_ssl, id(httpx.AsyncClient))
    client = _brave_clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(verify=verify_ssl)
        _brave_clients[key] = client
    return client


async def close_clients() -> None:
    """Close every pooled Brave client and drop the pool on shutdown."""
    for client in list(_brave_clients.values()):
        try:
            await client.aclose()
        except Exception:
            log.exception("Failed to close a shared Brave Search client.")
    _brave_clients.clear()


def _search_web_desc(prefix: str) -> str:
    """Build model-facing guidance with the configured sibling-tool prefix."""
    return (
        "Search the web with Brave LLM Context. Use for unknown facts, current "
        "events, research, or verification. Brave returns relevance-ranked "
        "excerpts extracted from source pages, including text, tables, code, "
        "and occasionally JSON-serialized structured data.\n\n"
        "Each result includes url/title/snippets and optional source metadata. "
        "Search does not fetch source pages separately. Use "
        + prefix + 'fetch_page with mode="structured" for page headings and a '
        "table of contents, or its default text mode to read a source in full.\n\n"
        "Query: 1-400 characters and at most 50 words. Prefer concise keywords. "
        "Combine Brave operators when useful: site:example.com (domain), "
        '"exact phrase", -exclude, foo OR bar (uppercase logical operator), '
        "filetype:pdf, intitle:word, inbody:word, lang:en, or loc:us. Operators "
        "are experimental and very restrictive combinations may return nothing.\n"
        "time_range: day/week/month/year/all, or an inclusive custom range "
        "YYYY-MM-DD to YYYY-MM-DD. country and search_lang influence result "
        "localization. safesearch: off/moderate/strict. context_threshold_mode: "
        "strict/balanced/lenient/disabled; omit it to use Brave's calibrated "
        "default. max_tokens controls the approximate total excerpt budget.\n\n"
        "Brave LLM Context does not support result-page pagination or search "
        "categories. Put constraints in the query instead (for example, "
        "site:youtube.com for videos).\n\n"
        "Returns JSON {query,provider,time_range,country,search_lang,safesearch?,"
        "context_threshold_mode?,max_tokens,results:[{url,title,snippets,"
        "published_date?,description?,site_name?}]}"
    )


def _clamp_count(
    requested: int | None,
    maximum: int,
    *,
    minimum: int,
) -> int:
    """Resolve a model-requested count against its configured maximum."""
    if requested is None:
        return maximum
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        return maximum
    if requested < minimum:
        return minimum
    return min(requested, maximum)


def _resolve_time_range(value: str | None, default: str) -> tuple[str, str]:
    """Return ``(model-facing range, Brave freshness value)``."""
    raw = default if value is None else value
    normalized = (raw or "").strip().lower()
    if normalized in _TIME_RANGE_NO_RESTRICTION:
        return "all", ""
    if normalized in _TIME_RANGE_TO_FRESHNESS:
        return normalized, _TIME_RANGE_TO_FRESHNESS[normalized]
    if normalized in _TIME_RANGE_TO_FRESHNESS.values():
        friendly = next(
            name for name, freshness in _TIME_RANGE_TO_FRESHNESS.items()
            if freshness == normalized
        )
        return friendly, normalized

    match = _DATE_RANGE_RE.fullmatch(normalized)
    if match:
        try:
            start = date.fromisoformat(match.group(1))
            end = date.fromisoformat(match.group(2))
        except ValueError as exc:
            raise ToolError(f"Invalid time_range {value!r}: invalid calendar date.") from exc
        if start > end:
            raise ToolError(
                f"Invalid time_range {value!r}: start date must not follow end date."
            )
        # Echo the documented spaced form; Brave's freshness param is concatenated.
        start_s, end_s = start.isoformat(), end.isoformat()
        return f"{start_s} to {end_s}", f"{start_s}to{end_s}"

    raise ToolError(
        f"Invalid time_range {value!r}. Use day, week, month, year, all, or "
        "YYYY-MM-DD to YYYY-MM-DD."
    )


def _resolve_country(value: str | None, default: str) -> str:
    country = (default if value is None else value).strip().upper()
    if len(country) == 2 and country.isalpha():
        return country
    raise ToolError("country must be a two-letter country code such as 'US'.")


def _resolve_search_lang(value: str | None, default: str) -> str:
    language = (default if value is None else value).strip().lower()
    if len(language) >= 2 and all(part.isalpha() for part in language.split("-")):
        return language
    raise ToolError("search_lang must be a language code such as 'en' or 'zh-hans'.")


def _resolve_optional_choice(
    value: str | None,
    default: str,
    allowed: set[str],
    name: str,
) -> str:
    resolved = (default if value is None else value).strip().lower()
    if not resolved or resolved in {"default", "auto", "none"}:
        return ""
    if resolved in allowed:
        return resolved
    choices = ", ".join(sorted(allowed))
    raise ToolError(f"Invalid {name} {value!r}. Use one of: {choices}, or omit it.")


def _brave_error_message(data: object) -> str:
    if not isinstance(data, dict):
        return "unknown Brave Search error"
    error = data.get("error")
    if isinstance(error, dict):
        for key in ("detail", "message", "code"):
            if error.get(key):
                return str(error[key])[:500]
    if error:
        return str(error)[:500]
    return "unknown Brave Search error"


_RETRYABLE_BRAVE_STATUSES = {429, 502, 503, 504}


def _number_list(value: str | None) -> list[float]:
    if not value:
        return []
    numbers: list[float] = []
    for part in value.split(","):
        try:
            numbers.append(max(0.0, float(part.strip())))
        except ValueError:
            return []
    return numbers


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return Brave/server-directed retry delay, if supplied.

    Brave documents X-RateLimit-Reset as comma-separated relative seconds for
    each quota window. When Remaining is available, wait for all exhausted
    windows to reset; do not include a month-long reset unless that quota is
    exhausted too. Retry-After supports both delta seconds and HTTP
    dates. The larger applicable hint wins so neither header is undercut.
    """
    hints: list[float] = []
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            hints.append(max(0.0, float(retry_after.strip())))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                hints.append(
                    max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                )
            except (TypeError, ValueError, OverflowError):
                pass

    resets = _number_list(response.headers.get("X-RateLimit-Reset"))
    remaining = _number_list(response.headers.get("X-RateLimit-Remaining"))
    if resets:
        exhausted_resets = [
            reset
            for left, reset in zip(remaining, resets)
            if left <= 0
        ]
        # Every exhausted quota must reset before another request can succeed.
        # Without exhaustion information, retain the shortest-window fallback.
        hints.append(max(exhausted_resets) if exhausted_resets else min(resets))

    return max(hints) if hints else None


def _log_brave_rate_limits(response: httpx.Response) -> None:
    remaining = response.headers.get("X-RateLimit-Remaining")
    if remaining is None or not log.isEnabledFor(logging.DEBUG):
        return
    log.debug(
        "Brave rate limit remaining=%s limit=%s reset_seconds=%s policy=%s",
        remaining,
        response.headers.get("X-RateLimit-Limit", "unknown"),
        response.headers.get("X-RateLimit-Reset", "unknown"),
        response.headers.get("X-RateLimit-Policy", "unknown"),
    )


async def _post_brave_with_retry(
    client: httpx.AsyncClient,
    api_url: str,
    *,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> httpx.Response:
    """POST to Brave with bounded exponential retries for transient failures."""
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TransportError as exc:
            if attempt >= max_retries:
                raise
            wait = retry_backoff_seconds * (2 ** attempt)
            if wait > timeout:
                raise
            log.warning(
                "Brave request transport failure; retrying in %.3fs (%d/%d): %s",
                wait,
                attempt + 1,
                max_retries,
                exc,
            )
            await anyio.sleep(wait)
            continue

        _log_brave_rate_limits(response)
        if response.status_code not in _RETRYABLE_BRAVE_STATUSES:
            return response
        if attempt >= max_retries:
            return response

        backoff = retry_backoff_seconds * (2 ** attempt)
        server_hint = _retry_after_seconds(response)
        wait = max(backoff, server_hint or 0.0)
        # A monthly quota reset can be days away. Do not pin one MCP call (and
        # the process-wide queue) for longer than its per-attempt HTTP timeout.
        if wait > timeout:
            log.warning(
                "Brave returned HTTP %d with retry delay %.3fs, exceeding the "
                "%.3fs request timeout; not retrying.",
                response.status_code,
                wait,
                timeout,
            )
            return response
        log.warning(
            "Brave returned HTTP %d; retrying in %.3fs (%d/%d).",
            response.status_code,
            wait,
            attempt + 1,
            max_retries,
        )
        await anyio.sleep(wait)

    raise RuntimeError("Brave retry loop exited unexpectedly.")


def _published_date(source: dict) -> str | None:
    age = source.get("age")
    if not isinstance(age, list):
        return None
    # The fourth fixed position is the only representation retaining time of
    # day; older API versions may only provide the YYYY-MM-DD second position.
    for index in (3, 1, 0):
        if len(age) > index and isinstance(age[index], str) and age[index].strip():
            return age[index].strip()
    return None


async def _brave_query(
    api_url: str,
    api_key: str,
    query: str,
    *,
    num_results: int,
    country: str,
    search_lang: str,
    freshness: str,
    safesearch: str,
    context_threshold_mode: str,
    max_tokens: int,
    search_count: int,
    max_tokens_per_url: int,
    timeout: float,
    verify_ssl: bool,
    user_agent: str,
    request_delay_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[dict]:
    """Call Brave LLM Context and normalize its grounding/source records."""
    payload: dict = {
        "q": query,
        "country": country,
        "search_lang": search_lang,
        "count": max(num_results, search_count),
        "maximum_number_of_urls": num_results,
        "maximum_number_of_tokens": max_tokens,
        "maximum_number_of_tokens_per_url": min(max_tokens_per_url, max_tokens),
        "enable_source_metadata": True,
    }
    if freshness:
        payload["freshness"] = freshness
    if safesearch:
        payload["safesearch"] = safesearch
    if context_threshold_mode:
        payload["context_threshold_mode"] = context_threshold_mode

    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    client = _brave_client(verify_ssl)
    async with _brave_request_queue.request_slot(request_delay_seconds):
        response = await _post_brave_with_retry(
            client,
            api_url,
            payload=payload,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    if not 200 <= response.status_code < 300:
        try:
            error_data = response.json()
        except ValueError:
            excerpt = " ".join(response.text.split())[:300] or "empty response body"
            raise RuntimeError(
                f"Brave LLM Context returned HTTP {response.status_code} with a "
                f"non-JSON response: {excerpt}"
            )
        raise RuntimeError(
            f"Brave LLM Context returned HTTP {response.status_code}: "
            f"{_brave_error_message(error_data)}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Brave LLM Context returned HTTP {response.status_code} with invalid JSON."
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Brave LLM Context returned HTTP {response.status_code} with a JSON "
            "value that is not an object."
        )

    grounding = data.get("grounding")
    if not isinstance(grounding, dict):
        raise RuntimeError("Brave LLM Context response did not contain a grounding object.")
    generic = grounding.get("generic") or []
    if not isinstance(generic, list):
        raise RuntimeError("Brave LLM Context returned non-list generic grounding.")
    sources = data.get("sources") or {}
    if not isinstance(sources, dict):
        raise RuntimeError("Brave LLM Context returned a non-object sources value.")

    results: list[dict] = []
    seen_urls: set[str] = set()
    for raw in generic:
        if not isinstance(raw, dict):
            continue
        url = raw.get("url")
        if not isinstance(url, str) or not url.strip() or url in seen_urls:
            continue
        seen_urls.add(url)
        source = sources.get(url)
        source = source if isinstance(source, dict) else {}
        snippets = raw.get("snippets") or []
        if not isinstance(snippets, list):
            snippets = []
        clean_snippets = [item for item in snippets if isinstance(item, str) and item]
        item = {
            "url": url,
            "title": raw.get("title") or source.get("title"),
            "snippets": clean_snippets,
        }
        published = _published_date(source)
        if published:
            item["published_date"] = published
        if source.get("description"):
            item["description"] = source["description"]
        if source.get("site_name"):
            item["site_name"] = source["site_name"]
        results.append(item)
        if len(results) >= num_results:
            break
    return results


def register(mcp: FastMCP) -> None:
    @mcp.tool(
        description=_search_web_desc(server_settings.tool_prefix),
        annotations=READ_ONLY_EXTERNAL_TOOL,
    )
    async def search_web(
        query: str,
        time_range: str | None = None,
        country: str | None = None,
        search_lang: str | None = None,
        safesearch: str | None = None,
        context_threshold_mode: str | None = None,
        num_results: Annotated[
            int | None,
            Field(
                description=f"Max source URLs to return, up to {cfg.max_num_results} "
                "(larger is clamped); omit for the max."
            ),
        ] = None,
        max_tokens: Annotated[
            int | None,
            Field(
                description=f"Approximate total Brave excerpt-token budget, from "
                f"1024 up to {cfg.max_context_tokens} (larger is clamped); omit "
                "for the max."
            ),
        ] = None,
    ) -> str:
        """Search the web. Model-facing guidance is in the tool description.

        :param query: Concise keywords; supports Brave operators such as site:,
            "exact phrase", -exclude, foo OR bar, filetype:, intitle:, inbody:, lang:, loc:.
        :param time_range: Recency (day/week/month/year/all) or inclusive YYYY-MM-DD to YYYY-MM-DD.
        :param country: Two-letter result country code, such as US or GB.
        :param search_lang: Result language code, such as en or zh-hans.
        :param safesearch: Adult-content filter: off, moderate, strict, or omitted.
        :param context_threshold_mode: Relevance filter: strict, balanced, lenient,
            disabled, or omitted for Brave's default.
        """
        log_call(
            log,
            "search_web",
            query=query,
            time_range=time_range,
            country=country,
            search_lang=search_lang,
            safesearch=safesearch,
            context_threshold_mode=context_threshold_mode,
            num_results=num_results,
            max_tokens=max_tokens,
        )
        query = (query or "").strip()
        if not query:
            raise ToolError("Empty query.")
        if len(query) > 400:
            raise ToolError("Query exceeds Brave's 400-character limit.")
        if len(query.split()) > 50:
            raise ToolError("Query exceeds Brave's 50-word limit.")

        api_key = cfg.brave_api_key.strip()
        if not api_key:
            raise ToolError(
                "Brave Search is not configured (set WEB_SEARCH_BRAVE_API_KEY)."
            )
        api_url = cfg.brave_api_url.strip()
        if not api_url:
            raise ToolError("WEB_SEARCH_BRAVE_API_URL must not be blank.")

        resolved_time_range, freshness = _resolve_time_range(
            time_range, cfg.brave_freshness
        )
        resolved_country = _resolve_country(country, cfg.brave_country)
        resolved_language = _resolve_search_lang(search_lang, cfg.brave_search_lang)
        resolved_safesearch = _resolve_optional_choice(
            safesearch, cfg.brave_safesearch, _BRAVE_SAFESEARCH, "safesearch"
        )
        resolved_threshold = _resolve_optional_choice(
            context_threshold_mode,
            cfg.brave_context_threshold_mode,
            _BRAVE_THRESHOLD_MODES,
            "context_threshold_mode",
        )
        resolved_num_results = _clamp_count(
            num_results, cfg.max_num_results, minimum=1
        )
        resolved_max_tokens = _clamp_count(
            max_tokens, cfg.max_context_tokens, minimum=1024
        )

        try:
            results = await _brave_query(
                api_url=api_url,
                api_key=api_key,
                query=query,
                num_results=resolved_num_results,
                country=resolved_country,
                search_lang=resolved_language,
                freshness=freshness,
                safesearch=resolved_safesearch,
                context_threshold_mode=resolved_threshold,
                max_tokens=resolved_max_tokens,
                search_count=cfg.brave_search_count,
                max_tokens_per_url=cfg.brave_max_tokens_per_url,
                timeout=cfg.brave_timeout_seconds,
                verify_ssl=cfg.verify_ssl,
                user_agent=cfg.user_agent,
                request_delay_seconds=cfg.brave_request_delay_seconds,
                max_retries=cfg.brave_max_retries,
                retry_backoff_seconds=cfg.brave_retry_backoff_seconds,
            )
        except Exception as exc:
            detail = redact_secrets(exc, api_key)
            raise ToolError(f"Web search failed for {query!r}: {detail}") from exc

        applied = {
            "provider": "brave_llm_context",
            "time_range": resolved_time_range,
            "country": resolved_country,
            "search_lang": resolved_language,
            "max_tokens": resolved_max_tokens,
        }
        if resolved_safesearch:
            applied["safesearch"] = resolved_safesearch
        if resolved_threshold:
            applied["context_threshold_mode"] = resolved_threshold

        return log_result(
            log,
            "search_web",
            to_json({"query": query, **applied, "results": results}),
        )
