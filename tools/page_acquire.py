"""Browser-first acquisition and fallback policy for fetch_page.

HTML is rendered through FlareSolverr after a cheap streamed type probe. Known
non-HTML resources remain on the direct path. This module is the single owner of
FlareSolverr -> optional Firecrawl fallback, hedging, quality assessment, and the
short-lived host circuit breaker.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from urllib.parse import urlparse

from config import web_search_settings as cfg
from .page_quality import PageAssessment, PageVerdict, assess_page
from .web_fetch import (
    DownloadTooLargeError,
    SSRFError,
    _assert_url_allowed,
    _cache_page,
    _decode_body,
    _direct_resource_fetch,
    _is_tika_document,
    _page_cache,
    _render_with_firecrawl,
    _render_with_flaresolverr,
    _sniff_document_bytes,
)


log = logging.getLogger(__name__)


class PageAcquisitionError(RuntimeError):
    """No acquisition provider returned acceptable page content."""


# host -> {URL: last failure monotonic time}; opened circuits have an expiry.
_host_failures: dict[str, dict[str, float]] = defaultdict(dict)
_open_circuits: dict[str, float] = {}
_acquire_inflight: dict[str, asyncio.Task] = {}


def _host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _prune_host(host: str, now: float) -> None:
    cutoff = now - max(0, cfg.circuit_breaker_window_seconds)
    failures = _host_failures.get(host)
    if failures is not None:
        for url, timestamp in list(failures.items()):
            if timestamp < cutoff:
                failures.pop(url, None)
        if not failures:
            _host_failures.pop(host, None)
    if _open_circuits.get(host, 0) <= now:
        _open_circuits.pop(host, None)


def _circuit_open(url: str) -> bool:
    if not cfg.circuit_breaker_enabled or not cfg.firecrawl_api_key.strip():
        return False
    host = _host(url)
    now = time.monotonic()
    _prune_host(host, now)
    return _open_circuits.get(host, 0) > now


def _record_browser_failure(url: str) -> None:
    if not cfg.circuit_breaker_enabled:
        return
    host = _host(url)
    now = time.monotonic()
    _prune_host(host, now)
    _host_failures[host][url] = now
    threshold = max(1, cfg.circuit_breaker_failure_threshold)
    if len(_host_failures[host]) >= threshold:
        _open_circuits[host] = now + max(0, cfg.circuit_breaker_ttl_seconds)
        log.warning(
            "Opening FlareSolverr circuit for host %s after %d distinct URL failures",
            host,
            len(_host_failures[host]),
        )


def _record_browser_success(url: str) -> None:
    host = _host(url)
    _host_failures.pop(host, None)
    _open_circuits.pop(host, None)


def _classifier_enabled() -> bool:
    return bool(cfg.classifier_api_url.strip() and cfg.classifier_model.strip())


def _accepted(assessment: PageAssessment) -> bool:
    if assessment.verdict is PageVerdict.ACCEPT:
        return True
    # If no semantic classifier is configured, intentionally skip that hybrid
    # stage and accept concise-but-not-obviously-blocked pages.
    return assessment.verdict is PageVerdict.UNCERTAIN and not _classifier_enabled()


async def _assess(fetched: dict, url: str) -> PageAssessment:
    assessment = await assess_page(url, fetched.get("status"), fetched.get("text") or "")
    fetched["assessment"] = {
        "verdict": assessment.verdict.value,
        "confidence": assessment.confidence,
        "reason": assessment.reason,
        "source": assessment.source,
    }
    fetched["blocked_detected"] = assessment.verdict is PageVerdict.BLOCKED
    log.debug(
        "Page assessment url=%s via=%s verdict=%s confidence=%.2f reason=%s metrics=%s",
        url,
        fetched.get("via"),
        assessment.verdict.value,
        assessment.confidence,
        assessment.reason,
        assessment.metrics,
    )
    return assessment


async def _firecrawl(url: str) -> tuple[dict | None, str | None]:
    if not cfg.firecrawl_api_key.strip():
        return None, "Firecrawl is not configured"
    try:
        fetched = await _render_with_firecrawl(url)
        assessment = await _assess(fetched, url)
        if _accepted(assessment):
            _cache_page(url, fetched)
            return fetched, None
        return None, f"Firecrawl page was {assessment.verdict.value} ({assessment.reason})"
    except Exception as exc:
        return None, str(exc)


async def _delayed_firecrawl(url: str) -> tuple[dict | None, str | None]:
    await asyncio.sleep(max(0.0, cfg.firecrawl_hedge_delay_seconds))
    return await _firecrawl(url)


async def _browser(url: str) -> tuple[dict | None, str | None]:
    """Render and assess FlareSolverr output, updating the host circuit."""
    try:
        rendered = await _render_with_flaresolverr(url)
        assessment = await _assess(rendered, url)
        if _accepted(assessment):
            _record_browser_success(url)
            _cache_page(url, rendered)
            return rendered, None
        _record_browser_failure(url)
        return None, f"FlareSolverr page was {assessment.verdict.value} ({assessment.reason})"
    except (SSRFError, DownloadTooLargeError):
        raise
    except Exception as exc:
        _record_browser_failure(url)
        return None, f"FlareSolverr failed: {exc}"


async def _cancel(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _hedged_browser_fetch(url: str) -> dict:
    browser_task = asyncio.create_task(_browser(url))
    hedge_task = asyncio.create_task(_delayed_firecrawl(url))
    try:
        done, _pending = await asyncio.wait(
            {browser_task, hedge_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # Use the first acceptable provider, not merely the first provider to
        # finish. If one fails quickly, leave the other running.
        if hedge_task in done:
            recovered, firecrawl_error = await hedge_task
            if recovered is not None:
                return recovered
            rendered, browser_error = await browser_task
            if rendered is not None:
                return rendered
        else:
            rendered, browser_error = await browser_task
            if rendered is not None:
                return rendered
            recovered, firecrawl_error = await hedge_task
            if recovered is not None:
                return recovered
        raise PageAcquisitionError(f"{browser_error}; {firecrawl_error}")
    finally:
        await _cancel(browser_task)
        await _cancel(hedge_task)


def _direct_result(url: str, status: int, body: bytes, ctype: str) -> dict:
    magic_document = _sniff_document_bytes(body)
    declared_html = "html" in ctype.lower()
    url_document = _is_tika_document("", url)
    # A document-looking URL can return a branded HTML denial with HTTP 200.
    # Do not pass that HTML to Tika merely because the path ends in .pdf.
    is_document = magic_document or (_is_tika_document(ctype, url) and not declared_html)
    resource_kind = (
        "document"
        if is_document
        else "html_at_document_url"
        if url_document and declared_html
        else "direct"
    )
    textlike = (
        not is_document
        and (
            ctype.lower().startswith("text/")
            or "json" in ctype.lower()
            or "xml" in ctype.lower()
        )
    )
    return {
        "url": url,
        "status": status,
        "content_type": ctype,
        "text": _decode_body(body, ctype) if textlike else None,
        "bytes": None if textlike else body,
        "via": "direct",
        "blocked_detected": False,
        "resource_kind": resource_kind,
    }


async def _acquire_page(url: str) -> dict:
    """Return one accepted raw fetch, preferring browser rendering for HTML."""
    cached = _page_cache.get(url)
    if cached is not None:
        # Search enrichment may cache direct HTML. It is useful to enrichment but
        # must not bypass fetch_page's browser-first policy.
        ctype = (cached.get("content_type") or "").lower()
        direct_html = cached.get("via") == "direct" and "html" in ctype
        if not (direct_html and cfg.flaresolverr_url):
            return cached

    await _assert_url_allowed(url)
    status, _headers, body, ctype, is_html = await _direct_resource_fetch(
        url,
        timeout=cfg.http_timeout_seconds,
        user_agent=cfg.user_agent,
        verify_ssl=cfg.verify_ssl,
        max_bytes=cfg.max_download_bytes,
    )
    if not is_html:
        fetched = _direct_result(url, status, body, ctype)
        if fetched["resource_kind"] == "html_at_document_url":
            raise PageAcquisitionError(
                "Document URL returned HTML instead of a document; refusing to "
                "send an error or challenge page to Tika."
            )
        if status < 400:
            _cache_page(url, fetched)
        return fetched

    # FlareSolverr is optional operationally. With it disabled, the direct probe
    # intentionally becomes a normal direct fetch rather than making HTML unreadable.
    if not cfg.flaresolverr_url:
        from .web_fetch import _httpx_fetch

        status, _headers, body, ctype = await _httpx_fetch(
            url, cfg.http_timeout_seconds, cfg.user_agent, cfg.verify_ssl,
            cfg.max_download_bytes,
        )
        fetched = _direct_result(url, status, body, ctype)
        assessment = await _assess(fetched, url)
        if _accepted(assessment):
            _cache_page(url, fetched)
            return fetched
        recovered, error = await _firecrawl(url)
        if recovered is not None:
            return recovered
        raise PageAcquisitionError(
            f"Direct HTML fetch was {assessment.verdict.value} ({assessment.reason}); {error}"
        )

    if _circuit_open(url):
        recovered, error = await _firecrawl(url)
        if recovered is not None:
            return recovered
        raise PageAcquisitionError(f"Host browser circuit is open; {error}")

    if cfg.firecrawl_hedge_enabled and cfg.firecrawl_api_key.strip():
        return await _hedged_browser_fetch(url)

    rendered, browser_error = await _browser(url)
    if rendered is not None:
        return rendered
    recovered, firecrawl_error = await _firecrawl(url)
    if recovered is not None:
        return recovered
    raise PageAcquisitionError(f"{browser_error}; {firecrawl_error}")


async def acquire_page(url: str) -> dict:
    """Coalesce concurrent acquisition of one URL around the browser-first core."""
    task = _acquire_inflight.get(url)
    if task is None:
        task = asyncio.create_task(_acquire_page(url))
        _acquire_inflight[url] = task
    try:
        return await asyncio.shield(task)
    finally:
        if _acquire_inflight.get(url) is task:
            _acquire_inflight.pop(url, None)
