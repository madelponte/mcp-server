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
    _render_document_with_firecrawl,
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


def _error_detail(exc: BaseException) -> str:
    """Useful provider error text even for exceptions such as bare timeouts."""
    return str(exc).strip() or type(exc).__name__


def _known_direct_url(url: str) -> bool:
    """Whether the URL itself identifies a resource that should skip browsers."""
    if _is_tika_document("", url):
        return True
    path = (urlparse(url).path or "").lower()
    return path.endswith((".json", ".xml", ".rss", ".atom", ".txt", ".csv"))


def _accepted(assessment: PageAssessment) -> bool:
    if assessment.verdict is PageVerdict.ACCEPT:
        return True
    # Deterministically uncertain pages are concise but not obviously blocked.
    # Accept them both when the optional classifier is disabled and when a
    # configured classifier failed and assess_page preserved the deterministic
    # result. A classifier that actually returned a low-confidence/uncertain
    # verdict has source="llm" and remains rejected.
    return (
        assessment.verdict is PageVerdict.UNCERTAIN
        and assessment.source == "deterministic"
    )


def _counts_toward_circuit(assessment: PageAssessment) -> bool:
    """Whether a rejected render is evidence of a host-level browser problem."""
    if assessment.verdict is PageVerdict.BLOCKED:
        return True
    if assessment.verdict is PageVerdict.UNUSABLE:
        # A normal origin response such as 404/410 is specific to that URL. It
        # says nothing about whether FlareSolverr can render another URL on the
        # host, so it must not poison the host circuit.
        return not assessment.reason.startswith("http_")
    # An uncertain classifier verdict is not evidence that the browser failed.
    return False


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
        return None, _error_detail(exc)


async def _firecrawl_document(url: str) -> tuple[dict | None, str | None]:
    if not cfg.firecrawl_api_key.strip():
        return None, "Firecrawl is not configured"
    try:
        fetched = await _render_document_with_firecrawl(url)
        assessment = await _assess(fetched, url)
        if not _accepted(assessment):
            return None, (
                "Firecrawl document recovery was "
                f"{assessment.verdict.value} ({assessment.reason})"
            )
        _cache_page(url, fetched)
        return fetched, None
    except Exception as exc:
        return None, _error_detail(exc)


async def _delayed_firecrawl(url: str) -> tuple[dict | None, str | None]:
    await asyncio.sleep(max(0.0, cfg.firecrawl_hedge_delay_seconds))
    return await _firecrawl(url)


async def _browser(url: str) -> tuple[dict | None, str | None]:
    """Render and assess FlareSolverr output, updating the host circuit."""
    try:
        if cfg.firecrawl_api_key.strip():
            rendered = await asyncio.wait_for(
                _render_with_flaresolverr(url),
                timeout=max(0.1, cfg.flaresolverr_attempt_timeout_seconds),
            )
        else:
            rendered = await _render_with_flaresolverr(url)
        assessment = await _assess(rendered, url)
        if _accepted(assessment):
            _record_browser_success(url)
            _cache_page(url, rendered)
            return rendered, None
        if _counts_toward_circuit(assessment):
            _record_browser_failure(url)
        return None, f"FlareSolverr page was {assessment.verdict.value} ({assessment.reason})"
    except (SSRFError, DownloadTooLargeError):
        raise
    except TimeoutError:
        _record_browser_failure(url)
        return None, (
            "FlareSolverr timed out after "
            f"{max(0.1, cfg.flaresolverr_attempt_timeout_seconds):g} seconds"
        )
    except Exception as exc:
        _record_browser_failure(url)
        return None, f"FlareSolverr failed: {_error_detail(exc)}"


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
    direct_only = _known_direct_url(url)
    probe_error: str | None = None
    try:
        status, _headers, body, ctype, is_html = await _direct_resource_fetch(
            url,
            timeout=(
                cfg.http_timeout_seconds
                if direct_only
                else cfg.direct_probe_timeout_seconds
            ),
            user_agent=cfg.user_agent,
            verify_ssl=cfg.verify_ssl,
            max_bytes=cfg.max_download_bytes,
        )
    except (SSRFError, DownloadTooLargeError):
        raise
    except Exception as exc:
        probe_error = _error_detail(exc)
        if direct_only:
            raise PageAcquisitionError(
                f"Direct resource fetch failed: {probe_error}"
            ) from exc
        # A type probe is an optimization, not a prerequisite. Sites commonly
        # stall or drop non-browser clients; assume an ordinary web page and
        # continue directly to the browser providers.
        is_html = True
        status, body, ctype = 0, b"", "text/html"
        log.debug("Direct resource probe failed for %s: %s", url, probe_error)
    if is_html and direct_only:
        if _is_tika_document("", url):
            recovered, document_error = await _firecrawl_document(url)
            if recovered is not None:
                return recovered
            raise PageAcquisitionError(
                "Document URL returned HTML instead of a document; "
                f"Firecrawl document recovery failed: {document_error}"
            )
        raise PageAcquisitionError(
            "Known direct-resource URL returned HTML instead of its expected format."
        )

    if not is_html:
        fetched = _direct_result(url, status, body, ctype)
        if fetched["resource_kind"] == "html_at_document_url":
            recovered, document_error = await _firecrawl_document(url)
            if recovered is not None:
                return recovered
            raise PageAcquisitionError(
                "Document URL returned HTML instead of a document; "
                f"Firecrawl document recovery failed: {document_error}"
            )
        if status < 400:
            _cache_page(url, fetched)
        return fetched

    # FlareSolverr is optional operationally. With it disabled, the direct probe
    # intentionally becomes a normal direct fetch rather than making HTML unreadable.
    if not cfg.flaresolverr_url:
        from .web_fetch import _httpx_fetch

        try:
            status, _headers, body, ctype = await _httpx_fetch(
                url, cfg.http_timeout_seconds, cfg.user_agent, cfg.verify_ssl,
                cfg.max_download_bytes,
            )
        except (SSRFError, DownloadTooLargeError):
            raise
        except Exception as exc:
            direct_error = _error_detail(exc)
            recovered, firecrawl_error = await _firecrawl(url)
            if recovered is not None:
                return recovered
            raise PageAcquisitionError(
                f"Direct HTML fetch failed: {direct_error}; {firecrawl_error}"
            ) from exc
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
        prefix = f"Direct probe failed: {probe_error}; " if probe_error else ""
        raise PageAcquisitionError(f"{prefix}host browser circuit is open; {error}")

    if cfg.firecrawl_hedge_enabled and cfg.firecrawl_api_key.strip():
        try:
            return await _hedged_browser_fetch(url)
        except PageAcquisitionError as exc:
            if probe_error:
                raise PageAcquisitionError(
                    f"Direct probe failed: {probe_error}; {exc}"
                ) from exc
            raise

    rendered, browser_error = await _browser(url)
    if rendered is not None:
        return rendered
    recovered, firecrawl_error = await _firecrawl(url)
    if recovered is not None:
        return recovered
    prefix = f"Direct probe failed: {probe_error}; " if probe_error else ""
    raise PageAcquisitionError(f"{prefix}{browser_error}; {firecrawl_error}")


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
