"""
HTTP fetching infrastructure, shared by the web-search and fetch-page tools.

This is the network layer both tools sit on top of: a resilient fetch that
SSRF-guards every URL (and redirect hop), detects bot/CAPTCHA walls and retries
through FlareSolverr, routes binary documents to Apache Tika, and caches the raw
result so a repeated fetch within a task skips the round-trip. `search_web` uses
it to enrich results with page metadata; `fetch_page` uses it to read pages. The
returned dict is the raw fetch (status / content-type / text-or-bytes / via);
each caller formats it for its own needs.
"""

import asyncio
import codecs
import ipaddress
import logging
import re
import socket
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import UnicodeDammit

from config import web_search_settings as cfg
from .cache import TTLCache

log = logging.getLogger(__name__)

# Process-wide cache of fetched pages, keyed by URL. Shared by fetch_page and
# search_web's result enrichment so a repeated fetch within a task skips the
# network round-trip. Fetch settings all come from static config, so the URL
# alone is a sufficient key. See the README "Caching" section.
_page_cache = TTLCache(cfg.cache_ttl_seconds, cfg.cache_max_entries)

# Shared httpx clients for direct fetches, keyed by the `verify` setting (the
# only client-construction option that varies). Reusing one client per setting
# keeps a keep-alive connection pool warm across fetches — so an agent loop that
# reads several pages from the same host reuses the TCP+TLS connection instead of
# reopening it each time — while per-request headers/timeout still vary per call.
# Built lazily so tests that patch `httpx.AsyncClient` are still honored.
_fetch_clients: dict[bool, httpx.AsyncClient] = {}


def _fetch_client(verify_ssl: bool) -> httpx.AsyncClient:
    """Return the shared direct-fetch client for `verify_ssl`, building it once.

    `follow_redirects` is off (redirects are followed by hand so each hop is
    SSRF-checked); the User-Agent, timeout, and Accept headers are supplied
    per-request by `_httpx_fetch`, since only `verify` must be fixed at
    construction time."""
    client = _fetch_clients.get(verify_ssl)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(follow_redirects=False, verify=verify_ssl)
        _fetch_clients[verify_ssl] = client
    return client


# ---------------------------------------------------------------------------
# Bot-wall / CAPTCHA detection
#
# Detecting a challenge page is what routes a fetch to FlareSolverr (which
# renders in a real browser) instead of returning the challenge itself as if it
# were content. Cloudflare is the common case, but it is not the only one:
# PerimeterX/HUMAN, DataDome, and Akamai Bot Manager all serve a 403 (sometimes a
# 200) whose body is a JS/CAPTCHA challenge bearing none of the Cloudflare
# markers — so they must be matched explicitly or the fallback never fires.
# ---------------------------------------------------------------------------

BLOCK_STATUS_CODES = {403, 503, 520, 521, 522, 523, 524, 525, 526, 527}
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
# Markers for the other major bot walls. Kept separate from the Cloudflare set
# only for clarity; detection treats them the same way.
CHALLENGE_MARKERS = (
    "px-cloud.net",            # PerimeterX / HUMAN (e.g. captcha.px-cloud.net)
    "/captcha/captcha.js",     # PerimeterX block script path
    "px-captcha",
    "perimeterx",
    "_pxhd",                   # PerimeterX cookie
    "datadome",                # DataDome
    "geo.captcha-delivery.com",
    "ak_bmsc",                 # Akamai Bot Manager
    "/_sec/cp_challenge",
    # Human-readable text of the *rendered* interstitial these walls serve — what
    # FlareSolverr gets back when it can't solve an interactive challenge (e.g.
    # PerimeterX "Press & Hold"). Distinctive enough that the >=2-marker rule on a
    # 200 keeps them from false-positiving on real pages.
    "access to this page has been denied",
    "confirm you are a human",
    "and not a bot",
)
ALL_BLOCK_MARKERS = CLOUDFLARE_MARKERS + CHALLENGE_MARKERS


def _is_blocked_response(status: int, text: str, headers: dict) -> bool:
    """Best-effort detection that a response is a bot wall / CAPTCHA challenge.

    Covers Cloudflare plus PerimeterX/HUMAN, DataDome, and Akamai Bot Manager —
    each of which serves a challenge under one of ``BLOCK_STATUS_CODES`` (or a
    bare 200) carrying its own marker rather than the page's real content — and
    any HTTP 429, which is definitionally a throttle and never real content.
    """
    # A 429 ("Too Many Requests") is always a rate-limit/throttle: it never
    # carries the page's real content, and is frequently fingerprint-based bot
    # detection that a real browser (FlareSolverr) — different TLS/JS fingerprint
    # — clears, even when a plain client is throttled. Always treat it as a block
    # so the fallback fires, and so a still-throttled 429 surfaces as an error
    # rather than its "Too Many Requests" page being returned as data.
    if status == 429:
        return True
    hdr_lower = {k.lower(): str(v).lower() for k, v in (headers or {}).items()}
    server = hdr_lower.get("server", "")
    set_cookie = hdr_lower.get("set-cookie", "")
    if "cloudflare" in server and status in BLOCK_STATUS_CODES:
        return True
    if status in BLOCK_STATUS_CODES:
        # Some walls signal in headers even when the body is opaque/minified.
        if "x-datadome" in hdr_lower or "datadome" in set_cookie or "_px" in set_cookie:
            return True
        t = (text or "")[:8000].lower()
        if any(m in t for m in ALL_BLOCK_MARKERS):
            return True
    if status == 200 and text:
        t = text[:4000].lower()
        hits = sum(1 for m in ALL_BLOCK_MARKERS if m in t)
        if hits >= 2:
            return True
    return False


# ---------------------------------------------------------------------------
# Document (Tika) detection and extraction
#
# Document types Apache Tika can extract that are NOT served as text/html. Tika
# auto-detects the format from the bytes, so we route any of these to it rather
# than treating the response as HTML/text.
# ---------------------------------------------------------------------------

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


def _sniff_document_bytes(body: bytes | None) -> bool:
    """True if the leading bytes look like a Tika-extractable binary document.

    The last line of defence when neither the content-type nor the URL extension
    reveals the type — e.g. a PDF served as ``application/octet-stream`` (or even
    mislabelled ``text/html``) from an extensionless ``/download?id=`` URL. Without
    this the binary would be UTF-8-decoded into garbage and returned as if it were
    page text. Only unambiguous signatures are matched; a plain ZIP is accepted
    only when it carries an Office/OpenDocument/EPUB marker, never as a bare
    archive.
    """
    if not body:
        return False
    head = body[:8]
    if head.startswith(b"%PDF"):
        return True
    if head.startswith(b"{\\rtf"):
        return True
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):  # OLE2: legacy doc/xls/ppt
        return True
    if head.startswith(b"PK\x03\x04"):
        # A ZIP container — only a document if it's an OOXML/ODF/EPUB package, not
        # an arbitrary archive. These markers appear in the first local-file entry.
        sample = body[:2000]
        return (
            b"word/" in sample
            or b"xl/" in sample
            or b"ppt/" in sample
            or b"mimetypeapplication/epub+zip" in sample
            or b"mimetypeapplication/vnd.oasis.opendocument" in sample
        )
    return False


# ---------------------------------------------------------------------------
# Charset-aware decoding
#
# A fetched page's bytes must be decoded with the right charset, not a blind
# UTF-8: a large slice of the web is served in another encoding (Cyrillic
# windows-1251, Japanese Shift_JIS, Korean EUC-KR, Western windows-1252, …), and
# decoding those as UTF-8 turns every non-ASCII character into a replacement
# glyph — content the rest of the stack would then return as if it were real.
# We resolve the charset like a browser: HTTP Content-Type header, then an
# in-document <meta charset>, then statistical detection, then UTF-8 as a floor.
# ---------------------------------------------------------------------------

_CHARSET_PARAM_RE = re.compile(r"charset\s*=\s*([a-zA-Z0-9_\-:.]+)", re.I)
# <meta charset="..."> or <meta http-equiv=content-type content="...;charset=...">.
# Sniffed from the raw head bytes (ASCII-safe) before we've committed to a codec.
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+?charset\s*=\s*["']?\s*([a-zA-Z0-9_\-:.]+)""", re.I)


def _normalize_codec(name: str | None) -> str | None:
    """Return the canonical codec name if `name` is a real encoding, else None."""
    if not name:
        return None
    try:
        return codecs.lookup(name.strip().strip("\"'")).name
    except (LookupError, TypeError, ValueError):
        return None


def _decode_body(body: bytes, ctype: str) -> str:
    """Decode page bytes to text using the declared/detected charset.

    Precedence mirrors a browser: (1) the HTTP ``Content-Type`` charset, (2) a
    ``<meta charset>`` declaration inside the document head, (3) BeautifulSoup's
    ``UnicodeDammit`` statistical detection (with the declared charsets fed in as
    hints), and (4) UTF-8 with replacement as a floor that never raises. Decoding
    blindly as UTF-8 would garble every non-UTF-8 page.
    """
    if not body:
        return ""
    header_enc = _normalize_codec(_charset_from_ctype(ctype))
    meta_match = _META_CHARSET_RE.search(body[:4096])
    meta_enc = _normalize_codec(meta_match.group(1).decode("ascii", "ignore")) if meta_match else None

    # A declared charset that actually decodes the bytes cleanly wins outright.
    for enc in (header_enc, meta_enc):
        if enc:
            try:
                return body.decode(enc)
            except (UnicodeDecodeError, LookupError):
                pass

    # Otherwise let UnicodeDammit detect, preferring the declared encodings as
    # overrides and handling BOMs / Microsoft smart-quote bytes along the way.
    overrides = [e for e in (header_enc, meta_enc) if e]
    try:
        dammit = UnicodeDammit(body, override_encodings=overrides, is_html=True)
        if dammit.unicode_markup is not None:
            return dammit.unicode_markup
    except Exception:
        pass

    return body.decode("utf-8", errors="replace")


def _charset_from_ctype(ctype: str) -> str | None:
    """Extract the ``charset=`` value from a Content-Type header, if any."""
    m = _CHARSET_PARAM_RE.search(ctype or "")
    return m.group(1) if m else None


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
# SSRF guard
#
# The model's URLs can come from search results or page content it just read, so
# an attacker can use indirect prompt injection to steer a fetch at internal
# targets — cloud metadata (169.254.169.254), localhost, or LAN hosts. We resolve
# the host and refuse any non-publicly-routable address, on the initial URL AND
# every redirect hop (follow_redirects can otherwise 302 a public URL into a
# private one). Cap redirects while we follow them by hand. Operators can opt
# specific hosts/IPs/CIDRs back in via WEB_SEARCH_SSRF_ALLOWLIST.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fetching (with FlareSolverr fallback)
# ---------------------------------------------------------------------------

class DownloadTooLargeError(RuntimeError):
    """A response body exceeded the configured download-size cap."""


async def _httpx_fetch(
    url: str, timeout: float, user_agent: str, verify_ssl: bool, max_bytes: int = 0
) -> tuple[int, dict, bytes, str]:
    """Direct fetch via httpx. Returns (status, headers, body_bytes, content_type).

    Redirects are followed by hand (not httpx's follow_redirects) so each hop's
    target passes the SSRF guard before we connect to it. The caller validates
    the initial URL; here we re-validate every redirect destination.

    The body is streamed and aborted once it passes ``max_bytes`` (0 = unbounded)
    so a multi-GB response — or a decompression bomb, since the cap is on the
    decoded stream httpx hands us — can't exhaust memory on the single-process
    server. Exceeding the cap raises ``DownloadTooLargeError``.
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "application/json;q=0.9,application/pdf;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    client = _fetch_client(verify_ssl)
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        async with client.stream("GET", current, headers=headers, timeout=timeout) as resp:
            location = resp.headers.get("location")
            if resp.is_redirect and location:
                current = str(resp.url.join(location))
                await _assert_url_allowed(current)
                continue
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if max_bytes and total > max_bytes:
                    raise DownloadTooLargeError(
                        f"Response from {current!r} exceeds the "
                        f"{max_bytes}-byte download cap (WEB_SEARCH_MAX_DOWNLOAD_BYTES)."
                    )
                chunks.append(chunk)
            ctype = resp.headers.get("content-type", "")
            return resp.status_code, dict(resp.headers), b"".join(chunks), ctype
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
    max_bytes: int = 0,
) -> dict:
    """Try direct httpx; on a Cloudflare wall, retry through FlareSolverr if configured."""
    # SSRF guard the initial URL before EITHER fetch path. FlareSolverr is the
    # more dangerous one (it renders in real Chrome and runs JS), so the check
    # must gate it too — hence here, not inside _httpx_fetch.
    await _assert_url_allowed(url)
    try:
        status, headers, body, ctype = await _httpx_fetch(
            url, timeout=timeout, user_agent=user_agent, verify_ssl=verify_ssl,
            max_bytes=max_bytes,
        )
    except (SSRFError, DownloadTooLargeError):
        # A redirect hop resolved to a blocked host, or the body blew past the
        # size cap — refuse outright rather than handing the same URL to
        # FlareSolverr, which would re-fetch (and re-download) it in-browser.
        raise
    except Exception as e:
        if flaresolverr_url:
            try:
                fs_status, fs_headers, fs_html = await _flaresolverr_fetch(
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
                    # Flag the result if FlareSolverr's page is itself a wall
                    # (interactive challenge it couldn't solve) rather than the
                    # recovered content.
                    "blocked_detected": _is_blocked_response(fs_status, fs_html, fs_headers),
                }
            except Exception as fe:
                raise RuntimeError(
                    f"Both direct and FlareSolverr fetches failed: {e!r} / {fe!r}"
                )
        raise

    # Office/OpenDocument content-types contain the substring "xml" (e.g.
    # application/vnd.openxmlformats-officedocument...), so the loose checks
    # below would mis-classify them as text and corrupt the bytes via UTF-8
    # decode. Documents we hand to Tika must stay binary — detected by
    # content-type/extension OR by a magic-byte sniff, which also catches a
    # document mislabelled with a text/* or octet-stream content-type.
    is_document = _is_tika_document(ctype, url) or _sniff_document_bytes(body)
    is_textlike = (
        not is_document
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

    text = _decode_body(body, ctype)

    blocked = _is_blocked_response(status, text, headers)
    if blocked and flaresolverr_url:
        try:
            fs_status, fs_headers, fs_html = await _flaresolverr_fetch(
                url,
                flaresolverr_url=flaresolverr_url,
                max_timeout_ms=flaresolverr_timeout_ms,
                http_timeout=max(timeout, flaresolverr_timeout_ms / 1000 + 10),
            )
            # FlareSolverr returning a page isn't the same as a bypass: an
            # interactive wall (PerimeterX "Press & Hold", a CAPTCHA) renders as
            # an ordinary page it can't solve. Re-run detection on what it
            # actually got so a still-walled result is flagged (and raised
            # downstream) instead of returned as content.
            return {
                "url": url,
                "status": fs_status,
                "content_type": "text/html",
                "text": fs_html,
                "bytes": None,
                "via": "flaresolverr",
                "blocked_detected": _is_blocked_response(fs_status, fs_html, fs_headers),
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
        max_bytes=cfg.max_download_bytes,
    )
    _page_cache.set(url, fetched)
    return fetched


async def _enrich_fetch(url: str) -> dict | None:
    """Lean fetch used only to enrich a search result with page metadata.

    Enrichment wants a hit's title/description/heading outline — all in the
    document head — not its whole body, and it isn't worth a multi-second browser
    render. So unlike `_cached_resilient_fetch` this:

    * reuses an already-cached full fetch when one exists (so a later
      ``fetch_page`` read and this share a download), but
    * on a miss does a single direct, byte-capped (``cfg.enrich_max_bytes``) httpx
      fetch through the same SSRF-guarded redirect path — and **skips** the
      FlareSolverr and Wayback fallbacks, so one bot-walled result among the top
      hits can't stall the whole ``search_web`` call.

    A page larger than the cap returns ``None`` (left un-enriched) rather than
    being pulled in full. Only a complete, textlike, un-blocked result is written
    to the shared page cache — never a truncated/blocked one, which would poison a
    later real ``fetch_page`` read. Returns a raw-fetch-shaped dict, or ``None``.
    """
    cached = _page_cache.get(url)
    if cached is not None:
        return cached
    await _assert_url_allowed(url)
    try:
        status, headers, body, ctype = await _httpx_fetch(
            url,
            timeout=cfg.http_timeout_seconds,
            user_agent=cfg.user_agent,
            verify_ssl=cfg.verify_ssl,
            max_bytes=cfg.enrich_max_bytes,
        )
    except (SSRFError, DownloadTooLargeError):
        # Blocked redirect host, or a page too big to enrich cheaply — skip it.
        return None

    is_document = _is_tika_document(ctype, url) or _sniff_document_bytes(body)
    is_textlike = (
        not is_document
        and (
            ctype.startswith("text/")
            or "json" in ctype
            or "xml" in ctype
            or "html" in ctype
        )
    )
    text = _decode_body(body, ctype) if is_textlike else None
    blocked = _is_blocked_response(status, text or "", headers)
    result = {
        "url": url,
        "status": status,
        "content_type": ctype,
        "text": text,
        "bytes": None if is_textlike else body,
        "via": "direct",
        "blocked_detected": blocked,
    }
    # Cache only a complete, readable, un-blocked page, so a later fetch_page read
    # can reuse it. A blocked direct response must not be cached: fetch_page would
    # then reuse it and skip its own FlareSolverr attempt.
    if is_textlike and not blocked:
        _page_cache.set(url, result)
    return result


async def _render_with_flaresolverr(url: str) -> dict:
    """Force a FlareSolverr (real-browser, JS-executing) render of `url`.

    Unlike `_resilient_fetch`, this skips the direct httpx attempt and renders
    straight through FlareSolverr. It's the second attempt a caller makes when a
    direct fetch *succeeded* but came back with no extractable content — the
    signature of a client-side-rendered SPA whose body the static fetch can't
    see. Returns the same raw-fetch dict shape as `_resilient_fetch`.

    The URL is SSRF-guarded first (FlareSolverr runs a real browser, so it must
    be gated like any other fetch path). On a usable render (text present and not
    itself a wall) the result replaces any cached direct fetch for this URL, so
    later fetches and search enrichment reuse the rendered page. Raises if
    FlareSolverr isn't configured or the render call fails.
    """
    flaresolverr_url = cfg.flaresolverr_url or None
    if not flaresolverr_url:
        raise RuntimeError("FlareSolverr is not configured (WEB_SEARCH_FLARESOLVERR_URL).")
    await _assert_url_allowed(url)
    fs_status, fs_headers, fs_html = await _flaresolverr_fetch(
        url,
        flaresolverr_url=flaresolverr_url,
        max_timeout_ms=cfg.flaresolverr_timeout_ms,
        http_timeout=max(cfg.http_timeout_seconds, cfg.flaresolverr_timeout_ms / 1000 + 10),
    )
    result = {
        "url": url,
        "status": fs_status,
        "content_type": "text/html",
        "text": fs_html,
        "bytes": None,
        "via": "flaresolverr",
        "blocked_detected": _is_blocked_response(fs_status, fs_html, fs_headers),
    }
    if fs_html and not result["blocked_detected"]:
        _page_cache.set(url, result)
    return result


# ---------------------------------------------------------------------------
# Wayback Machine (archive.org) fallback
# ---------------------------------------------------------------------------

WAYBACK_AVAILABILITY_API = "https://archive.org/wayback/available"


async def _fetch_from_wayback(url: str) -> dict | None:
    """Fetch `url` from the Internet Archive's Wayback Machine as a last resort.

    Used when the live page (even after a real-browser render) yields no readable
    content, or has since changed/disappeared: a prior snapshot may have captured
    text the live SPA hides behind JavaScript. Returns a raw-fetch dict in the
    `_resilient_fetch` shape — with ``via="archive.org"`` and the snapshot's
    ``wayback_timestamp`` / ``wayback_url`` — or ``None`` when no usable snapshot
    exists. Best-effort: any error returns ``None`` rather than raising, since
    this only ever runs after the live attempts already failed.

    The snapshot is requested in ``id_`` (identity) mode, which returns the
    original archived HTML with its original links and without the Wayback
    toolbar/URL-rewriting, so the existing extractors handle it like a live page.
    The snapshot fetch goes through `_cached_resilient_fetch`, so it is SSRF-
    guarded and cached like any other; the availability API is a fixed host.
    """
    now = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    try:
        async with httpx.AsyncClient(timeout=cfg.http_timeout_seconds) as client:
            resp = await client.get(
                WAYBACK_AVAILABILITY_API, params={"url": url, "timestamp": now}
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return None

    snap = ((data or {}).get("archived_snapshots") or {}).get("closest") or {}
    timestamp = snap.get("timestamp") or ""
    # Accept a snapshot only when it is available and (per the API) was a 200
    # capture; some snapshots omit `status`, which we tolerate.
    if not snap.get("available") or not timestamp:
        return None
    if str(snap.get("status") or "200") != "200":
        return None

    snapshot_url = f"https://web.archive.org/web/{timestamp}id_/{url}"
    try:
        fetched = await _cached_resilient_fetch(snapshot_url)
    except Exception:
        return None

    result = dict(fetched)
    result["via"] = "archive.org"
    result["wayback_timestamp"] = timestamp
    result["wayback_url"] = f"https://web.archive.org/web/{timestamp}/{url}"
    return result
