"""
HTTP fetching infrastructure, shared by the web-search and fetch-page tools.

This is the provider/transport layer both tools sit on top of. It SSRF-guards
URLs and redirect hops, streams direct resources with size limits, talks to
FlareSolverr and Firecrawl, extracts binary documents through Tika, and owns the
raw page cache. `page_acquire` defines fetch_page's browser-first policy and page
acceptance; `search_web` keeps a separate direct-only enrichment path. Returned
dicts are raw artifacts (status / content-type / text-or-bytes / via).
"""

import asyncio
import codecs
import ipaddress
import json
import logging
import re
import socket
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

# Lean search-result enrichment has its own in-flight map. It deliberately does
# not use the full fetch path because enrichment skips browser/Firecrawl fallbacks,
# but concurrent enrichment of the same URL should still share the one direct,
# byte-capped download.
_enrich_inflight: dict[str, asyncio.Task] = {}

# Shared httpx clients for direct fetches, keyed by the `verify` setting (the
# only client-construction option that varies). Reusing one client per setting
# keeps a keep-alive connection pool warm across fetches — so an agent loop that
# reads several pages from the same host reuses the TCP+TLS connection instead of
# reopening it each time — while per-request headers/timeout still vary per call.
# Built lazily so tests that patch `httpx.AsyncClient` are still honored.
_fetch_clients: dict[bool, httpx.AsyncClient] = {}


def _cache_page(url: str, fetched: dict) -> None:
    """Cache a raw fetch only when one entry cannot dominate process memory."""
    limit = cfg.cache_max_item_bytes
    if limit > 0:
        body = fetched.get("bytes")
        if body is not None:
            size = len(body)
        else:
            size = len((fetched.get("text") or "").encode("utf-8"))
        if size > limit:
            return
    _page_cache.set(url, fetched)


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
# Lean enrichment bot-wall detection
#
# Search enrichment is intentionally direct-only, but must avoid caching an
# obvious challenge as reusable page content. fetch_page does not use this
# marker list for fallback decisions; page_quality applies its generic gate.
# Cloudflare is the common direct-enrichment case, but it is not the only one:
# PerimeterX/HUMAN, DataDome, and Akamai Bot Manager all serve a 403 (sometimes a
# 401, 429, or even a 200) whose body is a JS/CAPTCHA challenge bearing none of
# the Cloudflare markers — so they must be matched explicitly or the fallback
# never fires. DataDome in particular (e.g. Reuters) answers its interstitial
# with HTTP 401, which is why 401 is a block status here even though it normally
# means "authentication required": a plain 401 with no challenge marker/cookie
# still falls through as not-blocked (the marker check below must also pass).
# ---------------------------------------------------------------------------

BLOCK_STATUS_CODES = {401, 403, 503, 520, 521, 522, 523, 524, 525, 526, 527}
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
    # Cloudflare's managed-challenge interstitial doesn't always carry a "just a
    # moment" title or a cf-* token in a stripped/minimal response — but its
    # visible body text is this. Distinctive enough not to false-positive on a
    # real page (and the >=2-marker rule still guards the bare-200 case).
    "enable javascript and cookies to continue",
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
    "your request originates from an undeclared automated tool",
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
    # detection. Always treat it as blocked so enrichment does not cache a
    # "Too Many Requests" page as reusable content.
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


def _declared_textlike(ctype: str) -> bool:
    """True when the Content-Type header declares a text/HTML/XML/JSON body."""
    ctype = (ctype or "").lower()
    return (
        ctype.startswith("text/")
        or "json" in ctype
        or "xml" in ctype
        or "html" in ctype
    )


def _generic_binary_ctype(ctype: str) -> bool:
    """True for missing/generic types that often hide HTML or JSON downloads."""
    base = (ctype or "").split(";", 1)[0].strip().lower()
    return base in {
        "",
        "application/octet-stream",
        "binary/octet-stream",
        "application/download",
        "application/x-download",
    }


def _sniff_textlike_ctype(body: bytes | None, ctype: str = "") -> str | None:
    """A synthetic content type if generic bytes look like HTML, XML, or JSON.

    Some sites serve ordinary HTML/JSON from extensionless download endpoints as
    ``application/octet-stream`` or with no Content-Type. Without this sniff the
    caller sees only ``bytes`` and may waste time on browser/archive fallbacks.
    Keep the sniff conservative: it only applies to missing/generic content
    types and only recognizes unambiguous leading syntax.
    """
    if not body or not _generic_binary_ctype(ctype):
        return None
    sample = body[:4096]
    if sample.startswith(codecs.BOM_UTF8):
        sample = sample[len(codecs.BOM_UTF8):]
    sample = sample.lstrip()
    if not sample:
        return None
    head = sample[:256].lower()
    if head.startswith((b"<!doctype html", b"<html")):
        return "text/html"
    if head.startswith((b"<?xml", b"<rss", b"<feed", b"<urlset")):
        return "application/xml"
    if head[:1] in (b"{", b"["):
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            return None
        return "application/json"
    return None


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
    max_output_bytes: int = 0,
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
    chunks: list[bytes] = []
    total = 0
    with httpx.stream(
        "PUT",
        f"{tika_url.rstrip('/')}/tika",
        content=data,
        headers=headers,
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if max_output_bytes and total > max_output_bytes:
                raise DownloadTooLargeError(
                    "Tika extraction output exceeds the configured "
                    f"{max_output_bytes}-byte download cap."
                )
            chunks.append(chunk)
        encoding = resp.encoding or "utf-8"
    text = b"".join(chunks).decode(encoding, errors="replace").strip()
    if not text:
        raise RuntimeError("Document contained no extractable text.")
    return text


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
    literal_ip = _ip_of(host)
    if literal_ip is not None:
        if _addr_is_blocked(host, allowed_nets):
            raise SSRFError(
                f"Refusing to fetch {host!r}: resolves to non-public address(es) "
                f"{host} (allowlist via WEB_SEARCH_SSRF_ALLOWLIST)"
            )
        return
    try:
        # Tests monkeypatch `socket.getaddrinfo` with tiny Python functions. In
        # this Python 3.13 test environment, offloading those patched functions
        # can hang during thread cleanup; run only the real socket resolver in a
        # worker thread. Production keeps DNS off the event loop.
        getaddrinfo = socket.getaddrinfo
        if getattr(getaddrinfo, "__module__", "") == "socket":
            infos = await asyncio.to_thread(getaddrinfo, host, None)
        else:
            infos = getaddrinfo(host, None)
    except socket.gaierror as e:
        raise SSRFError(f"Could not resolve host {host!r}: {e}")
    addresses = {info[4][0] for info in infos}
    if not addresses:
        raise SSRFError(f"Could not resolve host {host!r}: no addresses returned")
    blocked = sorted(
        address for address in addresses if _addr_is_blocked(address, allowed_nets)
    )
    if blocked:
        raise SSRFError(
            f"Refusing to fetch {host!r}: resolves to non-public address(es) "
            f"{', '.join(blocked)} (allowlist via WEB_SEARCH_SSRF_ALLOWLIST)"
        )


# ---------------------------------------------------------------------------
# Direct-resource probing and browser provider transports
# ---------------------------------------------------------------------------

class DownloadTooLargeError(RuntimeError):
    """A response body exceeded the configured download-size cap."""


class BrowserRenderError(RuntimeError):
    """FlareSolverr returned Chromium's own navigation-error document."""


# A failed Chromium navigation is sometimes reported by FlareSolverr as an
# otherwise successful response: its API says ``status: ok`` and the solution
# status is 200, but ``solution.response`` is Chrome's internal error page.  If
# that HTML reaches the extraction layer, fetch_page presents text such as
# "This site can't be reached / ERR_HTTP2_PROTOCOL_ERROR" as page content.
# Keep this separate from bot-wall detection: it is a browser/network failure,
# not a response produced by the target site.
_CHROMIUM_NET_ERROR_CODE_RE = re.compile(r"\bERR_[A-Z0-9_]+\b")
_CHROMIUM_NET_ERROR_MARKERS = (
    "chrome-error://chromewebdata/",
    'id="main-frame-error"',
    'class="error-code"',
    'id="error-information-button"',
)


def _chromium_network_error_code(html: str) -> str | None:
    """Return an error code when ``html`` is Chromium's navigation-error page.

    Requiring both an ``ERR_*`` token and either Chromium-specific markup or
    its standard visible heading avoids rejecting an ordinary page that merely
    discusses a browser error code.
    """
    if not html:
        return None
    match = _CHROMIUM_NET_ERROR_CODE_RE.search(html)
    if not match:
        return None
    lowered = html.lower()
    standard_heading = (
        "this site can't be reached" in lowered
        or "this site can\u2019t be reached" in lowered
        or "this page isn't working" in lowered
        or "this page isn\u2019t working" in lowered
    )
    if standard_heading or any(marker in lowered for marker in _CHROMIUM_NET_ERROR_MARKERS):
        return match.group(0)
    return None


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


async def _direct_resource_fetch(
    url: str, timeout: float, user_agent: str, verify_ssl: bool, max_bytes: int = 0,
    sniff_bytes: int = 16384,
) -> tuple[int, dict, bytes, str, bool]:
    """Probe a URL and download it only when it is not an HTML web page.

    Returns ``(status, headers, body, content_type, is_html)``. For a declared
    HTML response the stream is closed without consuming its body. Missing or
    generic content types are read only through a small sniff window before HTML
    is identified; JSON/XML and binary resources continue in the same request.
    A document-looking URL is always downloaded directly even if a server labels
    it HTML, so an error page at a .pdf URL can be rejected without feeding it to
    a browser or Tika.
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
            status = resp.status_code
            response_headers = dict(resp.headers)
            ctype = resp.headers.get("content-type", "")
            base = ctype.split(";", 1)[0].strip().lower()
            document_url = _is_tika_document("", current)
            if not document_url and (base in {"text/html", "application/xhtml+xml"} or "html" in base):
                return status, response_headers, b"", ctype, True

            chunks: list[bytes] = []
            total = 0
            undecided = not document_url and _generic_binary_ctype(ctype)
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if max_bytes and total > max_bytes:
                    raise DownloadTooLargeError(
                        f"Response from {current!r} exceeds the {max_bytes}-byte "
                        "download cap (WEB_SEARCH_MAX_DOWNLOAD_BYTES)."
                    )
                chunks.append(chunk)
                if undecided and total >= min(sniff_bytes, max_bytes or sniff_bytes):
                    sample = b"".join(chunks)
                    sniffed = _sniff_textlike_ctype(sample, ctype)
                    if sniffed == "text/html":
                        return status, response_headers, b"", sniffed, True
                    # JSON/XML can now continue in this same response.
                    if sniffed:
                        ctype = sniffed
                    undecided = False
            body = b"".join(chunks)
            if undecided:
                sniffed = _sniff_textlike_ctype(body, ctype)
                if sniffed == "text/html":
                    return status, response_headers, b"", sniffed, True
                if sniffed:
                    ctype = sniffed
            return status, response_headers, body, ctype, False
    raise RuntimeError(f"Exceeded {MAX_REDIRECTS} redirects fetching {url!r}")


async def _flaresolverr_fetch(
    url: str,
    flaresolverr_url: str,
    max_timeout_ms: int,
    http_timeout: float,
    max_bytes: int = 0,
) -> tuple[int, dict, str]:
    """Use FlareSolverr to fetch a page without bypassing the download cap."""
    endpoint = flaresolverr_url.rstrip("/") + "/v1"
    payload = {"cmd": "request.get", "url": url, "maxTimeout": max_timeout_ms}
    async with httpx.AsyncClient(timeout=http_timeout) as client:
        async with client.stream(
            "POST", endpoint, json=payload, headers={"Content-Type": "application/json"}
        ) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            # FlareSolverr wraps the rendered body in JSON, so permit modest
            # protocol overhead beyond the configured body limit.
            response_limit = max_bytes + 1048576 if max_bytes else 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if response_limit and total > response_limit:
                    raise DownloadTooLargeError(
                        f"FlareSolverr response for {url!r} exceeds the configured "
                        f"{max_bytes}-byte download cap."
                    )
                chunks.append(chunk)
        data = json.loads(b"".join(chunks))
    if not isinstance(data, dict):
        raise RuntimeError("FlareSolverr returned a JSON value that is not an object.")
    if data.get("status") != "ok":
        msg = str(data.get("message", "unknown FlareSolverr error"))[:500]
        raise RuntimeError(f"FlareSolverr failed: {msg}")
    sol = data.get("solution") or {}
    status = int(sol.get("status") or 0)
    hdrs = sol.get("headers") or {}
    body = sol.get("response") or ""
    if not isinstance(body, str):
        raise RuntimeError("FlareSolverr returned a non-text response body.")
    if max_bytes and len(body.encode("utf-8")) > max_bytes:
        raise DownloadTooLargeError(
            f"Rendered response from {url!r} exceeds the configured "
            f"{max_bytes}-byte download cap."
        )
    browser_error = _chromium_network_error_code(body)
    if browser_error:
        raise BrowserRenderError(
            "FlareSolverr's browser could not load "
            f"{url!r} ({browser_error})."
        )
    return status, hdrs, body


async def _direct_enrich_fetch(url: str) -> dict | None:
    """One direct, byte-capped enrichment fetch. Call via `_enrich_fetch`."""
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
    sniffed_ctype = (
        None
        if is_document or _declared_textlike(ctype)
        else _sniff_textlike_ctype(body, ctype)
    )
    is_textlike = (
        not is_document
        and (_declared_textlike(ctype) or sniffed_ctype is not None)
    )
    if sniffed_ctype:
        ctype = sniffed_ctype
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
        _cache_page(url, result)
    return result


async def _enrich_fetch(url: str) -> dict | None:
    """Lean cached/coalesced fetch used only to enrich a search result with page metadata.

    Enrichment wants a hit's title/description/heading outline — all in the
    document head — not its whole body, and it isn't worth a multi-second browser
    render. Unlike fetch_page's acquisition pipeline this:

    * reuses an already-cached accepted artifact when one exists, but
    * on a miss does a single direct, byte-capped (``cfg.enrich_max_bytes``) httpx
      fetch through the same SSRF-guarded redirect path — and **skips** the
      FlareSolverr and Firecrawl fallbacks, so one bot-walled result among the top
      hits can't stall the whole ``search_web`` call.

    A page larger than the cap returns ``None`` (left un-enriched) rather than
    being pulled in full. Only a complete, textlike, un-blocked result is written
    to the shared page cache — never a truncated/blocked one, which would poison a
    later real ``fetch_page`` read. Returns a raw-fetch-shaped dict, or ``None``.
    """
    cached = _page_cache.get(url)
    if cached is not None:
        return cached

    task = _enrich_inflight.get(url)
    if task is None:
        task = asyncio.create_task(_direct_enrich_fetch(url))
        _enrich_inflight[url] = task

    try:
        return await asyncio.shield(task)
    finally:
        if _enrich_inflight.get(url) is task:
            _enrich_inflight.pop(url, None)


async def _render_with_flaresolverr(url: str) -> dict:
    """Force a FlareSolverr (real-browser, JS-executing) render of `url`.

    This is a transport operation only. The URL is SSRF-guarded first; page_acquire
    applies the shared quality gate and decides whether to accept/cache the HTML
    or continue to Firecrawl. Raises if FlareSolverr is unavailable or fails.
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
        max_bytes=cfg.max_download_bytes,
    )
    result = {
        "url": url,
        "status": fs_status,
        "content_type": "text/html",
        "text": fs_html,
        "bytes": None,
        "via": "flaresolverr",
        "blocked_detected": False,
    }
    # Acceptance and caching belong to page_acquire; a transport-level API
    # success can still be a challenge page.
    return result


# ---------------------------------------------------------------------------
# Firecrawl browser transport
# ---------------------------------------------------------------------------

async def _firecrawl_fetch(
    url: str,
    *,
    api_url: str,
    api_key: str,
    timeout_seconds: float,
    max_bytes: int = 0,
) -> tuple[int, str, str, str | None]:
    """Render one URL through Firecrawl's synchronous v2 scrape endpoint.

    The response is requested as rendered HTML so fetch_page can keep using its
    existing BeautifulSoup extraction for text, structured metadata, sections,
    and query filtering. The JSON envelope and returned HTML both obey the same
    download cap as the direct and FlareSolverr paths.
    """
    timeout_ms = max(1000, min(300000, int(timeout_seconds * 1000)))
    payload = {
        "url": url,
        "formats": ["html"],
        "onlyMainContent": False,
        "timeout": timeout_ms,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    http_timeout = timeout_ms / 1000 + 10

    async with httpx.AsyncClient(timeout=http_timeout) as client:
        async with client.stream("POST", api_url, json=payload, headers=headers) as resp:
            chunks: list[bytes] = []
            total = 0
            # Firecrawl wraps the rendered page in JSON. Match the modest
            # protocol-overhead allowance used for FlareSolverr.
            response_limit = max_bytes + 1048576 if max_bytes else 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if response_limit and total > response_limit:
                    raise DownloadTooLargeError(
                        f"Firecrawl response for {url!r} exceeds the configured "
                        f"{max_bytes}-byte download cap."
                    )
                chunks.append(chunk)
            response_status = resp.status_code

    try:
        response_data = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("Firecrawl returned invalid JSON.") from exc
    if not isinstance(response_data, dict):
        raise RuntimeError("Firecrawl returned a JSON value that is not an object.")

    if response_status < 200 or response_status >= 300:
        message = str(response_data.get("error") or "unknown Firecrawl error")[:500]
        raise RuntimeError(f"Firecrawl returned HTTP {response_status}: {message}")
    if response_data.get("success") is not True:
        message = str(response_data.get("error") or "scrape was unsuccessful")[:500]
        raise RuntimeError(f"Firecrawl failed: {message}")

    data = response_data.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("Firecrawl response did not contain a data object.")
    html = data.get("html")
    if not isinstance(html, str) or not html:
        raise RuntimeError("Firecrawl returned no rendered HTML.")
    if max_bytes and len(html.encode("utf-8")) > max_bytes:
        raise DownloadTooLargeError(
            f"Rendered response from {url!r} exceeds the configured "
            f"{max_bytes}-byte download cap."
        )

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    try:
        status = int(metadata.get("statusCode") or 200)
    except (TypeError, ValueError):
        status = 200
    content_type = str(metadata.get("contentType") or "text/html")
    title = metadata.get("title")
    title = title.strip() if isinstance(title, str) and title.strip() else None
    browser_error = _chromium_network_error_code(html)
    if browser_error:
        raise BrowserRenderError(
            f"Firecrawl's browser could not load {url!r} ({browser_error})."
        )
    return status, content_type, html, title


async def _render_with_firecrawl(url: str) -> dict:
    """Fetch rendered HTML through Firecrawl for page_acquire to assess."""
    api_url = cfg.firecrawl_api_url.strip()
    api_key = cfg.firecrawl_api_key.strip()
    if not api_url or not api_key:
        raise RuntimeError(
            "Firecrawl is not configured (set WEB_SEARCH_FIRECRAWL_API_KEY)."
        )

    await _assert_url_allowed(url)
    status, content_type, html, title = await _firecrawl_fetch(
        url,
        api_url=api_url,
        api_key=api_key,
        timeout_seconds=cfg.firecrawl_timeout_seconds,
        max_bytes=cfg.max_download_bytes,
    )
    result = {
        "url": url,
        "status": status,
        "content_type": content_type,
        "text": html,
        "bytes": None,
        "via": "firecrawl",
        "title": title,
        "blocked_detected": False,
    }
    # page_acquire applies the shared quality gate before caching this result.
    return result
