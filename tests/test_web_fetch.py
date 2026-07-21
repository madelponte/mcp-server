"""Tests for tools/web_fetch.py — block detection, Tika routing, SSRF guard."""

import asyncio
import json
import socket

import httpx
import pytest

import tools.web_fetch as web_fetch
from tools.cache import TTLCache
from tools.web_fetch import (
    _is_blocked_response,
    _chromium_network_error_code,
    _is_tika_document,
    _sniff_document_bytes,
    _decode_body,
    _parse_allowlist,
    _ip_of,
    _addr_is_blocked,
    _assert_url_allowed,
    _cached_resilient_fetch,
    BrowserRenderError,
    DownloadTooLargeError,
    SSRFError,
)
from conftest import run


# --------------------------- block detection ---------------------------

def test_429_is_always_blocked():
    assert _is_blocked_response(429, "", {}) is True


def test_cloudflare_server_header_with_block_status():
    assert _is_blocked_response(503, "", {"server": "cloudflare"}) is True


def test_datadome_header_with_block_status():
    assert _is_blocked_response(403, "opaque", {"x-datadome": "protected"}) is True


def test_block_status_with_body_marker():
    assert _is_blocked_response(403, "<title>Just a moment...</title>", {}) is True


def test_clean_403_without_markers_is_not_blocked():
    assert _is_blocked_response(403, "<p>Plain forbidden page</p>", {}) is False


def test_datadome_401_challenge_is_blocked():
    # DataDome (e.g. Reuters) serves its interstitial as HTTP 401 with a
    # captcha-delivery marker in the body and a datadome= cookie — both signals,
    # and 401 is a block status, so it must be recognized rather than returned.
    body = (
        '<p id="cmsg">Please enable JS and disable any ad blocker</p>'
        "<script>var dd={'host':'geo.captcha-delivery.com'}</script>"
    )
    headers = {"set-cookie": "datadome=AbC~xyz; Max-Age=31536000; Path=/; Secure"}
    assert _is_blocked_response(401, body, headers) is True


def test_plain_401_without_markers_is_not_blocked():
    # A genuine auth-required 401 (no challenge marker/cookie) is NOT a bot wall —
    # adding 401 to the block statuses must not turn every 401 into a block.
    assert (
        _is_blocked_response(401, '{"error":"unauthorized"}', {"www-authenticate": "Bearer"})
        is False
    )


def test_cloudflare_managed_challenge_text_is_blocked():
    # Cloudflare's managed-challenge interstitial sometimes arrives as a 403 with
    # only this visible body text (no cf-* token / "just a moment" title). It must
    # be recognized as a wall, not returned as content.
    assert _is_blocked_response(403, "Enable JavaScript and cookies to continue", {}) is True


def test_200_with_two_markers_is_blocked():
    body = "cf-ray challenge-platform present here"
    assert _is_blocked_response(200, body, {}) is True


def test_200_with_one_marker_is_not_blocked():
    assert _is_blocked_response(200, "just a moment, loading the page", {}) is False


def test_clean_200_is_not_blocked():
    assert _is_blocked_response(200, "<html><body>Real content</body></html>", {}) is False


# --------------------------- browser navigation errors ---------------------------

def test_chromium_navigation_error_is_detected():
    html = """
    <html><body>
      <div id="main-frame-error" class="interstitial-wrapper">
        <h1>This site can\u2019t be reached</h1>
        <div class="error-code">ERR_HTTP2_PROTOCOL_ERROR</div>
      </div>
    </body></html>
    """
    assert _chromium_network_error_code(html) == "ERR_HTTP2_PROTOCOL_ERROR"


def test_page_discussing_chromium_error_is_not_misclassified():
    html = (
        "<html><article><h1>How to fix ERR_HTTP2_PROTOCOL_ERROR</h1>"
        "<p>This troubleshooting article explains several possible causes.</p>"
        "</article></html>"
    )
    assert _chromium_network_error_code(html) is None


# --------------------------- Tika document detection ---------------------------

@pytest.mark.parametrize(
    "ctype,url",
    [
        ("application/pdf", "https://e.com/x"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "https://e.com/x"),
        ("application/vnd.oasis.opendocument.text", "https://e.com/x"),
        ("application/epub+zip", "https://e.com/x"),
        ("application/octet-stream", "https://e.com/report.pdf"),  # extension fallback
        ("", "https://e.com/sheet.xlsx"),
    ],
)
def test_is_tika_document_true(ctype, url):
    assert _is_tika_document(ctype, url) is True


@pytest.mark.parametrize(
    "ctype,url",
    [
        ("text/html", "https://e.com/page"),
        ("application/json", "https://e.com/api"),
        ("text/plain", "https://e.com/file.txt"),
    ],
)
def test_is_tika_document_false(ctype, url):
    assert _is_tika_document(ctype, url) is False


# --------------------------- magic-byte document sniffing ---------------------------

def test_sniff_pdf_magic():
    assert _sniff_document_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3 rest") is True


def test_sniff_rtf_magic():
    assert _sniff_document_bytes(rb"{\rtf1\ansi hello}") is True


def test_sniff_ole2_legacy_office():
    assert _sniff_document_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32) is True


def test_sniff_ooxml_zip_is_document():
    # A ZIP whose entries name an OOXML part (word/) is a docx, not a bare archive.
    body = b"PK\x03\x04" + b"\x00" * 20 + b"[Content_Types].xml word/document.xml"
    assert _sniff_document_bytes(body) is True


def test_sniff_epub_zip_is_document():
    body = b"PK\x03\x04" + b"\x00" * 20 + b"mimetypeapplication/epub+zip"
    assert _sniff_document_bytes(body) is True


def test_sniff_plain_zip_is_not_document():
    # A generic ZIP with no Office/ODF/EPUB marker must NOT be routed to Tika.
    body = b"PK\x03\x04" + b"\x00" * 20 + b"notes.txt photos/cat.png"
    assert _sniff_document_bytes(body) is False


def test_sniff_html_is_not_document():
    assert _sniff_document_bytes(b"<!doctype html><html><body>hi</body></html>") is False


def test_sniff_empty_or_none():
    assert _sniff_document_bytes(b"") is False
    assert _sniff_document_bytes(None) is False


# --------------------------- charset-aware decoding ---------------------------

def test_decode_uses_content_type_charset():
    body = "Привет мир".encode("windows-1251")
    assert _decode_body(body, "text/html; charset=windows-1251") == "Привет мир"


def test_decode_uses_meta_charset_when_header_silent():
    body = (
        b"<html><head><meta charset='shift_jis'></head><body>"
        + "こんにちは".encode("shift_jis")
        + b"</body></html>"
    )
    assert "こんにちは" in _decode_body(body, "text/html")


def test_decode_detects_when_undeclared():
    # No header charset, no <meta>: detection (UnicodeDammit) should still avoid
    # the all-replacement-glyph garble a blind UTF-8 decode produces.
    body = "café résumé".encode("windows-1252")
    out = _decode_body(body, "text/html")
    assert "�" not in out
    assert "caf" in out and "sum" in out


def test_decode_plain_utf8_roundtrips():
    assert _decode_body("héllo".encode("utf-8"), "text/html; charset=utf-8") == "héllo"


def test_decode_bogus_charset_falls_back_to_utf8():
    # An unknown/garbage charset name must not raise; UTF-8 bytes still decode.
    assert _decode_body("hi".encode("utf-8"), "text/html; charset=totally-bogus") == "hi"


def test_decode_empty_body():
    assert _decode_body(b"", "text/html") == ""


def test_resilient_fetch_sniffs_octet_stream_html(monkeypatch, patch_httpx):
    """HTML served as octet-stream should still be decoded as page text."""
    _clear_allowlist(monkeypatch)
    monkeypatch.setattr(
        web_fetch.socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<!doctype html><html><body><p>Generic typed page.</p></body></html>",
            headers={"content-type": "application/octet-stream"},
        )

    patch_httpx(handler)
    out = run(
        web_fetch._resilient_fetch(
            "https://example.com/generic",
            timeout=5,
            user_agent="t",
            verify_ssl=False,
            flaresolverr_url=None,
            flaresolverr_timeout_ms=1000,
        )
    )
    assert out["text"] is not None
    assert "Generic typed page" in out["text"]
    assert out["bytes"] is None


def test_resilient_fetch_sniffs_octet_stream_json(monkeypatch, patch_httpx):
    _clear_allowlist(monkeypatch)
    monkeypatch.setattr(
        web_fetch.socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"ok": true}',
            headers={"content-type": "application/octet-stream"},
        )

    patch_httpx(handler)
    out = run(
        web_fetch._resilient_fetch(
            "https://example.com/generic-json",
            timeout=5,
            user_agent="t",
            verify_ssl=False,
            flaresolverr_url=None,
            flaresolverr_timeout_ms=1000,
        )
    )
    assert out["content_type"] == "application/json"
    assert out["text"] == '{"ok": true}'
    assert out["bytes"] is None


def test_resilient_fetch_leaves_unsupported_binary_as_bytes(monkeypatch, patch_httpx):
    _clear_allowlist(monkeypatch)
    monkeypatch.setattr(
        web_fetch.socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"\x89PNG\r\n\x1a\nbinary image",
            headers={"content-type": "image/png"},
        )

    patch_httpx(handler)
    out = run(
        web_fetch._resilient_fetch(
            "https://example.com/image.png",
            timeout=5,
            user_agent="t",
            verify_ssl=False,
            flaresolverr_url=None,
            flaresolverr_timeout_ms=1000,
        )
    )
    assert out["text"] is None
    assert out["bytes"].startswith(b"\x89PNG")


# --------------------------- download size cap ---------------------------

def test_download_cap_aborts_oversized_body(patch_httpx):
    """A body past max_bytes raises DownloadTooLargeError instead of buffering it."""
    big = b"x" * 5000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-type": "text/plain"})

    patch_httpx(handler)
    with pytest.raises(DownloadTooLargeError):
        run(
            web_fetch._httpx_fetch(
                "https://example.com/big", timeout=5, user_agent="t",
                verify_ssl=False, max_bytes=1000,
            )
        )


def test_download_cap_allows_within_limit(patch_httpx):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"small", headers={"content-type": "text/plain"})

    patch_httpx(handler)
    status, _headers, body, ctype = run(
        web_fetch._httpx_fetch(
            "https://example.com/ok", timeout=5, user_agent="t",
            verify_ssl=False, max_bytes=1000,
        )
    )
    assert status == 200 and body == b"small"


def test_download_cap_zero_is_unbounded(patch_httpx):
    big = b"y" * 100000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-type": "text/plain"})

    patch_httpx(handler)
    _status, _headers, body, _ctype = run(
        web_fetch._httpx_fetch(
            "https://example.com/big", timeout=5, user_agent="t",
            verify_ssl=False, max_bytes=0,
        )
    )
    assert len(body) == 100000


def test_flaresolverr_rendered_body_obeys_download_cap(patch_httpx):
    payload = {
        "status": "ok",
        "solution": {"status": 200, "headers": {}, "response": "x" * 100},
    }
    patch_httpx(lambda request: httpx.Response(200, content=json.dumps(payload).encode()))
    with pytest.raises(DownloadTooLargeError):
        run(
            web_fetch._flaresolverr_fetch(
                "https://example.com",
                "http://flaresolverr:8191",
                max_timeout_ms=1000,
                http_timeout=5,
                max_bytes=10,
            )
        )


def test_flaresolverr_rejects_chromium_navigation_error(patch_httpx):
    """A FlareSolverr API success must not turn Chrome's error page into data."""
    browser_error_html = (
        '<html><body><div id="main-frame-error">'
        '<h1>This site can\u2019t be reached</h1>'
        '<div class="error-code">ERR_HTTP2_PROTOCOL_ERROR</div>'
        '</div></body></html>'
    )
    payload = {
        "status": "ok",
        "solution": {
            "status": 200,
            "headers": {"content-type": "text/html"},
            "response": browser_error_html,
        },
    }
    patch_httpx(lambda request: httpx.Response(200, content=json.dumps(payload).encode()))

    with pytest.raises(BrowserRenderError, match="ERR_HTTP2_PROTOCOL_ERROR"):
        run(
            web_fetch._flaresolverr_fetch(
                "https://example.com",
                "http://flaresolverr:8191",
                max_timeout_ms=1000,
                http_timeout=5,
            )
        )


def test_tika_output_obeys_download_cap(monkeypatch):
    class Response:
        encoding = "utf-8"

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"1234"
            yield b"5678"

    class Stream:
        def __enter__(self):
            return Response()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(web_fetch.httpx, "stream", lambda *args, **kwargs: Stream())
    with pytest.raises(DownloadTooLargeError):
        web_fetch._tika_extract(
            b"%PDF",
            "http://tika:9998",
            max_output_bytes=5,
        )


def test_page_cache_skips_oversized_entry(monkeypatch):
    cache = TTLCache(60, 8)
    monkeypatch.setattr(web_fetch, "_page_cache", cache)
    monkeypatch.setattr(web_fetch.cfg, "cache_max_item_bytes", 5)
    web_fetch._cache_page(
        "https://example.com/large",
        {"text": "123456", "bytes": None},
    )
    assert cache.get("https://example.com/large") is None


def test_page_cache_keeps_entry_within_item_limit(monkeypatch):
    cache = TTLCache(60, 8)
    monkeypatch.setattr(web_fetch, "_page_cache", cache)
    monkeypatch.setattr(web_fetch.cfg, "cache_max_item_bytes", 6)
    fetched = {"text": "123456", "bytes": None}
    web_fetch._cache_page("https://example.com/small", fetched)
    assert cache.get("https://example.com/small") is fetched


# --------------------------- full fetch cache coordination ---------------------------

def test_cached_resilient_fetch_coalesces_concurrent_misses(monkeypatch):
    calls = 0

    async def fake_resilient_fetch(url, **kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {
            "url": url,
            "status": 200,
            "content_type": "text/html",
            "text": "<html>ok</html>",
            "bytes": None,
            "via": "direct",
            "blocked_detected": False,
        }

    async def scenario():
        return await asyncio.gather(
            _cached_resilient_fetch("https://example.com/page"),
            _cached_resilient_fetch("https://example.com/page"),
            _cached_resilient_fetch("https://example.com/page"),
        )

    monkeypatch.setattr(web_fetch, "_page_cache", TTLCache(60, 8))
    web_fetch._page_inflight.clear()
    monkeypatch.setattr(web_fetch, "_resilient_fetch", fake_resilient_fetch)

    results = run(scenario())
    assert calls == 1
    assert [r["url"] for r in results] == ["https://example.com/page"] * 3
    assert web_fetch._page_inflight == {}


def test_enrich_fetch_coalesces_concurrent_misses(monkeypatch):
    calls = 0

    async def fake_direct_enrich(url):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {
            "url": url,
            "status": 200,
            "content_type": "text/html",
            "text": "<html><title>ok</title></html>",
            "bytes": None,
            "via": "direct",
            "blocked_detected": False,
        }

    async def scenario():
        return await asyncio.gather(
            web_fetch._enrich_fetch("https://example.com/page"),
            web_fetch._enrich_fetch("https://example.com/page"),
            web_fetch._enrich_fetch("https://example.com/page"),
        )

    monkeypatch.setattr(web_fetch, "_page_cache", TTLCache(60, 8))
    web_fetch._enrich_inflight.clear()
    monkeypatch.setattr(web_fetch, "_direct_enrich_fetch", fake_direct_enrich)

    results = run(scenario())
    assert calls == 1
    assert [r["url"] for r in results] == ["https://example.com/page"] * 3
    assert web_fetch._enrich_inflight == {}


# --------------------------- allowlist parsing ---------------------------

def test_parse_allowlist_separates_hosts_and_nets():
    _parse_allowlist.cache_clear()
    hosts, nets = _parse_allowlist("localhost, 10.0.0.0/8 example.com 192.168.1.5")
    assert "localhost" in hosts
    assert "example.com" in hosts
    assert len(nets) == 2  # the CIDR and the bare IP both parse as networks


def test_parse_allowlist_empty():
    _parse_allowlist.cache_clear()
    hosts, nets = _parse_allowlist("")
    assert hosts == frozenset()
    assert nets == ()


# --------------------------- IP parsing ---------------------------

def test_ip_of_ipv4():
    ip = _ip_of("8.8.8.8")
    assert str(ip) == "8.8.8.8"


def test_ip_of_unwraps_ipv4_mapped_ipv6():
    ip = _ip_of("::ffff:169.254.169.254")
    assert str(ip) == "169.254.169.254"


def test_ip_of_strips_scope_id():
    ip = _ip_of("fe80::1%eth0")
    assert ip is not None


def test_ip_of_invalid_returns_none():
    assert _ip_of("not-an-ip") is None


# --------------------------- addr blocking ---------------------------

def test_public_ip_not_blocked():
    assert _addr_is_blocked("8.8.8.8") is False


def test_private_ip_blocked():
    assert _addr_is_blocked("192.168.1.1") is True


def test_loopback_blocked():
    assert _addr_is_blocked("127.0.0.1") is True


def test_metadata_ip_blocked():
    assert _addr_is_blocked("169.254.169.254") is True


def test_unparseable_addr_blocked():
    assert _addr_is_blocked("garbage") is True


def test_allowlisted_net_not_blocked():
    import ipaddress

    net = ipaddress.ip_network("10.0.0.0/8")
    assert _addr_is_blocked("10.1.2.3", (net,)) is False


# --------------------------- _assert_url_allowed (async) ---------------------------

def _clear_allowlist(monkeypatch, value=""):
    _parse_allowlist.cache_clear()
    monkeypatch.setattr(web_fetch.cfg, "ssrf_allowlist", value)


def test_assert_url_rejects_non_http(monkeypatch):
    _clear_allowlist(monkeypatch)
    with pytest.raises(SSRFError):
        run(_assert_url_allowed("ftp://example.com/x"))


def test_assert_url_rejects_no_host(monkeypatch):
    _clear_allowlist(monkeypatch)
    with pytest.raises(SSRFError):
        run(_assert_url_allowed("http:///nohost"))


def test_assert_url_allows_public_host(monkeypatch):
    _clear_allowlist(monkeypatch)
    # Resolve to a public address.
    monkeypatch.setattr(
        web_fetch.socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )
    # Should not raise.
    run(_assert_url_allowed("https://example.com/page"))


def test_assert_url_blocks_private_resolution(monkeypatch):
    _clear_allowlist(monkeypatch)
    monkeypatch.setattr(
        web_fetch.socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("192.168.0.10", 0))],
    )
    with pytest.raises(SSRFError):
        run(_assert_url_allowed("https://intranet.example.com/secret"))


def test_assert_url_blocks_metadata_endpoint(monkeypatch):
    _clear_allowlist(monkeypatch)
    monkeypatch.setattr(
        web_fetch.socket, "getaddrinfo",
        lambda host, port: [(socket.AF_INET, None, None, "", ("169.254.169.254", 0))],
    )
    with pytest.raises(SSRFError):
        run(_assert_url_allowed("http://metadata.internal/latest"))


def test_assert_url_allowlisted_host_bypasses_resolution(monkeypatch):
    _clear_allowlist(monkeypatch, "internal.example.com")

    def _boom(*a, **k):
        raise AssertionError("getaddrinfo should not be called for an allowlisted host")

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", _boom)
    run(_assert_url_allowed("https://internal.example.com/x"))


def test_assert_url_unresolvable_host_raises(monkeypatch):
    _clear_allowlist(monkeypatch)

    def _fail(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", _fail)
    with pytest.raises(SSRFError):
        run(_assert_url_allowed("https://nonexistent.invalid/x"))


def test_assert_url_empty_resolution_raises(monkeypatch):
    _clear_allowlist(monkeypatch)
    monkeypatch.setattr(web_fetch.socket, "getaddrinfo", lambda host, port: [])
    with pytest.raises(SSRFError):
        run(_assert_url_allowed("https://empty-resolution.invalid/x"))


# --------------------------- redirect SSRF re-validation ---------------------------

def test_httpx_fetch_revalidates_redirect_target(monkeypatch, patch_httpx):
    """A redirect to a private host must be refused mid-fetch."""
    _clear_allowlist(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/"})
        return httpx.Response(200, text="should not reach here")

    patch_httpx(handler)
    with pytest.raises(SSRFError):
        run(
            web_fetch._httpx_fetch(
                "https://example.com/start", timeout=5, user_agent="t", verify_ssl=False
            )
        )
