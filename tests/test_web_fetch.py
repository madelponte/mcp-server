"""Tests for tools/web_fetch.py — block detection, Tika routing, SSRF guard."""

import socket

import httpx
import pytest

import tools.web_fetch as web_fetch
from tools.web_fetch import (
    _is_blocked_response,
    _is_tika_document,
    _parse_allowlist,
    _ip_of,
    _addr_is_blocked,
    _assert_url_allowed,
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


def test_200_with_two_markers_is_blocked():
    body = "cf-ray challenge-platform present here"
    assert _is_blocked_response(200, body, {}) is True


def test_200_with_one_marker_is_not_blocked():
    assert _is_blocked_response(200, "just a moment, loading the page", {}) is False


def test_clean_200_is_not_blocked():
    assert _is_blocked_response(200, "<html><body>Real content</body></html>", {}) is False


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
