"""Tests for auth.py — the bearer-token ASGI middleware."""

import logging

import pytest

import auth
from auth import BearerAuthMiddleware
from conftest import run

TOKEN = "s3cret-token"


async def _dummy_app(scope, receive, send):
    """A trivial downstream ASGI app that records that it was reached."""
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"OK"})


def _collect_sends():
    events = []

    async def send(message):
        events.append(message)

    async def receive():
        return {"type": "http.request"}

    return events, send, receive


def _http_scope(headers=None):
    return {"type": "http", "headers": headers or []}


def test_empty_token_rejected_at_construction():
    with pytest.raises(ValueError):
        BearerAuthMiddleware(_dummy_app, "")


@pytest.mark.parametrize("token", ["caf\u00e9", "secret\u2603"])
def test_non_ascii_configured_token_rejected(token):
    with pytest.raises(ValueError, match="ASCII"):
        BearerAuthMiddleware(_dummy_app, token)


@pytest.mark.parametrize("value", [b"Bearer \xff", b"Bearer \xc3\xa9", b"Bearer \xa0" + TOKEN.encode()])
def test_non_ascii_header_returns_401(value, caplog):
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    events, send, receive = _collect_sends()
    with caplog.at_level(logging.WARNING, logger="auth"):
        run(mw(_http_scope([(b"authorization", value)]), receive, send))
    assert events[0]["status"] == 401
    assert events[1]["body"] == b'{"error": "unauthorized"}'
    assert "Rejected unauthorized MCP request" in caplog.text


def test_non_http_scope_passes_through():
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    events, send, receive = _collect_sends()
    # A lifespan scope must bypass auth entirely and reach the app.
    run(mw({"type": "lifespan"}, receive, send))
    assert events[0]["status"] == 200


def test_missing_authorization_header_401():
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    events, send, receive = _collect_sends()
    run(mw(_http_scope(), receive, send))
    assert events[0]["status"] == 401


def test_wrong_token_401():
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    events, send, receive = _collect_sends()
    headers = [(b"authorization", b"Bearer wrong-token")]
    run(mw(_http_scope(headers), receive, send))
    assert events[0]["status"] == 401


def test_non_bearer_scheme_401():
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    events, send, receive = _collect_sends()
    headers = [(b"authorization", b"Basic dXNlcjpwYXNz")]
    run(mw(_http_scope(headers), receive, send))
    assert events[0]["status"] == 401


def test_correct_token_passes():
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    events, send, receive = _collect_sends()
    headers = [(b"authorization", f"Bearer {TOKEN}".encode("latin-1"))]
    run(mw(_http_scope(headers), receive, send))
    assert events[0]["status"] == 200
    assert events[1]["body"] == b"OK"


def test_rejected_request_logs_a_warning_with_client_ip(caplog, monkeypatch):
    """The first rejection is logged even when the monotonic clock's arbitrary
    starting value is smaller than the throttle interval."""
    monkeypatch.setattr(auth.time, "monotonic", lambda: 1.0)
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    events, send, receive = _collect_sends()
    scope = _http_scope([(b"authorization", b"Bearer wrong")])
    scope["client"] = ("203.0.113.7", 51234)
    with caplog.at_level(logging.WARNING, logger="auth"):
        run(mw(scope, receive, send))
    assert events[0]["status"] == 401
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("203.0.113.7" in m for m in warnings)


def test_rejection_logging_is_throttled_and_counts_suppressed(caplog):
    """A flood of rejections must not flood the log: the first is logged, the
    rest counted, and the count reported with the next warning."""
    mw = BearerAuthMiddleware(_dummy_app, TOKEN)
    scope = _http_scope([(b"authorization", b"Bearer wrong")])
    scope["client"] = ("203.0.113.7", 51234)

    with caplog.at_level(logging.WARNING, logger="auth"):
        for _ in range(3):
            events, send, receive = _collect_sends()
            run(mw(scope, receive, send))
            assert events[0]["status"] == 401
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1  # first rejection logged, next two suppressed

    # Rewind the throttle window: the next rejection logs again, with the
    # suppressed count attached.
    mw._last_reject_log -= auth._REJECT_LOG_INTERVAL + 1
    with caplog.at_level(logging.WARNING, logger="auth"):
        events, send, receive = _collect_sends()
        run(mw(scope, receive, send))
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert len(warnings) == 2
    assert "2 more suppressed" in warnings[1]
