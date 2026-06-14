"""Tests for auth.py — the bearer-token ASGI middleware."""

import pytest

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
