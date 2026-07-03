"""Shared pytest fixtures and helpers for the MCP server test suite.

The suite is deliberately offline: every test that would otherwise hit the
network either exercises a pure helper, monkeypatches the module-level fetch
helper, or routes httpx through an in-memory ``MockTransport``. Async tool code
is driven with :func:`run` (a thin ``asyncio.run`` wrapper) so no async pytest
plugin is required.

Settings are loaded from the repo ``.env`` like in production, so tests never
assume a particular configured value — they read caps from the live ``cfg`` at
runtime, or monkeypatch the specific attribute they depend on.
"""

import asyncio

import httpx
import pytest


def run(coro):
    """Run an async coroutine to completion (one fresh event loop per call)."""
    return asyncio.run(coro)


def make_mock_async_client_cls(handler):
    """Build an ``httpx.AsyncClient`` subclass that serves every request from a
    ``MockTransport`` handler, so code that constructs its own client offline.

    ``verify`` is dropped because it is meaningless with a mock transport (and
    would otherwise be silently ignored); all other kwargs the callers pass
    (timeout, headers, follow_redirects, …) are preserved.
    """

    class _MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.pop("verify", None)
            super().__init__(*args, transport=httpx.MockTransport(handler), **kwargs)

    return _MockAsyncClient


@pytest.fixture(autouse=True)
def _reset_shared_clients():
    """Drop shared async clients and in-flight fetch maps between tests.

    `web_fetch` reuses one `httpx.AsyncClient` per `verify` setting for keep-alive.
    Tests build that client lazily under a patched `httpx.AsyncClient` (a fresh
    MockTransport per test), so the cache must be cleared each test or a later
    test would reuse an earlier test's mock handler.
    """
    from tools import web_fetch
    from tools import web_search
    from tools import geocoding
    from tools import wolfram_alpha

    web_fetch._fetch_clients.clear()
    web_fetch._enrich_inflight.clear()
    web_search._searxng_clients.clear()
    geocoding._http_clients.clear()
    wolfram_alpha._http_clients.clear()
    yield
    web_fetch._fetch_clients.clear()
    web_fetch._enrich_inflight.clear()
    web_search._searxng_clients.clear()
    geocoding._http_clients.clear()
    wolfram_alpha._http_clients.clear()


@pytest.fixture
def patch_httpx(monkeypatch):
    """Return a function that installs a MockTransport handler on httpx.AsyncClient.

    All tool modules share the one ``httpx`` module object, so patching
    ``httpx.AsyncClient`` here redirects every async client they build. The
    handler receives an ``httpx.Request`` and returns an ``httpx.Response``.
    """

    def _apply(handler):
        monkeypatch.setattr(httpx, "AsyncClient", make_mock_async_client_cls(handler))

    return _apply


@pytest.fixture(scope="session")
def server():
    """The real MCP server built exactly as production does."""
    from server import build_server

    return build_server()


@pytest.fixture(scope="session")
def tool_fns(server):
    """Map each registered tool name to its underlying (undecorated) async fn.

    ``tool.fn`` is the raw closure: calling it returns the tool's JSON string on
    success and raises ``ToolError`` on failure, bypassing MCP serialization —
    ideal for asserting on the tool's own contract.
    """
    names = [
        "search_web",
        "fetch_page",
        "get_company_data",
        "query_wolfram_alpha",
        "find_nearby_places",
        "send_email",
    ]

    async def _collect():
        return {n: (await server.get_tool(n)).fn for n in names}

    return run(_collect())
