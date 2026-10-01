"""Shared pytest fixtures and helpers for the MCP server test suite.

The suite is deliberately offline: every test that would otherwise hit the
network either exercises a pure helper, monkeypatches the module-level fetch
helper, or routes httpx through an in-memory ``MockTransport``. Async tool code
is driven with :func:`run` (a thin ``asyncio.run`` wrapper) so no async pytest
plugin is required.

Settings are loaded from the repo ``config.yaml`` when one exists, exactly as in
production, so tests never assume a particular configured value — they read caps
from the live ``cfg`` at runtime, or monkeypatch the specific attribute they
depend on. Tests of the loading logic itself pass an explicit file and env
mapping (see test_config.py) and never touch the deployment's own file.
"""

import asyncio
import json

import httpx
import pytest
from fastmcp.tools import ToolResult
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match


def run(coro):
    """Run an async coroutine to completion (one fresh event loop per call)."""
    return asyncio.run(coro)


def assert_result_matches_schema(tool, result) -> None:
    """Check a tool's ``ToolResult`` against the output schema it advertises.

    MCP clients validate ``structuredContent`` against ``outputSchema`` and fail
    the call on a mismatch, so a payload/schema drift is a real client-facing
    bug. Validation uses the schema exactly as served (``to_mcp_tool()``) and
    the draft the MCP client uses. On top of that, the test copy rejects
    undeclared top-level fields: production schemas stay permissive for forward
    compatibility, but a payload field missing from the schema is invisible to
    clients that type ``structuredContent`` from it (e.g. Pi codemode).
    """
    assert isinstance(result, ToolResult), (
        f"{tool.name} returned {type(result).__name__}, not ToolResult"
    )
    content = result.structured_content
    assert content is not None, f"{tool.name} returned no structured_content"
    text = "\n".join(block.text for block in result.content if block.type == "text")
    assert json.loads(text) == content, (
        f"{tool.name}: text content and structured_content differ"
    )

    schema = {
        **tool.to_mcp_tool().output_schema,
        "additionalProperties": False,
    }
    error = best_match(Draft202012Validator(schema).iter_errors(content))
    if error is not None:
        path = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise AssertionError(
            f"{tool.name} payload does not match its output schema at {path}: "
            f"{error.message}"
        )


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
    """Drop shared async clients and capacity limiters between tests.

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
    web_fetch._capacity_limiters.clear()
    web_fetch._tika_sema = None
    web_fetch._tika_sema_total = None
    web_search._brave_clients.clear()
    geocoding._http_clients.clear()
    wolfram_alpha._http_clients.clear()
    yield
    web_fetch._fetch_clients.clear()
    web_fetch._capacity_limiters.clear()
    web_fetch._tika_sema = None
    web_fetch._tika_sema_total = None
    web_search._brave_clients.clear()
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
    """The real MCP server with every tool enabled for tool-level tests.

    Availability itself is covered in test_server.py. Forcing the flags here
    keeps the rest of the suite deterministic when a developer's local
    config.yaml intentionally disables one or more tools.
    """
    from server import build_server, tool_settings

    for field_name in type(tool_settings).model_fields:
        setattr(tool_settings, field_name, True)
    return build_server()


@pytest.fixture(scope="session")
def tool_fns(server):
    """Map tool names to test adapters around their undecorated async functions.

    Production functions return ``ToolResult`` so MCP clients receive both JSON
    text and native ``structuredContent``. Each adapter checks every successful
    result against the tool's output schema (:func:`assert_result_matches_schema`),
    then returns the text block to keep tool behavior tests focused on the
    long-standing JSON contract while still bypassing MCP transport
    serialization. ``ToolError`` passes through.
    """
    names = [
        "search_web",
        "fetch_page",
        "get_company_data",
        "query_wolfram_alpha",
        "find_nearby_places",
        "send_email",
    ]

    def _json_text_adapter(tool):
        async def call(*args, **kwargs):
            result = await tool.fn(*args, **kwargs)
            assert_result_matches_schema(tool, result)
            return "\n".join(
                block.text for block in result.content if block.type == "text"
            )

        return call

    async def _collect():
        return {
            name: _json_text_adapter(await server.get_tool(name))
            for name in names
        }

    return run(_collect())
