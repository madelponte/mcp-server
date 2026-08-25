"""Tests for server.py wiring — tool registration and dynamic cap interpolation."""

import pytest
from fastmcp.exceptions import ToolError

from conftest import run

EXPECTED_TOOLS = {
    "search_web",
    "fetch_page",
    "get_company_data",
    "query_wolfram_alpha",
    "find_nearby_places",
    "send_email",
}


def _list_tools(server):
    return run(server.list_tools())


def test_all_expected_tools_registered(server):
    names = {t.name for t in _list_tools(server)}
    assert names == EXPECTED_TOOLS


def test_every_tool_has_a_description(server):
    for t in _list_tools(server):
        assert t.description and t.description.strip(), f"{t.name} has no description"


def test_fetch_page_description_explains_image_placeholders(server):
    desc = _tool_by_name(server, "fetch_page").description
    assert "[Image at this location: ...]" in desc
    assert "where the image appeared" in desc
    assert "not visual analysis" in desc
    assert "direct image" in desc


def test_fetch_page_description_explains_citation_anchors(server):
    desc = _tool_by_name(server, "fetch_page").description
    assert "{#anchor}" in desc
    assert "Source heading IDs" in desc
    assert "generated `cite-*` anchors" in desc
    assert "citation_url" in desc

    search_desc = _tool_by_name(server, "search_web").description
    assert "stable anchors" in search_desc
    assert "citation_url" in search_desc


def test_fetch_page_schema_exposes_query_window_controls(server):
    import tools.fetch_page as fp

    tool = _tool_by_name(server, "fetch_page").to_mcp_tool()
    props = tool.inputSchema["properties"]
    assert str(fp.cfg.max_query_matches) in props["max_matches"]["description"]
    assert str(fp.cfg.max_query_context_lines) in props["context_lines"]["description"]
    assert "include_match_toc" in props


def test_search_web_documents_common_operators(server):
    tool = _tool_by_name(server, "search_web")
    examples = (
        "site:example.com",
        '"exact phrase"',
        "-exclude",
        "foo OR bar",
        "filetype:pdf",
        "intitle:word",
        "inurl:word",
    )
    for example in examples:
        assert example in tool.description

    query_desc = tool.to_mcp_tool().inputSchema["properties"]["query"]["description"]
    for operator in (
        "site:",
        '"exact phrase"',
        "-exclude",
        " OR ",
        "filetype:",
        "intitle:",
        "inurl:",
    ):
        assert operator in query_desc


def _tool_by_name(server, name):
    return next(t for t in _list_tools(server) if t.name == name)


def test_tool_prefix_does_not_rename_tools(monkeypatch):
    """MCP_TOOL_PREFIX must NOT rename the server's own tools — the client adds
    its own prefix, so renaming here would double it (mcp_mcp_fetch_page)."""
    import server as server_mod

    monkeypatch.setattr(server_mod.server_settings, "tool_prefix", "mcp_")
    built = server_mod.build_server()
    names = {t.name for t in run(built.list_tools())}
    assert names == EXPECTED_TOOLS


def test_tool_prefix_interpolated_into_cross_references(monkeypatch):
    """The configured prefix is spliced into the docstring cross-references so
    they match what the model sees once the client has prefixed the names."""
    import server as server_mod

    monkeypatch.setattr(server_mod.server_settings, "tool_prefix", "owui_")
    built = server_mod.build_server()
    assert "owui_fetch_page" in _tool_by_name(built, "search_web").description
    assert "owui_search_web" in _tool_by_name(built, "fetch_page").description


def test_blank_tool_prefix_yields_bare_cross_references(monkeypatch):
    import server as server_mod

    monkeypatch.setattr(server_mod.server_settings, "tool_prefix", "")
    built = server_mod.build_server()
    desc = _tool_by_name(built, "search_web").description
    assert "fetch_page" in desc
    assert "mcp_fetch_page" not in desc


def test_find_nearby_places_description_interpolates_caps(server):
    import tools.geocoding as geo

    tool = next(t for t in _list_tools(server) if t.name == "find_nearby_places")
    # The configured nearby-towns radius is rendered as a concrete number.
    assert str(geo.cfg.nearby_towns_radius_m) in tool.description
    assert "Every POI search also lists nearby towns" in tool.description
    props = (tool.to_mcp_tool().inputSchema or {}).get("properties", {})
    assert "include_nearby_towns" not in props
    assert "nearby_towns_limit" in props


def test_get_company_data_schema_has_section_params(server):
    tool = next(t for t in _list_tools(server) if t.name == "get_company_data")
    mcp_tool = tool.to_mcp_tool()
    props = (mcp_tool.inputSchema or {}).get("properties", {})
    for param in ("symbol", "sections", "statement", "period", "history_interval"):
        assert param in props


def test_tool_run_invokes_the_function(server):
    """Exercise the full Tool.run() path once: an invalid argument trips the
    tool's own validation and raises ToolError (which the server layer would
    turn into an isError result for the client)."""
    tool = run(server.get_tool("query_wolfram_alpha"))
    with pytest.raises(ToolError):
        run(tool.run({"query": ""}))


def test_importing_server_has_no_module_level_mcp():
    """Importing server loads config/tool modules but must not construct and
    register a throwaway FastMCP instance before ``build_server`` is called."""
    import server as server_mod

    assert not hasattr(server_mod, "mcp")


def test_build_server_still_registers_every_tool():
    import server as server_mod

    names = {t.name for t in _list_tools(server_mod.build_server())}
    assert names == EXPECTED_TOOLS


# ---------------------------------------------------------------------------
# Lifespan cleanup (graceful shutdown of the tool modules' httpx client pools)
# ---------------------------------------------------------------------------


def _drive_lifespan(app, messages):
    """Drive one full lifespan exchange (startup + shutdown) against `app`."""

    async def scenario():
        queued = iter(messages)

        async def receive():
            return next(queued)

        sent = []

        async def send(message):
            sent.append(message)

        await app({"type": "lifespan"}, receive, send)
        return sent

    return run(scenario())


def test_lifespan_cleanup_runs_hooks_before_shutdown_reaches_inner_app():
    from server import _LifespanCleanup

    order = []

    class Inner:
        async def __call__(self, scope, receive, send):
            startup = await receive()
            assert startup["type"] == "lifespan.startup"
            await send({"type": "lifespan.startup.complete"})
            shutdown = await receive()
            assert shutdown["type"] == "lifespan.shutdown"
            order.append("inner")
            await send({"type": "lifespan.shutdown.complete"})

    async def hook():
        order.append("hook")

    sent = _drive_lifespan(
        _LifespanCleanup(Inner(), hooks=(hook,)),
        [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}],
    )
    assert order == ["hook", "inner"]
    assert [m["type"] for m in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


def test_lifespan_cleanup_passes_non_lifespan_scopes_through():
    from server import _LifespanCleanup

    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"OK"})

    async def hook():
        raise AssertionError("hooks must not run for non-lifespan scopes")

    async def scenario():
        async def receive():
            return {"type": "http.request"}

        sent = []

        async def send(message):
            sent.append(message)

        await _LifespanCleanup(inner, hooks=(hook,))(  # noqa: B027
            {"type": "http", "headers": []}, receive, send
        )
        return sent

    sent = run(scenario())
    assert seen == ["http"]
    assert sent[0]["status"] == 200


def test_lifespan_cleanup_hook_failure_does_not_block_shutdown():
    """A broken cleanup hook is logged, not raised: the shutdown message still
    reaches the inner app so the server terminates cleanly."""
    from server import _LifespanCleanup

    delivered = []

    async def inner(scope, receive, send):
        startup = await receive()
        assert startup["type"] == "lifespan.startup"
        await send({"type": "lifespan.startup.complete"})
        shutdown = await receive()
        delivered.append(shutdown["type"])
        await send({"type": "lifespan.shutdown.complete"})

    async def bad_hook():
        raise RuntimeError("boom")

    _drive_lifespan(
        _LifespanCleanup(inner, hooks=(bad_hook,)),
        [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}],
    )
    assert delivered == ["lifespan.shutdown"]


@pytest.mark.parametrize(
    ("module_name", "pool_attr"),
    [
        ("web_fetch", "_fetch_clients"),
        ("web_search", "_searxng_clients"),
        ("geocoding", "_http_clients"),
        ("wolfram_alpha", "_http_clients"),
    ],
)
def test_close_clients_closes_every_client_and_clears_pool(module_name, pool_attr):
    """Each client-owning module exposes the close_clients() hook that
    server.run_http runs on lifespan shutdown."""
    import importlib

    mod = importlib.import_module(f"tools.{module_name}")
    pool = getattr(mod, pool_attr)

    class FakeClient:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    fake = FakeClient()
    pool["test"] = fake
    run(mod.close_clients())
    assert fake.closed
    assert pool == {}
