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


def test_tool_prefix_namespaces_every_tool(monkeypatch):
    """MCP_TOOL_PREFIX prepends its value to every registered tool name."""
    import server as server_mod

    monkeypatch.setattr(server_mod.server_settings, "tool_prefix", "mcp_")
    built = server_mod.build_server()
    names = {t.name for t in run(built.list_tools())}
    assert names == {f"mcp_{n}" for n in EXPECTED_TOOLS}


def test_prefixed_tool_is_still_callable(monkeypatch):
    """A prefixed tool still routes to and runs the original function."""
    import server as server_mod

    monkeypatch.setattr(server_mod.server_settings, "tool_prefix", "mcp_")
    built = server_mod.build_server()
    # send_email validates its args before any network use, so an empty
    # recipients list raises ToolError — proving the prefixed name dispatches
    # to the real tool function rather than 404-ing.
    with pytest.raises(ToolError):
        run(built.call_tool("mcp_send_email", {"recipients": [], "subject": "x", "body": "y"}))


def test_blank_tool_prefix_leaves_names_unchanged(monkeypatch):
    import server as server_mod

    monkeypatch.setattr(server_mod.server_settings, "tool_prefix", "")
    built = server_mod.build_server()
    names = {t.name for t in run(built.list_tools())}
    assert names == EXPECTED_TOOLS


def test_find_nearby_places_description_interpolates_caps(server):
    import tools.geocoding as geo

    tool = next(t for t in _list_tools(server) if t.name == "find_nearby_places")
    # The configured nearby-towns radius is rendered as a concrete number.
    assert str(geo.cfg.nearby_towns_radius_m) in tool.description


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
