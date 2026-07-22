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
