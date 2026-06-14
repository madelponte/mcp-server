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
}


def _list_tools(server):
    return run(server.list_tools())


def test_all_expected_tools_registered(server):
    names = {t.name for t in _list_tools(server)}
    assert names == EXPECTED_TOOLS


def test_every_tool_has_a_description(server):
    for t in _list_tools(server):
        assert t.description and t.description.strip(), f"{t.name} has no description"


def test_find_nearby_places_description_interpolates_caps(server):
    import tools.geocoding as geo

    tool = next(t for t in _list_tools(server) if t.name == "find_nearby_places")
    # The configured nearby-towns radius is rendered as a concrete number.
    assert str(geo.cfg.nearby_towns_radius_m) in tool.description


def test_fetch_page_description_interpolates_url_cap(server):
    import tools.fetch_page as fp

    tool = next(t for t in _list_tools(server) if t.name == "fetch_page")
    assert str(fp.MAX_FETCH_URLS) in tool.description


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
