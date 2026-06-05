"""Tool modules for the MCP server. Each exposes a `register(mcp)` function."""

from . import web_search, stock_data, wolfram_alpha, youtube_transcript, geocoding

__all__ = [
    "web_search",
    "stock_data",
    "wolfram_alpha",
    "youtube_transcript",
    "geocoding",
]
