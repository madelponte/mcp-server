"""Tool modules for the MCP server. Each exposes a `register(mcp)` function.

`youtube_transcript` is the exception: it is a helper module (no `register`)
whose transcript logic is used by `web_search.fetch_page`, not a standalone tool.
"""

from . import web_search, stock_data, wolfram_alpha, youtube_transcript, geocoding

__all__ = [
    "web_search",
    "stock_data",
    "wolfram_alpha",
    "youtube_transcript",
    "geocoding",
]
