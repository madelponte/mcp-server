"""Tool modules for the MCP server. Each exposes a `register(mcp)` function.

The web tooling is split across several modules: `web_search` (the `search_web`
tool) and `fetch_page` (the `fetch_page` tool) are the two registered tools,
while `web_fetch` (HTTP fetching) and `web_extract` (HTML extraction) are shared
helper modules they both build on. `youtube_transcript` is likewise a helper (no
`register`) whose transcript logic is used by `fetch_page`, not a standalone tool.
"""

from . import (
    web_search,
    fetch_page,
    stock_data,
    wolfram_alpha,
    youtube_transcript,
    geocoding,
    email,
)

__all__ = [
    "web_search",
    "fetch_page",
    "stock_data",
    "wolfram_alpha",
    "youtube_transcript",
    "geocoding",
    "email",
]
