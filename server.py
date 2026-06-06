"""
MCP server entrypoint.

Bundles the Agentic Web Search, Stock Data, Wolfram Alpha, and Geocoding &
Place Search tools into a single MCP server. (YouTube video transcripts are
served by web_search's fetch_page rather than a standalone tool.) Configuration
comes entirely from environment variables — see config.py and .env.example.

Run locally:
    python server.py

Or via Docker / docker-compose (see Dockerfile and docker-compose.yml).
"""

import logging

from mcp.server.fastmcp import FastMCP

from auth import BearerAuthMiddleware
from config import server_settings
from tools import web_search, stock_data, wolfram_alpha, geocoding


def build_server() -> FastMCP:
    # Debug mode forces DEBUG-level logging regardless of MCP_LOG_LEVEL so the
    # verbose per-tool-call logs are actually emitted.
    level = (
        logging.DEBUG
        if server_settings.debug
        else getattr(logging, server_settings.log_level.upper(), logging.INFO)
    )
    logging.basicConfig(level=level)
    if server_settings.debug:
        logging.getLogger(__name__).debug(
            "MCP_DEBUG enabled: pretty-printed JSON output and verbose tool logging are ON."
        )

    mcp = FastMCP(
        "openwebui-tools",
        instructions=(
            "Tools for web search & page fetching (fetch_page also returns YouTube "
            "video transcripts), stock market data, Wolfram Alpha computations, and "
            "geocoding & nearby place search (OpenStreetMap)."
        ),
        host=server_settings.host,
        port=server_settings.port,
    )

    # Register every tool group. (YouTube transcripts are handled inside
    # web_search.fetch_page, not as a separate tool — see tools/youtube_transcript.py.)
    web_search.register(mcp)
    stock_data.register(mcp)
    wolfram_alpha.register(mcp)
    geocoding.register(mcp)

    return mcp


mcp = build_server()


def run_http(transport: str) -> None:
    """Serve an HTTP transport, optionally gated by a bearer token.

    Builds FastMCP's Starlette app ourselves (instead of ``mcp.run()``) so we
    can wrap it with ``BearerAuthMiddleware`` before handing it to uvicorn.
    """
    import uvicorn

    log = logging.getLogger(__name__)

    if transport == "sse":
        app = mcp.sse_app()
    else:  # streamable-http
        app = mcp.streamable_http_app()

    token = server_settings.auth_token
    if token:
        app = BearerAuthMiddleware(app, token)
        log.info("Bearer token authentication is ENABLED.")
    else:
        log.warning(
            "MCP_AUTH_TOKEN is not set — the server is UNAUTHENTICATED and open "
            "to anyone who can reach it. Set MCP_AUTH_TOKEN in your .env to "
            "require a bearer token."
        )

    uvicorn.run(
        app,
        host=server_settings.host,
        port=server_settings.port,
        log_level="debug" if server_settings.debug else server_settings.log_level.lower(),
    )


if __name__ == "__main__":
    transport = server_settings.transport.lower()
    if transport not in ("streamable-http", "sse", "stdio"):
        raise SystemExit(
            f"Unsupported MCP_TRANSPORT={transport!r}. "
            "Use 'streamable-http', 'sse', or 'stdio'."
        )
    log = logging.getLogger(__name__)
    log.info(
        "Starting MCP server (transport=%s, host=%s, port=%s)",
        transport,
        server_settings.host,
        server_settings.port,
    )

    if transport == "stdio":
        # No network surface — bearer auth doesn't apply.
        mcp.run(transport="stdio")
    else:
        run_http(transport)
