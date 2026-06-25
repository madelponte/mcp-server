"""
MCP server entrypoint.

Bundles the Agentic Web Search, Stock Data, Wolfram Alpha, and Geocoding &
Place Search tools into a single MCP server. (YouTube video transcripts are
served by the fetch_page tool rather than a standalone tool.) Configuration
comes entirely from environment variables — see config.py and .env.example.

Run locally:
    python server.py

Or via Docker / docker-compose (see Dockerfile and docker-compose.yml).
"""

import asyncio
import logging

from fastmcp import FastMCP
from fastmcp.tools import Tool

from auth import BearerAuthMiddleware
from config import server_settings
from tools import web_search, fetch_page, stock_data, wolfram_alpha, geocoding, email


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

    # FastMCP v3's constructor no longer accepts host/port — they're supplied
    # per-transport at serve time (see run_http() below, which passes them to
    # uvicorn directly, and mcp.run() for stdio which needs neither).
    mcp = FastMCP(
        "openwebui-tools",
        instructions=(
            "Tools for web search & page fetching (fetch_page also returns YouTube "
            "video transcripts), stock market data, Wolfram Alpha computations, "
            "geocoding & nearby place search (OpenStreetMap), and sending email."
        ),
    )

    # Register every tool group. (YouTube transcripts are handled inside the
    # fetch_page tool, not as a separate tool — see tools/youtube_transcript.py.)
    web_search.register(mcp)
    fetch_page.register(mcp)
    stock_data.register(mcp)
    wolfram_alpha.register(mcp)
    geocoding.register(mcp)
    email.register(mcp)

    _apply_tool_prefix(mcp, server_settings.tool_prefix)

    return mcp


def _apply_tool_prefix(mcp: FastMCP, prefix: str) -> None:
    """Prepend ``prefix`` to every registered tool's name.

    Lets a deployment namespace the whole tool surface from one env var
    (``MCP_TOOL_PREFIX``) — e.g. to satisfy a client like Open WebUI that adds /
    expects an ``mcp_`` prefix — without touching each tool's ``register``. A
    blank prefix is a no-op, preserving the bare names. Each tool is re-registered
    under its prefixed name via a name-only transform that delegates to the
    original, so behavior is unchanged.
    """
    prefix = (prefix or "").strip()
    if not prefix:
        return
    provider = mcp.local_provider
    for tool in _run_sync(mcp.list_tools()):
        if tool.name.startswith(prefix):
            continue
        provider.remove_tool(tool.name)
        provider.add_tool(Tool.from_tool(tool, name=f"{prefix}{tool.name}"))


def _run_sync(coro):
    """Drive an async coroutine to completion from sync code.

    ``build_server()`` is normally called at import time (no running loop), where
    ``asyncio.run`` works — but it may also be called from inside an already
    running loop (e.g. scripts/show_tool.py awaits a wrapper that builds the
    server). ``asyncio.run`` raises there, so fall back to running the coroutine
    on its own loop in a worker thread.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


mcp = build_server()


def run_http(transport: str) -> None:
    """Serve an HTTP transport, optionally gated by a bearer token.

    Builds FastMCP's Starlette app ourselves (instead of ``mcp.run()``) so we
    can wrap it with ``BearerAuthMiddleware`` before handing it to uvicorn.
    """
    import uvicorn

    log = logging.getLogger(__name__)

    # FastMCP v3 unifies app construction under http_app(transport=...); the
    # 1.0 streamable_http_app()/sse_app() helpers are gone. The streamable-http
    # app still mounts at /mcp and the SSE app at /sse by default.
    if transport == "sse":
        app = mcp.http_app(transport="sse")
    else:  # streamable-http
        app = mcp.http_app(transport="streamable-http")

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
