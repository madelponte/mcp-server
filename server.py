"""
MCP server entrypoint.

Bundles Agentic Web Search, Stock Data, Wolfram Alpha, Geocoding & Place
Search, and send-only Email tools into a single MCP server. (YouTube video
transcripts are served by the fetch_page tool rather than a standalone tool.)
Configuration comes entirely from environment variables — see config.py and
.env.example.

Run locally:
    python server.py

Or via Docker / docker-compose (see Dockerfile and docker-compose.yml).
"""

import logging
from collections.abc import Awaitable, Callable

from fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

from auth import BearerAuthMiddleware
from config import server_settings, tool_settings
from tools import (
    web_search,
    fetch_page,
    stock_data,
    wolfram_alpha,
    geocoding,
    email,
    web_fetch,
)

log = logging.getLogger(__name__)


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

    # FastMCP's constructor no longer accepts host/port — they're supplied
    # per-transport at serve time (see run_http() below, which passes them to
    # uvicorn directly, and mcp.run() for stdio which needs neither).
    catalog_ttl = server_settings.tool_catalog_cache_ttl_seconds
    mcp = FastMCP(
        "openwebui-tools",
        cache_ttl=catalog_ttl or None,
        cache_scope=(
            server_settings.tool_catalog_cache_scope if catalog_ttl > 0 else None
        ),
        instructions=(
            "Tools for web search & page fetching (fetch_page also returns YouTube "
            "video transcripts), stock market data, Wolfram Alpha computations, "
            "geocoding & nearby place search (OpenStreetMap), and sending email."
        ),
    )

    # Register enabled tools only. Each module currently exposes one MCP tool;
    # YouTube transcripts remain part of fetch_page rather than a separate tool.
    registrations = (
        ("search_web", tool_settings.search_web_enabled, web_search.register),
        ("fetch_page", tool_settings.fetch_page_enabled, fetch_page.register),
        (
            "get_company_data",
            tool_settings.get_company_data_enabled,
            stock_data.register,
        ),
        (
            "query_wolfram_alpha",
            tool_settings.query_wolfram_alpha_enabled,
            wolfram_alpha.register,
        ),
        (
            "find_nearby_places",
            tool_settings.find_nearby_places_enabled,
            geocoding.register,
        ),
        ("send_email", tool_settings.send_email_enabled, email.register),
    )
    for tool_name, enabled, register in registrations:
        if enabled:
            register(mcp)
        else:
            log.info("MCP tool %s is DISABLED by configuration.", tool_name)

    return mcp


class _LifespanCleanup:
    """Pure ASGI wrapper that runs async cleanup hooks on lifespan shutdown.

    The tool modules pool their httpx clients at module scope for keep-alive;
    without an explicit close those connections are only released when the
    process exits. This wrapper intercepts the lifespan *shutdown* message,
    runs each cleanup hook, then forwards the message to the inner app. It
    wraps ``receive`` rather than driving the protocol itself, so the inner
    app keeps full control of the lifespan exchange. Non-lifespan scopes
    (every actual request) pass straight through untouched.
    """

    def __init__(
        self,
        app: ASGIApp,
        hooks: tuple[Callable[[], Awaitable[None]], ...],
    ) -> None:
        self.app = app
        self.hooks = hooks

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return

        async def receiving():
            message = await receive()
            if message["type"] == "lifespan.shutdown":
                for hook in self.hooks:
                    try:
                        await hook()
                    except Exception:
                        log.exception(
                            "Lifespan cleanup hook %s failed", hook.__name__
                        )
            return message

        await self.app(scope, receiving, send)


def run_http(mcp: FastMCP, transport: str) -> None:
    """Serve an HTTP transport, optionally gated by a bearer token.

    Builds FastMCP's Starlette app ourselves (instead of ``mcp.run()``) so we
    can wrap it with ``BearerAuthMiddleware`` before handing it to uvicorn.
    """
    import uvicorn

    # FastMCP unifies app construction under http_app(transport=...); the
    # 1.0 streamable_http_app()/sse_app() helpers are gone. The streamable-http
    # app still mounts at /mcp and the SSE app at /sse by default.
    if transport == "sse":
        app = mcp.http_app(transport="sse")
    else:  # streamable-http
        app = mcp.http_app(transport="streamable-http")

    # Close the tool modules' shared httpx client pools on graceful shutdown
    # (uvicorn delivers the lifespan shutdown message before exit). The stdio
    # transport has no lifespan surface and relies on process exit.
    app = _LifespanCleanup(
        app,
        hooks=(
            web_fetch.close_clients,
            web_search.close_clients,
            geocoding.close_clients,
            wolfram_alpha.close_clients,
        ),
    )

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

    # The FastMCP instance is built here, not at module import time, so scripts
    # like show_tool.py don't pay for a throwaway build. Configuration singletons
    # are still loaded when config and the tool modules are imported.
    mcp = build_server()
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
        run_http(mcp, transport)
