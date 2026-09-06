"""
MCP server entrypoint.

Bundles Agentic Web Search, Stock Data, Wolfram Alpha, Geocoding & Place
Search, and send-only Email tools into a single MCP server. (YouTube video
transcripts are served by the fetch_page tool rather than a standalone tool.)

Configuration comes from one YAML file (see config.py and
config.example.yaml): ``MCP_CONFIG_FILE`` if set, otherwise ``config.yaml`` next
to this module, otherwise ``/etc/mcp-server/config.yaml``. Any individual
setting can still be overridden by its legacy environment variable.

Run locally:
    python server.py

Or via Docker / docker-compose, which bind-mounts ./config.yaml read-only (see
Dockerfile and docker-compose.yml).
"""

import logging
from collections.abc import Awaitable, Callable

from fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

from auth import BearerAuthMiddleware, BearerToken
from config import CONFIG_PATH, server_settings, tool_settings
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
    # Debug mode forces DEBUG-level logging regardless of the configured level so
    # the verbose per-tool-call logs are actually emitted.
    level = (
        logging.DEBUG
        if server_settings.debug
        else getattr(logging, server_settings.log_level.upper(), logging.INFO)
    )
    logging.basicConfig(level=level)
    log.info(
        "Configuration source: %s",
        CONFIG_PATH if CONFIG_PATH is not None else "built-in defaults plus environment overrides (no config file found)",
    )
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


def configure_http_auth(app: ASGIApp) -> ASGIApp:
    """Install bearer auth, or refuse to start an unauthenticated HTTP server.

    HTTP transports are fail-closed: with no configured ``auth_tokens`` entry
    (or legacy ``auth_token``) startup fails unless ``allow_unauthenticated`` is
    explicitly true. Every configured token authenticates, and each is named so
    the log shows which client made a request. stdio never uses this helper.
    """
    entries = server_settings.token_entries()
    if entries:
        log.info(
            "Bearer token authentication is ENABLED for %d client(s): %s.",
            len(entries),
            ", ".join(entry.name for entry in entries),
        )
        return BearerAuthMiddleware(
            app, [BearerToken(entry.name, entry.token) for entry in entries]
        )
    if server_settings.allow_unauthenticated:
        log.warning(
            "No bearer token is configured and server.allow_unauthenticated is "
            "true — the server is UNAUTHENTICATED and open to anyone who can "
            "reach it."
        )
        return app
    raise SystemExit(
        "No bearer token is configured. HTTP transports require at least one. "
        "Add an entry to server.auth_tokens in the config file "
        "(generate one with: openssl rand -hex 32), or set "
        "allow_unauthenticated: true for a tightly firewalled local setup."
    )


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

    app = configure_http_auth(app)

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
            f"Unsupported server.transport={transport!r}. "
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
