# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single [FastMCP v3](https://github.com/jlowin/fastmcp) server that bundles four tool groups — **Agentic Web Search** (the `search_web` and `fetch_page` tools, where `fetch_page` also returns **YouTube** video transcripts), **Stock Data**, **Wolfram Alpha**, and **Geocoding & Place Search** (OpenStreetMap Nominatim + Overpass) — originally written as Open WebUI tools and ported to MCP. Default transport is `streamable-http`, reachable at `http://<host>:8000/mcp`.

The web tool group spans four modules: `tools/web_search.py` (the `search_web` tool) and `tools/fetch_page.py` (the `fetch_page` tool) are the two registered tools, and they share `tools/web_fetch.py` (the SSRF-guarded HTTP fetch stack: direct httpx, FlareSolverr fallback, bot-wall detection, Tika document extraction, a Wayback Machine fallback, and the process-wide page cache) and `tools/web_extract.py` (pure HTML→content helpers: markdown/plain-text rendering, structured metadata, single-section extraction).

YouTube transcript retrieval used to be its own tool; it was folded into `fetch_page` (which detects a YouTube video URL and returns the transcript) so a model has one fewer tool to choose between. The logic still lives in `tools/youtube_transcript.py`, now exposed as the `fetch_transcript` / `is_youtube_video_url` helpers rather than a registered tool.

## Commands

```bash
# Run locally (default streamable-http transport)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit it
python server.py

# Run as a stdio MCP server instead (no network, no auth)
MCP_TRANSPORT=stdio python server.py

# Full stack incl. web-search support services (SearXNG, FlareSolverr, Tika)
docker compose up --build   # MCP at http://localhost:8000/mcp

# Server-only image (other three tools have no local-service deps)
docker build -t openwebui-tools-mcp .
docker run --rm -p 8000:8000 --env-file .env openwebui-tools-mcp
```

There is **no test suite, linter, or formatter** configured. CI (`.github/workflows/`) only builds and publishes the Docker image.

## Architecture

**Entry point → registration.** `server.py:build_server()` constructs one `FastMCP` instance and calls `register(mcp)` on each tool module in `tools/`. Every tool module exposes a single `register(mcp: FastMCP)` function that defines its `@mcp.tool()` functions inside that call (closures over module-level config/cache). To add a tool group: create `tools/<name>.py` with a `register`, then wire it into `tools/__init__.py` and the import + `register()` calls in `server.py`. (`tools/youtube_transcript.py`, `tools/web_fetch.py`, and `tools/web_extract.py` are the exceptions — they have no `register`; they're helper modules. `youtube_transcript` is consumed by `fetch_page`; `web_fetch` and `web_extract` are shared by both `web_search` and `fetch_page`.)

**Config is centralized and prefix-namespaced.** `config.py` defines one `pydantic-settings` `BaseSettings` subclass per tool, each with a distinct `env_prefix` (`MCP_`, `WEB_SEARCH_`, `STOCK_`, `WOLFRAM_`, `YOUTUBE_`, `GEO_`) so variables can't collide. It instantiates module-level singletons (`server_settings`, `web_search_settings`, …) that tool modules import as `cfg`. Settings load from the process env and `.env`. Every Open WebUI "valve" maps to one env var here; `.env.example` is the canonical documented list. When adding a setting, add the `Field` (with a description) to the relevant class — don't read `os.environ` directly.

**Transport & auth.** `server.py:run_http()` builds FastMCP's Starlette app itself (rather than `mcp.run()`) so it can wrap it in `auth.py:BearerAuthMiddleware` before handing it to uvicorn. Auth is a single shared secret (`MCP_AUTH_TOKEN`), enforced as pure ASGI middleware with a constant-time compare; if the token is blank the middleware is not installed and the server logs an unauthenticated-startup warning. The `stdio` transport bypasses all of this (no network surface).

**Caching.** `tools/cache.py:TTLCache` is a tiny, dependency-free, process-local TTL cache (`OrderedDict` + timestamps, lock-free). Each tool that caches instantiates its own at module scope from its config's `cache_ttl_seconds` / `cache_max_entries`. `ttl <= 0` disables caching; `max_entries == 0` is unbounded. Cache keys must include everything that affects the result (e.g. Wolfram keys on input+assumption+units+maxchars) but **never** secrets like API keys.

## Conventions to follow

**Error handling — this is the core contract.** Every genuine failure raises `ToolError` (from `fastmcp.exceptions`), which FastMCP turns into a result with `isError: true`. This deliberately prevents a model from mistaking a failure for real data. A *valid-but-empty* result (a search with zero hits, a successful call that found nothing) is **not** a failure — return it as normal output. For multi-part calls like `get_company_data`, partial success returns the data plus an `errors` map; only raise `ToolError` when *everything* requested failed.

**Context-budget caps.** List/range parameters exposed to the model (result counts, financial periods, enrichment/lookback windows like `news_days`, etc.) are **maximums, not fixed amounts**. The model may request fewer; anything above the server-configured cap is silently clamped so an oversized response can't blow up the model's context window. Omitting the value uses the cap. Preserve this clamping pattern when adding parameters. Where a cap would otherwise *drop* data the model still needs (page content truncated to `WEB_SEARCH_MAX_PAGE_CHARS`), pair it with a continuation escape hatch instead of losing the rest: `fetch_page`'s `offset` param pages through truncated `content` (`_set_content` echoes a `next_offset` to re-request with), guaranteeing the rest of *any* format is reachable — including JSON and headingless docs that `query=`/`section=` can't narrow.

**Sync libraries in async tools.** Tool functions are `async`. Blocking/synchronous clients (e.g. the Stock tool's `requests` and `yfinance` calls) must be offloaded with `anyio.to_thread.run_sync(...)` rather than called directly — see `tools/stock_data.py`. `httpx.AsyncClient` is used directly where the library is async-native (web search, Wolfram). The same offload applies to **CPU-bound** work that could stall the single-process event loop: `fetch_page`'s model-supplied `query` regex is run in a thread (and bounded — over-long or nested-quantifier patterns fall back to a literal search, and the scan has a wall-clock budget — so a ReDoS pattern can't pin the server). When several independent blocking calls don't depend on each other, dispatch them concurrently rather than serially: `get_company_data`'s `_gather_sections` fires each section through its own `to_thread.run_sync` and `await`s them together, so a multi-section call pays the slowest provider's latency, not the sum.

**JSON output is centralized.** Tools never call `json.dumps` for their results — they serialize through `tools/serialize.py:to_json()`, which emits **compact** JSON by default (to conserve the model's context window) and **indented** JSON when debug mode is on. Debug mode is a single server-level switch (`MCP_DEBUG`, on `ServerSettings`): when enabled it also forces DEBUG-level logging and each tool logs its invocation via `log_call(...)` / a result summary via `log_result(...)` to stdout. Wrap each tool's `return` with `log_result(log, "<tool>", to_json(payload))`, and call `log_call(log, "<tool>", **args)` at the top of the function. Tools that return plain text (e.g. Wolfram) skip `to_json` but still use `log_result`. (The YouTube transcript path returns plain text from `fetch_transcript`, which `fetch_page` then wraps in its JSON envelope with `format: "youtube_transcript"`.)

**Tool docstrings are the model's API.** The `@mcp.tool()` function docstring is what the model sees. Keep the detailed usage guidance there (query formatting rules, when-to-use / when-not-to-use, param semantics) as the existing tools do.

**Surface configured caps to the model as concrete numbers.** A docstring can't be an f-string, so a cap written as the word "capped" leaves the model guessing the actual limit (and getting silently clamped, or refused). FastMCP lets you inject the live `cfg` value at registration time two ways, and `register(mcp)` runs at startup with `cfg` loaded, so both interpolate real numbers:
- **Per-argument cap** → annotate the parameter with `Annotated[T, Field(description=f"... {cfg.max_x} ...")]`. A `Field` description wins over a `:param:` line, so drop that param's `:param:` from the docstring to avoid a stale duplicate (keep `:param:` for the un-capped params). This is the authoritative place — it's where the model fills in the value. See `find_nearby_places` (`radius_m`/`limit`/`nearby_towns_limit`), `search_web` (`num_results`/`enrich_results`), and `get_company_data` (`periods`/`news_items`/`insider_weeks`/`history_bars`/`news_days`).
- **A number that lives in prose, not on a param** → pass `@mcp.tool(description=...)` a runtime-built string and interpolate there. When you do this, FastMCP still parses `:param:` from the docstring, so slim the docstring to a one-line human pointer plus the `:param:`/return lines and move the model-facing prose (including the `Returns JSON {...}` shape) into the `description=` string; keep the brace-heavy return shape as a plain (non-f) string to avoid escaping. See `find_nearby_places` (towns radius) and `fetch_page` (the `WEB_SEARCH_MAX_FETCH_URLS` URL cap — formerly a hardcoded literal that had to be kept in sync by hand).

Verify what the model actually receives — the serialized `tools/list` entry, which no longer matches the source docstring — with `python scripts/show_tool.py <tool>` (`--json` for the full schema; prefix env vars, e.g. `GEO_MAX_RADIUS_M=50000 python scripts/show_tool.py find_nearby_places`, to confirm a cap is genuinely dynamic). The script builds the real server via `build_server()`, so it covers every tool.

## Web Search support services

The Agentic Web Search tool depends on external services pointed at by `WEB_SEARCH_*` config: **SearXNG** (search backend — JSON output must be enabled, which the bundled `searxng/settings.yml` does), **FlareSolverr** (automatic fallback for bot-walled pages — `web_fetch._is_blocked_response` detects Cloudflare, PerimeterX/HUMAN, DataDome, and Akamai challenges, plus any HTTP 429 rate-limit, and re-fetches through a real browser; when a wall can't be bypassed, `fetch_page` falls back to the Wayback Machine — see below — and only then raises rather than returning the challenge as data), and **Apache Tika** (text extraction for PDF/Office/OpenDocument/RTF/EPUB). `docker-compose.yml` starts all three; the defaults reference them by container hostname. The other three tool groups have no local-service dependencies.

`fetch_page` has a fallback ladder for two failure modes that leave it without usable content: a page that fetches successfully but yields no readable text — the signature of a client-side-rendered (JavaScript) SPA whose body isn't in the static HTML (`_is_contentless` detects this, distinct from a bot wall) — and a page that's bot-walled or rate-limited (a 4xx/5xx/429 `blocked_detected`). Both share the same escalation: FlareSolverr (a real browser that runs the page's JS; for blocks this is attempted inside `_resilient_fetch`, for empty shells via an explicit re-render) and, failing that, the **Wayback Machine** (`web_fetch._fetch_from_wayback`, a public API needing no local service; gated by `WEB_SEARCH_WAYBACK_FALLBACK`). The Wayback path tries the archived snapshot's static HTML first, and — since an SPA's *archived* main document is itself a JS shell — renders the Wayback *replay* URL through FlareSolverr when it is (`_wayback_content`), so the archived sub-resources/XHRs hydrate the body from the archive. A recovered archived copy is returned as text via `_archived_text_payload` and flagged (`via: "archive.org"` + an `archived_snapshot` field + a staleness note naming why the live page failed) so the model treats it as possibly out of date; if every tier still yields nothing, `fetch_page` raises rather than returning the shell/challenge as data.
