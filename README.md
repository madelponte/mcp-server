# openwebui-tools MCP server

A single [MCP](https://modelcontextprotocol.io) server that bundles four tools
originally written for Open WebUI, so they can be used from any MCP-capable
client (Claude Desktop, IDEs, custom agents, Open WebUI's MCP support, etc.).

Bundled tools:

| Tool group | MCP tools exposed |
|------------|-------------------|
| **Agentic Web Search** | `search_web`, `fetch_page` |
| **Stock Data** | `get_company_data`, `search_symbol` |
| **Wolfram Alpha** | `query_wolfram_alpha` |
| **YouTube Transcript** | `get_youtube_transcript` |

Built on [FastMCP](https://github.com/modelcontextprotocol/python-sdk). The
default transport is **streamable-http** so the server is reachable over the
network at `http://<host>:8000/mcp`.

## Error handling

All tools follow a single convention: **a genuine failure raises a
`ToolError`**, which FastMCP returns to the client as a tool result with
`isError: true` (the error message becomes the result's text content). Tools
never hand back a failure disguised as normal output — no `❌ ...` strings, no
in-band `{"error": ...}` payloads. This matters for smaller local models, which
will otherwise happily summarize an error string as if it were data.

What this means for callers:

- **Success** returns the tool's normal payload — a JSON string for the web
  search and stock tools, plain text for Wolfram Alpha and YouTube transcripts.
- **Failure** is signalled at the protocol level (`isError: true`); inspect the
  result's error flag, and show or retry based on the message rather than
  feeding it back to the model as content.
- A **valid-but-empty** result is *not* a failure — e.g. a web search with zero
  hits returns normally with an empty `results` list.

When extending the server, raise
`from mcp.server.fastmcp.exceptions import ToolError` for any real failure
instead of returning an error sentinel.

## Caching

Several tools keep a small in-memory, process-local TTL cache (a shared helper,
[`tools/cache.py`](tools/cache.py)) so an agent loop that hits the same thing
repeatedly within a task doesn't pay for it every time. Nothing is persisted to
disk and only successful results are cached.

| Tool | What's cached | Default TTL | Env prefix |
|------|---------------|-------------|------------|
| `fetch_page` (and `search_web` enrichment) | the fetched page, keyed by URL | 5 min (`300s`) | `WEB_SEARCH_CACHE_*` |
| `get_youtube_transcript` | the finished transcript, keyed by video id + languages | 24 h (`86400s`) | `YOUTUBE_CACHE_*` |
| Stock Data tools | each upstream API response | 60 s | `STOCK_CACHE_TTL_SECONDS` |

For each, `*_CACHE_TTL_SECONDS=0` disables caching, and `*_CACHE_MAX_ENTRIES`
bounds memory by evicting the oldest entry once full (`0` = unbounded). The page
cache deliberately stores the raw fetch, not the formatted output, so a repeat
fetch of the same URL skips the network even when a different `mode`/`section`
is requested. Transcripts get a long TTL because they essentially never change.

## Configuration

Every Open WebUI "valve" became an environment variable. Copy the example file
and edit it:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for the full list with defaults. Key things to
set:

- `WOLFRAM_APP_ID` — required for the Wolfram tool ([free AppID](https://developer.wolframalpha.com)).
- `STOCK_FINNHUB_API_KEY` — recommended for Stock Data (the `search_symbol`
  tool requires it; everything else falls back to keyless yfinance).
- `WEB_SEARCH_SEARXNG_URL` — points at the bundled SearXNG service by default.

Variables are grouped by prefix: `MCP_` (server), `WEB_SEARCH_`, `STOCK_`,
`WOLFRAM_`, `YOUTUBE_`.

The `WEB_SEARCH_SEARXNG_TIME_RANGE` and `WEB_SEARCH_SEARXNG_CATEGORIES` valves
set the *defaults* for `search_web`, but the tool also takes optional
`time_range` (`day`/`week`/`month`/`year`/`all`) and `category` (e.g. `news`,
`science`, `it`) parameters per call, so a model can ask for, say, "news from
today" without reconfiguring the server. Omitting them falls back to these env
values.

### Authentication

Set `MCP_AUTH_TOKEN` to require a bearer token on every HTTP request. Clients
must then send an `Authorization: Bearer <token>` header; anything else gets a
`401`. Leaving it blank disables auth and leaves the server open to anyone who
can reach it (the server logs a warning at startup in that case). The token is
ignored for the `stdio` transport, which has no network surface.

Generate a strong token, e.g.:

```bash
openssl rand -hex 32
```

and put it in your `.env`:

```
MCP_AUTH_TOKEN=<your-generated-token>
```

> Open WebUI per-user valves and UI-only behaviors that don't apply to MCP were
> dropped: status/progress events, citation events, the Wolfram HTML result
> "card" (it now returns plain text), and the stock tool's `verbose_status` /
> `include_raw_numbers` per-user toggles.

## Run with Docker Compose (recommended)

The compose file builds the server and also starts the supporting services the
web search tool expects — [SearXNG](https://docs.searxng.org/) (search),
[FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) (Cloudflare
fallback), and [Apache Tika](https://tika.apache.org/) (PDF extraction):

```bash
cp .env.example .env        # then edit it
docker compose up --build
```

The MCP endpoint is then available at `http://localhost:8000/mcp`.

If you don't need web search, delete the `searxng`/`flaresolverr`/`tika`
services (and the `depends_on` block) from `docker-compose.yml`. The other
three tools have no local-service dependencies.

> **SearXNG note:** JSON output must be enabled for `search_web` to work — the
> bundled [searxng/settings.yml](searxng/settings.yml) does this. Set a real
> `SEARXNG_SECRET` in your `.env`.

## Run with Docker (server only)

```bash
docker build -t openwebui-tools-mcp .
docker run --rm -p 8000:8000 --env-file .env openwebui-tools-mcp
```

## Run locally (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit it
python server.py
```

Set `MCP_TRANSPORT=stdio` to run as a stdio MCP server instead (useful for
clients that spawn the process directly rather than connecting over HTTP).

## Connecting a client

For an HTTP client, point it at `http://<host>:8000/mcp` (streamable-http). For
example, a Claude Desktop / generic client config using a stdio bridge or native
streamable-http support would reference that URL. If `MCP_AUTH_TOKEN` is set,
configure the client to send an `Authorization: Bearer <token>` header (most MCP
clients expose a "headers" or "auth token" field for HTTP servers). For stdio
mode, configure the client to launch `python server.py` with the environment
variables set.
