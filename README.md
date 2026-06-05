# openwebui-tools MCP server

A single [MCP](https://modelcontextprotocol.io) server that bundles four tool
groups originally written for Open WebUI, so they can be used from any
MCP-capable client (Claude Desktop, IDEs, custom agents, Open WebUI's MCP
support, etc.).

Built on [FastMCP](https://github.com/modelcontextprotocol/python-sdk). The
default transport is **streamable-http**, so the server is reachable over the
network at `http://<host>:8000/mcp`.

## Tools

| Tool group             | MCP tools exposed                     |
| ---------------------- | ------------------------------------- |
| **Agentic Web Search** | `search_web`, `fetch_page`            |
| **Stock Data**         | `get_company_data`, `search_symbol`   |
| **Wolfram Alpha**      | `query_wolfram_alpha`                 |
| **YouTube Transcript** | `get_youtube_transcript`              |

Every tool is context-budget aware: list/range parameters are **maximums**, not
fixed amounts. The model can request less per call, and anything above the
server-configured cap is silently clamped so an oversized response can't
overwhelm a model's context window. Omitting a value uses the cap.

### Agentic Web Search

`search_web(query, time_range=None, category=None, num_results=None, enrich_results=None)`
— Search the web via SearXNG and return a ranked list of results. Each result
carries a url, title, snippet, optional published date, and (for the top
results) page metadata: a description plus a heading/JSON-LD table-of-contents
outline, so the model can decide which links are worth fetching in full.
`time_range` accepts `day`/`week`/`month`/`year`/`all`; `category` accepts
SearXNG categories (`general`, `news`, `science`, `it`, `social media`,
`videos`, `images`, `music`, `files`, `map`, comma-separated to combine).
`enrich_results` controls how many top results get full page metadata (`0`
skips enrichment).

`fetch_page(url, mode="text", section=None)` — Fetch the contents of a page (or
a URL returned by `search_web`). `mode="text"` returns readable plain text;
`mode="structured"` returns metadata only (title, description, heading outline,
JSON-LD). Document links (PDF, Word, Excel, PowerPoint, OpenDocument, RTF, EPUB)
are extracted via Apache Tika and always returned as text. Passing a `section`
(a heading from a `page_headings` outline) returns just that section of an HTML
page instead of the whole thing.

Fetching is resilient: a direct `httpx` request first, an automatic
[FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) fallback for
Cloudflare-blocked pages, and a short page cache so an agent loop that
re-fetches the same URL skips the network round-trip.

### Stock Data

`get_company_data(symbol, sections=None, statement="income", period="annual", periods=None, news_items=None, insider_weeks=None)`
— One ticker, only the sections you ask for. Available `sections`:

- `quote` — latest price, day's change, open/high/low/previous close, volume.
- `profile` — name, sector, industry, market cap, employees, exchange, and key
  fundamentals (P/E, EPS, dividend yield, 52-week range, beta, margins).
- `financials` — income statement, balance sheet, or cash flow, controlled by
  `statement` (`income`/`balance`/`cashflow`) and `period`
  (`annual`/`quarterly`); `periods` sets how many to return.
- `earnings` — historical earnings: actual vs. estimated EPS, surprise %,
  revenue. `periods` sets how many to return.
- `news` — recent articles (headline, source, summary, url, published date).
  `news_items` sets how many to return.
- `insiders` — insider buying/selling with a buy/sell summary and individual
  transactions. `insider_weeks` sets how far back to look.

Defaults to `["quote", "profile"]` when `sections` is omitted. Data is sourced
across providers (Finnhub / yfinance / FMP) with optional yfinance fallback. On
partial success the response includes an `errors` map listing sections that
returned nothing; if every requested section fails, the call raises an error so
a failure is never mistaken for data.

`search_symbol(query)` — Look up a ticker by company name or partial symbol
(e.g. `"apple"` → `AAPL`). Uses Finnhub when a key is configured, otherwise
falls back to a keyless Yahoo Finance lookup.

> **Note:** earlier versions exposed `get_stock_quote`, `get_company_profile`,
> `get_financials`, `get_earnings`, and `get_company_news` as separate tools.
> These are now folded into the single `get_company_data` tool via the
> `sections` parameter, which keeps the tool count low (better for smaller
> models' tool selection) and lets one call fetch several sections at once.

### Wolfram Alpha

`query_wolfram_alpha(query, assumption=None)` — Exact computation and
authoritative reference data: math, unit/currency conversion, physics &
chemistry, astronomy, geography & demographics, dates & times, finance,
nutrition, weather history, linguistics, and structured entity comparisons.
Queries should be English keyword-style (`"France population"`, not a full
sentence). If a result returns assumptions, re-send the same input with the
relevant `assumption` value to disambiguate.

### YouTube Transcript

`get_youtube_transcript(url, languages=None)` — Fetch a video's transcript /
closed captions as plain text for summarizing, quoting, searching, or
translating. Accepts any YouTube URL form (`watch`, `youtu.be`, `/shorts/`,
`/embed/`, `/live/`) or a bare 11-character video ID. `languages` is an optional
comma-separated priority list (e.g. `"en,es"`); it falls back to any available
transcript. Transcripts are cached (they almost never change), and optional
Webshare / generic proxy settings are supported for networks where YouTube
blocks the server's IP.

## Configuration

Every Open WebUI "valve" became an environment variable. Copy the example file
and edit it:

```
cp .env.example .env
```

See [.env.example](https://github.com/madelponte/mcp-server/blob/main/.env.example)
for the full list with defaults. Key things to set:

- `WOLFRAM_APP_ID` — required for the Wolfram tool ([free AppID](https://developer.wolframalpha.com)).
- `STOCK_FINNHUB_API_KEY` — recommended for Stock Data (improves `search_symbol` and quote/profile coverage; everything falls back to keyless yfinance).
- `WEB_SEARCH_SEARXNG_URL` — points at the bundled SearXNG service by default.

Variables are grouped by prefix: `MCP_` (server), `WEB_SEARCH_`, `STOCK_`,
`WOLFRAM_`, `YOUTUBE_`.

### Authentication

Set `MCP_AUTH_TOKEN` to require a bearer token on every HTTP request. Clients
must then send an `Authorization: Bearer <token>` header; anything else gets a
`401`. Leaving it blank disables auth and leaves the server open to anyone who
can reach it (the server logs a warning at startup in that case). The token is
ignored for the `stdio` transport, which has no network surface.

Generate a strong token, e.g.:

```
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
fallback), and [Apache Tika](https://tika.apache.org/) (document text
extraction):

```
cp .env.example .env        # then edit it
docker compose up --build
```

The MCP endpoint is then available at `http://localhost:8000/mcp`.

If you don't need web search, delete the `searxng` / `flaresolverr` / `tika`
services (and the `depends_on` block) from `docker-compose.yml`. The other three
tools have no local-service dependencies.

> **SearXNG note:** JSON output must be enabled for `search_web` to work — the
> bundled [searxng/settings.yml](https://github.com/madelponte/mcp-server/blob/main/searxng/settings.yml)
> does this. Set a real `SEARXNG_SECRET` in your `.env`.

## Run with Docker (server only)

```
docker build -t openwebui-tools-mcp .
docker run --rm -p 8000:8000 --env-file .env openwebui-tools-mcp
```

## Run locally (no Docker)

```
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

## License

MIT