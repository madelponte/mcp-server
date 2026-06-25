# openwebui-tools MCP server

A single [MCP](https://modelcontextprotocol.io) server that bundles four tool
groups (originally written for Open WebUI), so they can be used from any
MCP-capable client (Claude Desktop, IDEs, custom agents, Open WebUI's MCP
support, etc.). `fetch_page` doubles as a YouTube transcript fetcher.

Built on [FastMCP v3](https://github.com/jlowin/fastmcp). The
default transport is **streamable-http**, so the server is reachable over the
network at `http://<host>:8000/mcp`.

## Tools

| Tool group             | MCP tools exposed                     |
| ---------------------- | ------------------------------------- |
| **Agentic Web Search** | `search_web`, `fetch_page`            |
| **Stock Data**         | `get_company_data`                    |
| **Wolfram Alpha**      | `query_wolfram_alpha`                 |
| **Place Search**       | `find_nearby_places`                  |

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

`fetch_page(url, mode="text", section=None, query=None)` — Fetch the contents
of a single page (or a URL returned by `search_web`). Reads one URL per call —
to read several pages, call the tool once per URL. `mode="text"` returns the
page as markdown — headings, lists, tables, and hyperlinks (resolved to absolute URLs)
are preserved, so the model sees the page's structure and can fetch a link it
found in the content (set `WEB_SEARCH_MARKDOWN=false` for bare plain text);
`mode="structured"` returns metadata only (title, description,
heading outline, JSON-LD). Document links (PDF, Word, Excel, PowerPoint,
OpenDocument, RTF, EPUB) are extracted via Apache Tika and always returned as
text. Passing a `section` (a heading from a `page_headings` outline) returns
just that section of an HTML page instead of the whole thing. Passing a `query`
(a keyword, phrase, or regex) returns only the matching passages plus surrounding
context instead of the full page — useful for pulling one topic from a long
article, document, or transcript; for YouTube, matched segments keep their
`[M:SS]` timestamps. A YouTube video URL returns the video's transcript rather
than the watch page (see [below](#youtube-transcripts-via-fetch_page)).

Reddit URLs are automatically rewritten to Reddit's `.json` API and returned as
compacted JSON (post + comments tree) rather than the HTML page.

Fetching is resilient: a direct `httpx` request first, an automatic
[FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) fallback for
Cloudflare-blocked pages, and a short page cache so an agent loop that
re-fetches the same URL skips the network round-trip.

Fetching is also SSRF-guarded. Because a URL can come from search results or page
content the model just read, it's attacker-influenceable via indirect prompt
injection — so `fetch_page` resolves the target host and **refuses any non-public
address** (loopback, private, link-local, etc.), blocking access to `localhost`,
cloud metadata endpoints like `169.254.169.254`, and LAN hosts. The check is
applied to the initial URL *and* every redirect hop (a public URL can't `302`
into an internal one) and gates the FlareSolverr path too. To deliberately allow
a trusted local/private target you host, list its host, IP, or CIDR in
`WEB_SEARCH_SSRF_ALLOWLIST` (e.g. `localhost,127.0.0.1,10.0.0.0/8`).

### Stock Data

`get_company_data(symbol, sections=None, statement="income", period="annual", periods=None, news_items=None, insider_weeks=None, history_bars=None)`
— One company, only the sections you ask for. `symbol` accepts a ticker
(`AAPL`) **or** a company name (`Apple`); a name is resolved to its ticker via
symbol search before any data is fetched, so there's no separate lookup step.
Available `sections`:

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
- `price_history` — recent daily OHLC price bars (newest first); `history_bars`
  sets how many trading days to return.

Defaults to `["quote", "profile"]` when `sections` is omitted. Data is sourced
across providers (Finnhub / yfinance / FMP) with optional yfinance fallback. On
partial success the response includes an `errors` map listing sections that
returned nothing; if every requested section fails, the call raises an error so
a failure is never mistaken for data. When `symbol` was a company name, the
response includes a `resolved_from` block naming the matched company (and any
alternatives) so you can confirm the right ticker was used.

> **Note:** earlier versions exposed `get_stock_quote`, `get_company_profile`,
> `get_financials`, `get_earnings`, `get_company_news`, and `search_symbol` as
> separate tools. These are now folded into the single `get_company_data` tool —
> the data tools via the `sections` parameter, and `search_symbol` via automatic
> name→ticker resolution on the `symbol` argument. This keeps the tool count low
> (better for smaller models' tool selection) and lets one call do what used to
> take two or more.

### Wolfram Alpha

`query_wolfram_alpha(query, assumption=None)` — Exact computation and
authoritative reference data: math, unit/currency conversion, physics &
chemistry, astronomy, geography & demographics, dates & times, finance,
nutrition, weather history, linguistics, and structured entity comparisons.
Queries should be English keyword-style (`"France population"`, not a full
sentence). If a result returns assumptions, re-send the same input with the
relevant `assumption` value to disambiguate.

### YouTube transcripts (via `fetch_page`)

There is no separate YouTube tool — pass a YouTube video URL to `fetch_page` and
it returns the video's transcript / closed captions instead of scraping the
watch page, for summarizing, quoting, searching, or translating. Any YouTube URL
form works (`watch`, `youtu.be`, `/shorts/`, `/embed/`, `/live/`). Preferred
languages come from `YOUTUBE_DEFAULT_LANGUAGES` (falling back to any available
transcript). Transcripts are cached (they almost never change), and optional
Webshare / generic proxy settings are supported for networks where YouTube
blocks the server's IP. Folding this into `fetch_page` keeps the tool count low,
which helps smaller models avoid tool-selection paralysis.

### Place Search

`find_nearby_places(category, near=None, latitude=None, longitude=None, radius_m=None, limit=None)`
— Find points of interest near a location via OpenStreetMap
[Overpass](https://overpass-api.de/). Specify the location either as `near` (a
place name, geocoded for you via [Nominatim](https://nominatim.org/) — so "vegan
restaurants in Portland" is a single call) or as explicit `latitude`/`longitude`
(which win if both are given). `category` is plain language, not OSM tags:
`restaurant`, `coffee`, `pharmacy`, `atm`, `hotel`, `museum`, `gas station`, etc.
A food category can be prefixed with a diet — `vegan`, `vegetarian`, `halal`,
`kosher`, or `gluten free` (`"vegan restaurant"`, or just `"vegan"`). An
unrecognized category falls back to matching place names, so brands like
`"Starbucks"` work too. Results are sorted nearest-first and include distance
plus useful tags (cuisine, address, phone, website, opening hours) when
available. An empty `results` list means nothing matched in range (not an error).

The server has no access to the user's location, so a relative `near` value
("near me", "nearby", "around here", etc.) is refused with a message telling the
model to ask the user where to search or pass explicit coordinates — rather than
silently guessing a location.

It uses the public OpenStreetMap APIs by default and honors Nominatim's
[usage policy](https://operations.osmfoundation.org/policies/nominatim/): a
descriptive `GEO_USER_AGENT` (set this!) and a ~1 req/sec throttle on the public
API. To self-host, point `GEO_NOMINATIM_URL` / `GEO_OVERPASS_URL` at your own
instances and set `GEO_MIN_REQUEST_INTERVAL_SECONDS=0`. Results are cached
(place data changes slowly), which also eases the rate limits.

## Configuration

Every Open WebUI "valve" became an environment variable. Copy the example file
and edit it:

```
cp .env.example .env
```

See [.env.example](https://github.com/madelponte/mcp-server/blob/main/.env.example)
for the full list with defaults. Key things to set:

- `WOLFRAM_APP_ID` — required for the Wolfram tool ([free AppID](https://developer.wolframalpha.com)).
- `STOCK_FINNHUB_API_KEY` — recommended for Stock Data (improves name→ticker resolution and quote/profile coverage; everything falls back to keyless yfinance).
- `STOCK_FMP_API_KEY` — optional [Financial Modeling Prep](https://financialmodelingprep.com) key; when set, financial statements (`financials` section) are sourced from FMP instead of yfinance.
- `WEB_SEARCH_SEARXNG_URL` — points at the bundled SearXNG service by default.
- `WEB_SEARCH_SSRF_ALLOWLIST` — optional; hosts/IPs/CIDRs that `fetch_page` may reach despite the SSRF guard's default block on non-public addresses (e.g. a local page you host). Empty by default (all private/loopback/link-local targets blocked).

- `GEO_USER_AGENT` — for Geocoding & Places: set a descriptive User-Agent (ideally with contact info) as required by Nominatim's usage policy. Also set `GEO_NOMINATIM_EMAIL` to a contact address (recommended by the policy so they can reach you before blocking on heavy use). Self-hosters should also set `GEO_NOMINATIM_URL` / `GEO_OVERPASS_URL` and `GEO_MIN_REQUEST_INTERVAL_SECONDS=0`.

Variables are grouped by prefix: `MCP_` (server), `WEB_SEARCH_`, `STOCK_`,
`WOLFRAM_`, `YOUTUBE_`, `GEO_`.

### Debug mode

Set `MCP_DEBUG=true` to enable debug mode: tool responses are serialized as
indented, human-readable JSON (instead of compact JSON) and each tool call emits
verbose per-call logs to stdout. Useful for troubleshooting; leave it off in
normal operation so responses stay compact in the model's context window.

### Tool-name prefix

Set `MCP_TOOL_PREFIX` to prepend a namespace to every tool's name. For example
`MCP_TOOL_PREFIX=mcp_` exposes the tools as `mcp_search_web`, `mcp_fetch_page`,
`mcp_get_company_data`, and so on. This is handy for clients (such as Open WebUI)
that expect or add a prefix to keep tool names from colliding across servers.
Leaving it blank (the default) keeps the bare names unchanged. The value is
prepended verbatim, so include any trailing separator you want (e.g. the `_`).

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
clients that spawn the process directly rather than connecting over HTTP). Or
`MCP_TRANSPORT=sse` for SSE (Server-Sent Events) transport.

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