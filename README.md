# openwebui-tools MCP server

A single [MCP](https://modelcontextprotocol.io) server that bundles five tool
groups (originally written for Open WebUI), so they can be used from any
MCP-capable client (Claude Desktop, IDEs, custom agents, Open WebUI's MCP
support, etc.). `fetch_page` doubles as a YouTube transcript fetcher, and the
email tool is send-only.

Built on [FastMCP v4](https://github.com/jlowin/fastmcp). The
default transport is **streamable-http**, so the server is reachable over the
network at `http://<host>:8000/mcp`.

## Tools

| Tool group             | MCP tools exposed                     |
| ---------------------- | ------------------------------------- |
| **Agentic Web Search** | `search_web`, `fetch_page`            |
| **Stock Data**         | `get_company_data`                    |
| **Wolfram Alpha**      | `query_wolfram_alpha`                 |
| **Place Search**       | `find_nearby_places`                  |
| **Email**              | `send_email`                          |

Each tool can be independently omitted from MCP registration with its
`<TOOL_NAME>_ENABLED` environment variable. For example, set
`SEARCH_WEB_ENABLED=false` and `FETCH_PAGE_ENABLED=false` to replace those two
with third-party tools. The available flags are `SEARCH_WEB_ENABLED`,
`FETCH_PAGE_ENABLED`, `GET_COMPANY_DATA_ENABLED`,
`QUERY_WOLFRAM_ALPHA_ENABLED`, `FIND_NEARBY_PLACES_ENABLED`, and
`SEND_EMAIL_ENABLED`. All default to `true`; restart the server after changing
them.

Every tool is context-budget aware: list/range parameters are **maximums**, not
fixed amounts. The model can request less per call, and anything above the
server-configured cap is silently clamped so an oversized response can't
overwhelm a model's context window. Omitting a value uses the cap.

### Agentic Web Search

`search_web(query, time_range=None, country=None, search_lang=None, safesearch=None, context_threshold_mode=None, num_results=None, max_tokens=None, enrich_results=None)`
— Search with Brave's [LLM Context API](https://api-dashboard.search.brave.com/documentation/services/llm-context),
which returns relevance-ranked excerpts extracted from source pages for direct
model consumption. Excerpts may contain text, tables, code, or JSON-serialized
structured data. Each result carries a URL, title, `snippets` list, optional
published date, description, and site name. The top-level `provider` is
`brave_llm_context`.

`time_range` accepts `day`/`week`/`month`/`year`/`all` or an inclusive custom
`YYYY-MM-DD to YYYY-MM-DD` range. `country`, `search_lang`, `safesearch`
(`off`/`moderate`/`strict`), and `context_threshold_mode`
(`strict`/`balanced`/`lenient`/`disabled`) map directly to Brave options.
`num_results` controls the source-URL count and `max_tokens` controls the
approximate total excerpt budget; both are clamped to configured server caps.
`enrich_results` optionally fetches top sources directly to add heading anchors
and a compact TOC (`0` skips this extra fetch).

Brave LLM Context does not expose result-page pagination or search categories.
Put those constraints in the query instead—for example `site:youtube.com` for
videos, which `fetch_page` can then read as transcripts. Brave supports operators
such as `site:`, `filetype:`, `intitle:`, `inbody:`, `lang:`, `loc:`, quoted
phrases, exclusion with `-`, and uppercase `AND`/`OR`/`NOT`; operators are
experimental and overly restrictive combinations may return no results.

`fetch_page(url, mode="text", section=None, query=None, max_matches=None, context_lines=None, include_match_toc=false, offset=None)` — Fetch the contents
of a single page (or a URL returned by `search_web`). Reads one URL per call —
to read several pages, call the tool once per URL. `mode="text"` returns the
page as markdown — headings, lists, tables, and hyperlinks (resolved to absolute URLs)
are preserved, so the model sees the page's structure and can fetch a link it
found in the content (set `WEB_SEARCH_MARKDOWN=false` for bare plain text).
Prominent images are replaced at their original positions by explicit
`[Image at this location: ...]` markers populated from page-provided alt text,
captions, or image metadata; these are textual stand-ins, not visual analysis.
Standalone image URLs return the same placeholder and any embedded SVG
description when available. Extracted headings carry visible `{#anchor}` markers
so a downstream agent can cite a precise section. Source-native heading IDs are
usable as URL fragments; generated `cite-*` anchors are stable identifiers only
within the returned extraction. Structured headings include `citation_url` when
the source page supplied a real fragment. `mode="structured"` returns metadata
only (title, description, heading outline, JSON-LD, and prominent image descriptions).
Document links (PDF, Word, Excel, PowerPoint,
OpenDocument, RTF, EPUB) are extracted via Apache Tika and always returned as
text. Passing a `section` (a heading from a `page_headings` outline) returns
just that section of an HTML page instead of the whole thing. Passing a `query`
(a keyword, phrase, or regex) returns only bounded extractive match windows.
`max_matches` controls how many windows are shown and `context_lines` controls
surrounding nonblank lines; both are clamped to server-configured safe limits.
Each window reports 1-based line ranges, exact match lines, its surrounding
heading, and a compact quality label (`exact_line`, `literal_substring`, or
`regex_pattern`). Set `include_match_toc=true` to return a small TOC containing
only matching headings, transcript timestamps, or document line ranges. YouTube
matched segments retain their `[M:SS]` timestamps. Regex evaluation has a hard
total time budget; a pattern
that exceeds it raises a tool error instead of returning an incomplete scan. If
a response is marked `truncated`, pass `offset` with the
returned `next_offset` to read the next chunk; this works for HTML, documents,
JSON, Reddit, and transcripts. A YouTube video URL returns the video's transcript
rather than the watch page (see [below](#youtube-transcripts-via-fetch_page)).

Reddit URLs are returned as compacted JSON. When Reddit OAuth is configured,
`fetch_page` uses the authenticated Data API first. It otherwise falls back in
order to Reddit's RSS feed, targeted `old.reddit.com` HTML extraction, and
official oEmbed metadata. RSS and old Reddit may expose only an initial comment
snapshot; the result reports comment counts/completeness when the source makes
them available.

Fetching is resilient: a direct `httpx` request first, an automatic
[FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) fallback for
bot/CAPTCHA walls and JavaScript-empty pages, then a last-resort
[Firecrawl](https://www.firecrawl.dev/) API fallback when local fetching cannot
recover usable content. This ordering keeps Firecrawl usage low enough for a
small API quota. A short page cache lets an agent loop that re-fetches the same
URL skip the network round-trip.

Fetching is also SSRF-guarded. Because a URL can come from search results or page
content the model just read, it's attacker-influenceable via indirect prompt
injection — so `fetch_page` resolves the target host and **refuses any non-public
address** (loopback, private, link-local, etc.), blocking access to `localhost`,
cloud metadata endpoints like `169.254.169.254`, and LAN hosts. The check is
applied to the initial URL *and* every redirect hop (a public URL can't `302`
into an internal one). Direct fetches also verify the connected peer IP so a DNS
rebinding race cannot swap a public lookup for a private connect. FlareSolverr
and Firecrawl follow redirects inside their own browsers — that request cannot
be intercepted — but the provider-reported final URL is SSRF-checked and the
body is discarded if it landed on a blocked host. To deliberately allow a
trusted local/private target you host, list its host, IP, or CIDR in
`WEB_SEARCH_SSRF_ALLOWLIST` (e.g. `localhost,127.0.0.1,10.0.0.0/8`).

Concurrent `fetch_page` work is also bounded: `WEB_SEARCH_MAX_CONCURRENT_DIRECT_FETCHES`,
`WEB_SEARCH_MAX_CONCURRENT_FLARESOLVERR`, `WEB_SEARCH_MAX_CONCURRENT_TIKA`, and
`WEB_SEARCH_MAX_CONCURRENT_FIRECRAWL` cap in-flight sidecar/API calls so a model
that fans out many reads cannot stampede FlareSolverr or Tika.

### Stock Data

`get_company_data(symbol, sections=None, statement="income", period="annual", periods=None, news_items=None, insider_weeks=None, history_bars=None, news_days=None, history_interval="1d", financial_metrics=None)`
— One company, or a short list of companies for comparison, with only the
sections you ask for. `symbol` accepts a ticker (`AAPL`), company name (`Apple`),
or a list of tickers/names; names are resolved to tickers via symbol search
before any data is fetched, so there's no separate lookup step.
Available `sections`:

- `quote` — latest price, day's change, open/high/low/previous close, volume.
- `profile` — name, sector, industry, market cap, employees, exchange, and key
  fundamentals (P/E, EPS, dividend yield, 52-week range, beta, margins).
- `financials` — income statement, balance sheet, or cash flow, controlled by
  `statement` (`income`/`balance`/`cashflow`) and `period`
  (`annual`/`quarterly`); `periods` sets how many to return. Set
  `financial_metrics` to a list or comma-separated string (for example,
  `["revenue", "gross profit", "free cash flow"]`) to return only matching
  rows and keep the response compact. The response includes a `metrics_filter`
  block listing matched and unmatched requested metrics.
- `earnings` — historical earnings: actual vs. estimated EPS, surprise %,
  revenue. `periods` sets how many to return.
- `news` — recent articles (headline, source, summary, url, published date).
  `news_items` sets how many to return; `news_days` sets the lookback window.
- `insiders` — insider buying/selling with a buy/sell summary and individual
  transactions. `insider_weeks` sets how far back to look.
- `price_history` — recent OHLC price bars (newest first); `history_bars` sets
  how many bars to return, and `history_interval` controls bar size (`1d`,
  `1wk`, or `1mo`).
- `peers` — competitor/peer tickers in the same sector or industry (Finnhub).
- `dividends` — dividend payment history plus stock splits (yfinance).
- `ownership` — ownership summary and top institutional holders (yfinance).

Defaults to `["quote", "profile"]` when `sections` is omitted. Data is sourced
across providers (Finnhub / yfinance / FMP) with optional yfinance fallback. On
partial success the response includes an `errors` map listing sections that
returned nothing; if every requested section fails, the call raises an error so
a failure is never mistaken for data. When `symbol` was a company name, the
response includes a `resolved_from` block naming the matched company (and any
alternatives) so you can confirm the right ticker was used. For a list input,
the response is `{"results":[...]}` and one failed ticker does not sink the
whole comparison unless every ticker fails.

> **Note:** earlier versions exposed `get_stock_quote`, `get_company_profile`,
> `get_financials`, `get_earnings`, `get_company_news`, and `search_symbol` as
> separate tools. These are now folded into the single `get_company_data` tool —
> the data tools via the `sections` parameter, and `search_symbol` via automatic
> name→ticker resolution on the `symbol` argument. This keeps the tool count low
> (better for smaller models' tool selection) and lets one call do what used to
> take two or more.

### Wolfram Alpha

`query_wolfram_alpha(query, assumption=None, units=None)` — Exact computation and
authoritative reference data: math, unit/currency conversion, physics &
chemistry, astronomy, geography & demographics, dates & times, finance,
nutrition, weather history, linguistics, and structured entity comparisons.
Queries should be English keyword-style (`"France population"`, not a full
sentence). If a result returns assumptions, re-send the same input with the
relevant `assumption` value to disambiguate. `units` may be `metric` or
`nonmetric`; omit it to use the server default.

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

`find_nearby_places(category="", near=None, latitude=None, longitude=None, radius_m=None, limit=None, nearby_towns_limit=None, place_details=False)`
— Find points of interest near a location via OpenStreetMap
[Overpass](https://overpass-api.de/). Specify the location either as `near` (a
place name, geocoded for you via [Nominatim](https://nominatim.org/) — so "vegan
restaurants in Portland" is a single call), as `near="lat,lon"`, as a map URL
containing coordinates, as an OpenStreetMap node/way/relation URL, or as
explicit `latitude`/`longitude` (which win if both are given). `category` is
plain language, not OSM tags:
`restaurant`, `coffee`, `pharmacy`, `atm`, `hotel`, `museum`, `gas station`, etc.
A food category can be prefixed with a diet — `vegan`, `vegetarian`, `halal`,
`kosher`, or `gluten free` (`"vegan restaurant"`, or just `"vegan"`). An
unrecognized category falls back to matching place names, so brands like
`"Starbucks"` work too. Results are sorted nearest-first and include distance
plus useful tags (cuisine, address, phone, website, opening hours) when
available. An empty `results` list means nothing matched in range (not an error).
Every POI search automatically includes nearby city/town/village centers that can
seed follow-up searches in neighboring municipalities; `nearby_towns_limit` can
reduce the number returned.

Set `place_details=true` to look up rich information about the place named in
`near` or at the supplied coordinates instead of searching for POIs around it.
That mode returns coordinates,
bounding box, address details, population when available, Wikidata/Wikipedia
links, website, phone, and alternatives; it ignores `category`, `radius_m`, and
`limit`.

The server has no access to the user's location, so a relative `near` value
("near me", "nearby", "around here", etc.) is refused with a message telling the
model to ask the user where to search or pass explicit coordinates — rather than
silently guessing a location.

It uses the public OpenStreetMap APIs by default and honors Nominatim's
[usage policy](https://operations.osmfoundation.org/policies/nominatim/): a
descriptive `GEO_USER_AGENT` (set this!) and a ~1 req/sec throttle on the public
API. To self-host, point `GEO_NOMINATIM_URL` / `GEO_OVERPASS_URL` at your own
instances, clear `GEO_OVERPASS_FALLBACK_URLS` if queries must stay private, and
set `GEO_MIN_REQUEST_INTERVAL_SECONDS=0`. Results are cached
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
- `WEB_SEARCH_BRAVE_API_KEY` — required for `search_web`; create a Search API subscription token at [Brave Search API](https://api.search.brave.com/). `WEB_SEARCH_BRAVE_API_URL`, localization/filter defaults, search candidate count, token budgets, and timeout are separately configurable. Calls are serialized with a default one-second quiet period (`WEB_SEARCH_BRAVE_REQUEST_DELAY_SECONDS`) for low-throughput plans; HTTP 429/502/503/504 and transient transport failures receive bounded exponential retries configured by `WEB_SEARCH_BRAVE_MAX_RETRIES` / `WEB_SEARCH_BRAVE_RETRY_BACKOFF_SECONDS`, honoring Brave's reset headers.
- `WEB_SEARCH_FIRECRAWL_API_KEY` — optional `fetch_page` credential; enables the last-resort fetch fallback when the first-line FlareSolverr HTML render is blocked/unusable or a known document is hidden behind an HTML challenge. Firecrawl is not used by `search_web`.
- `WEB_SEARCH_FIRECRAWL_HEDGE_ENABLED` / `WEB_SEARCH_FIRECRAWL_HEDGE_DELAY_SECONDS` — optionally start Firecrawl while a slow FlareSolverr render is still running; disabled by default to avoid unnecessary credits.
- `WEB_SEARCH_CLASSIFIER_API_URL` / `WEB_SEARCH_CLASSIFIER_MODEL` — optional OpenAI-compatible small-model classifier for ambiguous rendered pages; both must be set to enable it. `WEB_SEARCH_CLASSIFIER_API_KEY` supplies an optional bearer token.
- `WEB_SEARCH_CIRCUIT_BREAKER_*` — configure the short-lived host circuit that skips FlareSolverr after repeated failures when Firecrawl is available.
- `WEB_SEARCH_REDDIT_CLIENT_ID` / `WEB_SEARCH_REDDIT_CLIENT_SECRET` / `WEB_SEARCH_REDDIT_USER_AGENT` — optional Reddit OAuth credentials; strongly recommended for reliable Reddit post/comment fetching. See [Reddit Data API setup](#reddit-data-api-setup).
- `WEB_SEARCH_REDDIT_REQUEST_DELAY_SECONDS` — serializes Reddit acquisitions and leaves a quiet period between calls (default `1`) to reduce anonymous RSS/HTML burst throttling; `0` disables queueing. Post RSS/oEmbed URLs are also canonicalized so share/context parameters cannot bypass the raw-page cache.
- `WEB_SEARCH_REDDIT_RATE_LIMIT_RETRY_SECONDS` — after an anonymous RSS `HTTP 429`, wait this long (default `3`) and retry once before degrading to old Reddit/oEmbed; `0` disables the retry. These throttling mitigations are not a substitute for OAuth on hosted-server IPs.
- `WEB_SEARCH_SSRF_ALLOWLIST` — optional; hosts/IPs/CIDRs that `fetch_page` may reach despite the SSRF guard's default block on non-public addresses (e.g. a local page you host). Empty by default (all private/loopback/link-local targets blocked).
- `GEO_USER_AGENT` — for Geocoding & Places: set a descriptive User-Agent (ideally with contact info) as required by Nominatim's usage policy. Also set `GEO_NOMINATIM_EMAIL` to a contact address (recommended by the policy so they can reach you before blocking on heavy use). Self-hosters should also set `GEO_NOMINATIM_URL` / `GEO_OVERPASS_URL`, clear `GEO_OVERPASS_FALLBACK_URLS` when queries must stay private, and set `GEO_MIN_REQUEST_INTERVAL_SECONDS=0`.
- `EMAIL_USERNAME` / `EMAIL_PASSWORD` — required for `send_email`. For Gmail,
  `EMAIL_PASSWORD` must be a 16-character App Password, not the normal account
  password. `EMAIL_FROM_ADDRESS`, `EMAIL_FROM_NAME`, SMTP host/port/TLS, timeout,
  and recipient/attachment caps are configurable. Set `EMAIL_ALLOWED_RECIPIENTS`
  (addresses and/or domains) on any network-exposed server. Attachments stay off
  until `EMAIL_ATTACHMENT_ROOT` points at a directory the tool may read.
- `WEB_SEARCH_MAX_CONCURRENT_*` — in-flight caps for direct fetches, FlareSolverr,
  Tika, and Firecrawl (defaults 8 / 2 / 2 / 2).

Provider variables are grouped by prefix: `MCP_` (server), `WEB_SEARCH_`,
`STOCK_`, `WOLFRAM_`, `YOUTUBE_`, `GEO_`, `EMAIL_`. Tool availability uses the
exact MCP tool name followed by `_ENABLED` (for example,
`SEARCH_WEB_ENABLED=false`).

### Reddit Data API setup

Reddit blocks unidentified API traffic from many hosted-server networks. A
personal, non-commercial project can request free Data API access, subject to
Reddit's approval and rate limits:

1. Sign in to the Reddit account that will own the application.
2. Read Reddit's [Developer Platform and Data API access guidance](https://support.reddithelp.com/hc/en-us/articles/14945211791892-Developer-Platform-Accessing-Reddit-Data) and [Responsible Builder Policy](https://support.reddithelp.com/hc/en-us/articles/42728983564564-Responsible-Builder-Policy).
3. Submit Reddit's [Data API access request](https://support.reddithelp.com/hc/en-us/requests/new?ticket_form_id=14868593862164). Describe this as a personal, non-commercial MCP page reader and provide the repository URL if requested. Follow any approval instructions Reddit sends you.
4. Open [Reddit app preferences](https://www.reddit.com/prefs/apps), select
   **create another app**, and create a confidential application. For a personal
   server, the **script** type is normally appropriate. The redirect URI is not
   used by this server's application-only flow, but Reddit may still require a
   valid URL such as `http://localhost:8080`.
5. Copy the short value displayed beneath the application name as the client ID,
   and copy the value labeled `secret` as the client secret.
6. Add the credentials to `.env`, replacing `your_username` with the owning
   Reddit username:

   ```dotenv
   WEB_SEARCH_REDDIT_CLIENT_ID=your_client_id
   WEB_SEARCH_REDDIT_CLIENT_SECRET=your_client_secret
   WEB_SEARCH_REDDIT_USER_AGENT=linux:mcp-server:1.0 (by /u/your_username)
   ```

7. Restart the MCP server. Do not commit `.env` or expose the client secret.

The server exchanges these credentials for a short-lived application-only OAuth
token, caches it until shortly before expiration, and sends requests to
`oauth.reddit.com`. It does not store or require your Reddit password. Reddit's
current free-access limit is 100 queries per minute per OAuth client ID; consult
the [Data API Wiki](https://support.reddithelp.com/hc/en-us/articles/16160319875092-Reddit-Data-API-Wiki) for current requirements.

### Email

`send_email(recipients, subject, body, cc=None, bcc=None, reply_to=None, attachments=None)`
sends a plain-text email through the configured SMTP account. It is send-only:
it cannot read, list, or delete mailbox contents. `recipients`, `cc`, and `bcc`
are lists of email addresses; BCC recipients are included in the SMTP envelope
but not written into message headers. Set `EMAIL_ALLOWED_RECIPIENTS` to a list of
addresses and/or domains so a prompt-injected model cannot mail arbitrary people;
when that list is set, To/Cc/Bcc/Reply-To outside it are rejected. Attachments
are disabled unless `EMAIL_ATTACHMENT_ROOT` is set; paths must stay inside that
directory (symlink escapes are rejected), and the result reports only the
filename, not the resolved filesystem path. Counts are still capped by
`EMAIL_MAX_ATTACHMENTS` and `EMAIL_MAX_ATTACHMENT_BYTES`.

The result reports `status` (`sent` or `partial`), intended recipients by field,
attempted recipients, accepted recipients, refused recipients with SMTP codes and
server responses, invalid addresses, dropped addresses, and attachment metadata.
SMTP authentication, sender, connection, or total-recipient-refusal failures
raise tool errors instead of being returned as successful sends.

### Debug mode

Set `MCP_DEBUG=true` to enable debug mode: tool responses are serialized as
indented, human-readable JSON (instead of compact JSON) and each tool call emits
verbose per-call logs to stdout. Reddit `fetch_page` results also append a
redacted fallback trace to `note`, showing whether OAuth JSON, RSS, old Reddit,
and oEmbed were skipped, failed, or succeeded. Useful for troubleshooting; leave
it off in normal operation so responses stay compact in the model's context
window.

### Tool-catalog caching

FastMCP 4 advertises that opted-in modern clients may reuse the static MCP
component catalog for `MCP_TOOL_CATALOG_CACHE_TTL_SECONDS` (default 300 seconds),
reducing repeated `tools/list` round trips. Set it to `0` to disable the hint.
`MCP_TOOL_CATALOG_CACHE_SCOPE` defaults to `public` because every authenticated
caller currently sees the same tools; use `private` if visibility ever varies by
caller. These settings do **not** cache tool-call results—the provider-specific
TTL caches remain separate.

### Tool-name prefix in cross-references

Some MCP clients prepend a namespace to every tool name before showing it to the
model — Open WebUI, for example, forces an `mcp_` prefix, so `fetch_page` appears
to the model as `mcp_fetch_page`. The server keeps its tool names **bare**
(prefixing them here too would double it, e.g. `mcp_mcp_fetch_page`), but a few
docstrings point one tool at another (e.g. `search_web` tells the model to use
`fetch_page` to read a result). `MCP_TOOL_PREFIX` is the prefix spliced into
those cross-references so they match what the model actually sees. It defaults to
blank (no prefix); set it to `mcp_` when serving Open WebUI, or to whatever
prefix your client adds. The value is inserted verbatim, so include any trailing
separator (e.g. the `_`).

### Authentication

HTTP transports **require** `MCP_AUTH_TOKEN`. Clients must send an
`Authorization: Bearer <token>` header; anything else gets a `401`. Starting
streamable-http or SSE without a token is a startup error. Set
`MCP_ALLOW_UNAUTHENTICATED=true` only for a tightly firewalled local setup (the
server logs a warning in that case). The token is ignored for the `stdio`
transport, which has no network surface.

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

The compose file starts the server plus the local services used by `fetch_page`:
[FlareSolverr](https://github.com/FlareSolverr/FlareSolverr) (Cloudflare
fallback) and [Apache Tika](https://tika.apache.org/) (document text extraction).
Sidecar images are digest-pinned (Tika 4.0.0 minimal, FlareSolverr v3.5.0);
bump the digest when you intentionally upgrade. `search_web` uses Brave's hosted
API and requires `WEB_SEARCH_BRAVE_API_KEY`. HTTP mode also requires
`MCP_AUTH_TOKEN` in `.env`:

```
cp .env.example .env        # then edit it
docker compose up --build
```

The MCP endpoint is then available at `http://localhost:8000/mcp`.

If you don't need `fetch_page`, delete the `flaresolverr` / `tika` services (and
the `depends_on` block) from `docker-compose.yml`. The stock, Wolfram,
geocoding, and email tools have no local-service dependencies, though they may
need API keys, SMTP credentials, or internet access.

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
