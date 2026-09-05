"""
Central configuration for the MCP server.

Every Open WebUI "valve" from the original tools is exposed here as an
environment variable. Provider settings use distinct environment prefixes so
variables can't collide; tool availability flags use each MCP tool's complete
name (for example, ``SEARCH_WEB_ENABLED``). Values are read from the process
environment and, when present, the `.env` file that sits next to this module
(see `.env.example`). Fields carry range constraints (``ge=``/``gt=``/``le=``)
so a misconfigured cap — e.g. a negative download cap that would abort every
fetch — fails fast at startup with a pydantic validation error instead of
silently changing runtime behavior.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor the .env file next to this module instead of resolving it against the
# process working directory: pydantic-settings resolves a bare ".env" relative
# to the CWD, so starting the server from any other directory would silently
# fall back to every default (notably MCP_AUTH_TOKEN, which would boot the HTTP
# transport unauthenticated).
ENV_FILE = Path(__file__).resolve().parent / ".env"


DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class ToolSettings(BaseSettings):
    """Startup availability flags for the registered MCP tools.

    These fields deliberately have no additional prefix: each field already
    contains the complete public tool name, producing environment variables
    such as ``SEARCH_WEB_ENABLED`` and ``GET_COMPANY_DATA_ENABLED``.
    Disabled tools are omitted from MCP registration entirely.
    """

    model_config = SettingsConfigDict(
        env_prefix="", env_file=ENV_FILE, extra="ignore"
    )

    search_web_enabled: bool = Field(
        True, description="Register the search_web tool."
    )
    fetch_page_enabled: bool = Field(
        True, description="Register the fetch_page tool."
    )
    get_company_data_enabled: bool = Field(
        True, description="Register the get_company_data tool."
    )
    query_wolfram_alpha_enabled: bool = Field(
        True, description="Register the query_wolfram_alpha tool."
    )
    find_nearby_places_enabled: bool = Field(
        True, description="Register the find_nearby_places tool."
    )
    send_email_enabled: bool = Field(
        True, description="Register the send_email tool."
    )


class ServerSettings(BaseSettings):
    """Transport / networking settings for the MCP server itself."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_", env_file=ENV_FILE, extra="ignore"
    )

    host: str = Field("0.0.0.0", description="Interface to bind to.")
    port: int = Field(8000, ge=1, le=65535, description="Port to listen on.")
    transport: str = Field(
        "streamable-http",
        description="MCP transport: 'streamable-http', 'sse', or 'stdio'.",
    )
    log_level: str = Field("INFO", description="Python logging level.")
    debug: bool = Field(
        False,
        description=(
            "Debug mode. When true, tools serialize their JSON results as "
            "indented, human-readable JSON (instead of the default compact JSON) "
            "and emit verbose DEBUG-level logs of each tool call and its arguments "
            "to stdout. Off by default so results stay compact in the model's "
            "context window. Enabling it forces the effective log level to DEBUG "
            "regardless of MCP_LOG_LEVEL."
        ),
    )
    auth_token: str = Field(
        "",
        description=(
            "Shared bearer token required on every HTTP request "
            "(Authorization: Bearer <token>). HTTP transports refuse to start "
            "when this is blank unless allow_unauthenticated is true. Ignored "
            "for the 'stdio' transport, which has no network surface."
        ),
    )
    allow_unauthenticated: bool = Field(
        False,
        description=(
            "Permit HTTP transports to start without MCP_AUTH_TOKEN. Default "
            "false: streamable-http and sse refuse to start if the token is "
            "blank. Set true only for a tightly firewalled local setup. stdio "
            "is never authenticated."
        ),
    )
    tool_prefix: str = Field(
        "",
        description=(
            "Prefix your MCP client prepends to tool names when it presents them "
            "to the model (Open WebUI, for example, forces an 'mcp_' namespace). "
            "The server does NOT rename its own tools — this value is only "
            "interpolated into docstrings where one tool refers to another (e.g. "
            "search_web pointing at fetch_page), so those cross-references read as "
            "'mcp_fetch_page' and match what the model actually sees. Set it to "
            "match your client's prefix, or blank if it adds none."
        ),
    )
    tool_catalog_cache_ttl_seconds: int = Field(
        300,
        ge=0,
        description=(
            "Advertise that clients may reuse the static MCP component catalog for "
            "this many seconds (0 disables the cache hint). This affects tools/list "
            "and other cacheable component metadata, not tool-call results."
        ),
    )
    tool_catalog_cache_scope: Literal["public", "private"] = Field(
        "public",
        description=(
            "Scope for FastMCP's component-catalog cache hint. 'public' permits "
            "sharing across authorization contexts; use 'private' if component "
            "visibility ever varies by caller. Ignored when the TTL is 0."
        ),
    )


class WebSearchSettings(BaseSettings):
    """Valves for the Agentic Web Search tool."""

    model_config = SettingsConfigDict(
        env_prefix="WEB_SEARCH_", env_file=ENV_FILE, extra="ignore"
    )

    brave_api_url: str = Field(
        "https://api.search.brave.com/res/v1/llm/context",
        description="Brave LLM Context endpoint used by search_web.",
    )
    brave_api_key: str = Field(
        "",
        description=(
            "Brave Search API subscription token. Required by search_web and "
            "sent only in the X-Subscription-Token request header."
        ),
    )
    brave_country: str = Field(
        "US",
        description="Default Brave result country (two-letter code).",
    )
    brave_search_lang: str = Field(
        "en", description="Default Brave result-language code."
    )
    brave_freshness: str = Field(
        "",
        description=(
            "Default Brave freshness filter: blank/all, day/week/month/year "
            "(or pd/pw/pm/py), or YYYY-MM-DD to YYYY-MM-DD."
        ),
    )
    brave_safesearch: str = Field(
        "",
        description=(
            "Default Brave adult-content filter: off, moderate, strict, or "
            "blank to let Brave apply its endpoint default."
        ),
    )
    brave_context_threshold_mode: str = Field(
        "",
        description=(
            "Default Brave relevance threshold: strict, balanced, lenient, "
            "disabled, or blank for Brave's calibrated default."
        ),
    )
    brave_search_count: int = Field(
        20,
        ge=1,
        le=50,
        description=(
            "Number of search candidates Brave may consider before selecting "
            "the smaller model-requested result set."
        ),
    )
    max_context_tokens: int = Field(
        8192,
        ge=1024,
        le=32768,
        description=(
            "Maximum approximate Brave excerpt-token budget the model may "
            "request; omission uses this value."
        ),
    )
    brave_max_tokens_per_url: int = Field(
        4096,
        ge=512,
        le=8192,
        description="Maximum Brave excerpt tokens retained from one source URL.",
    )
    brave_timeout_seconds: float = Field(
        30.0, gt=0, description="Timeout for one Brave LLM Context request."
    )
    brave_request_delay_seconds: float = Field(
        1.0,
        ge=0,
        description=(
            "Quiet period between completed Brave search calls. Calls are "
            "serialized when positive to respect one-request-per-second plans; "
            "0 disables spacing for higher-throughput plans."
        ),
    )
    brave_max_retries: int = Field(
        2,
        ge=0,
        le=5,
        description=(
            "Maximum retries after Brave HTTP 429/502/503/504 or a transient "
            "transport failure."
        ),
    )
    brave_retry_backoff_seconds: float = Field(
        1.0,
        ge=0,
        description="Initial Brave retry delay; later retries double it.",
    )

    # A maximum, not a fixed amount: search_web can request fewer source URLs.
    max_num_results: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of Brave source URLs to return.",
    )
    flaresolverr_url: str = Field(
        "http://flaresolverr:8191",
        description="Base URL of the first-line HTML renderer. Blank uses direct HTML fetching.",
    )
    flaresolverr_timeout_ms: int = Field(
        60000, ge=1, description="maxTimeout passed to FlareSolverr, in milliseconds."
    )
    flaresolverr_attempt_timeout_seconds: float = Field(
        20.0, gt=0,
        description=(
            "When Firecrawl is configured, stop waiting for one FlareSolverr "
            "attempt after this many seconds so the sequential fallback can fit "
            "within an MCP request timeout."
        ),
    )

    firecrawl_api_url: str = Field(
        "https://api.firecrawl.dev/v2/scrape",
        description="Firecrawl v2 scrape endpoint used as the last-resort page fallback.",
    )
    firecrawl_api_key: str = Field(
        "",
        description=(
            "Firecrawl API key used only by fetch_page after the FlareSolverr "
            "render is blocked, unusable, or unresolved, including for known "
            "documents blocked by an HTML challenge. Blank disables that fallback."
        ),
    )
    firecrawl_timeout_seconds: float = Field(
        60.0, gt=0,
        description=(
            "Timeout for a Firecrawl fetch_page scrape request, in seconds "
            "(clamped to Firecrawl's supported 1-300 second range)."
        ),
    )
    firecrawl_hedge_enabled: bool = Field(
        False,
        description=(
            "Start Firecrawl speculatively when FlareSolverr exceeds the hedge "
            "delay. Disabled by default because a cancelled hedge may still consume credits."
        ),
    )
    firecrawl_hedge_delay_seconds: float = Field(
        8.0, ge=0,
        description="Seconds to wait for FlareSolverr before starting a hedged Firecrawl scrape.",
    )

    classifier_api_url: str = Field(
        "",
        description=(
            "OpenAI-compatible API base URL or /chat/completions endpoint used only "
            "to classify ambiguous rendered pages. Blank disables LLM classification."
        ),
    )
    classifier_api_key: str = Field(
        "", description="Optional bearer token for the page-classifier API."
    )
    classifier_model: str = Field(
        "",
        description="Model name for ambiguous-page classification. Blank disables it.",
    )
    classifier_timeout_seconds: float = Field(
        5.0, gt=0, description="Timeout for one page-classifier request."
    )
    classifier_max_input_chars: int = Field(
        8000, ge=0, description="Maximum visible page characters sent to the classifier."
    )
    classifier_min_confidence: float = Field(
        0.7, ge=0, le=1, description="Minimum classifier confidence required to use its verdict."
    )

    circuit_breaker_enabled: bool = Field(
        True,
        description=(
            "Temporarily skip FlareSolverr for hosts with repeated failed renders "
            "when Firecrawl is configured."
        ),
    )
    circuit_breaker_failure_threshold: int = Field(
        3, ge=1, description="Distinct failed URLs on one host required to open the circuit."
    )
    circuit_breaker_window_seconds: int = Field(
        300, ge=0, description="Window in which host-level FlareSolverr failures are counted."
    )
    circuit_breaker_ttl_seconds: int = Field(
        300, ge=0, description="How long an opened host circuit bypasses FlareSolverr."
    )

    max_concurrent_direct_fetches: int = Field(
        8, ge=1,
        description=(
            "Maximum in-flight direct httpx fetches for fetch_page. "
            "Bounds a model that fans out many reads."
        ),
    )
    max_concurrent_flaresolverr: int = Field(
        2, ge=1,
        description="Maximum in-flight FlareSolverr renders."
    )
    max_concurrent_tika: int = Field(
        2, ge=1,
        description="Maximum in-flight Apache Tika extractions."
    )
    max_concurrent_firecrawl: int = Field(
        2, ge=1,
        description="Maximum in-flight Firecrawl scrapes."
    )

    tika_url: str = Field(
        "http://tika:9998",
        description="Base URL of an Apache Tika server used for document text extraction.",
    )
    tika_timeout_seconds: float = Field(
        90.0, gt=0, description="Timeout for a single Tika extraction request, in seconds."
    )
    tika_ocr_strategy: str = Field(
        "no_ocr",
        description=(
            "Tika PDF OCR strategy (X-Tika-PDFOcrStrategy): 'no_ocr' extracts only "
            "embedded text and is fast; 'auto'/'ocr_and_text_extraction'/'ocr_only' "
            "enable OCR of images but are much slower."
        ),
    )

    http_timeout_seconds: float = Field(25.0, gt=0, description="HTTP timeout for fetches, in seconds.")
    direct_probe_timeout_seconds: float = Field(
        5.0, gt=0,
        description=(
            "Timeout for fetch_page's lightweight direct resource-type probe. "
            "On timeout, HTML acquisition continues through FlareSolverr."
        ),
    )
    max_download_bytes: int = Field(
        104857600, ge=0,  # 100 MiB
        description=(
            "Maximum bytes a single fetch_page response may download before the "
            "fetch is aborted (0 = unbounded). The cap is on the decompressed "
            "stream, so it also bounds a decompression bomb. Sized for the largest "
            "reasonable document — an image-heavy/scanned PDF, which Tika still "
            "reduces to plain text — while protecting the single-process server "
            "from a multi-GB body exhausting memory."
        ),
    )
    verify_ssl: bool = Field(True, description="Verify TLS certificates.")
    user_agent: str = Field(DEFAULT_UA, description="User-Agent sent with direct fetches.")
    reddit_client_id: str = Field(
        "",
        description=(
            "Reddit OAuth application client ID. When this, reddit_client_secret, "
            "and reddit_user_agent are set, fetch_page uses Reddit's authenticated "
            "Data API before its anonymous RSS/HTML fallbacks."
        ),
    )
    reddit_client_secret: str = Field(
        "", description="Reddit OAuth application client secret."
    )
    reddit_user_agent: str = Field(
        "",
        description=(
            "Descriptive User-Agent required by Reddit, for example "
            "'linux:mcp-server:1.0 (by /u/your_username)'."
        ),
    )
    reddit_request_delay_seconds: float = Field(
        1.0,
        ge=0,
        description=(
            "Minimum quiet period between Reddit fetch_page acquisitions. "
            "Requests are serialized to reduce anonymous RSS/HTML throttling; "
            "0 disables queueing."
        ),
    )
    reddit_rate_limit_retry_seconds: float = Field(
        3.0,
        ge=0,
        description=(
            "Wait this many seconds and retry Reddit RSS once after HTTP 429; "
            "0 disables the retry."
        ),
    )
    ssrf_allowlist: str = Field(
        "",
        description=(
            "Comma/space-separated hosts, IPs, or CIDRs that bypass the SSRF guard "
            "so fetch_page may reach a trusted local/private page you host "
            "(e.g. 'localhost,127.0.0.1,192.168.1.50,10.0.0.0/8'). Applies to "
            "redirect targets too. Empty = block all non-public addresses."
        ),
    )

    cache_ttl_seconds: int = Field(
        300, ge=0,
        description=(
            "Cache fetched pages this many seconds so an agent loop that re-fetches "
            "the same URL skips the network round-trip (0 disables the cache)."
        ),
    )
    cache_max_entries: int = Field(
        128, ge=0, description="Max number of cached pages before the oldest is evicted (0 = unbounded)."
    )
    cache_max_item_bytes: int = Field(
        5242880, ge=0,  # 5 MiB
        description=(
            "Maximum size of one raw fetched page retained in the process cache "
            "(0 = unbounded). Larger pages are still returned normally but are "
            "not cached, preventing a handful of large downloads from exhausting "
            "memory before the entry-count limit is reached."
        ),
    )

    markdown: bool = Field(
        True,
        description=(
            "Return fetch_page text-mode content as markdown — headings, lists, "
            "tables, and hyperlinks (resolved to absolute URLs) are preserved, so "
            "the model sees the page's structure and can follow links. Set false "
            "for the old structure-free plain-text extraction."
        ),
    )
    max_page_chars: int = Field(15000, ge=1, description="Max characters of page content before truncation.")
    max_image_descriptions: int = Field(
        10, ge=0,
        description=(
            "Maximum prominent image placeholders returned from one HTML page "
            "(0 disables image descriptions)."
        ),
    )
    # Keep the historical env name for existing fetch_page configurations.
    max_enrich_headings: int = Field(
        25, ge=1, description="Max headings in fetch_page structured/section responses."
    )

    # `fetch_page`'s optional `query` does server-side extractive filtering:
    # it returns only the segments (paragraphs / transcript caption lines) that
    # lexically match the query, each with this many neighbouring segments of
    # context on either side. A larger context reads more naturally but costs
    # more of the model's context window.
    query_context_segments: int = Field(
        2, ge=0,
        description=(
            "Default surrounding nonblank lines on each side of a fetch_page "
            "query match when the model omits context_lines."
        ),
    )
    max_query_context_lines: int = Field(
        8, ge=0,
        description=(
            "Maximum surrounding nonblank lines on each side of a fetch_page "
            "query match; larger model requests are clamped."
        ),
    )
    # MAXIMUM, not a fixed amount: a context-budget cap on how many distinct
    # match windows a single filtered fetch_page response may contain.
    max_query_matches: int = Field(
        10, ge=1,
        description=(
            "Maximum number of distinct match windows fetch_page's `query` "
            "filter returns before the remainder are dropped (a context-budget "
            "cap; the reported match_count still counts every match)."
        ),
    )


class StockSettings(BaseSettings):
    """Valves for the Stock Data tool."""

    model_config = SettingsConfigDict(
        env_prefix="STOCK_", env_file=ENV_FILE, extra="ignore"
    )

    finnhub_api_key: str = Field("", description="Finnhub API key (free at finnhub.io).")
    fmp_api_key: str = Field("", description="Financial Modeling Prep API key (optional).")

    default_provider: str = Field(
        "auto", description="Default provider: 'auto', 'finnhub', 'yfinance', or 'fmp'."
    )
    financials_provider: str = Field(
        "auto", description="Provider for financial statements: 'auto', 'fmp', 'yfinance', 'finnhub'."
    )
    prefer_yfinance_fallback: bool = Field(
        True, description="Retry with yfinance when the primary provider fails."
    )

    request_timeout: int = Field(15, gt=0, description="HTTP request timeout in seconds.")
    cache_ttl_seconds: int = Field(60, ge=0, description="Cache responses this long (0 disables).")

    # The following are MAXIMUMS, not fixed amounts. `get_company_data` lets the
    # model request a smaller range per call; anything above these caps is
    # clamped down so an oversized response can't overwhelm the model's context
    # window. When the model doesn't specify, the cap is used (the prior behavior).
    max_symbols: int = Field(
        2, ge=1,
        description=(
            "Maximum number of tickers/names a single get_company_data call will "
            "process when passed a list; extras are skipped. Lets the model compare "
            "a few companies in one call without blowing its context window. The "
            "tool description states this number to the model."
        ),
    )
    max_news_items: int = Field(5, ge=1, description="Maximum news articles returned per query.")
    max_news_lookback_days: int = Field(
        30, ge=1, description="Maximum days of company news to look back on."
    )
    max_financial_periods: int = Field(
        4, ge=1, description="Maximum historical financial periods (income/balance/cashflow) returned."
    )
    max_earnings_periods: int = Field(
        8, ge=1, description="Maximum historical earnings periods returned."
    )
    max_insider_lookback_weeks: int = Field(
        12, ge=1, description="Maximum weeks of insider buying/selling to look back on."
    )
    max_history_bars: int = Field(
        30, ge=1, description="Maximum daily OHLC price-history bars returned."
    )
    # Caps for the peers / dividends / ownership sections. These are pure
    # server-side safety caps (not model-tunable params): they bound the
    # response size for sections whose natural length is open-ended.
    max_peers: int = Field(
        15, ge=1, description="Maximum peer tickers returned by the 'peers' section."
    )
    max_dividend_events: int = Field(
        24, ge=1,
        description=(
            "Maximum dividend payments (and stock splits) returned by the "
            "'dividends' section, most recent first."
        ),
    )
    max_institutional_holders: int = Field(
        10, ge=1,
        description=(
            "Maximum top institutional holders returned by the 'ownership' section."
        ),
    )


class WolframSettings(BaseSettings):
    """Valves for the Wolfram Alpha tool."""

    model_config = SettingsConfigDict(
        env_prefix="WOLFRAM_", env_file=ENV_FILE, extra="ignore"
    )

    app_id: str = Field("", description="Wolfram Alpha AppID (free at developer.wolframalpha.com).")
    default_units: str = Field("metric", description="Default unit system: 'metric' or 'nonmetric'.")
    max_chars: int = Field(6800, ge=1, description="Max characters in Wolfram's response.")
    http_timeout_seconds: float = Field(
        30.0, gt=0, description="HTTP timeout for the Wolfram Alpha request, in seconds."
    )

    cache_ttl_seconds: int = Field(
        3600, ge=0,
        description=(
            "Cache results this many seconds so an agent loop that re-asks the same "
            "computation skips the network round-trip. Wolfram results for a given "
            "(input, assumption, units) are effectively deterministic, so a long TTL "
            "is safe (0 disables the cache)."
        ),
    )
    cache_max_entries: int = Field(
        256, ge=0, description="Max number of cached results before the oldest is evicted (0 = unbounded)."
    )


class YouTubeSettings(BaseSettings):
    """Valves for the YouTube Transcript tool."""

    model_config = SettingsConfigDict(
        env_prefix="YOUTUBE_", env_file=ENV_FILE, extra="ignore"
    )

    default_languages: str = Field(
        "en", description="Comma-separated language codes to try, in priority order."
    )
    include_timestamps: bool = Field(
        False, description="Prefix each line with a [M:SS]/[H:MM:SS] timestamp."
    )
    max_characters: int = Field(0, ge=0, description="Truncate transcript to this many chars (0 = no limit).")
    cache_ttl_seconds: int = Field(
        86400, ge=0,
        description=(
            "Cache transcripts this many seconds (0 disables). Transcripts almost "
            "never change, so a long TTL is safe and avoids re-fetching."
        ),
    )
    cache_max_entries: int = Field(
        256, ge=0, description="Max number of cached transcripts before the oldest is evicted (0 = unbounded)."
    )
    webshare_proxy_username: str = Field("", description="Webshare Residential proxy username.")
    webshare_proxy_password: str = Field("", description="Webshare Residential proxy password.")
    http_proxy_url: str = Field(
        "", description="Generic HTTP/SOCKS proxy URL (used if Webshare fields are empty)."
    )


class GeocodingSettings(BaseSettings):
    """Valves for the Geocoding & Place Search tool (OpenStreetMap)."""

    model_config = SettingsConfigDict(
        env_prefix="GEO_", env_file=ENV_FILE, extra="ignore"
    )

    # Geocoding backend. Defaults to OpenStreetMap's public Nominatim instance;
    # point this at your own deployment to self-host (and then set
    # min_request_interval_seconds to 0 to drop the public-API throttle).
    nominatim_url: str = Field(
        "https://nominatim.openstreetmap.org",
        description="Base URL of a Nominatim instance (no trailing /search).",
    )
    # Point-of-interest backend. Defaults to the public Overpass API; can be
    # pointed at a self-hosted Overpass instance.
    overpass_url: str = Field(
        "https://overpass-api.de/api/interpreter",
        description="Full URL of the primary Overpass API interpreter endpoint.",
    )
    overpass_fallback_urls: str = Field(
        "https://overpass.openstreetmap.fr/api/interpreter",
        description=(
            "Comma-separated fallback Overpass interpreter URLs tried when the "
            "primary times out, rate-limits, or returns 502/503/504. Set blank to "
            "disable, especially when a self-hosted query must remain private."
        ),
    )

    # Nominatim's usage policy REQUIRES a descriptive User-Agent that identifies
    # the application (ideally with contact info). The shared browser-style UA
    # used elsewhere is NOT acceptable here — set this to identify your deployment.
    user_agent: str = Field(
        "openwebui-tools-mcp/1.0 (OpenStreetMap geocoding; "
        "+https://github.com/madelponte/mcp-server)",
        description=(
            "User-Agent sent to Nominatim/Overpass. Required by Nominatim's usage "
            "policy and must identify your application — customize it with contact info."
        ),
    )
    nominatim_email: str = Field(
        "",
        description=(
            "Optional contact email passed to Nominatim (email= param). Recommended "
            "by the usage policy so they can reach you before blocking on heavy use."
        ),
    )
    language: str = Field(
        "en",
        description="Preferred result language (sent as the Accept-Language header).",
    )

    http_timeout_seconds: float = Field(
        20.0, gt=0, description="HTTP timeout for a Nominatim request, in seconds."
    )
    overpass_timeout_seconds: float = Field(
        30.0, gt=0,
        description=(
            "Timeout for an Overpass request, in seconds. Also passed into the "
            "Overpass query's [timeout:N] so the server stops its own work in time."
        ),
    )
    # Nominatim's public API allows at most 1 request/second. We serialize calls
    # and space them by this interval. Set to 0 when self-hosting to disable it.
    # One throttle covers both OpenStreetMap backends (Nominatim and Overpass).
    # The public APIs rate-limit aggressive callers (Nominatim caps at ~1/sec;
    # Overpass rejects bursts with 429/504), so requests to both are serialized
    # and spaced by this single interval. Set to 0 to disable when self-hosting.
    min_request_interval_seconds: float = Field(
        1.0, ge=0,
        description=(
            "Minimum seconds between OpenStreetMap requests (Nominatim + Overpass "
            "share one throttle; the public APIs cap aggressive callers). Set to 0 "
            "to disable throttling when self-hosting."
        ),
    )

    # The following are MAXIMUMS, not fixed amounts. The tool lets the model
    # request fewer per call; anything larger is clamped so an oversized response
    # can't overwhelm the model's context window. Omitting the value uses the cap.
    max_nearby_results: int = Field(
        20, ge=1, description="Maximum nearby places returned per query."
    )
    default_nearby_results: int = Field(
        8, ge=1, description="Nearby places returned when the model doesn't specify."
    )

    default_radius_m: int = Field(
        1500, ge=1, description="Search radius (meters) used when the model doesn't specify."
    )
    max_radius_m: int = Field(
        20000, ge=1, description="Maximum search radius (meters) for a nearby query."
    )

    # Every POI search includes a nearby-towns companion list (city/town/village
    # around the center) to seed follow-up searches in neighboring municipalities.
    # MAX is also the default, so omitting the count returns up to this many.
    max_nearby_towns: int = Field(
        10, ge=1,
        description=(
            "Maximum (and default) number of nearby towns automatically returned "
            "with each POI search."
        ),
    )
    nearby_towns_radius_m: int = Field(
        40000, ge=1,
        description=(
            "Radius (meters) for the automatic nearby-towns search. Larger than the "
            "POI radius because satellite towns sit well outside a city's core."
        ),
    )
    overpass_max_elements: int = Field(
        120, ge=1,
        description=(
            "Safety cap on elements fetched from Overpass before they are sorted by "
            "distance and trimmed to the requested count. Bounds the response size "
            "in dense areas (the nearest few may be missed if a tiny radius holds "
            "more than this)."
        ),
    )

    max_place_matches: int = Field(
        5, ge=1,
        description=(
            "Maximum candidate places a place_details lookup returns (the top "
            "match plus alternatives). Caps the alternatives list size."
        ),
    )

    cache_ttl_seconds: int = Field(
        86400, ge=0,
        description=(
            "Cache geocoding/Overpass results this many seconds (0 disables). "
            "Place data changes slowly, so a long TTL is safe and helps honor the "
            "public APIs' rate limits."
        ),
    )
    cache_max_entries: int = Field(
        256, ge=0, description="Max cached results before the oldest is evicted (0 = unbounded)."
    )


class EmailSettings(BaseSettings):
    """Valves for the Email (send-only) tool.

    Defaults target Gmail. Gmail no longer accepts your normal account
    password over SMTP — you must create an **App Password** (Google Account →
    Security → 2-Step Verification → App passwords) and put that 16-character
    value in EMAIL_PASSWORD, with your full address in EMAIL_USERNAME.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMAIL_", env_file=ENV_FILE, extra="ignore"
    )

    smtp_host: str = Field(
        "smtp.gmail.com",
        description="SMTP server hostname. Default is Gmail; change to use another provider.",
    )
    smtp_port: int = Field(
        465, ge=1, le=65535,
        description=(
            "SMTP server port. 465 for implicit SSL (use_ssl=true), 587 for "
            "STARTTLS (use_ssl=false)."
        ),
    )
    username: str = Field(
        "",
        description=(
            "SMTP login username — for Gmail, your full email address "
            "(e.g. you@gmail.com). Required for the tool to work."
        ),
    )
    password: str = Field(
        "",
        description=(
            "SMTP login password. For Gmail this MUST be a 16-character App "
            "Password (not your normal account password); requires 2-Step "
            "Verification enabled on the account. Required for the tool to work."
        ),
    )
    from_address: str = Field(
        "",
        description=(
            "Address the mail is sent From. Defaults to `username` when blank. "
            "For Gmail this must be your own address (or a verified alias) or the "
            "send is rejected."
        ),
    )
    from_name: str = Field(
        "",
        description="Optional display name shown in the From header (e.g. 'My Bot').",
    )
    use_ssl: bool = Field(
        True,
        description=(
            "True → connect with implicit SSL/TLS (SMTP_SSL, typically port 465). "
            "False → connect plaintext then upgrade with STARTTLS (typically port 587)."
        ),
    )
    timeout_seconds: float = Field(
        30.0, gt=0, description="Timeout for connecting to and talking to the SMTP server, in seconds."
    )
    # MAXIMUM, not a fixed amount: a guard against a single call fanning out to an
    # unbounded recipient list. Recipients past this cap are dropped (and named in
    # the result) rather than silently sent to.
    max_recipients: int = Field(
        25, ge=1,
        description=(
            "Maximum number of recipient addresses a single send_email call will "
            "send to; addresses past this are dropped and reported, not sent."
        ),
    )
    max_attachments: int = Field(
        5, ge=0,
        description=(
            "Maximum number of file attachments a single send_email call will "
            "include; extra attachment paths are rejected."
        ),
    )
    max_attachment_bytes: int = Field(
        10485760, ge=1,
        description=(
            "Maximum size in bytes for each individual email attachment "
            "(default 10 MiB)."
        ),
    )
    allowed_recipients: str = Field(
        "",
        description=(
            "Comma/space-separated allowlist of recipient addresses and/or "
            "domains (e.g. 'you@example.com, example.com, @corp.com'). When "
            "set, To/Cc/Bcc/Reply-To outside the list are rejected. Blank "
            "allows any address — set this on any network-exposed server."
        ),
    )
    attachment_root: str = Field(
        "",
        description=(
            "Directory the send_email tool may read attachments from. Blank "
            "disables attachments entirely so a prompt-injected model cannot "
            "exfiltrate local files. Relative attachment paths are resolved "
            "inside this directory; absolute paths must stay within it, and "
            "symlink escapes are rejected."
        ),
    )


# Singletons imported by the tool modules and the server entrypoint.
tool_settings = ToolSettings()
server_settings = ServerSettings()
web_search_settings = WebSearchSettings()
stock_settings = StockSettings()
wolfram_settings = WolframSettings()
youtube_settings = YouTubeSettings()
geocoding_settings = GeocodingSettings()
email_settings = EmailSettings()
