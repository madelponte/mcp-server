"""
Central configuration for the MCP server.

Every Open WebUI "valve" from the original tools is exposed here as an
environment variable. Each tool has its own settings class with a distinct
env prefix so the variables can't collide. Values are read from the process
environment and, when present, an `.env` file (see `.env.example`).
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class ServerSettings(BaseSettings):
    """Transport / networking settings for the MCP server itself."""

    model_config = SettingsConfigDict(
        env_prefix="MCP_", env_file=".env", extra="ignore"
    )

    host: str = Field("0.0.0.0", description="Interface to bind to.")
    port: int = Field(8000, description="Port to listen on.")
    transport: str = Field(
        "streamable-http",
        description="MCP transport: 'streamable-http', 'sse', or 'stdio'.",
    )
    log_level: str = Field("INFO", description="Python logging level.")
    auth_token: str = Field(
        "",
        description=(
            "Shared bearer token required on every HTTP request "
            "(Authorization: Bearer <token>). Blank disables auth — the server "
            "is then open to anyone who can reach it. Ignored for the 'stdio' "
            "transport, which has no network surface."
        ),
    )


class WebSearchSettings(BaseSettings):
    """Valves for the Agentic Web Search tool."""

    model_config = SettingsConfigDict(
        env_prefix="WEB_SEARCH_", env_file=".env", extra="ignore"
    )

    searxng_url: str = Field(
        "http://searxng:8080",
        description="Base URL of your SearXNG instance (no trailing /search).",
    )
    # Both of the following are MAXIMUMS, not fixed amounts. `search_web` lets
    # the model request fewer results / less enrichment per call; anything above
    # these caps is clamped down so an oversized response (or a pile of
    # table-of-contents outlines) can't overwhelm the model's context window.
    # When the model doesn't specify, the cap is used (the prior behavior).
    max_num_results: int = Field(
        5, description="Maximum number of search results to return."
    )
    max_enrich_results: int = Field(
        5,
        description=(
            "Maximum number of top results to fetch structured metadata "
            "(description + table-of-contents outline) for (0 disables)."
        ),
    )
    default_enrich_results: int = Field(
        3,
        description=(
            "Number of top results to enrich when the model doesn't specify "
            "(clamped to max_enrich_results; 0 disables enrichment by default)."
        ),
    )
    searxng_categories: str = Field("general", description="Comma-separated SearXNG categories.")
    searxng_language: str = Field("en", description="SearXNG language code (e.g. 'en', 'all').")
    searxng_time_range: str = Field("", description="'', 'day', 'week', 'month', or 'year'.")
    searxng_safesearch: int = Field(0, description="0=off, 1=moderate, 2=strict.")

    flaresolverr_url: str = Field(
        "http://flaresolverr:8191",
        description="Base URL of FlareSolverr (no trailing /v1). Blank disables fallback.",
    )
    flaresolverr_timeout_ms: int = Field(
        60000, description="maxTimeout passed to FlareSolverr, in milliseconds."
    )

    tika_url: str = Field(
        "http://tika:9998",
        description="Base URL of an Apache Tika server used for document text extraction.",
    )
    tika_timeout_seconds: float = Field(
        90.0, description="Timeout for a single Tika extraction request, in seconds."
    )
    tika_ocr_strategy: str = Field(
        "no_ocr",
        description=(
            "Tika PDF OCR strategy (X-Tika-PDFOcrStrategy): 'no_ocr' extracts only "
            "embedded text and is fast; 'auto'/'ocr_and_text_extraction'/'ocr_only' "
            "enable OCR of images but are much slower."
        ),
    )

    http_timeout_seconds: float = Field(25.0, description="HTTP timeout for fetches, in seconds.")
    verify_ssl: bool = Field(True, description="Verify TLS certificates.")
    user_agent: str = Field(DEFAULT_UA, description="User-Agent sent with direct fetches.")

    cache_ttl_seconds: int = Field(
        300,
        description=(
            "Cache fetched pages this many seconds so an agent loop that re-fetches "
            "the same URL skips the network round-trip (0 disables the cache)."
        ),
    )
    cache_max_entries: int = Field(
        128, description="Max number of cached pages before the oldest is evicted (0 = unbounded)."
    )

    max_page_chars: int = Field(25000, description="Max characters of page content before truncation.")
    max_enrich_headings: int = Field(25, description="Max headings per enriched result.")
    max_snippet_chars: int = Field(400, description="Max characters of each result snippet.")


class StockSettings(BaseSettings):
    """Valves for the Stock Data tool."""

    model_config = SettingsConfigDict(
        env_prefix="STOCK_", env_file=".env", extra="ignore"
    )

    finnhub_api_key: str = Field("", description="Finnhub API key (free at finnhub.io).")
    fmp_api_key: str = Field("", description="Financial Modeling Prep API key (optional).")
    alpha_vantage_api_key: str = Field("", description="Alpha Vantage API key (reserved/optional).")

    default_provider: str = Field(
        "auto", description="Default provider: 'auto', 'finnhub', 'yfinance', or 'fmp'."
    )
    financials_provider: str = Field(
        "auto", description="Provider for financial statements: 'auto', 'fmp', 'yfinance', 'finnhub'."
    )
    prefer_yfinance_fallback: bool = Field(
        True, description="Retry with yfinance when the primary provider fails."
    )

    request_timeout: int = Field(15, description="HTTP request timeout in seconds.")
    cache_ttl_seconds: int = Field(60, description="Cache responses this long (0 disables).")

    # The following are MAXIMUMS, not fixed amounts. `get_company_data` lets the
    # model request a smaller range per call; anything above these caps is
    # clamped down so an oversized response can't overwhelm the model's context
    # window. When the model doesn't specify, the cap is used (the prior behavior).
    max_news_items: int = Field(5, description="Maximum news articles returned per query.")
    max_financial_periods: int = Field(
        4, description="Maximum historical financial periods (income/balance/cashflow) returned."
    )
    max_earnings_periods: int = Field(
        8, description="Maximum historical earnings periods returned."
    )
    max_insider_lookback_weeks: int = Field(
        12, description="Maximum weeks of insider buying/selling to look back on."
    )


class WolframSettings(BaseSettings):
    """Valves for the Wolfram Alpha tool."""

    model_config = SettingsConfigDict(
        env_prefix="WOLFRAM_", env_file=".env", extra="ignore"
    )

    app_id: str = Field("", description="Wolfram Alpha AppID (free at developer.wolframalpha.com).")
    default_units: str = Field("metric", description="Default unit system: 'metric' or 'nonmetric'.")
    max_chars: int = Field(6800, description="Max characters in Wolfram's response.")
    http_timeout_seconds: float = Field(
        30.0, description="HTTP timeout for the Wolfram Alpha request, in seconds."
    )

    cache_ttl_seconds: int = Field(
        3600,
        description=(
            "Cache results this many seconds so an agent loop that re-asks the same "
            "computation skips the network round-trip. Wolfram results for a given "
            "(input, assumption, units) are effectively deterministic, so a long TTL "
            "is safe (0 disables the cache)."
        ),
    )
    cache_max_entries: int = Field(
        256, description="Max number of cached results before the oldest is evicted (0 = unbounded)."
    )


class YouTubeSettings(BaseSettings):
    """Valves for the YouTube Transcript tool."""

    model_config = SettingsConfigDict(
        env_prefix="YOUTUBE_", env_file=".env", extra="ignore"
    )

    default_languages: str = Field(
        "en", description="Comma-separated language codes to try, in priority order."
    )
    include_timestamps: bool = Field(
        False, description="Prefix each line with a [M:SS]/[H:MM:SS] timestamp."
    )
    max_characters: int = Field(0, description="Truncate transcript to this many chars (0 = no limit).")
    cache_ttl_seconds: int = Field(
        86400,
        description=(
            "Cache transcripts this many seconds (0 disables). Transcripts almost "
            "never change, so a long TTL is safe and avoids re-fetching."
        ),
    )
    cache_max_entries: int = Field(
        256, description="Max number of cached transcripts before the oldest is evicted (0 = unbounded)."
    )
    webshare_proxy_username: str = Field("", description="Webshare Residential proxy username.")
    webshare_proxy_password: str = Field("", description="Webshare Residential proxy password.")
    http_proxy_url: str = Field(
        "", description="Generic HTTP/SOCKS proxy URL (used if Webshare fields are empty)."
    )


class GeocodingSettings(BaseSettings):
    """Valves for the Geocoding & Place Search tool (OpenStreetMap)."""

    model_config = SettingsConfigDict(
        env_prefix="GEO_", env_file=".env", extra="ignore"
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
        description="Full URL of an Overpass API interpreter endpoint.",
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
        20.0, description="HTTP timeout for a Nominatim request, in seconds."
    )
    overpass_timeout_seconds: float = Field(
        30.0,
        description=(
            "Timeout for an Overpass request, in seconds. Also passed into the "
            "Overpass query's [timeout:N] so the server stops its own work in time."
        ),
    )
    # Nominatim's public API allows at most 1 request/second. We serialize calls
    # and space them by this interval. Set to 0 when self-hosting to disable it.
    min_request_interval_seconds: float = Field(
        1.0,
        description=(
            "Minimum seconds between Nominatim requests (the public API caps at "
            "1/sec). Set to 0 to disable throttling when self-hosting."
        ),
    )

    # The following are MAXIMUMS, not fixed amounts. The tool lets the model
    # request fewer per call; anything larger is clamped so an oversized response
    # can't overwhelm the model's context window. Omitting the value uses the cap.
    max_nearby_results: int = Field(
        20, description="Maximum nearby places returned per query."
    )
    default_nearby_results: int = Field(
        8, description="Nearby places returned when the model doesn't specify."
    )

    default_radius_m: int = Field(
        1500, description="Search radius (meters) used when the model doesn't specify."
    )
    max_radius_m: int = Field(
        20000, description="Maximum search radius (meters) for a nearby query."
    )
    overpass_max_elements: int = Field(
        120,
        description=(
            "Safety cap on elements fetched from Overpass before they are sorted by "
            "distance and trimmed to the requested count. Bounds the response size "
            "in dense areas (the nearest few may be missed if a tiny radius holds "
            "more than this)."
        ),
    )

    cache_ttl_seconds: int = Field(
        86400,
        description=(
            "Cache geocoding/Overpass results this many seconds (0 disables). "
            "Place data changes slowly, so a long TTL is safe and helps honor the "
            "public APIs' rate limits."
        ),
    )
    cache_max_entries: int = Field(
        256, description="Max cached results before the oldest is evicted (0 = unbounded)."
    )


# Singletons imported by the tool modules and the server entrypoint.
server_settings = ServerSettings()
web_search_settings = WebSearchSettings()
stock_settings = StockSettings()
wolfram_settings = WolframSettings()
youtube_settings = YouTubeSettings()
geocoding_settings = GeocodingSettings()
