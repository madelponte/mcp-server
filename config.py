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


class WebSearchSettings(BaseSettings):
    """Valves for the Agentic Web Search tool."""

    model_config = SettingsConfigDict(
        env_prefix="WEB_SEARCH_", env_file=".env", extra="ignore"
    )

    searxng_url: str = Field(
        "http://searxng:8080",
        description="Base URL of your SearXNG instance (no trailing /search).",
    )
    num_results: int = Field(5, description="Number of search results to return.")
    enrich_top_n: int = Field(
        3,
        description="Fetch structured metadata for this many top results (0 disables).",
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
        description="Base URL of an Apache Tika server used for PDF text extraction.",
    )

    http_timeout_seconds: float = Field(25.0, description="HTTP timeout for fetches, in seconds.")
    verify_ssl: bool = Field(True, description="Verify TLS certificates.")
    user_agent: str = Field(DEFAULT_UA, description="User-Agent sent with direct fetches.")

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
    max_news_items: int = Field(5, description="Max news articles per query.")
    max_financial_periods: int = Field(4, description="Max historical financial periods returned.")


class WolframSettings(BaseSettings):
    """Valves for the Wolfram Alpha tool."""

    model_config = SettingsConfigDict(
        env_prefix="WOLFRAM_", env_file=".env", extra="ignore"
    )

    app_id: str = Field("", description="Wolfram Alpha AppID (free at developer.wolframalpha.com).")
    default_units: str = Field("metric", description="Default unit system: 'metric' or 'nonmetric'.")
    max_chars: int = Field(6800, description="Max characters in Wolfram's response.")


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
    webshare_proxy_username: str = Field("", description="Webshare Residential proxy username.")
    webshare_proxy_password: str = Field("", description="Webshare Residential proxy password.")
    http_proxy_url: str = Field(
        "", description="Generic HTTP/SOCKS proxy URL (used if Webshare fields are empty)."
    )


# Singletons imported by the tool modules and the server entrypoint.
server_settings = ServerSettings()
web_search_settings = WebSearchSettings()
stock_settings = StockSettings()
wolfram_settings = WolframSettings()
youtube_settings = YouTubeSettings()
