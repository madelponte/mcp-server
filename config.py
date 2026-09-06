"""
Central configuration for the MCP server.
The server is configured by a single YAML file — bind-mounted into the container
at ``/app/config.yaml`` (see ``docker-compose.yml``), or picked up next to this
module when running from source — instead of a pile of environment variables.
Every setting lives under a named section whose keys are the field names below:

    tools:                      # which MCP tools get registered
      search_web_enabled: true
    server:                     # transport, logging, bearer auth
      port: 8000
      auth_tokens:              # any number of named client credentials
        - name: open-webui
          token: "..."
        - name: claude-desktop
          token: "..."
    web_search:                 # search_web / fetch_page
      brave_api_key: "..."
    stock: {}    wolfram: {}    youtube: {}    geocoding: {}    email: {}

Resolution order for any value is **process environment variable → YAML file →
field default**. Each section keeps the environment prefix it used before
(``MCP_``, ``WEB_SEARCH_``, ``STOCK_``, ``WOLFRAM_``, ``YOUTUBE_``, ``GEO_``,
``EMAIL_``; the ``tools`` flags have none) with the variable name being that
prefix plus the uppercased key — so an existing ``WEB_SEARCH_BRAVE_API_KEY``
still works and can keep a secret out of the mounted file. An environment
variable set to a blank value counts as unset, so a stale ``FOO=`` left in a
shell or image layer cannot wipe a configured YAML value.

The file is located, first match wins:

1. ``MCP_CONFIG_FILE`` — when set, the file **must** exist. A typo in an
   explicitly requested path is a startup error, never a silent fallback to
   defaults.
2. ``config.yaml`` / ``config.yml`` next to this module (the repo root, and
   ``/app`` inside the image).
3. ``/etc/mcp-server/config.yaml``.

Finding no file at all is not an error: every field has a default and the HTTP
transport still refuses to start without a bearer token. Configured values are
validated by pydantic at startup; range constraints (``ge=``/``gt=``/``le=``)
make a misconfigured cap — e.g. a negative download cap that would abort every
fetch — fail fast with a :class:`ConfigError` naming the section and key
instead of silently changing runtime behavior. Unknown keys are logged as
warnings and ignored, so a config file written for an older release still
starts a newer server.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args, get_origin

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

log = logging.getLogger(__name__)

#: Environment variable naming the config file to load.
CONFIG_PATH_ENV_VAR = "MCP_CONFIG_FILE"

#: Directory searched for ``config.yaml`` when MCP_CONFIG_FILE is unset, so a
#: host-managed file can be mounted without touching the image or the compose
#: file's environment.
SYSTEM_CONFIG_DIR = Path("/etc/mcp-server")

#: Directory this module lives in — the repo root, and /app in the image.
CONFIG_DIR = Path(__file__).resolve().parent

#: File names probed inside CONFIG_DIR / SYSTEM_CONFIG_DIR, in priority order.
CONFIG_FILENAMES = ("config.yaml", "config.yml")


DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class ConfigError(RuntimeError):
    """A missing explicitly-named config file, malformed YAML, or bad value.

    Raised while loading configuration, i.e. at import time, so a broken
    deployment fails at startup instead of mid-request.
    """


class BaseSection(BaseModel):
    """One top-level YAML section (``server:``, ``web_search:``, …).

    Field names are the YAML keys. ``_env_prefix`` records the environment
    prefix that keeps the pre-YAML variable names working as overrides, and
    ``_list_join_fields`` lists keys that may be written as a YAML list even
    though the consuming code parses a comma-separated string.

    Unknown keys are ignored by pydantic but reported by
    :func:`_warn_unknown_keys` while the file is read: a typo should be visible
    in the log without bricking a server that was upgraded past a setting the
    file still mentions.
    """

    model_config = ConfigDict(extra="ignore")

    _env_prefix: ClassVar[str] = ""
    _list_join_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def _join_list_values(cls, data: Any) -> Any:
        """Accept ``overpass_fallback_urls: [a, b]`` as well as ``"a,b"``."""
        if isinstance(data, dict) and cls._list_join_fields:
            for name in cls._list_join_fields:
                value = data.get(name)
                if isinstance(value, (list, tuple)):
                    if not all(isinstance(item, str) for item in value):
                        raise ValueError(
                            f"{name} list entries must be strings; quote numeric "
                            "values and YAML boolean words such as 'no'"
                        )
                    data = {**data, name: ",".join(value)}
        return data


class AuthToken(BaseModel):
    """A named bearer credential accepted by the HTTP transports.

    Unlike the settings sections, an unrecognized key here is a hard error: a
    misspelled field would silently drop a client's credential, and auth is not
    the place to accept a best guess.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        "client",
        description=(
            "Label used in logs to identify which client presented its token, so "
            "traffic can be told apart and one credential can be revoked without "
            "touching the others. Not a secret."
        ),
    )
    token: str = Field(
        ...,
        min_length=1,
        description=(
            "Bearer token this client must present "
            "(Authorization: Bearer <token>). Generate one per client with "
            "'openssl rand -hex 32'."
        ),
    )

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("auth token name must not be blank")
        return value

    @field_validator("token")
    @classmethod
    def _token_must_be_ascii(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("auth token must not be blank")
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "auth token must contain only ASCII characters"
            ) from exc
        return value


class ToolSettings(BaseSection):
    """Startup availability flags for the registered MCP tools (``tools:``).

    These fields deliberately carry no environment prefix: each already contains
    the complete public tool name, so the override variables stay
    ``SEARCH_WEB_ENABLED``, ``GET_COMPANY_DATA_ENABLED``, and so on.
    Disabled tools are omitted from MCP registration entirely.
    """

    _env_prefix = ""

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


class ServerSettings(BaseSection):
    """Transport, logging, and authentication settings (``server:``)."""

    _env_prefix = "MCP_"

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
    auth_tokens: list[AuthToken] = Field(
        default_factory=list,
        description=(
            "Named bearer credentials accepted on every HTTP request "
            "(Authorization: Bearer <token>). Any listed token authenticates, so "
            "one entry per client (or per trust boundary) lets traffic be told "
            "apart in the log and a single credential be revoked by editing this "
            "file. HTTP transports refuse to start when the list is empty unless "
            "allow_unauthenticated is true. Ignored for the 'stdio' transport, "
            "which has no network surface."
        ),
    )
    auth_token: str = Field(
        "",
        description=(
            "Legacy single shared bearer token, equivalent to one auth_tokens "
            "entry named 'default'. Kept so an existing MCP_AUTH_TOKEN "
            "environment variable (or a pre-YAML file) keeps working; prefer "
            "auth_tokens once you have more than one client."
        ),
    )
    allow_unauthenticated: bool = Field(
        False,
        description=(
            "Permit HTTP transports to start with no configured bearer token. "
            "Default false: streamable-http and sse refuse to start until at "
            "least one auth_tokens entry exists. Set true only for a tightly "
            "firewalled local setup. stdio is never authenticated."
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

    @field_validator("auth_token")
    @classmethod
    def _validate_legacy_token(cls, value: str) -> str:
        # Keep a blank legacy field as the documented opt-out, but validate a
        # nonblank one at its own field so errors name MCP_AUTH_TOKEN correctly.
        return AuthToken(token=value).token if value.strip() else ""

    def token_entries(self) -> list[AuthToken]:
        """Every accepted credential: the named list plus the legacy field.

        Computed on demand rather than stored, so a value changed after
        construction (a test monkeypatch, for example) is still reflected here.
        Identical token values are collapsed, which is what makes the legacy
        ``auth_token``/``MCP_AUTH_TOKEN`` field harmless when a file also lists
        the same credential explicitly.
        """
        entries = list(self.auth_tokens)
        legacy = (self.auth_token or "").strip()
        if legacy and legacy not in {entry.token for entry in entries}:
            entries.append(AuthToken(name="default", token=legacy))
        seen: set[str] = set()
        unique = []
        for entry in entries:
            if entry.token not in seen:
                seen.add(entry.token)
                unique.append(entry)
        return unique

    @property
    def auth_configured(self) -> bool:
        """True when at least one bearer token is available."""
        return bool(self.token_entries())

    @model_validator(mode="after")
    def _token_names_are_unique(self) -> "ServerSettings":
        """Duplicate labels would make the audit log ambiguous, so refuse to start."""
        # Validate names before deduplicating secrets, including the legacy
        # credential only when it adds a distinct accepted token.
        names = [entry.name for entry in self.auth_tokens]
        legacy = (self.auth_token or "").strip()
        if legacy and legacy not in {entry.token for entry in self.auth_tokens}:
            names.append("default")
        duplicate = next((n for n in names if names.count(n) > 1), None)
        if duplicate is not None:
            raise ValueError(
                f"duplicate auth token name {duplicate!r}: every server.auth_tokens "
                "entry needs a distinct name so log lines identify one client"
            )
        return self


class WebSearchSettings(BaseSection):
    """Valves for the Agentic Web Search tools (``web_search:``).

    The section name predates the split tooling: it covers ``fetch_page`` as
    well as ``search_web``.
    """

    _env_prefix = "WEB_SEARCH_"
    _list_join_fields = frozenset({"ssrf_allowlist"})

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
            "Comma/space-separated (or YAML list of) hosts, IPs, or CIDRs that bypass the SSRF guard "
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
    # Keep the historical key/env name for existing fetch_page configurations.
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


class StockSettings(BaseSection):
    """Valves for the Stock Data tool (``stock:``)."""

    _env_prefix = "STOCK_"

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


class WolframSettings(BaseSection):
    """Valves for the Wolfram Alpha tool (``wolfram:``)."""

    _env_prefix = "WOLFRAM_"

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


class YouTubeSettings(BaseSection):
    """Valves for YouTube transcript retrieval inside fetch_page (``youtube:``)."""

    _env_prefix = "YOUTUBE_"
    _list_join_fields = frozenset({"default_languages"})

    default_languages: str = Field(
        "en",
        description=(
            "Comma-separated (or YAML list of) language codes to try, in priority "
            "order."
        ),
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


class GeocodingSettings(BaseSection):
    """Valves for Geocoding & Place Search (OpenStreetMap) (``geocoding:``)."""

    _env_prefix = "GEO_"
    _list_join_fields = frozenset({"overpass_fallback_urls"})

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
            "Comma-separated (or YAML list of) fallback Overpass interpreter URLs tried when the "
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


class EmailSettings(BaseSection):
    """Valves for the Email (send-only) tool (``email:``).

    Defaults target Gmail. Gmail no longer accepts your normal account
    password over SMTP — you must create an **App Password** (Google Account →
    Security → 2-Step Verification → App passwords) and put that 16-character
    value in ``email.password``, with your full address in ``email.username``.
    """

    _env_prefix = "EMAIL_"
    _list_join_fields = frozenset({"allowed_recipients"})

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
            "Comma/space-separated (or YAML list of) allowlist entries: recipient addresses and/or "
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


# ---------------------------------------------------------------------------
# Section registry, file discovery, and loading
# ---------------------------------------------------------------------------

#: Top-level YAML section name -> settings class. Each class' ``_env_prefix``
#: names the environment variables that override that section.
SECTIONS: dict[str, type[BaseSection]] = {
    "tools": ToolSettings,
    "server": ServerSettings,
    "web_search": WebSearchSettings,
    "stock": StockSettings,
    "wolfram": WolframSettings,
    "youtube": YouTubeSettings,
    "geocoding": GeocodingSettings,
    "email": EmailSettings,
}


class AppConfig(BaseModel):
    """The fully validated configuration: one instance per section.

    A missing section is not an error — it simply keeps its field defaults (and
    any environment overrides for it).
    """

    model_config = ConfigDict(extra="ignore")

    tools: ToolSettings = Field(default_factory=ToolSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    stock: StockSettings = Field(default_factory=StockSettings)
    wolfram: WolframSettings = Field(default_factory=WolframSettings)
    youtube: YouTubeSettings = Field(default_factory=YouTubeSettings)
    geocoding: GeocodingSettings = Field(default_factory=GeocodingSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)


def _normalize_key(key: Any) -> str:
    """Fold case and hyphens, so ``WEB_SEARCH:`` / ``ssrf-allowlist`` also work.

    Canonical names are lower snake_case; the fold costs nothing and spares
    someone porting an old `.env` file from rewriting every line by hand.
    """
    return str(key).strip().lower().replace("-", "_")


def _normalize_keys(value: Any) -> Any:
    """Recursively normalize mapping keys anywhere in a parsed YAML document."""
    if isinstance(value, Mapping):
        return {_normalize_key(k): _normalize_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(item) for item in value]
    return value


def _is_env_overridable(annotation: Any) -> bool:
    """Whether an environment variable can express this field's type.

    Scalars and ``Literal``s can (pydantic coerces the string). Nested models and
    lists are YAML-only: there is no sane syntax for a list of named tokens in a
    variable, and pretending otherwise invites a confusing parse error.
    """
    if get_origin(annotation) is Literal:
        return all(isinstance(option, (str, int, float, bool)) for option in get_args(annotation))
    return isinstance(annotation, type) and issubclass(annotation, (str, int, float, bool))


def _env_overrides(
    cls: type[BaseSection], env: Mapping[str, str]
) -> dict[str, tuple[str, str]]:
    """Map field name -> (variable name, raw value) for the variables that are set.

    A variable holding only whitespace counts as unset, so a stray ``FOO=``
    inherited from a shell, an image layer, or a half-migrated compose file
    cannot wipe out the value configured in YAML.
    """
    overrides: dict[str, tuple[str, str]] = {}
    for name, field in cls.model_fields.items():
        if not _is_env_overridable(field.annotation):
            continue
        env_name = f"{cls._env_prefix}{name.upper()}"
        raw = env.get(env_name)
        if raw is None or not raw.strip():
            continue
        overrides[name] = (env_name, raw)
    return overrides


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """The nested model behind a field annotation, if it has exactly one."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], BaseModel):
        return args[0]
    return None


def _describe_error_location(section: str, loc: tuple[Any, ...]) -> str:
    # A whole-section validator (empty loc) reports against the section itself.
    return ".".join([section] + [str(part) for part in loc])


def _value_hint(error: Mapping[str, Any]) -> str:
    """Explain the one YAML mistake that catches almost every migrated file.

    YAML 1.1 resolves bare ``off`` / ``no`` / ``yes`` / ``on`` to booleans, so a
    textual value like ``brave_safesearch: off`` arrives as ``False`` and fails
    string validation. Guessing a replacement ("false"?) would silently change
    what the provider is asked for, so the fix is to say it out loud: quote it.
    """
    if error.get("type") == "string_type" and isinstance(error.get("input"), bool):
        return (
            "YAML read this as a boolean. Quote the word to keep it text, e.g. "
            'off -> "off".'
        )
    return ""


def _format_validation_error(
    section: str,
    exc: ValidationError,
    path: Path | None,
    overrides: Mapping[str, tuple[str, str]],
) -> str:
    """Turn a pydantic error into an operator-readable startup message.

    Each line names the setting and — when the value came from the environment
    rather than the file — the variable responsible, which is otherwise very
    hard to spot when a legacy variable silently wins.
    """
    lines = []
    for error in exc.errors():
        loc = error.get("loc") or ()
        setting = str(loc[0]) if loc else ""
        override = overrides.get(setting)
        source = f" [from {override[0]}]" if override else ""
        line = f"  - {_describe_error_location(section, loc)}: {error['msg']}{source}"
        hint = _value_hint(error)
        lines.append(line + (f"\n    {hint}" if hint else ""))
    where = f"in {path}" if path is not None else "from environment variables"
    return (
        f"Invalid {section} configuration {where}:\n"
        + "\n".join(lines)
        + "\nFix the value(s) above; config.example.yaml documents the accepted range of each."
    )


def _build_section(
    name: str,
    cls: type[BaseSection],
    raw: Mapping[str, Any] | None,
    path: Path | None,
    env: Mapping[str, str],
) -> BaseSection:
    """Validate one section from YAML + environment (env wins)."""
    values: dict[str, Any] = dict(raw) if isinstance(raw, Mapping) else {}
    overrides = _env_overrides(cls, env)
    values.update({field: raw_value for field, (_, raw_value) in overrides.items()})
    try:
        return cls(**values)
    except ValidationError as exc:
        # Pydantic's traceback includes input values (including credentials).
        # Only expose the sanitized setting/message summary, never its cause.
        raise ConfigError(_format_validation_error(name, exc, path, overrides)) from None


def _warn_unknown_keys(
    data: Mapping[str, Any], cls: type[BaseModel], where: str, path: Path | None
) -> None:
    """Log (and otherwise ignore) keys no field claims.

    Strict rejection would be friendlier to typos but hostile to upgrades: a
    file written for an older release mentioning a removed setting must still
    boot. The warning keeps the typo visible either way.
    """
    fields = cls.model_fields
    for key, value in data.items():
        if key not in fields:
            log.warning(
                "%s: ignoring unknown setting '%s' in section '%s' (no such "
                "setting — check the spelling against config.example.yaml)",
                path,
                key,
                where,
            )
            continue
        nested = _nested_model(fields[key].annotation)
        if nested is not None and isinstance(value, Mapping):
            _warn_unknown_keys(value, nested, f"{where}.{key}", path)


class _ConfigLoader(yaml.SafeLoader):
    """Safe YAML with duplicate explicit keys rejected, not silently overwritten.

    YAML merge keys remain supported: an explicit setting may override an
    inherited one, but two explicit spellings of the same setting are an error.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[str] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = _normalize_key(self.construct_object(key_node, deep=deep))
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, "duplicate setting key", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError:
        raise ConfigError(f"Could not read config file {path}: expected UTF-8 text") from None
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from None
    try:
        data = yaml.load(text, Loader=_ConfigLoader)
    except yaml.YAMLError as exc:
        # YAML exception strings include source lines, which may hold secrets.
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        detail = " (duplicate setting key)" if getattr(exc, "problem", "") == "duplicate setting key" else ""
        raise ConfigError(f"{path}: not valid YAML{location}{detail}") from None
    except RecursionError:
        raise ConfigError(f"{path}: excessively nested YAML is not supported") from None
    if data is None:  # empty file — every default applies
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{path}: expected a top-level mapping of sections "
            f"({', '.join(SECTIONS)}), got {type(data).__name__}"
        )
    try:
        return _normalize_keys(data)
    except RecursionError:
        raise ConfigError(f"{path}: recursive or excessively nested YAML is not supported") from None


def candidate_config_paths() -> tuple[Path, ...]:
    """Paths probed when no file is named explicitly, in priority order."""
    return tuple(CONFIG_DIR / name for name in CONFIG_FILENAMES) + tuple(
        SYSTEM_CONFIG_DIR / name for name in CONFIG_FILENAMES
    )


def resolve_config_path(env: Mapping[str, str] | None = None) -> Path | None:
    """Locate the config file, or None when the server runs on defaults.

    An explicitly requested path must exist. Falling back to defaults there
    would boot a server with no API keys and (fail-closed aside) potentially no
    authentication, which is a far worse outcome than refusing to start.
    """
    env = os.environ if env is None else env
    explicit = (env.get(CONFIG_PATH_ENV_VAR) or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"{CONFIG_PATH_ENV_VAR} points at {path}, which is not a readable "
                "file. Correct the path (relative paths resolve against the working "
                "directory), mount the file, or unset the variable to probe "
                + ", ".join(str(p) for p in candidate_config_paths())
                + "."
            )
        return path
    return next((p for p in candidate_config_paths() if p.is_file()), None)


def load_config(
    path: Path | None = None, *, env: Mapping[str, str] | None = None
) -> AppConfig:
    """Read, validate, and return the configuration.

    `path` defaults to :func:`resolve_config_path` and `env` to ``os.environ``;
    passing both is how the tests load a temp file without touching the process.
    Raises :class:`ConfigError` on unreadable/malformed YAML or an invalid value.
    """
    env = os.environ if env is None else env
    if path is None:
        path = resolve_config_path(env)
    raw = _read_yaml(path) if path is not None else {}

    for key in raw:
        if key not in SECTIONS:
            log.warning(
                "%s: ignoring unknown section '%s' (known sections: %s)",
                path,
                key,
                ", ".join(SECTIONS),
            )

    sections: dict[str, BaseSection] = {}
    for name, cls in SECTIONS.items():
        section_raw = raw.get(name)
        if section_raw is not None and not isinstance(section_raw, Mapping):
            raise ConfigError(
                f"{path}: section '{name}' must be a mapping of setting names to "
                f"values, got {type(section_raw).__name__}"
            )
        # Absent sections still get built: their environment overrides apply.
        _warn_unknown_keys(section_raw or {}, cls, name, path)
        sections[name] = _build_section(name, cls, section_raw, path, env)
    return AppConfig(**sections)


def _bootstrap(env: Mapping[str, str] | None = None) -> tuple[AppConfig, Path | None]:
    env = os.environ if env is None else env
    path = resolve_config_path(env)
    if path is None:
        log.warning(
            "No configuration file found (checked %s and %s); starting on built-in "
            "defaults plus environment overrides. Copy config.example.yaml to "
            "config.yaml to configure providers.",
            CONFIG_PATH_ENV_VAR,
            ", ".join(str(p) for p in candidate_config_paths()),
        )
    config = load_config(path, env=env)
    if path is not None:
        log.info("Loaded configuration from %s", path)
    return config, path


#: The loaded configuration, and the file it came from (None on pure defaults).
CONFIG, CONFIG_PATH = _bootstrap()

# Singletons imported by the tool modules and the server entrypoint.
tool_settings = CONFIG.tools
server_settings = CONFIG.server
web_search_settings = CONFIG.web_search
stock_settings = CONFIG.stock
wolfram_settings = CONFIG.wolfram
youtube_settings = CONFIG.youtube
geocoding_settings = CONFIG.geocoding
email_settings = CONFIG.email
