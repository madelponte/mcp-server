"""Tests for config.py — .env anchoring and fail-fast field constraints.

These exercise *fresh* settings objects (the module singletons are already
loaded by the tool modules at import), so they prove what a real startup does:
environment variables and the anchored .env are parsed and validated before
any tool runs.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

import config
from config import EmailSettings, GeocodingSettings, ServerSettings, ToolSettings, WebSearchSettings


def test_env_file_is_anchored_next_to_config_module():
    """A bare ".env" resolves against the CWD, so starting the server from
    another directory would silently lose every configured value (notably
    MCP_AUTH_TOKEN). The path must be pinned to config.py's directory."""
    assert config.ENV_FILE == Path(config.__file__).resolve().parent / ".env"


def test_every_settings_class_uses_the_anchored_env_file():
    classes = (
        config.ToolSettings,
        config.ServerSettings,
        config.WebSearchSettings,
        config.StockSettings,
        config.WolframSettings,
        config.YouTubeSettings,
        config.GeocodingSettings,
        config.EmailSettings,
    )
    for cls in classes:
        assert cls.model_config["env_file"] == config.ENV_FILE, cls.__name__


TOOL_FLAGS = (
    "SEARCH_WEB_ENABLED",
    "FETCH_PAGE_ENABLED",
    "GET_COMPANY_DATA_ENABLED",
    "QUERY_WOLFRAM_ALPHA_ENABLED",
    "FIND_NEARBY_PLACES_ENABLED",
    "SEND_EMAIL_ENABLED",
)


def test_tool_flags_default_to_enabled(monkeypatch):
    for env_name in TOOL_FLAGS:
        monkeypatch.delenv(env_name, raising=False)

    settings = ToolSettings(_env_file=None)
    assert all(getattr(settings, env_name.lower()) for env_name in TOOL_FLAGS)


@pytest.mark.parametrize("env_name", TOOL_FLAGS)
def test_each_tool_flag_can_be_disabled(monkeypatch, env_name):
    monkeypatch.setenv(env_name, "false")
    settings = ToolSettings(_env_file=None)
    assert getattr(settings, env_name.lower()) is False


def test_negative_download_cap_fails_fast(monkeypatch):
    """A negative cap would make every fetch abort (the guards are written
    `if max_bytes and total > max_bytes`, and a negative bound is always
    exceeded), so it must be a startup error, not a runtime one."""
    monkeypatch.setenv("WEB_SEARCH_MAX_DOWNLOAD_BYTES", "-1")
    with pytest.raises(ValidationError):
        WebSearchSettings()


def test_zero_download_cap_is_still_valid(monkeypatch):
    """0 = unbounded is the documented behavior for the download cap."""
    monkeypatch.setenv("WEB_SEARCH_MAX_DOWNLOAD_BYTES", "0")
    assert WebSearchSettings().max_download_bytes == 0


def test_brave_search_count_stays_within_api_range(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_BRAVE_SEARCH_COUNT", "0")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_BRAVE_SEARCH_COUNT", "51")
    with pytest.raises(ValidationError):
        WebSearchSettings()


def test_brave_result_cap_stays_within_api_range(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MAX_NUM_RESULTS", "51")
    with pytest.raises(ValidationError):
        WebSearchSettings()


def test_brave_context_token_cap_stays_within_api_range(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MAX_CONTEXT_TOKENS", "1023")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_MAX_CONTEXT_TOKENS", "32768")
    assert WebSearchSettings().max_context_tokens == 32768


def test_brave_retry_and_spacing_settings_are_bounded(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_BRAVE_REQUEST_DELAY_SECONDS", "-0.1")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_BRAVE_REQUEST_DELAY_SECONDS", "0")
    monkeypatch.setenv("WEB_SEARCH_BRAVE_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("WEB_SEARCH_BRAVE_MAX_RETRIES", "5")
    settings = WebSearchSettings()
    assert settings.brave_request_delay_seconds == 0
    assert settings.brave_retry_backoff_seconds == 0
    assert settings.brave_max_retries == 5

    monkeypatch.setenv("WEB_SEARCH_BRAVE_MAX_RETRIES", "6")
    with pytest.raises(ValidationError):
        WebSearchSettings()


def test_reddit_request_delay_must_be_nonnegative(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_REDDIT_REQUEST_DELAY_SECONDS", "-0.1")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_REDDIT_REQUEST_DELAY_SECONDS", "0")
    assert WebSearchSettings().reddit_request_delay_seconds == 0


def test_reddit_rate_limit_retry_must_be_nonnegative(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_REDDIT_RATE_LIMIT_RETRY_SECONDS", "-0.1")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_REDDIT_RATE_LIMIT_RETRY_SECONDS", "0")
    assert WebSearchSettings().reddit_rate_limit_retry_seconds == 0


def test_image_description_cap_must_be_nonnegative(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MAX_IMAGE_DESCRIPTIONS", "-1")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_MAX_IMAGE_DESCRIPTIONS", "0")
    assert WebSearchSettings().max_image_descriptions == 0


def test_query_context_line_cap_must_be_nonnegative(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MAX_QUERY_CONTEXT_LINES", "-1")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_MAX_QUERY_CONTEXT_LINES", "0")
    assert WebSearchSettings().max_query_context_lines == 0


def test_classifier_confidence_must_be_in_unit_interval(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_CLASSIFIER_MIN_CONFIDENCE", "1.5")
    with pytest.raises(ValidationError):
        WebSearchSettings()


def test_port_must_be_a_valid_tcp_port(monkeypatch):
    monkeypatch.setenv("MCP_PORT", "0")
    with pytest.raises(ValidationError):
        ServerSettings()
    monkeypatch.setenv("MCP_PORT", "65536")
    with pytest.raises(ValidationError):
        ServerSettings()
    monkeypatch.setenv("MCP_PORT", "8443")
    assert ServerSettings().port == 8443


def test_tool_catalog_cache_settings(monkeypatch):
    monkeypatch.setenv("MCP_TOOL_CATALOG_CACHE_TTL_SECONDS", "0")
    monkeypatch.setenv("MCP_TOOL_CATALOG_CACHE_SCOPE", "private")
    settings = ServerSettings(_env_file=None)
    assert settings.tool_catalog_cache_ttl_seconds == 0
    assert settings.tool_catalog_cache_scope == "private"

    monkeypatch.setenv("MCP_TOOL_CATALOG_CACHE_TTL_SECONDS", "-1")
    with pytest.raises(ValidationError):
        ServerSettings(_env_file=None)

    monkeypatch.setenv("MCP_TOOL_CATALOG_CACHE_TTL_SECONDS", "300")
    monkeypatch.setenv("MCP_TOOL_CATALOG_CACHE_SCOPE", "shared")
    with pytest.raises(ValidationError):
        ServerSettings(_env_file=None)


def test_allow_unauthenticated_defaults_to_false(monkeypatch):
    monkeypatch.delenv("MCP_ALLOW_UNAUTHENTICATED", raising=False)
    assert ServerSettings(_env_file=None).allow_unauthenticated is False


def test_concurrent_fetch_caps_reject_zero(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_MAX_CONCURRENT_FLARESOLVERR", "0")
    with pytest.raises(ValidationError):
        WebSearchSettings()
    monkeypatch.setenv("WEB_SEARCH_MAX_CONCURRENT_FLARESOLVERR", "2")
    monkeypatch.setenv("WEB_SEARCH_MAX_CONCURRENT_TIKA", "1")
    settings = WebSearchSettings()
    assert settings.max_concurrent_flaresolverr == 2
    assert settings.max_concurrent_tika == 1


def test_email_allowlist_and_attachment_root_default_empty(monkeypatch):
    monkeypatch.delenv("EMAIL_ALLOWED_RECIPIENTS", raising=False)
    monkeypatch.delenv("EMAIL_ATTACHMENT_ROOT", raising=False)
    settings = EmailSettings(_env_file=None)
    assert settings.allowed_recipients == ""
    assert settings.attachment_root == ""


def test_zero_context_caps_are_rejected(monkeypatch):
    """Caps that would silently disable a feature (max=0) are startup errors;
    the documented opt-outs live on fields where 0 means "off" (e.g. TTLs)."""
    monkeypatch.setenv("STOCK_MAX_SYMBOLS", "0")
    with pytest.raises(ValidationError):
        config.StockSettings()
    monkeypatch.setenv("GEO_MAX_RADIUS_M", "0")
    with pytest.raises(ValidationError):
        GeocodingSettings()
