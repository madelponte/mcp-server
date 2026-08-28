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
from config import GeocodingSettings, ServerSettings, WebSearchSettings


def test_env_file_is_anchored_next_to_config_module():
    """A bare ".env" resolves against the CWD, so starting the server from
    another directory would silently lose every configured value (notably
    MCP_AUTH_TOKEN). The path must be pinned to config.py's directory."""
    assert config.ENV_FILE == Path(config.__file__).resolve().parent / ".env"


def test_every_settings_class_uses_the_anchored_env_file():
    classes = (
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


def test_searxng_request_delay_must_be_nonnegative(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_SEARXNG_REQUEST_DELAY_SECONDS", "-0.1")
    with pytest.raises(ValidationError):
        WebSearchSettings()

    monkeypatch.setenv("WEB_SEARCH_SEARXNG_REQUEST_DELAY_SECONDS", "0")
    assert WebSearchSettings().searxng_request_delay_seconds == 0


def test_searxng_can_be_disabled_for_firecrawl_only_search(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_SEARXNG_ENABLED", "false")
    assert WebSearchSettings().searxng_enabled is False


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


def test_zero_context_caps_are_rejected(monkeypatch):
    """Caps that would silently disable a feature (max=0) are startup errors;
    the documented opt-outs live on fields where 0 means "off" (e.g. TTLs)."""
    monkeypatch.setenv("STOCK_MAX_SYMBOLS", "0")
    with pytest.raises(ValidationError):
        config.StockSettings()
    monkeypatch.setenv("GEO_MAX_RADIUS_M", "0")
    with pytest.raises(ValidationError):
        GeocodingSettings()
