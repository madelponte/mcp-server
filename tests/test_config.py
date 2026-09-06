"""Tests for config.py — YAML file loading, env overrides, and fail-fast constraints.

The suite never reads the deployment's own ``config.yaml``: every load passes an
explicit ``path`` (usually a tmp file) and an explicit ``env`` mapping, so a live
file with real values — or a developer's local one — cannot change what these
assert. The settings *classes* are constructed directly to prove a constraint is
enforced, and :func:`config.load_config` is exercised to prove the file/variable
plumbing reaches that constraint at startup.
"""

from pathlib import Path
import traceback

import pytest
from pydantic import ValidationError

import config
from config import (
    EmailSettings,
    GeocodingSettings,
    ServerSettings,
    ToolSettings,
    WebSearchSettings,
    WolframSettings,
    load_config,
)

NO_ENV: dict[str, str] = {}


def write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def test_config_file_is_anchored_next_to_the_module():
    """A bare "config.yaml" resolved against the CWD would silently drop every
    configured value when the server is started from another directory (the
    bug the old .env anchoring guarded against). The probe list must be pinned to
    config.py's directory plus the system config dir."""
    here = Path(config.__file__).resolve().parent
    candidates = config.candidate_config_paths()
    assert candidates[0] == here / "config.yaml"
    assert here / "config.yml" in candidates
    assert config.SYSTEM_CONFIG_DIR / "config.yaml" in candidates


def test_no_file_anywhere_yields_defaults_not_an_error(monkeypatch, tmp_path):
    """Every field has a default; the HTTP transport is separately fail-closed.

    The candidate list has to be stubbed out: this repo may legitimately contain
    a real config.yaml, which is exactly what the probe finds.
    """
    monkeypatch.setattr(
        config, "candidate_config_paths", lambda: (tmp_path / "absent.yaml",)
    )
    assert config.resolve_config_path(NO_ENV) is None
    loaded = load_config(None, env=NO_ENV)
    assert loaded.server.port == 8000
    assert loaded.web_search.max_page_chars >= 1
    assert loaded.server.auth_configured is False


@pytest.mark.parametrize(
    "env",
    [
        {"MCP_PORT": "9123", "MCP_TRANSPORT": "stdio", "MCP_AUTH_TOKEN": "test-token"},
        {"SEND_EMAIL_ENABLED": "false", "WEB_SEARCH_BRAVE_API_KEY": "test-key"},
    ],
)
def test_bootstrap_without_file_applies_environment(monkeypatch, tmp_path, env):
    monkeypatch.setattr(config, "candidate_config_paths", lambda: (tmp_path / "absent.yaml",))
    loaded, path = config._bootstrap(env=env)
    assert path is None
    assert loaded == load_config(env=env)
    if "MCP_PORT" in env:
        assert loaded.server.port == 9123
        assert loaded.server.transport == "stdio"
        assert loaded.server.auth_configured
    else:
        assert loaded.tools.send_email_enabled is False
        assert loaded.web_search.brave_api_key == "test-key"


def test_bootstrap_without_file_validates_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "candidate_config_paths", lambda: (tmp_path / "absent.yaml",))
    with pytest.raises(config.ConfigError, match="MCP_PORT"):
        config._bootstrap(env={"MCP_PORT": "70000"})


def test_blank_config_path_variable_counts_as_unset(monkeypatch, tmp_path):
    """An empty MCP_CONFIG_FILE (a half-filled compose env) probes the default
    locations instead of failing on an empty filename."""
    monkeypatch.setattr(
        config, "candidate_config_paths", lambda: (tmp_path / "absent.yaml",)
    )
    assert config.resolve_config_path({"MCP_CONFIG_FILE": "   "}) is None


def test_explicit_path_must_exist():
    """A typo in an explicitly requested path must not silently boot on defaults."""
    with pytest.raises(config.ConfigError, match="MCP_CONFIG_FILE"):
        config.resolve_config_path({"MCP_CONFIG_FILE": "/definitely/not/here.yaml"})


def test_explicit_env_path_is_used(tmp_path):
    path = write(tmp_path, "server:\n  port: 8642\n", "elsewhere.yml")
    resolved = config.resolve_config_path({"MCP_CONFIG_FILE": str(path)})
    assert resolved == path
    assert load_config(resolved, env=NO_ENV).server.port == 8642


def test_first_existing_candidate_wins(monkeypatch, tmp_path):
    """The module-directory file takes priority over the system config dir, so a
    compose mount cannot be shadowed by a leftover host file."""
    monkeypatch.setattr(
        config,
        "candidate_config_paths",
        lambda: (tmp_path / "a.yaml", tmp_path / "b.yaml"),
    )
    write(tmp_path, "server:\n  port: 8001\n", "b.yaml")
    assert config.resolve_config_path(NO_ENV) == tmp_path / "b.yaml"
    write(tmp_path, "server:\n  port: 8002\n", "a.yaml")
    assert config.resolve_config_path(NO_ENV) == tmp_path / "a.yaml"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_empty_file_is_all_defaults(tmp_path):
    assert load_config(write(tmp_path, ""), env=NO_ENV).server.port == 8000


def test_sections_are_optional_and_partial(tmp_path):
    path = write(tmp_path, "wolfram:\n  app_id: ABC123\n")
    loaded = load_config(path, env=NO_ENV)
    assert loaded.wolfram.app_id == "ABC123"
    assert loaded.wolfram.default_units == WolframSettings().default_units
    assert loaded.email.smtp_port == EmailSettings().smtp_port


def test_missing_section_still_applies_env_overrides(tmp_path):
    """An absent YAML section is not a reason to ignore its variables."""
    loaded = load_config(write(tmp_path, ""), env={"STOCK_MAX_SYMBOLS": "4"})
    assert loaded.stock.max_symbols == 4


def test_key_names_are_case_and_hyphen_insensitive(tmp_path):
    """An .env migrated by hand (UPPER case, hyphens) still loads."""
    path = write(
        tmp_path,
        "WEB_SEARCH:\n  MAX-PAGE-CHARS: 1234\ngeocoding:\n  MAX_RADIUS_M: 5000\n",
    )
    loaded = load_config(path, env=NO_ENV)
    assert loaded.web_search.max_page_chars == 1234
    assert loaded.geocoding.max_radius_m == 5000


def test_comma_separated_settings_accept_a_yaml_list(tmp_path):
    path = write(
        tmp_path,
        "web_search:\n"
        "  ssrf_allowlist: [localhost, 10.0.0.0/8]\n"
        "geocoding:\n"
        "  overpass_fallback_urls:\n"
        "    - https://one.example/api\n"
        "    - https://two.example/api\n"
        "youtube:\n"
        "  default_languages: [en, de]\n"
        "email:\n"
        "  allowed_recipients: [you@example.com, corp.com]\n",
    )
    loaded = load_config(path, env=NO_ENV)
    assert loaded.web_search.ssrf_allowlist == "localhost,10.0.0.0/8"
    assert (
        loaded.geocoding.overpass_fallback_urls
        == "https://one.example/api,https://two.example/api"
    )
    assert loaded.youtube.default_languages == "en,de"
    assert loaded.email.allowed_recipients == "you@example.com,corp.com"


def test_unknown_keys_warn_but_do_not_block_startup(tmp_path, caplog):
    """A file written for another release must still boot; the typo is logged."""
    path = write(
        tmp_path,
        "web_search:\n  max_snippet_chars: 400\n  max_page_chars: 999\nbogus:\n  x: 1\n",
    )
    with caplog.at_level("WARNING", logger="config"):
        loaded = load_config(path, env=NO_ENV)
    assert loaded.web_search.max_page_chars == 999  # the known key still applied
    assert "max_snippet_chars" in caplog.text
    assert "unknown section 'bogus'" in caplog.text


def test_malformed_yaml_is_a_startup_error(tmp_path):
    path = write(tmp_path, "server: [oops\n  x: ]]]\n")
    with pytest.raises(config.ConfigError, match="not valid YAML"):
        load_config(path, env=NO_ENV)


@pytest.mark.parametrize(
    "body",
    [
        "server: {port: 8001}\nserver: {port: 8002}\n",
        "server: {port: 8001, port: 8002}\n",
        "server: {port: 8001, PORT: 8002}\n",
        "web_search: {max-page-chars: 100, max_page_chars: 200}\n",
        "server:\n  auth_tokens:\n    - {name: a, token: one, TOKEN: two}\n",
    ],
)
def test_duplicate_yaml_keys_fail_instead_of_overwriting(tmp_path, body):
    with pytest.raises(config.ConfigError, match="duplicate setting key"):
        load_config(write(tmp_path, body), env=NO_ENV)


def test_yaml_merge_defaults_can_be_explicitly_overridden(tmp_path):
    path = write(
        tmp_path,
        "stock: &defaults {cache_ttl_seconds: 60}\n"
        "wolfram: {<<: *defaults, cache_ttl_seconds: 120}\n",
    )
    loaded = load_config(path, env=NO_ENV)
    assert loaded.stock.cache_ttl_seconds == 60
    assert loaded.wolfram.cache_ttl_seconds == 120


def test_recursive_yaml_is_a_config_error(tmp_path):
    with pytest.raises(config.ConfigError, match="recursive"):
        load_config(write(tmp_path, "server: &loop {other: *loop}\n"), env=NO_ENV)


def test_non_utf8_yaml_is_a_config_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes(b"email:\n  password: secret-\xff\n")
    with pytest.raises(config.ConfigError, match="UTF-8") as excinfo:
        load_config(path, env=NO_ENV)
    assert "secret-" not in "".join(traceback.format_exception(excinfo.value))


@pytest.mark.parametrize(
    "body",
    [
        "youtube: {default_languages: [en, no]}",
        "geocoding: {overpass_fallback_urls: [null]}",
        "web_search: {ssrf_allowlist: [{host: localhost}]}",
        "email: {allowed_recipients: [123]}",
    ],
)
def test_list_settings_reject_non_string_items(tmp_path, body):
    with pytest.raises(config.ConfigError, match="list entries must be strings"):
        load_config(write(tmp_path, body), env=NO_ENV)


def test_empty_yaml_lists_clear_comma_separated_settings(tmp_path):
    loaded = load_config(
        write(tmp_path, "geocoding: {overpass_fallback_urls: []}"), env=NO_ENV
    )
    assert loaded.geocoding.overpass_fallback_urls == ""


def test_non_mapping_top_level_is_a_startup_error(tmp_path):
    with pytest.raises(config.ConfigError, match="mapping of sections"):
        load_config(write(tmp_path, "- just\n- a\n- list\n"), env=NO_ENV)


def test_non_mapping_section_is_a_startup_error(tmp_path):
    with pytest.raises(config.ConfigError, match="section 'server' must be a mapping"):
        load_config(write(tmp_path, "server: 5\n"), env=NO_ENV)


def test_unreadable_explicit_path_is_a_startup_error(tmp_path):
    with pytest.raises(config.ConfigError, match="Could not read config file"):
        load_config(tmp_path / "does-not-exist.yaml", env=NO_ENV)


# ---------------------------------------------------------------------------
# Environment overrides (the compatibility layer on top of the file)
# ---------------------------------------------------------------------------


def test_env_var_overrides_the_file_value(tmp_path):
    path = write(tmp_path, "web_search:\n  max_page_chars: 1000\nserver:\n  port: 8000\n")
    loaded = load_config(
        path, env={"WEB_SEARCH_MAX_PAGE_CHARS": "4321", "MCP_PORT": "9443"}
    )
    assert loaded.web_search.max_page_chars == 4321
    assert loaded.server.port == 9443


def test_blank_env_var_is_treated_as_unset(tmp_path):
    """A stray FOO= in a shell, image layer, or half-migrated compose file must
    not wipe the configured YAML value."""
    path = write(tmp_path, "geocoding:\n  max_radius_m: 7000\n")
    loaded = load_config(
        path, env={"GEO_MAX_RADIUS_M": "", "GEO_DEFAULT_RADIUS_M": "   "}
    )
    assert loaded.geocoding.max_radius_m == 7000


def test_tool_flags_are_env_overridable(tmp_path):
    loaded = load_config(write(tmp_path, ""), env={"SEND_EMAIL_ENABLED": "false"})
    assert loaded.tools.send_email_enabled is False


def test_nested_list_settings_are_yaml_only(tmp_path):
    """There is no sane variable syntax for a list of named tokens, so a variable
    named after the field must not be interpreted as one."""
    loaded = load_config(
        write(tmp_path, ""), env={"MCP_AUTH_TOKENS": "not-a-token-list"}
    )
    assert loaded.server.auth_tokens == []


# ---------------------------------------------------------------------------
# Bearer tokens
# ---------------------------------------------------------------------------


def test_multiple_named_tokens_are_all_accepted(tmp_path):
    path = write(
        tmp_path,
        "server:\n"
        "  auth_tokens:\n"
        "    - name: open-webui\n"
        '      token: "token-one"\n'
        "    - name: claude-desktop\n"
        '      token: "token-two"\n',
    )
    entries = load_config(path, env=NO_ENV).server.token_entries()
    assert [(e.name, e.token) for e in entries] == [
        ("open-webui", "token-one"),
        ("claude-desktop", "token-two"),
    ]


def test_duplicate_token_names_are_a_startup_error(tmp_path):
    """Duplicate labels would make the audit log unable to attribute a request."""
    path = write(
        tmp_path,
        "server:\n"
        "  auth_tokens:\n"
        "    - {name: same, token: aaa}\n"
        "    - {name: same, token: bbb}\n",
    )
    with pytest.raises(config.ConfigError, match="duplicate auth token name"):
        load_config(path, env=NO_ENV)


def test_unknown_key_in_a_token_entry_is_an_error(tmp_path):
    """Unlike settings sections, a misspelled credential field must fail loudly."""
    path = write(
        tmp_path, "server:\n  auth_tokens:\n    - {name: webui, tokne: aaa}\n"
    )
    with pytest.raises(config.ConfigError, match="auth_tokens.0"):
        load_config(path, env=NO_ENV)


def test_blank_or_non_ascii_token_is_an_error(tmp_path):
    for body in (
        'server:\n  auth_tokens:\n    - {name: webui, token: ""}\n',
        'server:\n  auth_tokens:\n    - {name: webui, token: "pässword"}\n',
        'server:\n  auth_tokens:\n    - {name: "  ", token: "aaa"}\n',
    ):
        with pytest.raises(config.ConfigError):
            load_config(write(tmp_path, body), env=NO_ENV)


def test_duplicate_named_secrets_collapse_before_server_wiring(tmp_path):
    path = write(
        tmp_path,
        "server:\n  auth_tokens:\n"
        "    - {name: first, token: shared}\n"
        "    - {name: second, token: shared}\n",
    )
    entries = load_config(path, env=NO_ENV).server.token_entries()
    assert [(e.name, e.token) for e in entries] == [("first", "shared")]


def test_duplicate_names_fail_even_when_tokens_match(tmp_path):
    path = write(
        tmp_path,
        "server:\n  auth_tokens:\n"
        "    - {name: same, token: shared}\n"
        "    - {name: same, token: shared}\n",
    )
    with pytest.raises(config.ConfigError, match="duplicate auth token name"):
        load_config(path, env=NO_ENV)


def test_legacy_token_name_conflict_fails(tmp_path):
    path = write(tmp_path, "server: {auth_tokens: [{name: default, token: named}]}")
    with pytest.raises(config.ConfigError, match="duplicate auth token name"):
        load_config(path, env={"MCP_AUTH_TOKEN": "legacy"})


def test_invalid_legacy_token_names_responsible_variable(tmp_path):
    with pytest.raises(config.ConfigError) as excinfo:
        load_config(write(tmp_path, ""), env={"MCP_AUTH_TOKEN": "secret-é"})
    assert "MCP_AUTH_TOKEN" in str(excinfo.value)
    assert "server.auth_token" in str(excinfo.value)


def test_legacy_single_token_still_works(tmp_path):
    """MCP_AUTH_TOKEN (or server.auth_token) becomes one entry named 'default'."""
    from_env = load_config(write(tmp_path, ""), env={"MCP_AUTH_TOKEN": "legacy"})
    assert [(e.name, e.token) for e in from_env.server.token_entries()] == [
        ("default", "legacy")
    ]
    from_file = load_config(write(tmp_path, 'server:\n  auth_token: "legacy"\n'), env=NO_ENV)
    assert from_file.server.token_entries()[0].token == "legacy"
    assert from_file.server.auth_configured is True


def test_legacy_token_matching_a_named_entry_is_not_duplicated(tmp_path):
    """Setting both the legacy variable and the same secret in the file must not
    present the same credential under two names."""
    path = write(
        tmp_path, 'server:\n  auth_tokens:\n    - {name: webui, token: "shared"}\n'
    )
    entries = load_config(path, env={"MCP_AUTH_TOKEN": "shared"}).server.token_entries()
    assert [(e.name, e.token) for e in entries] == [("webui", "shared")]


# ---------------------------------------------------------------------------
# Validation messages and field constraints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        'server: {auth_tokens: [{name: client, token: "SECRET_MARKER_é"}]}',
        'email: {password: [SECRET_MARKER]}',
        'email:\n  password: "SECRET_MARKER\n',
        'server: {auth_token: "SECRET_MARKER_é"}',
    ],
)
def test_configuration_error_tracebacks_do_not_expose_secrets(tmp_path, body):
    with pytest.raises(config.ConfigError) as excinfo:
        load_config(write(tmp_path, body), env=NO_ENV)
    assert "SECRET_MARKER" not in "".join(traceback.format_exception(excinfo.value))


def test_yaml_syntax_error_keeps_line_location(tmp_path):
    with pytest.raises(config.ConfigError, match=r"line \d+, column \d+"):
        load_config(write(tmp_path, 'email:\n  password: "unfinished\n'), env=NO_ENV)


def test_invalid_value_error_names_the_file_and_setting(tmp_path):
    path = write(tmp_path, "server:\n  port: 70000\n")
    with pytest.raises(config.ConfigError) as excinfo:
        load_config(path, env=NO_ENV)
    message = str(excinfo.value)
    assert str(path) in message
    assert "server.port" in message


def test_yaml_boolean_in_a_text_setting_explains_quoting(tmp_path):
    """`brave_safesearch: off` is a YAML boolean, not the word "off". The error
    must say to quote it rather than silently coercing False to some string the
    provider would then be sent."""
    path = write(tmp_path, 'web_search:\n  brave_safesearch: off\n')
    with pytest.raises(config.ConfigError) as excinfo:
        load_config(path, env=NO_ENV)
    assert "Quote the word" in str(excinfo.value)


def test_invalid_value_error_names_the_responsible_variable(tmp_path):
    """Env beats YAML, so an error must say which source supplied the value —
    otherwise a legacy variable looks like a bug in the file."""
    path = write(tmp_path, "server:\n  port: 8000\n")
    with pytest.raises(config.ConfigError) as excinfo:
        load_config(path, env={"MCP_PORT": "70000"})
    assert "MCP_PORT" in str(excinfo.value)


def test_negative_download_cap_fails_fast(monkeypatch):
    """A negative cap would make every fetch abort (the guards are written
    `if max_bytes and total > max_bytes`, and a negative bound is always
    exceeded), so it must be a startup error, not a runtime one."""
    with pytest.raises(ValidationError):
        WebSearchSettings(max_download_bytes=-1)


def test_zero_download_cap_is_still_valid():
    """0 = unbounded is the documented behavior for the download cap."""
    assert WebSearchSettings(max_download_bytes=0).max_download_bytes == 0


def test_brave_search_count_stays_within_api_range():
    with pytest.raises(ValidationError):
        WebSearchSettings(brave_search_count=0)
    with pytest.raises(ValidationError):
        WebSearchSettings(brave_search_count=51)


def test_brave_result_cap_stays_within_api_range():
    with pytest.raises(ValidationError):
        WebSearchSettings(max_num_results=51)


def test_brave_context_token_cap_stays_within_api_range():
    with pytest.raises(ValidationError):
        WebSearchSettings(max_context_tokens=1023)
    assert WebSearchSettings(max_context_tokens=32768).max_context_tokens == 32768


def test_brave_retry_and_spacing_settings_are_bounded():
    with pytest.raises(ValidationError):
        WebSearchSettings(brave_request_delay_seconds=-0.1)
    settings = WebSearchSettings(
        brave_request_delay_seconds=0,
        brave_retry_backoff_seconds=0,
        brave_max_retries=5,
    )
    assert settings.brave_request_delay_seconds == 0
    assert settings.brave_retry_backoff_seconds == 0
    assert settings.brave_max_retries == 5
    with pytest.raises(ValidationError):
        WebSearchSettings(brave_max_retries=6)


def test_reddit_request_delay_must_be_nonnegative():
    with pytest.raises(ValidationError):
        WebSearchSettings(reddit_request_delay_seconds=-0.1)
    assert WebSearchSettings(reddit_request_delay_seconds=0).reddit_request_delay_seconds == 0


def test_reddit_rate_limit_retry_must_be_nonnegative():
    with pytest.raises(ValidationError):
        WebSearchSettings(reddit_rate_limit_retry_seconds=-0.1)
    assert (
        WebSearchSettings(reddit_rate_limit_retry_seconds=0).reddit_rate_limit_retry_seconds
        == 0
    )


def test_image_description_cap_must_be_nonnegative():
    with pytest.raises(ValidationError):
        WebSearchSettings(max_image_descriptions=-1)
    assert WebSearchSettings(max_image_descriptions=0).max_image_descriptions == 0


def test_query_context_line_cap_must_be_nonnegative():
    with pytest.raises(ValidationError):
        WebSearchSettings(max_query_context_lines=-1)
    assert WebSearchSettings(max_query_context_lines=0).max_query_context_lines == 0


def test_classifier_confidence_must_be_in_unit_interval():
    with pytest.raises(ValidationError):
        WebSearchSettings(classifier_min_confidence=1.5)


def test_port_must_be_a_valid_tcp_port():
    with pytest.raises(ValidationError):
        ServerSettings(port=0)
    with pytest.raises(ValidationError):
        ServerSettings(port=65536)
    assert ServerSettings(port=8443).port == 8443


def test_tool_catalog_cache_settings():
    settings = ServerSettings(tool_catalog_cache_ttl_seconds=0, tool_catalog_cache_scope="private")
    assert settings.tool_catalog_cache_ttl_seconds == 0
    assert settings.tool_catalog_cache_scope == "private"
    with pytest.raises(ValidationError):
        ServerSettings(tool_catalog_cache_ttl_seconds=-1)
    with pytest.raises(ValidationError):
        ServerSettings(tool_catalog_cache_scope="shared")


def test_allow_unauthenticated_defaults_to_false():
    assert ServerSettings().allow_unauthenticated is False


def test_concurrent_fetch_caps_reject_zero():
    with pytest.raises(ValidationError):
        WebSearchSettings(max_concurrent_flaresolverr=0)
    settings = WebSearchSettings(max_concurrent_flaresolverr=2, max_concurrent_tika=1)
    assert settings.max_concurrent_flaresolverr == 2
    assert settings.max_concurrent_tika == 1


def test_email_allowlist_and_attachment_root_default_empty():
    settings = EmailSettings()
    assert settings.allowed_recipients == ""
    assert settings.attachment_root == ""


def test_zero_context_caps_are_rejected():
    """Caps that would silently disable a feature (max=0) are startup errors;
    the documented opt-outs live on fields where 0 means "off" (e.g. TTLs)."""
    with pytest.raises(ValidationError):
        config.StockSettings(max_symbols=0)
    with pytest.raises(ValidationError):
        GeocodingSettings(max_radius_m=0)


def test_tool_flags_default_to_enabled():
    settings = ToolSettings()
    assert all(
        getattr(settings, name)
        for name in (
            "search_web_enabled",
            "fetch_page_enabled",
            "get_company_data_enabled",
            "query_wolfram_alpha_enabled",
            "find_nearby_places_enabled",
            "send_email_enabled",
        )
    )


# ---------------------------------------------------------------------------
# Cross-checks that keep the file format and the docs honest
# ---------------------------------------------------------------------------


def test_every_section_is_registered_and_prefix_unique():
    prefixes = [cls._env_prefix for cls in config.SECTIONS.values()]
    assert set(config.SECTIONS) == {
        "tools",
        "server",
        "web_search",
        "stock",
        "wolfram",
        "youtube",
        "geocoding",
        "email",
    }
    assert len(set(prefixes)) == len(prefixes)
    assert all(issubclass(cls, config.BaseSection) for cls in config.SECTIONS.values())


def test_module_singletons_match_the_loaded_config():
    """Tools import the singletons; they must be the sections of CONFIG."""
    assert config.tool_settings is config.CONFIG.tools
    assert config.server_settings is config.CONFIG.server
    assert config.web_search_settings is config.CONFIG.web_search
    assert config.stock_settings is config.CONFIG.stock
    assert config.wolfram_settings is config.CONFIG.wolfram
    assert config.youtube_settings is config.CONFIG.youtube
    assert config.geocoding_settings is config.CONFIG.geocoding
    assert config.email_settings is config.CONFIG.email


# Fields the example cannot show uncommented: a blank bearer token is rejected at
# startup, so its entries live commented out in the file.
EXAMPLE_COMMENTED_OUT = {"server.auth_tokens"}


def test_example_config_documents_every_setting(tmp_path, caplog):
    """Every field of every section must appear in config.example.yaml, so the
    example stays the complete list the old .env.example was."""
    example = Path(config.__file__).resolve().parent / "config.example.yaml"
    staged = tmp_path / "config.yaml"
    staged.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    loaded = load_config(staged, env=NO_ENV)
    raw = config._read_yaml(staged)
    missing = [
        f"{section}.{name}"
        for section, cls in config.SECTIONS.items()
        for name in cls.model_fields
        if f"{section}.{name}" not in EXAMPLE_COMMENTED_OUT
        and name not in (raw.get(section) or {})
    ]
    assert missing == []
    assert loaded.tools.search_web_enabled is True


def test_example_config_file_is_loadable_and_fully_current(tmp_path, caplog):
    """config.example.yaml is the documented shape of the file: if it stops
    parsing, or mentions a key no longer in the models, the docs are wrong."""
    example = Path(config.__file__).resolve().parent / "config.example.yaml"
    assert example.is_file(), "config.example.yaml is missing"
    # Copy it into tmp_path so a value the deployment edited in place cannot
    # change the assertion, only the shape matters here.
    staged = tmp_path / "config.yaml"
    staged.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    with caplog.at_level("WARNING", logger="config"):
        loaded = load_config(staged, env=NO_ENV)
    assert "ignoring unknown" not in caplog.text
    assert "unknown section" not in caplog.text
    assert loaded.server.transport in ("streamable-http", "sse", "stdio")
