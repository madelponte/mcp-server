"""Tests for tools/serialize.py — centralized JSON output and debug logging."""

import json
import logging
import math

import tools.serialize as serialize
from tools.serialize import to_json, log_call, log_result, debug_enabled, redact_secrets


def _set_debug(monkeypatch, value):
    monkeypatch.setattr(serialize.server_settings, "debug", value)


def test_to_json_compact_when_debug_off(monkeypatch):
    _set_debug(monkeypatch, False)
    out = to_json({"a": 1, "b": [1, 2]})
    # Compact: no spaces after separators.
    assert out == '{"a":1,"b":[1,2]}'


def test_to_json_indented_when_debug_on(monkeypatch):
    _set_debug(monkeypatch, True)
    out = to_json({"a": 1})
    assert "\n" in out
    assert json.loads(out) == {"a": 1}


def test_to_json_preserves_non_ascii(monkeypatch):
    _set_debug(monkeypatch, False)
    out = to_json({"city": "São Paulo", "emoji": "🚀"})
    assert "São Paulo" in out
    assert "🚀" in out


def test_to_json_default_str_for_unserializable(monkeypatch):
    _set_debug(monkeypatch, False)

    class Weird:
        def __str__(self):
            return "weird-value"

    out = to_json({"x": Weird()})
    assert "weird-value" in out


def test_to_json_replaces_nonfinite_floats(monkeypatch):
    _set_debug(monkeypatch, False)
    out = to_json({"values": [math.nan, math.inf, -math.inf, 1.5]})
    assert out == '{"values":[null,null,null,1.5]}'
    assert json.loads(out) == {"values": [None, None, None, 1.5]}


def test_redact_secrets_replaces_every_configured_value():
    out = redact_secrets("token=abc proxy=http://user:pass@host", "abc", "pass")
    assert "abc" not in out
    assert "pass" not in out
    assert out == "token=REDACTED proxy=http://user:REDACTED@host"


def test_debug_enabled_reflects_setting(monkeypatch):
    _set_debug(monkeypatch, True)
    assert debug_enabled() is True
    _set_debug(monkeypatch, False)
    assert debug_enabled() is False


def test_log_result_returns_input_unchanged(monkeypatch):
    _set_debug(monkeypatch, False)
    log = logging.getLogger("test")
    payload = '{"x":1}'
    assert log_result(log, "tool", payload) is payload


def test_log_call_and_result_emit_when_debug(monkeypatch, caplog):
    _set_debug(monkeypatch, True)
    log = logging.getLogger("test.serialize")
    log.setLevel(logging.DEBUG)
    with caplog.at_level(logging.DEBUG, logger="test.serialize"):
        log_call(log, "mytool", a=1, b="x")
        log_result(log, "mytool", "result-body")
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "mytool" in messages


def test_log_call_silent_when_debug_off(monkeypatch, caplog):
    _set_debug(monkeypatch, False)
    log = logging.getLogger("test.serialize.off")
    with caplog.at_level(logging.DEBUG, logger="test.serialize.off"):
        log_call(log, "mytool", a=1)
    assert caplog.records == []
