"""Tests for tools/email.py — the send_email tool.

Offline: the blocking SMTP send (`email._send`) is monkeypatched so no socket
is ever opened. Credentials are pinned on the live cfg per-test.
"""

import json
import smtplib

import pytest
from fastmcp.exceptions import ToolError

import tools.email as email_mod
from conftest import run


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    """Give the tool valid-looking credentials by default."""
    monkeypatch.setattr(email_mod.cfg, "username", "bot@gmail.com")
    monkeypatch.setattr(email_mod.cfg, "password", "app-password")
    monkeypatch.setattr(email_mod.cfg, "from_address", "")
    monkeypatch.setattr(email_mod.cfg, "from_name", "")


def _capture_send(monkeypatch):
    """Patch the blocking send to record the message instead of sending it."""
    sent = {}

    def fake_send(msg):
        sent["msg"] = msg

    monkeypatch.setattr(email_mod, "_send", fake_send)
    return sent


def test_not_configured_raises(monkeypatch, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "password", "")
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(fn(recipients=["a@b.com"], subject="hi", body="yo"))
    assert "not configured" in str(exc.value).lower()


def test_empty_recipients_raises(tool_fns):
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError):
        run(fn(recipients=[], subject="hi", body="yo"))


def test_empty_subject_raises(tool_fns):
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError):
        run(fn(recipients=["a@b.com"], subject="  ", body="yo"))


def test_empty_body_raises(tool_fns):
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError):
        run(fn(recipients=["a@b.com"], subject="hi", body=""))


def test_all_invalid_recipients_raises(monkeypatch, tool_fns):
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(fn(recipients=["not-an-email", "also bad"], subject="hi", body="yo"))
    assert "valid recipient" in str(exc.value).lower()


def test_happy_path_sends_and_returns_json(monkeypatch, tool_fns):
    sent = _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(fn(recipients=["a@b.com", "c@d.com"], subject="Report", body="Body text"))
    )
    assert out["status"] == "sent"
    assert out["recipients"] == ["a@b.com", "c@d.com"]
    assert out["dropped"] == []
    assert out["subject"] == "Report"

    msg = sent["msg"]
    assert msg["To"] == "a@b.com, c@d.com"
    assert msg["Subject"] == "Report"
    assert msg["From"] == "bot@gmail.com"  # falls back to username
    assert msg.get_content().strip() == "Body text"


def test_from_name_and_address_used(monkeypatch, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "from_address", "noreply@corp.com")
    monkeypatch.setattr(email_mod.cfg, "from_name", "My Bot")
    sent = _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    run(fn(recipients=["a@b.com"], subject="hi", body="yo"))
    assert sent["msg"]["From"] == "My Bot <noreply@corp.com>"


def test_invalid_addresses_dropped_but_valid_sent(monkeypatch, tool_fns):
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(fn(recipients=["good@b.com", "garbage"], subject="hi", body="yo"))
    )
    assert out["recipients"] == ["good@b.com"]
    assert "garbage" in out["dropped"]


def test_duplicate_recipients_deduped(monkeypatch, tool_fns):
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(fn(recipients=["a@b.com", "A@B.com", " a@b.com "], subject="hi", body="yo"))
    )
    assert out["recipients"] == ["a@b.com"]


def test_recipients_clamped_to_cap(monkeypatch, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "max_recipients", 2)
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(fn(recipients=["a@b.com", "c@d.com", "e@f.com"], subject="hi", body="yo"))
    )
    assert out["recipients"] == ["a@b.com", "c@d.com"]
    assert out["dropped"] == ["e@f.com"]


def test_auth_error_raises_toolerror(monkeypatch, tool_fns):
    def boom(msg):
        raise smtplib.SMTPAuthenticationError(535, b"bad creds")

    monkeypatch.setattr(email_mod, "_send", boom)
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(fn(recipients=["a@b.com"], subject="hi", body="yo"))
    assert "app password" in str(exc.value).lower()


def test_connection_error_raises_toolerror(monkeypatch, tool_fns):
    def boom(msg):
        raise OSError("connection refused")

    monkeypatch.setattr(email_mod, "_send", boom)
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(fn(recipients=["a@b.com"], subject="hi", body="yo"))
    assert "smtp server" in str(exc.value).lower()
