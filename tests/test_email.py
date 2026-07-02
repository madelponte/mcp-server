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
    monkeypatch.setattr(email_mod.cfg, "max_attachments", 5)
    monkeypatch.setattr(email_mod.cfg, "max_attachment_bytes", 10_000_000)


def _capture_send(monkeypatch):
    """Patch the blocking send to record the message instead of sending it."""
    sent = {}

    def fake_send(msg, envelope_recipients):
        sent["msg"] = msg
        sent["envelope_recipients"] = envelope_recipients
        return {}

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
    assert out["recipients"]["to"] == ["a@b.com", "c@d.com"]
    assert out["attempted_recipients"] == ["a@b.com", "c@d.com"]
    assert out["accepted_recipients"] == ["a@b.com", "c@d.com"]
    assert out["refused_recipients"] == []
    assert out["dropped"] == []
    assert out["subject"] == "Report"

    msg = sent["msg"]
    assert sent["envelope_recipients"] == ["a@b.com", "c@d.com"]
    assert msg["To"] == "a@b.com, c@d.com"
    assert msg["Subject"] == "Report"
    assert msg["From"] == "bot@gmail.com"  # falls back to username
    assert msg.get_content().strip() == "Body text"


def test_cc_bcc_reply_to_and_attachment(monkeypatch, tmp_path, tool_fns):
    attachment = tmp_path / "report.txt"
    attachment.write_text("hello attachment")
    sent = _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(
            fn(
                recipients=["to@example.com"],
                cc=["cc@example.com"],
                bcc=["bcc@example.com"],
                reply_to="reply@example.com",
                subject="Report",
                body="Body text",
                attachments=[str(attachment)],
            )
        )
    )

    assert out["status"] == "sent"
    assert out["recipients"] == {
        "to": ["to@example.com"],
        "cc": ["cc@example.com"],
        "bcc": ["bcc@example.com"],
    }
    assert out["attempted_recipients"] == [
        "to@example.com",
        "cc@example.com",
        "bcc@example.com",
    ]
    assert out["attachments"] == [
        {
            "path": str(attachment),
            "filename": "report.txt",
            "content_type": "text/plain",
            "size_bytes": len("hello attachment"),
        }
    ]

    msg = sent["msg"]
    assert msg["To"] == "to@example.com"
    assert msg["Cc"] == "cc@example.com"
    assert "Bcc" not in msg
    assert msg["Reply-To"] == "reply@example.com"
    assert sent["envelope_recipients"] == [
        "to@example.com",
        "cc@example.com",
        "bcc@example.com",
    ]
    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_filename() == "report.txt"
    assert attachments[0].get_content() == "hello attachment"


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
    assert out["recipients"]["to"] == ["good@b.com"]
    assert out["accepted_recipients"] == ["good@b.com"]
    assert out["invalid_recipients"] == [{"field": "to", "address": "garbage"}]
    assert "garbage" in out["dropped"]


def test_duplicate_recipients_deduped(monkeypatch, tool_fns):
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(fn(recipients=["a@b.com", "A@B.com", " a@b.com "], subject="hi", body="yo"))
    )
    assert out["recipients"]["to"] == ["a@b.com"]


def test_duplicate_across_recipient_fields_reported(monkeypatch, tool_fns):
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(
            fn(
                recipients=["a@b.com"],
                cc=["A@B.com", "c@d.com"],
                subject="hi",
                body="yo",
            )
        )
    )
    assert out["attempted_recipients"] == ["a@b.com", "c@d.com"]
    assert out["dropped_recipients"] == [
        {"field": "cc", "address": "A@B.com", "reason": "duplicate"}
    ]


def test_recipients_clamped_to_cap(monkeypatch, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "max_recipients", 2)
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(fn(recipients=["a@b.com", "c@d.com", "e@f.com"], subject="hi", body="yo"))
    )
    assert out["recipients"]["to"] == ["a@b.com", "c@d.com"]
    assert out["attempted_recipients"] == ["a@b.com", "c@d.com"]
    assert out["dropped"] == ["e@f.com"]
    assert out["dropped_recipients"] == [
        {"field": "to", "address": "e@f.com", "reason": "over_recipient_limit"}
    ]


def test_partial_refusals_are_returned(monkeypatch, tool_fns):
    def partial(msg, envelope_recipients):
        return {"bad@example.com": (550, b"no such user")}

    monkeypatch.setattr(email_mod, "_send", partial)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(
            fn(
                recipients=["good@example.com", "bad@example.com"],
                subject="hi",
                body="yo",
            )
        )
    )
    assert out["status"] == "partial"
    assert out["accepted_recipients"] == ["good@example.com"]
    assert out["refused_recipients"] == [
        {"address": "bad@example.com", "code": 550, "response": "no such user"}
    ]


def test_missing_attachment_raises(tool_fns):
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(
            fn(
                recipients=["a@b.com"],
                subject="hi",
                body="yo",
                attachments=["/tmp/does-not-exist-for-email-test.txt"],
            )
        )
    assert "attachment path" in str(exc.value).lower()


def test_auth_error_raises_toolerror(monkeypatch, tool_fns):
    def boom(msg, envelope_recipients):
        raise smtplib.SMTPAuthenticationError(535, b"bad creds")

    monkeypatch.setattr(email_mod, "_send", boom)
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(fn(recipients=["a@b.com"], subject="hi", body="yo"))
    assert "app password" in str(exc.value).lower()


def test_connection_error_raises_toolerror(monkeypatch, tool_fns):
    def boom(msg, envelope_recipients):
        raise OSError("connection refused")

    monkeypatch.setattr(email_mod, "_send", boom)
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(fn(recipients=["a@b.com"], subject="hi", body="yo"))
    assert "smtp server" in str(exc.value).lower()
