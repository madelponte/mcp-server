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
    async def run_sync(func, *args, **kwargs):
        return func(*args, **kwargs)

    # These tests patch the SMTP boundary and exercise validation/formatting.
    # Run worker targets inline because this Python 3.13 environment can hang
    # while cleaning up AnyIO worker threads during asyncio.run().
    monkeypatch.setattr(email_mod.anyio.to_thread, "run_sync", run_sync)
    monkeypatch.setattr(email_mod.cfg, "username", "bot@gmail.com")
    monkeypatch.setattr(email_mod.cfg, "password", "app-password")
    monkeypatch.setattr(email_mod.cfg, "from_address", "")
    monkeypatch.setattr(email_mod.cfg, "from_name", "")
    monkeypatch.setattr(email_mod.cfg, "max_attachments", 5)
    monkeypatch.setattr(email_mod.cfg, "max_attachment_bytes", 10_000_000)
    monkeypatch.setattr(email_mod.cfg, "allowed_recipients", "")
    monkeypatch.setattr(email_mod.cfg, "attachment_root", "")


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
    monkeypatch.setattr(email_mod.cfg, "attachment_root", str(tmp_path))
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
            "filename": "report.txt",
            "content_type": "text/plain",
            "size_bytes": len("hello attachment"),
        }
    ]
    assert "path" not in out["attachments"][0]

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


def test_attachments_disabled_without_root(tool_fns):
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
    assert "disabled" in str(exc.value).lower()


def test_missing_attachment_inside_root_raises(monkeypatch, tmp_path, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "attachment_root", str(tmp_path))
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(
            fn(
                recipients=["a@b.com"],
                subject="hi",
                body="yo",
                attachments=["missing-for-email-test.txt"],
            )
        )
    assert "readable file" in str(exc.value).lower()
    assert "missing-for-email-test.txt" not in str(exc.value)


def test_attachment_outside_root_is_rejected(monkeypatch, tmp_path, tool_fns):
    jail = tmp_path / "jail"
    jail.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak")
    monkeypatch.setattr(email_mod.cfg, "attachment_root", str(jail))
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(
            fn(
                recipients=["a@b.com"],
                subject="hi",
                body="yo",
                attachments=[str(secret)],
            )
        )
    assert "outside" in str(exc.value).lower()
    assert "secret.txt" not in str(exc.value)


def test_attachment_symlink_escape_is_rejected(monkeypatch, tmp_path):
    jail = tmp_path / "jail"
    jail.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not leak")
    link = jail / "link.txt"
    link.symlink_to(secret)
    monkeypatch.setattr(email_mod.cfg, "attachment_root", str(jail))
    with pytest.raises(ToolError) as exc:
        email_mod._prepare_attachments(["link.txt"])
    assert "outside" in str(exc.value).lower()


def test_attachment_relative_path_stays_in_root(monkeypatch, tmp_path):
    jail = tmp_path / "jail"
    jail.mkdir()
    (jail / "note.txt").write_text("ok")
    monkeypatch.setattr(email_mod.cfg, "attachment_root", str(jail))
    prepared = email_mod._prepare_attachments(["note.txt"])
    assert prepared[0]["filename"] == "note.txt"
    assert prepared[0]["data"] == b"ok"
    assert "path" not in prepared[0]


def test_recipient_allowlist_drops_unlisted_addresses(monkeypatch, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "allowed_recipients", "good@example.com, example.com")
    sent = _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    out = json.loads(
        run(
            fn(
                recipients=["good@example.com", "evil@attacker.test", "other@example.com"],
                subject="hi",
                body="yo",
            )
        )
    )
    assert out["attempted_recipients"] == ["good@example.com", "other@example.com"]
    assert sent["envelope_recipients"] == ["good@example.com", "other@example.com"]
    assert out["dropped_recipients"] == [
        {"field": "to", "address": "evil@attacker.test", "reason": "not_allowed"}
    ]


def test_recipient_allowlist_rejects_when_none_remain(monkeypatch, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "allowed_recipients", "@corp.com")
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(fn(recipients=["evil@attacker.test"], subject="hi", body="yo"))
    assert "allowed_recipients" in str(exc.value).lower()


def test_reply_to_must_match_allowlist(monkeypatch, tool_fns):
    monkeypatch.setattr(email_mod.cfg, "allowed_recipients", "a@b.com")
    _capture_send(monkeypatch)
    fn = tool_fns["send_email"]
    with pytest.raises(ToolError) as exc:
        run(
            fn(
                recipients=["a@b.com"],
                reply_to="evil@attacker.test",
                subject="hi",
                body="yo",
            )
        )
    assert "reply_to" in str(exc.value).lower()


def test_attachment_growth_after_stat_is_bounded(monkeypatch, tmp_path):
    path = tmp_path / "growing.bin"
    path.write_bytes(b"small")
    monkeypatch.setattr(email_mod.cfg, "attachment_root", str(tmp_path))
    monkeypatch.setattr(email_mod.cfg, "max_attachment_bytes", 5)

    original_open = email_mod.Path.open

    def growing_open(self, *args, **kwargs):
        with original_open(path, "wb") as stream:
            stream.write(b"too-large")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(email_mod.Path, "open", growing_open)
    with pytest.raises(ToolError) as exc:
        email_mod._prepare_attachments([str(path)])
    assert "per-file limit" in str(exc.value)


def test_subject_header_injection_raises_toolerror(monkeypatch, tool_fns):
    _capture_send(monkeypatch)
    with pytest.raises(ToolError) as exc:
        run(
            tool_fns["send_email"](
                recipients=["a@b.com"], subject="hello\nBcc: victim@example.com", body="yo"
            )
        )
    assert "newline" in str(exc.value).lower()


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
