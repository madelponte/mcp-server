"""
Email (send-only) MCP tool.

Exposes a single `send_email` tool that delivers a plain-text email through an
authenticated SMTP account (Gmail by default — see config.EmailSettings for the
App Password requirement). It only ever *sends*: there is no mailbox reading,
listing, or deletion surface.

`smtplib` is a blocking, synchronous library, so the actual send runs in a
worker thread via `anyio.to_thread.run_sync` to avoid stalling the event loop
(see the "Sync libraries in async tools" convention in CLAUDE.md).
"""

import logging
import mimetypes
import re
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Annotated

import anyio
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from config import email_settings as cfg
from .serialize import log_call, log_result, to_json

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake a delivery
# failure for a successful send. A partial send (some recipients accepted, some
# refused) is reported as data, not raised — see below.

# Deliberately permissive: a real address-syntax check (RFC 5322) is not worth
# the complexity here. We only reject the obvious junk (missing '@', whitespace,
# empty local/domain part) up front so a typo fails fast with a clear message;
# the SMTP server is the real authority on whether an address is deliverable.
_ADDR_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalize_recipients(recipients: list[str] | None) -> tuple[list[str], list[str]]:
    """Split, dedupe, and validate recipient addresses within one field.

    Returns ``(valid, invalid)`` preserving first-seen order. Surrounding
    whitespace is stripped; empties are ignored.
    """
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in recipients or []:
        addr = (raw or "").strip()
        if not addr:
            continue
        key = addr.lower()
        if key in seen:
            continue
        seen.add(key)
        if _ADDR_RE.match(addr):
            valid.append(addr)
        else:
            invalid.append(addr)
    return valid, invalid


def _dedupe_address_entries(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Dedupe recipients across To/Cc/Bcc, preserving the first occurrence."""
    kept: list[dict] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry["address"].lower()
        if key in seen:
            dropped.append({**entry, "reason": "duplicate"})
            continue
        seen.add(key)
        kept.append(entry)
    return kept, dropped


def _recipient_addresses(entries: list[dict], field: str) -> list[str]:
    return [e["address"] for e in entries if e["field"] == field]


def _format_refused(refused: dict | None) -> list[dict]:
    """Turn smtplib's refused-recipient mapping into JSON-safe records."""
    out: list[dict] = []
    for address, detail in (refused or {}).items():
        code = None
        response = detail
        if isinstance(detail, tuple) and len(detail) >= 2:
            code, response = detail[0], detail[1]
        if isinstance(response, bytes):
            response = response.decode("utf-8", errors="replace")
        out.append({"address": address, "code": code, "response": str(response)})
    return out


def _prepare_attachments(paths: list[str] | None) -> list[dict]:
    """Validate attachment paths and read bytes for EmailMessage.add_attachment."""
    if not paths:
        return []
    if not isinstance(paths, list):
        raise ToolError("`attachments` must be a list of local file paths.")
    if len(paths) > cfg.max_attachments:
        raise ToolError(
            f"`attachments` may include at most {cfg.max_attachments} files."
        )

    prepared: list[dict] = []
    for raw in paths:
        path = Path(str(raw or "").strip()).expanduser()
        if not str(path):
            continue
        try:
            if not path.is_file():
                raise ToolError(f"Attachment path is not a readable file: {path}")
            declared_size = path.stat().st_size
            if declared_size > cfg.max_attachment_bytes:
                raise ToolError(
                    f"Attachment {path} is {declared_size} bytes, above the "
                    f"{cfg.max_attachment_bytes}-byte per-file limit."
                )
            # Read at most one byte past the limit. The file can change between
            # stat() and open(), so the metadata check alone is not a memory cap.
            with path.open("rb") as stream:
                data = stream.read(cfg.max_attachment_bytes + 1)
        except ToolError:
            raise
        except OSError as exc:
            raise ToolError(f"Attachment path is not a readable file: {path}: {exc}")
        size = len(data)
        if size > cfg.max_attachment_bytes:
            raise ToolError(
                f"Attachment {path} is {size} bytes, above the "
                f"{cfg.max_attachment_bytes}-byte per-file limit."
            )
        ctype, _ = mimetypes.guess_type(path.name)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        prepared.append(
            {
                "path": str(path),
                "filename": path.name,
                "content_type": ctype or "application/octet-stream",
                "size_bytes": size,
                "maintype": maintype,
                "subtype": subtype,
                "data": data,
            }
        )
    return prepared


def _build_message(
    to_recipients: list[str],
    cc_recipients: list[str],
    subject: str,
    body: str,
    *,
    reply_to: str | None = None,
    attachments: list[dict] | None = None,
) -> EmailMessage:
    msg = EmailMessage()
    from_addr = (cfg.from_address or cfg.username).strip()
    msg["From"] = formataddr((cfg.from_name.strip(), from_addr)) if cfg.from_name.strip() else from_addr
    msg["To"] = ", ".join(to_recipients)
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)
    if reply_to:
        msg["Reply-To"] = reply_to
    msg["Subject"] = subject
    msg.set_content(body)
    for att in attachments or []:
        msg.add_attachment(
            att["data"],
            maintype=att["maintype"],
            subtype=att["subtype"],
            filename=att["filename"],
        )
    return msg


def _prepare_message(
    to_recipients: list[str],
    cc_recipients: list[str],
    subject: str,
    body: str,
    reply_to: str | None,
    attachment_paths: list[str] | None,
) -> tuple[list[dict], EmailMessage]:
    """Read attachments and build MIME content outside the async event loop."""
    attachments = _prepare_attachments(attachment_paths)
    message = _build_message(
        to_recipients,
        cc_recipients,
        subject,
        body,
        reply_to=reply_to,
        attachments=attachments,
    )
    return attachments, message


def _send(msg: EmailMessage, envelope_recipients: list[str]) -> dict:
    """Blocking SMTP send. Runs in a worker thread (see send_email)."""
    if cfg.use_ssl:
        with smtplib.SMTP_SSL(
            cfg.smtp_host, cfg.smtp_port, timeout=cfg.timeout_seconds
        ) as server:
            server.login(cfg.username, cfg.password)
            return server.send_message(msg, to_addrs=envelope_recipients)
    else:
        with smtplib.SMTP(
            cfg.smtp_host, cfg.smtp_port, timeout=cfg.timeout_seconds
        ) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(cfg.username, cfg.password)
            return server.send_message(msg, to_addrs=envelope_recipients)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def send_email(
        recipients: Annotated[
            list[str],
            Field(
                description=(
                    f"List of recipient email addresses (max {cfg.max_recipients}; "
                    "addresses past the cap are dropped)."
                ),
            ),
        ],
        subject: str,
        body: str,
        cc: Annotated[
            list[str] | None,
            Field(
                description=(
                    f"Optional CC recipient addresses. Total To+Cc+Bcc cap is "
                    f"{cfg.max_recipients}; extras are dropped and reported."
                ),
            ),
        ] = None,
        bcc: Annotated[
            list[str] | None,
            Field(
                description=(
                    f"Optional BCC recipient addresses. Total To+Cc+Bcc cap is "
                    f"{cfg.max_recipients}; extras are dropped and reported. "
                    "BCC addresses are not written into message headers."
                ),
            ),
        ] = None,
        reply_to: str | None = None,
        attachments: Annotated[
            list[str] | None,
            Field(
                description=(
                    f"Optional local file paths to attach, up to "
                    f"{cfg.max_attachments} files; each file max "
                    f"{cfg.max_attachment_bytes} bytes."
                ),
            ),
        ] = None,
    ) -> str:
        """Send a plain-text email from the server's configured account.

        Send-only: this delivers a message; no other interaction.
        Use it to notify a person of a result, forward a summary, or
        deliver content you have already produced. Supports CC, BCC, Reply-To,
        and local file attachments.

        :param subject: The email subject line.
        :param body: The plain-text message body.
        :param reply_to: Optional Reply-To email address.
        :return: JSON with status, intended/accepted/refused recipients,
            invalid/dropped recipients, and attachment metadata.
        """
        log_call(
            log,
            "send_email",
            recipients=recipients,
            cc=cc,
            bcc=bcc,
            subject=subject,
            body_chars=len(body or ""),
            attachment_count=len(attachments or []),
        )

        if not (cfg.username or "").strip() or not (cfg.password or "").strip():
            raise ToolError(
                "Email is not configured. Set EMAIL_USERNAME and EMAIL_PASSWORD "
                "(for Gmail, EMAIL_PASSWORD must be a 16-character App Password — "
                "see https://support.google.com/accounts/answer/185833)."
            )

        if not isinstance(recipients, list) or not recipients:
            raise ToolError("`recipients` must be a non-empty list of email addresses.")
        if cc is not None and not isinstance(cc, list):
            raise ToolError("`cc` must be a list of email addresses when provided.")
        if bcc is not None and not isinstance(bcc, list):
            raise ToolError("`bcc` must be a list of email addresses when provided.")
        if not (subject or "").strip():
            raise ToolError("`subject` must not be empty.")
        if "\r" in subject or "\n" in subject:
            raise ToolError("`subject` must not contain newline characters.")
        if not (body or "").strip():
            raise ToolError("`body` must not be empty.")

        to_valid, to_invalid = _normalize_recipients(recipients)
        cc_valid, cc_invalid = _normalize_recipients(cc)
        bcc_valid, bcc_invalid = _normalize_recipients(bcc)
        invalid = (
            [{"field": "to", "address": a} for a in to_invalid]
            + [{"field": "cc", "address": a} for a in cc_invalid]
            + [{"field": "bcc", "address": a} for a in bcc_invalid]
        )

        reply_to = (reply_to or "").strip() or None
        if reply_to and not _ADDR_RE.match(reply_to):
            raise ToolError(f"`reply_to` is not a valid email address: {reply_to}")

        entries = (
            [{"field": "to", "address": a} for a in to_valid]
            + [{"field": "cc", "address": a} for a in cc_valid]
            + [{"field": "bcc", "address": a} for a in bcc_valid]
        )
        entries, duplicate_dropped = _dedupe_address_entries(entries)
        if not entries:
            raise ToolError(
                "No valid recipient addresses. Rejected: "
                + ", ".join(e["address"] for e in invalid)
            )

        # Clamp to the configured cap (a context/abuse guard). Over-cap addresses
        # are dropped and reported rather than sent — see CLAUDE.md caps convention.
        over_limit_dropped: list[dict] = []
        if len(entries) > cfg.max_recipients:
            over_limit_dropped = [
                {**e, "reason": "over_recipient_limit"}
                for e in entries[cfg.max_recipients :]
            ]
            entries = entries[: cfg.max_recipients]

        if not entries:
            raise ToolError(
                f"No recipients remained after applying the {cfg.max_recipients}-recipient cap."
            )

        to_final = _recipient_addresses(entries, "to")
        cc_final = _recipient_addresses(entries, "cc")
        bcc_final = _recipient_addresses(entries, "bcc")
        envelope_recipients = [e["address"] for e in entries]

        try:
            prepared_attachments, msg = await anyio.to_thread.run_sync(
                _prepare_message,
                to_final,
                cc_final,
                subject,
                body,
                reply_to,
                attachments,
            )
        except ToolError:
            raise
        except (TypeError, ValueError) as exc:
            # EmailMessage rejects newline-bearing headers and other malformed
            # values. Surface that as a tool failure instead of an internal error.
            raise ToolError(f"Invalid email header value: {exc}")

        try:
            refused_raw = await anyio.to_thread.run_sync(_send, msg, envelope_recipients)
        except smtplib.SMTPAuthenticationError as exc:
            raise ToolError(
                "SMTP authentication failed. Check EMAIL_USERNAME and EMAIL_PASSWORD. "
                "For Gmail you must use a 16-character App Password (with 2-Step "
                f"Verification enabled), not your normal password. Server said: {exc}"
            )
        except smtplib.SMTPRecipientsRefused as exc:
            raise ToolError(
                "All recipients were refused by the server: "
                + to_json({"refused_recipients": _format_refused(exc.recipients)})
            )
        except smtplib.SMTPSenderRefused as exc:
            raise ToolError(
                "The From address was refused by the server. For Gmail, "
                f"EMAIL_FROM_ADDRESS must be your own address or a verified alias. {exc}"
            )
        except smtplib.SMTPException as exc:
            raise ToolError(f"SMTP error while sending email: {exc}")
        except (OSError, TimeoutError) as exc:
            raise ToolError(
                f"Could not connect to the SMTP server {cfg.smtp_host}:{cfg.smtp_port}: {exc}"
            )

        refused = _format_refused(refused_raw)
        refused_set = {r["address"].lower() for r in refused}
        accepted = [a for a in envelope_recipients if a.lower() not in refused_set]
        payload: dict = {
            "status": "partial" if refused else "sent",
            "subject": subject,
            "recipients": {
                "to": to_final,
                "cc": cc_final,
                "bcc": bcc_final,
            },
            "attempted_recipients": envelope_recipients,
            "accepted_recipients": accepted,
            "refused_recipients": refused,
            "invalid_recipients": invalid,
            "dropped_recipients": duplicate_dropped + over_limit_dropped,
            "attachments": [
                {
                    "path": att["path"],
                    "filename": att["filename"],
                    "content_type": att["content_type"],
                    "size_bytes": att["size_bytes"],
                }
                for att in prepared_attachments
            ],
            # Backward-compatible flat summary for older callers.
            "dropped": [e["address"] for e in invalid + duplicate_dropped + over_limit_dropped],
        }
        if refused and not accepted:
            payload["status"] = "failed"
        return log_result(log, "send_email", to_json(payload))
