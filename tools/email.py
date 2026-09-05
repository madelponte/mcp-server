"""
Email (send-only) MCP tool.

Exposes a single `send_email` tool that delivers a plain-text email through an
authenticated SMTP account (Gmail by default — see config.EmailSettings for the
App Password requirement). It only ever *sends*: there is no mailbox reading,
listing, or deletion surface.

`smtplib` is a blocking, synchronous library, so the actual send runs in a
worker thread via `anyio.to_thread.run_sync` to avoid stalling the event loop
(see the "Sync libraries in async tools" convention in AGENTS.md).
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
from .tool_annotations import SIDE_EFFECTING_EXTERNAL_TOOL

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


def _recipient_allowlist() -> tuple[frozenset[str], frozenset[str]] | None:
    """Parse EMAIL_ALLOWED_RECIPIENTS into (exact emails, domains), or None.

    ``None`` means unrestricted (the list is blank). Domain entries may be
    written ``example.com`` or ``@example.com``; anything containing ``@`` is
    an exact address.
    """
    raw = (cfg.allowed_recipients or "").strip()
    if not raw:
        return None
    emails: set[str] = set()
    domains: set[str] = set()
    for item in re.split(r"[,\s]+", raw):
        if not item:
            continue
        item = item.lower()
        if item.startswith("@"):
            domains.add(item[1:])
        elif "@" in item:
            emails.add(item)
        else:
            domains.add(item)
    return frozenset(emails), frozenset(domains)


def _recipient_allowed(
    addr: str, allowlist: tuple[frozenset[str], frozenset[str]] | None
) -> bool:
    if allowlist is None:
        return True
    emails, domains = allowlist
    key = addr.lower()
    if key in emails:
        return True
    domain = key.rsplit("@", 1)[-1]
    return domain in domains


def _filter_allowed_recipients(
    entries: list[dict],
    allowlist: tuple[frozenset[str], frozenset[str]] | None,
) -> tuple[list[dict], list[dict]]:
    """Drop addresses that are not on EMAIL_ALLOWED_RECIPIENTS."""
    if allowlist is None:
        return entries, []
    kept: list[dict] = []
    rejected: list[dict] = []
    for entry in entries:
        if _recipient_allowed(entry["address"], allowlist):
            kept.append(entry)
        else:
            rejected.append({**entry, "reason": "not_allowed"})
    return kept, rejected


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


def _attachment_root() -> Path | None:
    """Resolved EMAIL_ATTACHMENT_ROOT, or None when attachments are disabled."""
    raw = (cfg.attachment_root or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _jail_attachment_path(raw: str, root: Path) -> Path:
    """Resolve ``raw`` inside ``root``; reject symlink/absolute escapes.

    Error messages do not echo the caller-supplied path so a probe for
    ``/etc/passwd`` cannot confirm that a file exists outside the jail.
    """
    text = str(raw or "").strip()
    if not text:
        raise ToolError("Attachment path is empty.")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ToolError("Attachment path is outside EMAIL_ATTACHMENT_ROOT.")
    if not resolved.is_file():
        raise ToolError(
            "Attachment path is not a readable file inside EMAIL_ATTACHMENT_ROOT."
        )
    return resolved


def _prepare_attachments(paths: list[str] | None) -> list[dict]:
    """Validate attachment paths and read bytes for EmailMessage.add_attachment."""
    if not paths:
        return []
    if not isinstance(paths, list):
        raise ToolError("`attachments` must be a list of local file paths.")
    root = _attachment_root()
    if root is None:
        raise ToolError(
            "Attachments are disabled. Set EMAIL_ATTACHMENT_ROOT to a directory "
            "the tool may read."
        )
    if len(paths) > cfg.max_attachments:
        raise ToolError(
            f"`attachments` may include at most {cfg.max_attachments} files."
        )

    prepared: list[dict] = []
    for raw in paths:
        if not str(raw or "").strip():
            continue
        path = _jail_attachment_path(str(raw), root)
        try:
            declared_size = path.stat().st_size
            if declared_size > cfg.max_attachment_bytes:
                raise ToolError(
                    f"Attachment {path.name} is {declared_size} bytes, above the "
                    f"{cfg.max_attachment_bytes}-byte per-file limit."
                )
            # Read at most one byte past the limit. The file can change between
            # stat() and open(), so the metadata check alone is not a memory cap.
            with path.open("rb") as stream:
                data = stream.read(cfg.max_attachment_bytes + 1)
        except ToolError:
            raise
        except OSError:
            raise ToolError(
                "Attachment path is not a readable file inside EMAIL_ATTACHMENT_ROOT."
            )
        size = len(data)
        if size > cfg.max_attachment_bytes:
            raise ToolError(
                f"Attachment {path.name} is {size} bytes, above the "
                f"{cfg.max_attachment_bytes}-byte per-file limit."
            )
        ctype, _ = mimetypes.guess_type(path.name)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        prepared.append(
            {
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
    from_name = (cfg.from_name or "").strip()
    msg["From"] = formataddr((from_name, from_addr)) if from_name else from_addr
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
    allowlist_note = (
        " Addresses must match the server recipient allowlist."
        if (cfg.allowed_recipients or "").strip()
        else ""
    )
    root_set = bool((cfg.attachment_root or "").strip())
    attachments_desc = (
        f"Optional files under the server attachment directory, up to "
        f"{cfg.max_attachments} files; each file max "
        f"{cfg.max_attachment_bytes} bytes. Use a filename or a path inside "
        "that directory; paths outside it are rejected."
        if root_set
        else "Disabled on this server (EMAIL_ATTACHMENT_ROOT is unset)."
    )

    @mcp.tool(annotations=SIDE_EFFECTING_EXTERNAL_TOOL)
    async def send_email(
        recipients: Annotated[
            list[str],
            Field(
                description=(
                    f"List of recipient email addresses (max {cfg.max_recipients}; "
                    f"addresses past the cap are dropped).{allowlist_note}"
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
                    f"{allowlist_note}"
                ),
            ),
        ] = None,
        bcc: Annotated[
            list[str] | None,
            Field(
                description=(
                    f"Optional BCC recipient addresses. Total To+Cc+Bcc cap is "
                    f"{cfg.max_recipients}; extras are dropped and reported. "
                    f"BCC addresses are not written into message headers."
                    f"{allowlist_note}"
                ),
            ),
        ] = None,
        reply_to: str | None = None,
        attachments: Annotated[
            list[str] | None,
            Field(description=attachments_desc),
        ] = None,
    ) -> str:
        """Send a plain-text email from the server's configured account.

        Send-only: this delivers a message; no other interaction.
        Use it to notify a person of a result, forward a summary, or
        deliver content you have already produced. Supports CC, BCC, Reply-To,
        and optional local file attachments when the server allows them.

        Returns JSON {status, subject, recipients:{to,cc,bcc},
        attempted_recipients, accepted_recipients, refused_recipients,
        invalid_recipients, dropped_recipients, attachments, dropped}.
        status is "sent", "partial", or "failed".

        :param subject: The email subject line.
        :param body: The plain-text message body.
        :param reply_to: Optional Reply-To email address.
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
        # from_name becomes a MIME header value; newlines would allow header
        # injection (a misconfigured EMAIL_FROM_NAME could smuggle extra headers).
        from_name = (cfg.from_name or "").strip()
        if "\r" in from_name or "\n" in from_name:
            raise ToolError(
                "EMAIL_FROM_NAME must not contain newline characters."
            )
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
        allowlist = _recipient_allowlist()
        if reply_to and not _recipient_allowed(reply_to, allowlist):
            raise ToolError(
                "`reply_to` is not permitted by EMAIL_ALLOWED_RECIPIENTS."
            )

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
        entries, not_allowed_dropped = _filter_allowed_recipients(entries, allowlist)
        if not entries:
            raise ToolError(
                "No recipients remain after applying EMAIL_ALLOWED_RECIPIENTS. "
                "Rejected: "
                + ", ".join(e["address"] for e in not_allowed_dropped)
            )

        # Clamp to the configured cap (a context/abuse guard). Over-cap addresses
        # are dropped and reported rather than sent — see AGENTS.md caps convention.
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
            "dropped_recipients": (
                duplicate_dropped + not_allowed_dropped + over_limit_dropped
            ),
            "attachments": [
                {
                    "filename": att["filename"],
                    "content_type": att["content_type"],
                    "size_bytes": att["size_bytes"],
                }
                for att in prepared_attachments
            ],
            # Backward-compatible flat summary for older callers.
            "dropped": [
                e["address"]
                for e in invalid + duplicate_dropped + not_allowed_dropped + over_limit_dropped
            ],
        }
        if refused and not accepted:
            payload["status"] = "failed"
        return log_result(log, "send_email", to_json(payload))
