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
import re
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
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


def _normalize_recipients(recipients: list[str]) -> tuple[list[str], list[str]]:
    """Split, dedupe, and validate recipient addresses.

    Returns ``(valid, invalid)`` preserving first-seen order. Surrounding
    whitespace is stripped; empties are ignored.
    """
    valid: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()
    for raw in recipients:
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


def _build_message(recipients: list[str], subject: str, body: str) -> EmailMessage:
    msg = EmailMessage()
    from_addr = (cfg.from_address or cfg.username).strip()
    msg["From"] = formataddr((cfg.from_name.strip(), from_addr)) if cfg.from_name.strip() else from_addr
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)
    return msg


def _send(msg: EmailMessage) -> None:
    """Blocking SMTP send. Runs in a worker thread (see send_email)."""
    if cfg.use_ssl:
        with smtplib.SMTP_SSL(
            cfg.smtp_host, cfg.smtp_port, timeout=cfg.timeout_seconds
        ) as server:
            server.login(cfg.username, cfg.password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(
            cfg.smtp_host, cfg.smtp_port, timeout=cfg.timeout_seconds
        ) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(cfg.username, cfg.password)
            server.send_message(msg)


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
    ) -> str:
        """Send a plain-text email from the server's configured account.

        Send-only: this delivers a message; no other interaction.
        Use it to notify a person of a result, forward a summary, or
        deliver content you have already produced.

        :param subject: The email subject line.
        :param body: The plain-text message body.
        :return: JSON {"status":"sent","recipients":[...],"dropped":[...],"subject":...}.
        """
        log_call(
            log,
            "send_email",
            recipients=recipients,
            subject=subject,
            body_chars=len(body or ""),
        )

        if not (cfg.username or "").strip() or not (cfg.password or "").strip():
            raise ToolError(
                "Email is not configured. Set EMAIL_USERNAME and EMAIL_PASSWORD "
                "(for Gmail, EMAIL_PASSWORD must be a 16-character App Password — "
                "see https://support.google.com/accounts/answer/185833)."
            )

        if not isinstance(recipients, list) or not recipients:
            raise ToolError("`recipients` must be a non-empty list of email addresses.")
        if not (subject or "").strip():
            raise ToolError("`subject` must not be empty.")
        if not (body or "").strip():
            raise ToolError("`body` must not be empty.")

        valid, invalid = _normalize_recipients(recipients)
        if not valid:
            raise ToolError(
                "No valid recipient addresses. Rejected: " + ", ".join(invalid)
            )

        # Clamp to the configured cap (a context/abuse guard). Over-cap addresses
        # are dropped and reported rather than sent — see CLAUDE.md caps convention.
        dropped = list(invalid)
        if len(valid) > cfg.max_recipients:
            dropped.extend(valid[cfg.max_recipients :])
            valid = valid[: cfg.max_recipients]

        msg = _build_message(valid, subject, body)

        try:
            await anyio.to_thread.run_sync(_send, msg)
        except smtplib.SMTPAuthenticationError as exc:
            raise ToolError(
                "SMTP authentication failed. Check EMAIL_USERNAME and EMAIL_PASSWORD. "
                "For Gmail you must use a 16-character App Password (with 2-Step "
                f"Verification enabled), not your normal password. Server said: {exc}"
            )
        except smtplib.SMTPRecipientsRefused as exc:
            raise ToolError(
                f"All recipients were refused by the server: {exc.recipients}"
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

        payload = {
            "status": "sent",
            "recipients": valid,
            "dropped": dropped,
            "subject": subject,
        }
        return log_result(log, "send_email", to_json(payload))
