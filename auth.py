"""
Bearer-token authentication for the HTTP transports.

Any number of **named** tokens may be configured (``server.auth_tokens`` in the
YAML config file). A request is authorized when its ``Authorization: Bearer
<token>`` header matches one of them, and the matched name is recorded on the ASGI
scope and logged, so several clients can share the server while still being
distinguishable in the log — and one credential can be revoked by editing the
file. When no token is configured the middleware is not installed and the server
is open; ``server.py`` refuses to start an HTTP transport in that state unless
unauthenticated operation is explicitly allowed.

Implemented as pure ASGI middleware so it sits in front of FastMCP's Starlette
app without depending on Starlette's request/response objects. Non-HTTP scopes
(notably ``lifespan``) are passed straight through.
"""

import hmac
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

_BEARER_PREFIX = b"Bearer "
# Rejections are logged at most this often (seconds) so a misbehaving client or
# an active probe can't flood the log; rejections in between are counted and
# reported with the next warning.
_REJECT_LOG_INTERVAL = 30.0


@dataclass(frozen=True)
class BearerToken:
    """One accepted credential: a log-friendly name plus the secret itself."""

    name: str
    token: str


# Anything the middleware accepts where credentials are expected: a named entry,
# a bare secret, or an iterable of either.
TokenInput = BearerToken | str | Iterable[BearerToken | str]


def _coerce_tokens(tokens: TokenInput | None) -> list[BearerToken]:
    """Normalize the caller's credentials into named entries, dropping blanks.

    A bare string is accepted so a single-token deployment (the legacy
    ``MCP_AUTH_TOKEN`` path, and every caller that has one secret) needs no
    wrapper. An entry carrying a blank secret is dropped rather than installed:
    an empty credential must never match anything.
    """
    if tokens is None:
        return []
    items = [tokens] if isinstance(tokens, (str, BearerToken)) else list(tokens)
    entries: list[BearerToken] = []
    for item in items:
        if isinstance(item, str):
            item = BearerToken(name="default", token=item)
        secret = (item.token or "").strip()
        if not secret:
            continue
        entries.append(BearerToken(name=(item.name or "").strip() or "default", token=secret))
    # Collapse a secret listed twice (say the legacy field plus an explicit entry)
    # so it authenticates under exactly one name.
    seen: set[str] = set()
    unique: list[BearerToken] = []
    for entry in entries:
        if entry.token in seen:
            continue
        seen.add(entry.token)
        unique.append(entry)
    return unique


def _bearer_value(headers: Iterable[tuple[bytes, bytes]]) -> bytes | None:
    """Extract the bearer token from a raw ASGI header list (first one wins)."""
    for name, value in headers:
        if name == b"authorization":
            if not value.startswith(_BEARER_PREFIX):
                return None
            return value[len(_BEARER_PREFIX) :].strip()
    return None


class BearerAuthMiddleware:
    """Enforce that every HTTP request presents one of the configured tokens."""

    def __init__(self, app: ASGIApp, tokens: TokenInput) -> None:
        entries = _coerce_tokens(tokens)
        if not entries:
            raise ValueError(
                "BearerAuthMiddleware requires at least one non-empty token."
            )
        self.app = app
        # Pre-encode to bytes: the header arrives as bytes, and a non-ASCII
        # secret could never be presented correctly over HTTP anyway.
        self._tokens: list[tuple[str, bytes]] = []
        for entry in entries:
            try:
                secret = entry.token.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"BearerAuthMiddleware token for {entry.name!r} must contain "
                    "only ASCII characters."
                ) from exc
            self._tokens.append((entry.name, secret))
        self.client_names: tuple[str, ...] = tuple(name for name, _ in self._tokens)
        self._last_reject_log: float | None = None
        self._rejects_suppressed = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        client = self._authorized_client(scope)
        if client is None:
            self._log_rejection(scope)
            await self._reject(send)
            return

        # Let anything downstream (request logging, etc.) attribute the call to
        # the credential it was made with.
        scope.setdefault("state", {})["mcp_client"] = client
        logger.debug("Authorized MCP request from client %s", client)

        await self.app(scope, receive, send)

    def _authorized_client(self, scope: Scope) -> str | None:
        """Return the name of the token the request presented, or None."""
        presented = _bearer_value(scope.get("headers") or [])
        if not presented:
            return None
        matched: str | None = None
        for name, expected in self._tokens:
            # Compare against every configured token rather than stopping at the
            # first hit, so the time taken reveals neither which entry matched nor
            # how much of a guess was correct.
            if hmac.compare_digest(presented, expected):
                matched = name
        return matched

    def _log_rejection(self, scope: Scope) -> None:
        """Log an unauthorized request, throttled so a flood can't spam it.

        The client address comes from the ASGI ``client`` tuple. It is normally
        the direct peer address; trusted proxy-header middleware may rewrite it
        to the forwarded client address. If the tuple is absent, the source is
        reported as "unknown".
        """
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        now = time.monotonic()
        if (
            self._last_reject_log is None
            or now - self._last_reject_log >= _REJECT_LOG_INTERVAL
        ):
            extra = (
                f"; {self._rejects_suppressed} more suppressed since the last log"
                if self._rejects_suppressed
                else ""
            )
            logger.warning(
                "Rejected unauthorized MCP request from %s%s", client_ip, extra
            )
            self._last_reject_log = now
            self._rejects_suppressed = 0
        else:
            self._rejects_suppressed += 1

    @staticmethod
    async def _reject(send: Send) -> None:
        body = b'{"error": "unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
