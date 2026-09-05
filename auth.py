"""
Bearer-token authentication for the HTTP transports.

A single shared secret (``MCP_AUTH_TOKEN``) gates every request. Clients must
send ``Authorization: Bearer <token>``. When no token is configured the
middleware is not installed and the server is open — see ``server.py``.

Implemented as pure ASGI middleware so it sits in front of FastMCP's Starlette
app without depending on Starlette's request/response objects. Non-HTTP scopes
(notably ``lifespan``) are passed straight through.
"""

import hmac
import logging
import time

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

_BEARER_PREFIX = b"Bearer "
# Rejections are logged at most this often (seconds) so a misbehaving client or
# an active probe can't flood the log; rejections in between are counted and
# reported with the next warning.
_REJECT_LOG_INTERVAL = 30.0


class BearerAuthMiddleware:
    """Enforce a static bearer token on every HTTP request."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("BearerAuthMiddleware requires a non-empty token.")
        try:
            self._token = token.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "BearerAuthMiddleware token must contain only ASCII characters."
            ) from exc
        self.app = app
        self._last_reject_log: float | None = None
        self._rejects_suppressed = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self._is_authorized(scope):
            self._log_rejection(scope)
            await self._reject(send)
            return

        await self.app(scope, receive, send)

    def _is_authorized(self, scope: Scope) -> bool:
        for name, value in scope.get("headers") or []:
            if name == b"authorization":
                if not value.startswith(_BEARER_PREFIX):
                    return False
                presented = value[len(_BEARER_PREFIX):].strip()
                # Constant-time compare so a wrong token can't be guessed via timing.
                return hmac.compare_digest(presented, self._token)
        return False

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
