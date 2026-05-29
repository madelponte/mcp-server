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

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "Bearer "


class BearerAuthMiddleware:
    """Enforce a static bearer token on every HTTP request."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("BearerAuthMiddleware requires a non-empty token.")
        self.app = app
        self._token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self._is_authorized(scope):
            await self._reject(send)
            return

        await self.app(scope, receive, send)

    def _is_authorized(self, scope: Scope) -> bool:
        for name, value in scope.get("headers") or []:
            if name == b"authorization":
                header = value.decode("latin-1")
                if not header.startswith(_BEARER_PREFIX):
                    return False
                presented = header[len(_BEARER_PREFIX):].strip()
                # Constant-time compare so a wrong token can't be guessed via timing.
                return hmac.compare_digest(presented, self._token)
        return False

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
