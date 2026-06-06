"""
Shared JSON serialization and debug helpers for the tool modules.

Every tool returns its structured result through :func:`to_json` so the output
format is consistent across the whole server and driven by a single switch:

- **Default (MCP_DEBUG off):** compact JSON with no insignificant whitespace,
  to keep results as small as possible in the model's context window.
- **Debug (MCP_DEBUG on):** indented, human-readable JSON, plus verbose
  DEBUG-level logging of each tool call (see :func:`log_call`) to stdout.

`ensure_ascii` is always False so non-ASCII text is preserved literally rather
than escaped, and `default=str` lets non-JSON-native values (datetimes,
Decimals, etc.) serialize instead of raising.
"""

import json
import logging
from typing import Any

from config import server_settings


def debug_enabled() -> bool:
    """True when the server is running in debug mode (MCP_DEBUG)."""
    return server_settings.debug


def to_json(payload: Any, *, default=str) -> str:
    """Serialize a tool result to a JSON string.

    Compact by default; indented (``indent=2``) when debug mode is enabled. Use
    this for every structured tool result instead of calling ``json.dumps``
    directly so the compact/pretty switch stays centralized.
    """
    if server_settings.debug:
        return json.dumps(payload, ensure_ascii=False, indent=2, default=default)
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), default=default
    )


def log_call(logger: logging.Logger, tool: str, **params: Any) -> None:
    """In debug mode, log a tool invocation and its arguments. No-op otherwise.

    Keeps the ``if server_settings.debug`` check and argument formatting in one
    place so tool functions can announce themselves with a single line.
    """
    if not server_settings.debug:
        return
    if params:
        rendered = ", ".join(f"{k}={v!r}" for k, v in params.items())
        logger.debug("tool call %s(%s)", tool, rendered)
    else:
        logger.debug("tool call %s()", tool)


def log_result(logger: logging.Logger, tool: str, result: str) -> str:
    """In debug mode, log a one-line summary of a tool's return value.

    Logs the serialized size and a short preview so stdout shows what each call
    produced (useful for spotting oversized responses) without dumping a huge
    payload. Returns ``result`` unchanged so it can wrap a ``return`` statement.
    No-op when debug is off.
    """
    if server_settings.debug:
        preview = result if len(result) <= 200 else result[:200] + "…"
        logger.debug("tool result %s: %d chars; %s", tool, len(result), preview)
    return result
