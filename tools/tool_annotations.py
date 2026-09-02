"""Shared MCP safety annotations for the server's tools.

These hints describe effects, not result freshness: an idempotent read may
return newer data on a later call, but repeating it does not create additional
external side effects. All tools contact external systems, so ``open_world`` is
true for both profiles.
"""

from mcp.types import ToolAnnotations


READ_ONLY_EXTERNAL_TOOL = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

SIDE_EFFECTING_EXTERNAL_TOOL = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)
