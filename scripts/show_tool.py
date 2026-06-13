#!/usr/bin/env python3
"""Print a tool exactly as the model sees it (its MCP ``tools/list`` entry).

Tool descriptions and per-argument descriptions are now built at registration
time from ``config.py`` (e.g. the geocoding caps are interpolated into the text),
so the source docstring no longer matches what the model receives. This script
builds the real server via ``server.build_server()`` and dumps the serialized
schema, so you can verify the rendered result — including how env vars change it.

  Usage (from the repo root):

  # List every registered tool with its first description line
  .venv/bin/python scripts/show_tool.py

  # Readable view: description block + each argument's description + required/optional
  .venv/bin/python scripts/show_tool.py find_nearby_places

  # Full MCP schema as JSON (exact tools/list payload)
  .venv/bin/python scripts/show_tool.py find_nearby_places --json

  # Caps come from config, so env vars change the output live
  GEO_MAX_RADIUS_M=50000 .venv/bin/python scripts/show_tool.py find_nearby_places

Because the caps come from config, env vars change the output live:

    GEO_MAX_RADIUS_M=50000 python scripts/show_tool.py find_nearby_places
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running as `python scripts/show_tool.py` from the repo root by putting the
# project root (this file's parent's parent) on the path before importing server.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import build_server  # noqa: E402


async def _run(name: str | None, as_json: bool) -> int:
    mcp = build_server()
    tools = await mcp.list_tools()
    by_name = {t.name: t for t in tools}

    if name is None:
        # No tool given: list them all with their first description line.
        print(f"{len(tools)} tool(s):\n")
        for t in sorted(tools, key=lambda t: t.name):
            first_line = (t.description or "").strip().splitlines()
            summary = first_line[0] if first_line else ""
            print(f"  {t.name:<22} {summary}")
        return 0

    tool = by_name.get(name)
    if tool is None:
        print(f"No tool named {name!r}. Available: {', '.join(sorted(by_name))}")
        return 1

    mcp_tool = tool.to_mcp_tool()

    if as_json:
        # The exact payload FastMCP serializes into a tools/list response.
        print(json.dumps(mcp_tool.model_dump(exclude_none=True), indent=2))
        return 0

    # Readable view: the description block plus each argument's description.
    print("=" * 70)
    print(f"TOOL: {mcp_tool.name}")
    print("=" * 70)
    print(mcp_tool.description or "(no description)")
    print()
    print("ARGUMENTS:")
    props = (mcp_tool.inputSchema or {}).get("properties", {})
    required = set((mcp_tool.inputSchema or {}).get("required", []))
    if not props:
        print("  (none)")
    for arg, spec in props.items():
        flag = "required" if arg in required else "optional"
        print(f"  - {arg} ({flag}): {spec.get('description', '')}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show a tool as the model sees it (its MCP tools/list entry)."
    )
    parser.add_argument(
        "tool", nargs="?", help="Tool name (omit to list all tools)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full MCP schema as JSON."
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(args.tool, args.json)))


if __name__ == "__main__":
    main()
