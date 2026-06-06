"""
Wolfram Alpha MCP tool.

Runs computations and looks up factual data via the Wolfram Alpha LLM API.
Translated from the Open WebUI tool — the HTML "card" rendering was dropped
(MCP returns plain text), the rest of the behavior is preserved.
"""

import logging

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import wolfram_settings as cfg
from .cache import TTLCache
from .serialize import log_call, log_result

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`. This keeps failures from being mistaken
# for normal output by a model. See the README "Error handling" section.

BASE_URL = "https://www.wolframalpha.com/api/v1/llm-api"

# A Wolfram result for a given (input, assumption, units, maxchars) is
# effectively deterministic, and agent loops re-ask the same computation
# surprisingly often, so we cache the finished response. See the README
# "Caching" section.
_result_cache = TTLCache(cfg.cache_ttl_seconds, cfg.cache_max_entries)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def query_wolfram_alpha(query: str, assumption: str | None = None) -> str:
        """
        Compute or look up exact factual data with Wolfram Alpha: math, unit/currency
        conversions, physics, chemistry, astronomy, geography, demographics, dates &
        times, finance, nutrition, weather history, and entity comparisons.

        Query rules:
        - Use keyword style, not sentences: "France population", not "how many
          people live in France".
        - Math: single-letter variables (n, x), exponents like 6*10^14 (not 6e14),
          named constants ("speed of light") not raw numbers.
        - One property per call.
        - If a result lists 'Assumptions', re-send the SAME query with `assumption`
          set to that value — don't rephrase.

        Don't use for opinions, current news, code, or things you already know.

        :param query: Keyword-style query, single line.
        :param assumption: Assumption value from a prior result, to disambiguate
            (e.g. "mercury" = planet vs. element).
        :return: Text result from Wolfram Alpha.
        """
        log_call(log, "query_wolfram_alpha", query=query, assumption=assumption)
        app_id = (cfg.app_id or "").strip()
        if not app_id:
            raise ToolError(
                "Wolfram Alpha AppID is not configured. Set the WOLFRAM_APP_ID "
                "environment variable. Get a free AppID at https://developer.wolframalpha.com"
            )

        if not query or not query.strip():
            raise ToolError("Empty query. Provide a Wolfram Alpha query string.")

        clean_query = query.strip().replace("\n", " ")

        params = {
            "input": clean_query,
            "appid": app_id,
            "maxchars": cfg.max_chars,
            "units": cfg.default_units,
        }
        if assumption:
            params["assumption"] = assumption

        # Key on everything that affects the result (but never the AppID). A
        # null byte separates fields so values can't collide across boundaries.
        cache_key = "\x00".join(
            (clean_query, assumption or "", cfg.default_units, str(cfg.max_chars))
        )
        cached = _result_cache.get(cache_key)
        if cached is not None:
            return log_result(log, "query_wolfram_alpha", cached)

        try:
            async with httpx.AsyncClient(timeout=cfg.http_timeout_seconds) as client:
                response = await client.get(
                    BASE_URL,
                    params=params,
                    headers={"User-Agent": "MCP-WolframAlpha/1.0"},
                )
        except httpx.TimeoutException:
            raise ToolError(
                f"Wolfram Alpha request timed out after {cfg.http_timeout_seconds}s."
            )
        except httpx.HTTPError as exc:
            raise ToolError(f"Network error contacting Wolfram Alpha: {exc}")

        status = response.status_code
        body = response.text or ""

        # 501: Wolfram couldn't interpret the input. No answer was produced, so
        # this is an error — but the body often carries useful suggestions, so
        # surface them in the error message for the model to retry with.
        if status == 501:
            raise ToolError(
                f"Wolfram Alpha could not interpret the query: '{clean_query}'.\n"
                f"Suggestions from the API:\n{body.strip()}\n\n"
                "Try rephrasing as a simpler keyword-style query, or pick one of "
                "the suggested inputs above."
            )

        if status == 403:
            raise ToolError(
                "Wolfram Alpha rejected the AppID (HTTP 403). "
                "Check that WOLFRAM_APP_ID is set correctly."
            )

        body_snippet = body[:200]
        if status == 400:
            raise ToolError(f"Wolfram Alpha rejected the request (HTTP 400): {body_snippet}")

        if status >= 400:
            raise ToolError(f"Wolfram Alpha error (HTTP {status}): {body_snippet}")

        if not body.strip():
            raise ToolError("Wolfram Alpha returned an empty response.")

        _result_cache.set(cache_key, body)
        return log_result(log, "query_wolfram_alpha", body)
