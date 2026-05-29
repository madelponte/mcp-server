"""
Wolfram Alpha MCP tool.

Runs computations and looks up factual data via the Wolfram Alpha LLM API.
Translated from the Open WebUI tool — the HTML "card" rendering was dropped
(MCP returns plain text), the rest of the behavior is preserved.
"""

import httpx
from mcp.server.fastmcp import FastMCP

from config import wolfram_settings as cfg

BASE_URL = "https://www.wolframalpha.com/api/v1/llm-api"


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def query_wolfram_alpha(query: str, assumption: str | None = None) -> str:
        """
        Compute or look up factual data using Wolfram Alpha. Use this for anything
        requiring exact computation or authoritative reference data, including:
        math (algebra, calculus, linear algebra, statistics, number theory),
        unit/currency conversions, physics & chemistry (constants, formulas,
        properties, reactions), astronomy (planetary positions, eclipses, distances),
        geography & demographics (populations, GDP, distances between cities),
        dates & times (timezone conversion, day of week, time between dates),
        finance (stock data, historical prices), nutrition, weather history,
        words & linguistics, and structured comparisons of named entities.

        Query formatting (important):
        - Send English keyword-style queries, not full sentences:
          "France population" not "how many people live in France".
        - Use single-letter variable names (n, n_1, x) in math.
        - Use exponent notation like 6*10^14, never 6e14.
        - Use named constants ("speed of light") rather than substituting numbers.
        - One property per call — make separate calls for separate properties.
        - If a previous result returned 'Assumptions', re-send the SAME input
          with the assumption parameter set to the relevant value, do not rephrase.

        Do NOT use for: opinions, current news, code generation, creative writing,
        or simple lookups already answerable from general knowledge.

        :param query: The Wolfram Alpha query (English, keyword-style, single line).
        :param assumption: Optional assumption value from a previous result, used
            to disambiguate (e.g. when "mercury" could mean the planet or element).
        :return: The text result from Wolfram Alpha.
        """
        app_id = (cfg.app_id or "").strip()
        if not app_id:
            return (
                "❌ Wolfram Alpha AppID is not configured. Set the WOLFRAM_APP_ID "
                "environment variable. Get a free AppID at https://developer.wolframalpha.com"
            )

        if not query or not query.strip():
            return "❌ Empty query. Provide a Wolfram Alpha query string."

        clean_query = query.strip().replace("\n", " ")

        params = {
            "input": clean_query,
            "appid": app_id,
            "maxchars": cfg.max_chars,
            "units": cfg.default_units,
        }
        if assumption:
            params["assumption"] = assumption

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    BASE_URL,
                    params=params,
                    headers={"User-Agent": "MCP-WolframAlpha/1.0"},
                )
        except httpx.TimeoutException:
            return "❌ Wolfram Alpha request timed out after 30s."
        except httpx.HTTPError as exc:
            return f"❌ Network error contacting Wolfram Alpha: {exc}"

        status = response.status_code
        body = response.text or ""

        # 501: Wolfram couldn't interpret the input. Body often has suggestions.
        if status == 501:
            return (
                f"Wolfram Alpha could not interpret the query: '{clean_query}'.\n"
                f"Suggestions from the API:\n{body.strip()}\n\n"
                "Try rephrasing as a simpler keyword-style query, or pick one of "
                "the suggested inputs above."
            )

        if status == 403:
            return (
                "❌ Wolfram Alpha rejected the AppID (HTTP 403). "
                "Check that WOLFRAM_APP_ID is set correctly."
            )

        body_snippet = body[:200]
        if status == 400:
            return f"❌ Wolfram Alpha rejected the request (HTTP 400): {body_snippet}"

        if status >= 400:
            return f"❌ Wolfram Alpha error (HTTP {status}): {body_snippet}"

        if not body.strip():
            return "❌ Wolfram Alpha returned an empty response."

        return body
