"""
Wolfram Alpha MCP tool.

Runs computations and looks up factual data via the Wolfram Alpha LLM API.
Translated from the Open WebUI tool — the HTML "card" rendering was dropped.
The LLM API answers as a flat block of labeled-section plaintext; we parse that
into the same compact JSON envelope the other tools use (see `_structure_result`)
so a model gets one consistent shape and the relevant values aren't buried in
boilerplate (the echoed query, image URLs, the verbose "Assuming …" lines).
"""

import logging
import re
import asyncio
from typing import Any, Literal

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from config import wolfram_settings as cfg
from .cache import TTLCache
from .serialize import to_json, log_call, log_result, redact_secrets

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

# The Wolfram endpoint is often queried in short bursts. Reuse a loop-local
# AsyncClient so repeated calls keep the connection pool warm. Include the
# current AsyncClient class id for test isolation when httpx is monkeypatched.
_http_clients: dict[tuple[int, int], httpx.AsyncClient] = {}


def _http_client() -> httpx.AsyncClient:
    key = (id(asyncio.get_running_loop()), id(httpx.AsyncClient))
    client = _http_clients.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient()
        _http_clients[key] = client
    return client


async def close_clients() -> None:
    """Close every pooled Wolfram client and drop the pool.

    Called from the server's lifespan shutdown (``server.run_http``); see
    ``close_clients`` in ``web_fetch`` for the rationale.
    """
    for client in list(_http_clients.values()):
        try:
            await client.aclose()
        except Exception:
            log.exception("Failed to close a shared Wolfram client.")
    _http_clients.clear()


# --------------------------- Response structuring ---------------------------
#
# The LLM API returns blank-line-separated blocks, each a "Label:" line followed
# by its content. We turn that into JSON, dropping the parts a text model can't
# use: the echoed query, image/render-directive lines, and the multi-line
# "Assuming … / To use as … set assumption=…" boilerplate (kept, but folded into
# a compact structured `assumptions` field only when the query was ambiguous).

# Lines that are pure rendering artifacts — image URLs and Wolfram Language
# render directives — carry nothing a text model can use.
_NOISE_LINE_RE = re.compile(r"^\s*(?:image:|Wolfram Language code:)", re.IGNORECASE)
# The trailing "Wolfram|Alpha website result for …:" section is just a permalink.
_URL_LABEL_RE = re.compile(r"^Wolfram\|Alpha website result", re.IGNORECASE)
_ASSUMING_RE = re.compile(r'^Assuming\b.*?\bis\s+(.+)$', re.IGNORECASE)
_USE_AS_RE = re.compile(r'^To use as\s+(.+?)\s+set assumption=(\S+)', re.IGNORECASE)


def _clean_body(text: str) -> str:
    """Drop image/render-directive lines from a section body."""
    kept = [ln for ln in text.split("\n") if ln.strip() and not _NOISE_LINE_RE.match(ln)]
    return "\n".join(kept).strip()


def _parse_assumptions(body: str) -> dict | None:
    """Structure an 'Assumption(s)' block into ``{used, alternatives}``, or None.

    The first line is ``Assuming "X" is <used>``; each following line is
    ``To use as <description> set assumption=<value>``. Pulling these out of the
    answer flow is the "suppress the boilerplate when an answer is available"
    fix: the chosen interpretation stays as a one-word ``used`` hint and the
    alternatives become structured retry values instead of six prose lines.

    Returns None when there are no ``alternatives`` — Wolfram emits an Assumption
    block even for unambiguous queries (just the chosen interpretation, nothing to
    switch to), and a ``used``-only object is non-actionable metadata bloat. The
    field is therefore present only when the query is genuinely ambiguous, i.e.
    there's an alternative to retry with (matching the tool's documented contract).
    """
    used: str | None = None
    alternatives: list[dict] = []
    for ln in body.split("\n"):
        ln = ln.strip()
        m = _ASSUMING_RE.match(ln)
        if m:
            used = m.group(1).strip()
            continue
        m = _USE_AS_RE.match(ln)
        if m:
            alternatives.append({"description": m.group(1).strip(), "assumption": m.group(2).strip()})
    if not alternatives:
        return None
    out: dict[str, Any] = {}
    if used:
        out["used"] = used
    out["alternatives"] = alternatives
    return out


def _structure_result(body: str, query: str) -> dict:
    """Parse the LLM API's labeled-section text into a compact JSON payload:
    ``{query, data:{label: value, …}, assumptions?, url?}``. Falls back to a
    single ``result`` field holding the cleaned text if no sections parse, so a
    short or unexpectedly-shaped answer is never reduced to nothing."""
    data: dict[str, str] = {}
    assumptions: dict | None = None
    url: str | None = None
    last_key: str | None = None  # section an unlabeled block continues

    for block in re.split(r"\n[ \t]*\n", body.strip()):
        block = block.strip()
        if not block:
            continue
        first, _, rest = block.partition("\n")
        if first.rstrip().endswith(":"):
            label, content = first.rstrip()[:-1].strip(), rest
        else:
            label, content = "", block

        low = label.lower()
        if low == "query":
            continue  # the caller already knows what it asked
        if _URL_LABEL_RE.match(label):
            url = content.strip() or url
            continue
        if low in ("assumption", "assumptions"):
            assumptions = _parse_assumptions(content)
            continue

        content = _clean_body(content)
        if not content:
            continue  # section was image-only boilerplate
        if not label and last_key is not None:
            # An unlabeled block continues the previous section (e.g. the
            # alternate-unit forms Wolfram lists under "Value"), rather than
            # being its own entry.
            data[last_key] += "\n" + content
            continue
        key = label or "answer"
        while key in data:  # a genuinely repeated label
            key += " (cont.)"
        data[key] = content
        last_key = key

    payload: dict[str, Any] = {"query": query}
    if assumptions:
        payload["assumptions"] = assumptions
    if data:
        payload["data"] = data
    else:
        cleaned = _clean_body(body)
        payload["result"] = cleaned or body.strip()
    if url:
        payload["url"] = url
    return payload


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def query_wolfram_alpha(
        query: str,
        assumption: str | None = None,
        units: Literal["metric", "nonmetric"] | None = None,
    ) -> str:
        """Compute/lookup exact facts: math, units, physics, chemistry, finance,
        geography, demographics, nutrition, weather, astronomy.

        Query: keyword style only ("France population", "convert 5 miles to km",
        "derivative of x^2", "speed of light"). No sentences. Math: use * for
        multiply, ^ for exponent (6*10^14, not 6e14). One property per call.
        If the result has an "assumptions" field, the query was ambiguous: retry
        the SAME query with assumption=<value> from assumptions.alternatives.
        units="metric"(km, kg, °C) or "nonmetric"(miles, lb, °F) sets the unit
        system for the answer — set it to match what the question asks for.
        NOT for opinions, news, code, or known facts.

        :param query: Keyword query.
        :param assumption: Disambiguation value from a prior result's
            assumptions.alternatives (if any).
        :param units: "metric" or "nonmetric" (default: server config).
        :return: JSON {query, data:{section: value, ...}, assumptions?, url?}.
            The answer is the first entry in data; assumptions appears only when
            the query was ambiguous.
        """
        log_call(
            log, "query_wolfram_alpha", query=query, assumption=assumption, units=units
        )
        app_id = (cfg.app_id or "").strip()
        if not app_id:
            raise ToolError(
                "Wolfram Alpha AppID is not configured. Set the WOLFRAM_APP_ID "
                "environment variable. Get a free AppID at https://developer.wolframalpha.com"
            )

        if not query or not query.strip():
            raise ToolError("Empty query. Provide a Wolfram Alpha query string.")

        clean_query = query.strip().replace("\n", " ")

        # Units are per-question, not per-deployment (a US user asking for a
        # distance wants miles), so the tool param overrides the configured
        # default when supplied.
        if units is not None and units not in ("metric", "nonmetric"):
            raise ToolError("units must be 'metric' or 'nonmetric'.")
        resolved_units = units or cfg.default_units

        params = {
            "input": clean_query,
            "appid": app_id,
            "maxchars": cfg.max_chars,
            "units": resolved_units,
        }
        if assumption:
            params["assumption"] = assumption

        # Key on everything that affects the result (but never the AppID). A
        # null byte separates fields so values can't collide across boundaries.
        cache_key = "\x00".join(
            (clean_query, assumption or "", resolved_units, str(cfg.max_chars))
        )
        cached = _result_cache.get(cache_key)
        if cached is not None:
            return log_result(log, "query_wolfram_alpha", cached)

        try:
            client = _http_client()
            response = await client.get(
                BASE_URL,
                params=params,
                headers={"User-Agent": "MCP-WolframAlpha/1.0"},
                timeout=cfg.http_timeout_seconds,
            )
        except httpx.TimeoutException:
            raise ToolError(
                f"Wolfram Alpha request timed out after {cfg.http_timeout_seconds}s."
            )
        except httpx.HTTPError as exc:
            detail = redact_secrets(exc, cfg.app_id)
            raise ToolError(f"Network error contacting Wolfram Alpha: {detail}")

        status = response.status_code
        body = response.text or ""

        # 501: Wolfram couldn't interpret the input. No answer was produced, so
        # this is an error — but the body often carries useful suggestions, so
        # surface them in the error message for the model to retry with.
        if status == 501:
            raise ToolError(
                f"Wolfram Alpha could not interpret the query: '{clean_query}'.\n"
                f"Suggestions from the API:\n{redact_secrets(body.strip(), app_id)}\n\n"
                "Try rephrasing as a simpler keyword-style query, or pick one of "
                "the suggested inputs above."
            )

        if status == 403:
            raise ToolError(
                "Wolfram Alpha rejected the AppID (HTTP 403). "
                "Check that WOLFRAM_APP_ID is set correctly."
            )

        body_snippet = redact_secrets(body[:200], app_id)
        if status == 400:
            raise ToolError(f"Wolfram Alpha rejected the request (HTTP 400): {body_snippet}")

        if status >= 400:
            raise ToolError(f"Wolfram Alpha error (HTTP {status}): {body_snippet}")

        if not body.strip():
            raise ToolError("Wolfram Alpha returned an empty response.")

        result_json = to_json(_structure_result(body, clean_query))
        _result_cache.set(cache_key, result_json)
        return log_result(log, "query_wolfram_alpha", result_json)
