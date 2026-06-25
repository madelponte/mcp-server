"""Tests for tools/stock_data.py — helpers, provider resolution, dispatch."""

import json

import pytest
from fastmcp.exceptions import ToolError

import tools.stock_data as stock
from tools.cache import TTLCache
from tools.stock_data import (
    _safe_float,
    _safe_int,
    _ts_to_iso,
    _format_large_number,
    _clamp_amount,
    _resolve_provider,
    _is_special_symbol,
    _is_plain_ticker,
    _retrieval_error,
    _resolve_symbol,
    _fetch_section,
    _gather_sections,
    _scrub_secrets,
    _provider_error,
)
from conftest import run


# --------------------------- safe coercion ---------------------------

def test_safe_float():
    assert _safe_float("3.14") == 3.14
    assert _safe_float(2) == 2.0
    assert _safe_float(None) is None
    assert _safe_float("") is None
    assert _safe_float("abc") is None


def test_safe_int():
    assert _safe_int("5") == 5
    assert _safe_int(5.0) == 5
    assert _safe_int(None) is None
    assert _safe_int("") is None
    assert _safe_int("xyz") is None


def test_ts_to_iso():
    assert _ts_to_iso(0) is None  # falsy
    assert _ts_to_iso(None) is None
    out = _ts_to_iso(1_600_000_000)
    assert out is not None and out.startswith("2020-")


def test_ts_to_iso_invalid():
    assert _ts_to_iso("not-a-ts") is None


# --------------------------- _format_large_number ---------------------------

def test_format_large_number_scales():
    assert _format_large_number(3.2e12) == "3.20T"
    assert _format_large_number(5e9) == "5.00B"
    assert _format_large_number(2.5e6) == "2.50M"
    assert _format_large_number(1500) == "1.50K"
    assert _format_large_number(42) == "42.00"


def test_format_large_number_negative():
    assert _format_large_number(-2e9) == "-2.00B"


def test_format_large_number_none():
    assert _format_large_number(None) is None
    assert _format_large_number("bad") is None


# --------------------------- _clamp_amount ---------------------------

def test_clamp_amount_none_uses_max():
    assert _clamp_amount(None, 4) == 4


def test_clamp_amount_within_range():
    assert _clamp_amount(2, 4) == 2


def test_clamp_amount_above_max():
    assert _clamp_amount(100, 4) == 4


def test_clamp_amount_below_one():
    assert _clamp_amount(0, 4) == 1
    assert _clamp_amount(-5, 4) == 1


def test_clamp_amount_invalid_uses_max():
    assert _clamp_amount("x", 4) == 4


# --------------------------- _resolve_provider ---------------------------

def test_resolve_provider_passthrough():
    assert _resolve_provider("finnhub") == "finnhub"
    assert _resolve_provider("yfinance") == "yfinance"


def test_resolve_provider_auto_default_prefers_finnhub_when_keyed(monkeypatch):
    monkeypatch.setattr(stock.cfg, "finnhub_api_key", "key")
    assert _resolve_provider("auto") == "finnhub"


def test_resolve_provider_auto_default_no_key_uses_yfinance(monkeypatch):
    monkeypatch.setattr(stock.cfg, "finnhub_api_key", "")
    assert _resolve_provider("auto") == "yfinance"


def test_resolve_provider_auto_financials_prefers_fmp_when_keyed(monkeypatch):
    monkeypatch.setattr(stock.cfg, "fmp_api_key", "key")
    assert _resolve_provider("auto", for_financials=True) == "fmp"


def test_resolve_provider_auto_financials_no_key_uses_yfinance(monkeypatch):
    monkeypatch.setattr(stock.cfg, "fmp_api_key", "")
    assert _resolve_provider("auto", for_financials=True) == "yfinance"


# --------------------------- symbol shape predicates ---------------------------

@pytest.mark.parametrize("s", ["BTC-USD", "^GSPC", "EURUSD=X", "RY.TO"])
def test_is_special_symbol_true(s):
    assert _is_special_symbol(s) is True


@pytest.mark.parametrize("s", ["AAPL", "Apple Inc", "", "MSFT"])
def test_is_special_symbol_false(s):
    assert _is_special_symbol(s) is False


@pytest.mark.parametrize("s", ["AAPL", "MSFT", "BRKB"])
def test_is_plain_ticker_true(s):
    assert _is_plain_ticker(s) is True


@pytest.mark.parametrize("s", ["Apple Inc", "TOOLONGTICKER", "BTC-USD", ""])
def test_is_plain_ticker_false(s):
    assert _is_plain_ticker(s) is False


# --------------------------- _retrieval_error ---------------------------

def test_retrieval_error_with_errors():
    msg = _retrieval_error("quote", "AAPL", ["finnhub: boom", "yfinance: nope"])
    assert "AAPL" in msg
    assert "finnhub: boom" in msg


def test_retrieval_error_without_errors():
    msg = _retrieval_error("quote", "AAPL", [])
    assert msg == "Could not retrieve quote for AAPL."


# --------------------------- secret redaction ---------------------------

def test_scrub_secrets_redacts_fmp_apikey():
    raw = (
        "402 Client Error: Payment Required for url: "
        "https://financialmodelingprep.com/stable/income-statement"
        "?symbol=BTC-USD&limit=4&apikey=SUPERSECRETKEY"
    )
    scrubbed = _scrub_secrets(raw)
    assert "SUPERSECRETKEY" not in scrubbed
    assert "apikey=REDACTED" in scrubbed
    # Non-secret query params are preserved for diagnostics.
    assert "symbol=BTC-USD" in scrubbed


def test_scrub_secrets_redacts_finnhub_token():
    raw = "HTTPError for url: https://finnhub.io/api/v1/quote?symbol=AAPL&token=abc123"
    scrubbed = _scrub_secrets(raw)
    assert "abc123" not in scrubbed
    assert "token=REDACTED" in scrubbed


def test_provider_error_redacts_embedded_key():
    exc = Exception("boom for url: https://x/stable/q?apikey=LEAK&api_key=ALSOLEAK")
    out = _provider_error("fmp", exc)
    assert "LEAK" not in out and "ALSOLEAK" not in out
    assert out.startswith("fmp: Exception:")


# --------------------------- _resolve_symbol ---------------------------

def test_resolve_symbol_special_passthrough():
    assert _resolve_symbol("btc-usd") == ("BTC-USD", None)


def test_resolve_symbol_empty_raises():
    with pytest.raises(ToolError):
        _resolve_symbol("  ")


def test_resolve_symbol_exact_ticker_short_circuits(monkeypatch):
    monkeypatch.setattr(
        stock, "_search_symbols",
        lambda q, lim: ([{"symbol": "AAPL", "description": "Apple Inc"}], [], "finnhub"),
    )
    assert _resolve_symbol("AAPL") == ("AAPL", None)


def test_resolve_symbol_name_resolves_to_first_hit(monkeypatch):
    results = [
        {"symbol": "AAPL", "description": "Apple Inc", "type": "Common Stock"},
        {"symbol": "APLE", "description": "Apple Hospitality", "type": "REIT"},
    ]
    monkeypatch.setattr(stock, "_search_symbols", lambda q, lim: (results, [], "finnhub"))
    symbol, match = _resolve_symbol("Apple")
    assert symbol == "AAPL"
    assert match["query"] == "Apple"
    assert match["matched"]["symbol"] == "AAPL"
    assert match["alternatives"][0]["symbol"] == "APLE"


def test_resolve_symbol_provider_down_but_ticker_shaped(monkeypatch):
    monkeypatch.setattr(stock, "_search_symbols", lambda q, lim: ([], ["finnhub: down"], None))
    assert _resolve_symbol("AAPL") == ("AAPL", None)


def test_resolve_symbol_provider_down_and_name_raises(monkeypatch):
    monkeypatch.setattr(stock, "_search_symbols", lambda q, lim: ([], ["finnhub: down"], None))
    with pytest.raises(ToolError):
        _resolve_symbol("Some Company Name")


def test_resolve_symbol_no_results_name_raises(monkeypatch):
    monkeypatch.setattr(stock, "_search_symbols", lambda q, lim: ([], [], "finnhub"))
    with pytest.raises(ToolError):
        _resolve_symbol("Nonexistent Company")


def test_resolve_symbol_no_results_ticker_passthrough(monkeypatch):
    monkeypatch.setattr(stock, "_search_symbols", lambda q, lim: ([], [], "finnhub"))
    assert _resolve_symbol("ZZZZ") == ("ZZZZ", None)


# --------------------------- _fetch_section ---------------------------

def test_fetch_section_primary_success():
    result, errors = _fetch_section(
        "finnhub",
        {"finnhub": lambda s: {"ok": True, "provider": "finnhub"}},
        lambda s: {"ok": True, "provider": "yfinance"},
        ("AAPL",),
    )
    assert result["provider"] == "finnhub"
    assert errors == []


def test_fetch_section_provider_not_in_map_uses_yfinance():
    result, errors = _fetch_section(
        "fmp",  # not in the primary map
        {"finnhub": lambda s: {"provider": "finnhub"}},
        lambda s: {"provider": "yfinance"},
        ("AAPL",),
    )
    assert result["provider"] == "yfinance"


def test_fetch_section_primary_none_falls_back(monkeypatch):
    monkeypatch.setattr(stock.cfg, "prefer_yfinance_fallback", True)
    result, errors = _fetch_section(
        "finnhub",
        {"finnhub": lambda s: None},  # primary returns nothing
        lambda s: {"provider": "yfinance"},
        ("AAPL",),
    )
    assert result["provider"] == "yfinance"


def test_fetch_section_primary_raises_then_falls_back(monkeypatch):
    monkeypatch.setattr(stock.cfg, "prefer_yfinance_fallback", True)

    def boom(s):
        raise RuntimeError("provider exploded")

    result, errors = _fetch_section(
        "finnhub",
        {"finnhub": boom},
        lambda s: {"provider": "yfinance"},
        ("AAPL",),
    )
    assert result["provider"] == "yfinance"
    assert any("finnhub" in e for e in errors)


# --------------------------- provider parsing (mocked HTTP) ---------------------------

def test_finnhub_quote_parses(monkeypatch):
    monkeypatch.setattr(stock.cfg, "finnhub_api_key", "key")
    monkeypatch.setattr(
        stock, "_http_get_json",
        lambda url, params=None: {"c": 100.0, "pc": 90.0, "o": 95.0, "h": 101.0, "l": 94.0, "t": 1_600_000_000},
    )
    out = stock._finnhub_quote("AAPL")
    assert out["price"] == 100.0
    assert out["previous_close"] == 90.0
    assert out["change"] == 10.0
    assert round(out["change_percent"], 2) == 11.11


def test_finnhub_quote_all_zeros_returns_none(monkeypatch):
    monkeypatch.setattr(stock.cfg, "finnhub_api_key", "key")
    monkeypatch.setattr(
        stock, "_http_get_json",
        lambda url, params=None: {"c": 0, "o": 0, "h": 0},
    )
    assert stock._finnhub_quote("AAPL") is None


def test_fmp_quote_parses_list(monkeypatch):
    monkeypatch.setattr(stock.cfg, "fmp_api_key", "key")
    monkeypatch.setattr(
        stock, "_http_get_json",
        lambda url, params=None: [{"name": "Apple", "price": 150.0, "change": 1.0,
                                   "marketCap": 3e12, "volume": 1000}],
    )
    out = stock._fmp_quote("AAPL")
    assert out["price"] == 150.0
    assert out["market_cap_formatted"] == "3.00T"


def test_fmp_quote_empty_returns_none(monkeypatch):
    monkeypatch.setattr(stock.cfg, "fmp_api_key", "key")
    monkeypatch.setattr(stock, "_http_get_json", lambda url, params=None: [])
    assert stock._fmp_quote("AAPL") is None


# --------------------------- _finnhub_peers ---------------------------

def test_finnhub_peers_drops_self_dedupes_and_caps(monkeypatch):
    monkeypatch.setattr(stock.cfg, "finnhub_api_key", "key")
    monkeypatch.setattr(
        stock, "_http_get_json",
        lambda url, params=None: ["AAPL", "MSFT", "msft", "GOOGL", "AMZN"],
    )
    out = stock._finnhub_peers("AAPL", limit=2)
    # Self ("AAPL") dropped, case-insensitive de-dupe of MSFT, capped to 2.
    assert out["peers"] == ["MSFT", "GOOGL"]
    assert out["provider"] == "finnhub"


def test_finnhub_peers_only_self_returns_none(monkeypatch):
    monkeypatch.setattr(stock.cfg, "finnhub_api_key", "key")
    monkeypatch.setattr(stock, "_http_get_json", lambda url, params=None: ["AAPL"])
    assert stock._finnhub_peers("AAPL", limit=10) is None


def test_finnhub_peers_empty_returns_none(monkeypatch):
    monkeypatch.setattr(stock.cfg, "finnhub_api_key", "key")
    monkeypatch.setattr(stock, "_http_get_json", lambda url, params=None: [])
    assert stock._finnhub_peers("AAPL", limit=10) is None


# --------------------------- _http_get_json caching ---------------------------

def test_http_get_json_caches(monkeypatch):
    monkeypatch.setattr(stock, "_cache", TTLCache(60))
    calls = {"n": 0}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def fake_get(url, params=None, timeout=None, headers=None):
        calls["n"] += 1
        return FakeResp()

    monkeypatch.setattr(stock._session, "get", fake_get)
    a = stock._http_get_json("https://x.com/api", {"q": 1})
    b = stock._http_get_json("https://x.com/api", {"q": 1})
    assert a == b == {"ok": True}
    assert calls["n"] == 1  # second call served from cache


# --------------------------- _yfinance_dividends / _yfinance_ownership ---------------------------

def test_yfinance_dividends_parses_and_orders(monkeypatch):
    import pandas as pd

    class FakeTicker:
        dividends = pd.Series(
            [0.22, 0.23, 0.24],
            index=pd.to_datetime(["2024-02-01", "2024-05-01", "2024-08-01"]),
        )
        splits = pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))

    monkeypatch.setattr(stock, "_yfinance_ticker", lambda s: FakeTicker())
    out = stock._yfinance_dividends("AAPL", max_events=24)
    # Most recent dividend first.
    assert out["dividends"][0]["date"] == "2024-08-01"
    assert out["dividends"][0]["amount"] == 0.24
    assert out["dividend_count"] == 3
    assert out["splits"][0]["ratio"] == 4.0


def test_yfinance_dividends_caps_events(monkeypatch):
    import pandas as pd

    idx = pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 11)])

    class FakeTicker:
        dividends = pd.Series(range(1, 11), index=idx).astype(float)
        splits = pd.Series(dtype=float)

    monkeypatch.setattr(stock, "_yfinance_ticker", lambda s: FakeTicker())
    out = stock._yfinance_dividends("AAPL", max_events=3)
    assert len(out["dividends"]) == 3


def test_yfinance_dividends_none_when_empty(monkeypatch):
    import pandas as pd

    class FakeTicker:
        dividends = pd.Series(dtype=float)
        splits = pd.Series(dtype=float)

    monkeypatch.setattr(stock, "_yfinance_ticker", lambda s: FakeTicker())
    assert stock._yfinance_dividends("AAPL", max_events=24) is None


def test_yfinance_ownership_parses(monkeypatch):
    import pandas as pd

    major = pd.DataFrame(
        {"Value": [0.0007, 0.62]},
        index=["insidersPercentHeld", "institutionsPercentHeld"],
    )
    inst = pd.DataFrame([
        {"Date Reported": pd.Timestamp("2024-03-31"), "Holder": "Vanguard",
         "Shares": 1234, "pctHeld": 0.08, "Value": 5000},
    ])

    class FakeTicker:
        major_holders = major
        institutional_holders = inst

    monkeypatch.setattr(stock, "_yfinance_ticker", lambda s: FakeTicker())
    out = stock._yfinance_ownership("AAPL", max_holders=10)
    assert out["ownership_summary"]["institutionsPercentHeld"] == 0.62
    assert out["institutional_holders"][0]["holder"] == "Vanguard"
    assert out["institutional_holders"][0]["date_reported"] == "2024-03-31"


def test_yfinance_ownership_none_when_empty(monkeypatch):
    import pandas as pd

    class FakeTicker:
        major_holders = pd.DataFrame()
        institutional_holders = pd.DataFrame()

    monkeypatch.setattr(stock, "_yfinance_ticker", lambda s: FakeTicker())
    assert stock._yfinance_ownership("AAPL", max_holders=10) is None


def test_new_sections_are_valid():
    for sec in ("peers", "dividends", "ownership"):
        assert sec in stock.VALID_SECTIONS
        assert sec in stock._SECTION_FETCHERS


# --------------------------- _gather_sections (async) ---------------------------

def test_gather_sections_clamps_and_collects(monkeypatch):
    captured = {}

    def fake_quote(symbol, opts):
        captured["opts"] = opts
        return {"provider": "x", "symbol": symbol}, []

    def fake_news(symbol, opts):
        return None, ["finnhub: empty"]

    monkeypatch.setitem(stock._SECTION_FETCHERS, "quote", fake_quote)
    monkeypatch.setitem(stock._SECTION_FETCHERS, "news", fake_news)

    data, errors = run(_gather_sections(
        "AAPL", ["quote", "news"], "income", "annual",
        periods=9999, news_items=9999, insider_weeks=None,
        history_bars=None, news_days=None, history_interval="1d",
    ))
    assert "quote" in data
    assert errors["news"] == ["finnhub: empty"]
    # periods clamped to the configured financial cap.
    assert captured["opts"]["financial_periods"] == stock.cfg.max_financial_periods
    assert captured["opts"]["news_items"] == stock.cfg.max_news_items


# --------------------------- get_company_data tool (validation) ---------------------------

def test_get_company_data_invalid_section_raises(tool_fns):
    fn = tool_fns["get_company_data"]
    with pytest.raises(ToolError) as exc:
        run(fn(symbol="AAPL", sections=["bogus"]))
    assert "Invalid section" in str(exc.value)


def test_get_company_data_invalid_statement_raises(tool_fns):
    fn = tool_fns["get_company_data"]
    with pytest.raises(ToolError):
        run(fn(symbol="AAPL", sections=["financials"], statement="bogus"))


def test_get_company_data_empty_symbol_raises(tool_fns):
    fn = tool_fns["get_company_data"]
    with pytest.raises(ToolError):
        run(fn(symbol="   "))


def test_get_company_data_single_happy_path(monkeypatch, tool_fns):
    async def fake_fetch_company(raw_query, *args):
        return {"symbol": raw_query.upper(), "sections": ["quote"],
                "data": {"quote": {"price": 1.0}}}

    monkeypatch.setattr(stock, "_fetch_company", fake_fetch_company)
    fn = tool_fns["get_company_data"]
    out = json.loads(run(fn(symbol="AAPL", sections=["quote"])))
    assert out["symbol"] == "AAPL"
    assert out["data"]["quote"]["price"] == 1.0


def test_get_company_data_decodes_json_string_list(monkeypatch, tool_fns):
    async def fake_fetch_company(raw_query, *args):
        return {"symbol": raw_query.upper(), "sections": ["quote"], "data": {"quote": {}}}

    monkeypatch.setattr(stock, "_fetch_company", fake_fetch_company)
    fn = tool_fns["get_company_data"]
    # A list passed as a JSON-encoded string should be decoded into a batch.
    out = json.loads(run(fn(symbol='["AAPL","MSFT"]', sections=["quote"])))
    assert "results" in out
    symbols = {r["symbol"] for r in out["results"]}
    assert symbols == {"AAPL", "MSFT"}


def test_get_company_data_skips_beyond_max_symbols(monkeypatch, tool_fns):
    monkeypatch.setattr(stock.cfg, "max_symbols", 1)

    async def fake_fetch_company(raw_query, *args):
        return {"symbol": raw_query.upper(), "sections": ["quote"], "data": {"quote": {}}}

    monkeypatch.setattr(stock, "_fetch_company", fake_fetch_company)
    fn = tool_fns["get_company_data"]
    out = json.loads(run(fn(symbol=["AAPL", "MSFT"], sections=["quote"])))
    assert out["results"][0]["symbol"] == "AAPL"
    # MSFT skipped past the per-call limit.
    assert any("Skipped" in r.get("error", "") for r in out["results"])
    assert "note" in out
