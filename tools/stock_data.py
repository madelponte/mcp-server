"""
Stock Data MCP tool.

Exposes a single tool:

- ``get_company_data(symbol, sections=[...])`` — a single entry point for
  per-company data. ``symbol`` accepts a ticker (``AAPL``) *or* a company name
  (``Apple``); a name is resolved to its ticker via symbol search before any
  data is fetched, so the model never has to make a separate lookup call. The
  caller chooses which sections to fetch: ``quote``, ``profile``,
  ``financials``, ``earnings``, ``news``, ``insiders``, ``price_history``,
  ``sector_comparison`` (the last compares the company's P/E to its sector and
  industry averages and is FMP-only).

Uses Finnhub (primary, free API key), yfinance (no-key fallback), and optionally
Financial Modeling Prep for deep financial statements. Translated from the Open
WebUI tool; per-user valves and status emitters were removed.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

import anyio
import requests
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from config import stock_settings as cfg
from .cache import TTLCache
from .serialize import to_json, log_call, log_result

log = logging.getLogger(__name__)

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake the failure for
# market data. See the README "Error handling" section.

# -------------------------- Helpers --------------------------

def _safe_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _ts_to_iso(ts: int | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _format_large_number(n: float | None) -> str | None:
    """Render large numbers like market cap in human-readable form."""
    if n is None:
        return None
    try:
        n = float(n)
    except (TypeError, ValueError):
        return None
    abs_n = abs(n)
    if abs_n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if abs_n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs_n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs_n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.2f}"


def _retrieval_error(what: str, symbol: str, errors: list[str]) -> str:
    """Build a ToolError message for a failed data retrieval, keeping the
    per-provider error detail (otherwise lost when we raise instead of return)."""
    msg = f"Could not retrieve {what} for {symbol}."
    if errors:
        msg += " Provider errors: " + "; ".join(errors)
    return msg


def _clamp_amount(requested: int | None, maximum: int) -> int:
    """Resolve a model-requested range/count against its configured maximum.

    ``None`` (the model didn't ask) yields ``maximum``, preserving the old
    fixed-amount behavior. Otherwise the value is clamped to ``[1, maximum]`` so
    the model can dial the amount down but never request more than the cap — the
    guard that keeps an oversized response from overwhelming its context window.
    """
    if requested is None:
        return maximum
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        return maximum
    if requested < 1:
        return 1
    return min(requested, maximum)


# -------------------------- Cache + HTTP --------------------------

# Unbounded (max_entries=0) to preserve the original behavior; quote/profile
# responses are small and the TTL is short.
_cache = TTLCache(cfg.cache_ttl_seconds)


def _http_get_json(url: str, params: dict | None = None) -> Any:
    cache_key = f"GET::{url}::{json.dumps(params or {}, sort_keys=True)}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    resp = requests.get(
        url,
        params=params,
        timeout=cfg.request_timeout,
        headers={"User-Agent": "MCP-StockDataTool/1.0"},
    )
    resp.raise_for_status()
    data = resp.json()
    _cache.set(cache_key, data)
    return data


# -------------------------- Provider resolution --------------------------

def _resolve_provider(requested: str, *, for_financials: bool = False) -> str:
    """Resolve 'auto' to a concrete provider based on configured keys."""
    if requested != "auto":
        return requested
    if for_financials:
        if cfg.fmp_api_key:
            return "fmp"
        return "yfinance"
    if cfg.finnhub_api_key:
        return "finnhub"
    return "yfinance"


# ===================================================================
#                       PROVIDER: FINNHUB
# ===================================================================

def _finnhub_require_key() -> str:
    if not cfg.finnhub_api_key:
        raise RuntimeError("Finnhub API key not configured.")
    return cfg.finnhub_api_key


def _finnhub_quote(symbol: str) -> dict | None:
    token = _finnhub_require_key()
    data = _http_get_json("https://finnhub.io/api/v1/quote", {"symbol": symbol, "token": token})
    if not data or all(v in (0, None) for v in (data.get("c"), data.get("o"), data.get("h"))):
        return None
    current = _safe_float(data.get("c"))
    prev_close = _safe_float(data.get("pc"))
    change = (current - prev_close) if (current is not None and prev_close is not None) else None
    change_pct = ((change / prev_close) * 100) if (change is not None and prev_close) else None
    return {
        "provider": "finnhub",
        "symbol": symbol,
        "price": current,
        "change": round(change, 4) if change is not None else None,
        "change_percent": round(change_pct, 4) if change_pct is not None else None,
        "open": _safe_float(data.get("o")),
        "high": _safe_float(data.get("h")),
        "low": _safe_float(data.get("l")),
        "previous_close": prev_close,
        "timestamp": _ts_to_iso(data.get("t")),
    }


def _finnhub_profile(symbol: str) -> dict | None:
    token = _finnhub_require_key()
    profile = _http_get_json(
        "https://finnhub.io/api/v1/stock/profile2", {"symbol": symbol, "token": token}
    )
    if not profile:
        return None

    metrics = {}
    try:
        metrics_data = _http_get_json(
            "https://finnhub.io/api/v1/stock/metric",
            {"symbol": symbol, "metric": "all", "token": token},
        )
        metrics = (metrics_data or {}).get("metric") or {}
    except Exception:
        metrics = {}

    market_cap_m = _safe_float(profile.get("marketCapitalization"))
    market_cap = market_cap_m * 1_000_000 if market_cap_m else None

    return {
        "provider": "finnhub",
        "symbol": symbol,
        "name": profile.get("name"),
        "exchange": profile.get("exchange"),
        "country": profile.get("country"),
        "currency": profile.get("currency"),
        "industry": profile.get("finnhubIndustry"),
        "ipo_date": profile.get("ipo"),
        "logo": profile.get("logo"),
        "weburl": profile.get("weburl"),
        "phone": profile.get("phone"),
        "share_outstanding_millions": _safe_float(profile.get("shareOutstanding")),
        "market_cap": market_cap,
        "market_cap_formatted": _format_large_number(market_cap),
        "key_metrics": {
            "pe_ttm": _safe_float(metrics.get("peTTM")),
            "ps_ttm": _safe_float(metrics.get("psTTM")),
            "pb_ratio": _safe_float(metrics.get("pbAnnual")),
            "eps_ttm": _safe_float(metrics.get("epsTTM")),
            "dividend_yield_ttm_percent": _safe_float(metrics.get("dividendYieldIndicatedAnnual")),
            "beta": _safe_float(metrics.get("beta")),
            "52_week_high": _safe_float(metrics.get("52WeekHigh")),
            "52_week_low": _safe_float(metrics.get("52WeekLow")),
            "52_week_price_return_daily": _safe_float(metrics.get("52WeekPriceReturnDaily")),
            "roe_ttm": _safe_float(metrics.get("roeTTM")),
            "roa_ttm": _safe_float(metrics.get("roaTTM")),
            "current_ratio_annual": _safe_float(metrics.get("currentRatioAnnual")),
            "debt_to_equity_annual": _safe_float(metrics.get("totalDebt/totalEquityAnnual")),
            "gross_margin_ttm_percent": _safe_float(metrics.get("grossMarginTTM")),
            "operating_margin_ttm_percent": _safe_float(metrics.get("operatingMarginTTM")),
            "net_margin_ttm_percent": _safe_float(metrics.get("netProfitMarginTTM")),
        },
    }


def _finnhub_financials(symbol: str, statement: str, period: str, limit: int) -> dict | None:
    token = _finnhub_require_key()
    freq = "annual" if period == "annual" else "quarterly"
    data = _http_get_json(
        "https://finnhub.io/api/v1/stock/financials-reported",
        {"symbol": symbol, "freq": freq, "token": token},
    )
    if not data or not data.get("data"):
        return None

    statement_key = {"income": "ic", "balance": "bs", "cashflow": "cf"}[statement]

    periods = []
    for entry in data["data"][:limit]:
        report = (entry.get("report") or {}).get(statement_key) or []
        simplified = {item.get("label") or item.get("concept"): item.get("value") for item in report if item}
        periods.append({
            "period_end": entry.get("endDate"),
            "year": entry.get("year"),
            "quarter": entry.get("quarter"),
            "form": entry.get("form"),
            "data": simplified,
        })

    return {
        "provider": "finnhub",
        "symbol": symbol,
        "statement": statement,
        "period": period,
        "periods": periods,
    }


def _finnhub_earnings(symbol: str, limit: int) -> dict | None:
    token = _finnhub_require_key()
    data = _http_get_json(
        "https://finnhub.io/api/v1/stock/earnings", {"symbol": symbol, "token": token}
    )
    if not data:
        return None

    rows = []
    for row in data[:limit]:
        rows.append({
            "period": row.get("period"),
            "year": row.get("year"),
            "quarter": row.get("quarter"),
            "actual_eps": _safe_float(row.get("actual")),
            "estimated_eps": _safe_float(row.get("estimate")),
            "surprise": _safe_float(row.get("surprise")),
            "surprise_percent": _safe_float(row.get("surprisePercent")),
        })

    return {"provider": "finnhub", "symbol": symbol, "earnings": rows}


def _finnhub_news(symbol: str, limit: int, days: int = 7) -> dict | None:
    token = _finnhub_require_key()
    from datetime import date, timedelta
    today = date.today()
    from_date = (today - timedelta(days=days)).isoformat()
    to_date = today.isoformat()
    data = _http_get_json(
        "https://finnhub.io/api/v1/company-news",
        {"symbol": symbol, "from": from_date, "to": to_date, "token": token},
    )
    if not data:
        return None

    articles = []
    for item in data[:limit]:
        articles.append({
            "headline": item.get("headline"),
            "source": item.get("source"),
            "summary": (item.get("summary") or "")[:500],
            "url": item.get("url"),
            "published": _ts_to_iso(item.get("datetime")),
            "category": item.get("category"),
        })
    return {
        "provider": "finnhub",
        "symbol": symbol,
        "from_date": from_date,
        "to_date": to_date,
        "count": len(articles),
        "articles": articles,
    }


# SEC Form 4 transaction codes -> human-readable label.
_INSIDER_TX_CODES = {
    "P": "Open market purchase",
    "S": "Open market sale",
    "A": "Grant/award",
    "M": "Option exercise",
    "X": "Option exercise",
    "G": "Gift",
    "F": "Tax withholding",
    "C": "Conversion of derivative",
    "D": "Disposition to issuer",
    "W": "Acquisition/disposition by will",
}


def _finnhub_insider_transactions(symbol: str, weeks: int) -> dict | None:
    token = _finnhub_require_key()
    from datetime import date, timedelta
    today = date.today()
    from_date = (today - timedelta(weeks=weeks)).isoformat()
    to_date = today.isoformat()
    data = _http_get_json(
        "https://finnhub.io/api/v1/stock/insider-transactions",
        {"symbol": symbol, "from": from_date, "to": to_date, "token": token},
    )
    rows = (data or {}).get("data")
    if not rows:
        return None

    transactions = []
    shares_bought = shares_sold = 0
    buy_count = sell_count = 0
    for row in rows:
        change = _safe_int(row.get("change")) or 0
        if change > 0:
            direction = "buy"
            buy_count += 1
            shares_bought += change
        elif change < 0:
            direction = "sell"
            sell_count += 1
            shares_sold += -change
        else:
            direction = "neutral"
        code = row.get("transactionCode")
        transactions.append({
            "name": row.get("name"),
            "transaction_date": row.get("transactionDate"),
            "filing_date": row.get("filingDate"),
            "direction": direction,
            "transaction_code": code,
            "transaction_type": _INSIDER_TX_CODES.get(code, "Other"),
            "share_change": change,
            "shares_held_after": _safe_int(row.get("share")),
            "price": _safe_float(row.get("transactionPrice")),
        })

    # Most recent transactions first.
    transactions.sort(key=lambda t: t.get("transaction_date") or "", reverse=True)

    return {
        "provider": "finnhub",
        "symbol": symbol,
        "from_date": from_date,
        "to_date": to_date,
        "lookback_weeks": weeks,
        "summary": {
            "total_transactions": len(transactions),
            "buy_transactions": buy_count,
            "sell_transactions": sell_count,
            "shares_bought": shares_bought,
            "shares_sold": shares_sold,
            "net_shares": shares_bought - shares_sold,
        },
        "transactions": transactions,
    }


def _finnhub_search(query: str, limit: int) -> list[dict]:
    """Look up tickers by company name via Finnhub's symbol search."""
    token = _finnhub_require_key()
    data = _http_get_json("https://finnhub.io/api/v1/search", {"q": query, "token": token})
    results = []
    for item in (data.get("result") or [])[:limit]:
        results.append({
            "symbol": item.get("symbol"),
            "description": item.get("description"),
            "type": item.get("type"),
        })
    return results


# ===================================================================
#                       PROVIDER: YFINANCE
# ===================================================================

def _yfinance_ticker(symbol: str):
    import yfinance as yf
    return yf.Ticker(symbol)


def _yfinance_quote(symbol: str) -> dict | None:
    ticker = _yfinance_ticker(symbol)
    try:
        fast = ticker.fast_info or {}
    except Exception:
        fast = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    price = _safe_float(fast.get("last_price") or info.get("regularMarketPrice") or info.get("currentPrice"))
    prev_close = _safe_float(fast.get("previous_close") or info.get("regularMarketPreviousClose") or info.get("previousClose"))
    if price is None and prev_close is None:
        return None

    change = (price - prev_close) if (price is not None and prev_close is not None) else None
    change_pct = ((change / prev_close) * 100) if (change is not None and prev_close) else None

    return {
        "provider": "yfinance",
        "symbol": symbol,
        "price": price,
        "change": round(change, 4) if change is not None else None,
        "change_percent": round(change_pct, 4) if change_pct is not None else None,
        "open": _safe_float(fast.get("open") or info.get("regularMarketOpen") or info.get("open")),
        "high": _safe_float(fast.get("day_high") or info.get("regularMarketDayHigh") or info.get("dayHigh")),
        "low": _safe_float(fast.get("day_low") or info.get("regularMarketDayLow") or info.get("dayLow")),
        "previous_close": prev_close,
        "volume": _safe_int(fast.get("last_volume") or info.get("regularMarketVolume") or info.get("volume")),
        "currency": fast.get("currency") or info.get("currency"),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


def _yfinance_profile(symbol: str) -> dict | None:
    ticker = _yfinance_ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    if not info or not (info.get("longName") or info.get("shortName") or info.get("symbol")):
        return None

    market_cap = _safe_float(info.get("marketCap"))
    dividend_yield = _safe_float(info.get("dividendYield"))
    return {
        "provider": "yfinance",
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName"),
        "exchange": info.get("exchange") or info.get("fullExchangeName"),
        "country": info.get("country"),
        "currency": info.get("currency") or info.get("financialCurrency"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "employees": _safe_int(info.get("fullTimeEmployees")),
        "summary": (info.get("longBusinessSummary") or "")[:1000] or None,
        "market_cap": market_cap,
        "market_cap_formatted": _format_large_number(market_cap),
        "key_metrics": {
            "pe_trailing": _safe_float(info.get("trailingPE")),
            "pe_forward": _safe_float(info.get("forwardPE")),
            "ps_ttm": _safe_float(info.get("priceToSalesTrailing12Months")),
            "pb_ratio": _safe_float(info.get("priceToBook")),
            "eps_trailing": _safe_float(info.get("trailingEps")),
            "eps_forward": _safe_float(info.get("forwardEps")),
            "dividend_yield_percent": (
                dividend_yield * 100
                if dividend_yield is not None and dividend_yield < 1
                else dividend_yield
            ),
            "dividend_rate": _safe_float(info.get("dividendRate")),
            "beta": _safe_float(info.get("beta")),
            "52_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "52_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "50_day_avg": _safe_float(info.get("fiftyDayAverage")),
            "200_day_avg": _safe_float(info.get("twoHundredDayAverage")),
            "profit_margin_percent": (
                _safe_float(info.get("profitMargins")) * 100
                if _safe_float(info.get("profitMargins")) is not None
                else None
            ),
            "operating_margin_percent": (
                _safe_float(info.get("operatingMargins")) * 100
                if _safe_float(info.get("operatingMargins")) is not None
                else None
            ),
            "return_on_equity_percent": (
                _safe_float(info.get("returnOnEquity")) * 100
                if _safe_float(info.get("returnOnEquity")) is not None
                else None
            ),
            "debt_to_equity": _safe_float(info.get("debtToEquity")),
            "revenue_ttm": _safe_float(info.get("totalRevenue")),
            "ebitda": _safe_float(info.get("ebitda")),
            "shares_outstanding": _safe_int(info.get("sharesOutstanding")),
        },
    }


def _yfinance_financials(symbol: str, statement: str, period: str, limit: int) -> dict | None:
    ticker = _yfinance_ticker(symbol)
    try:
        if statement == "income":
            df = ticker.quarterly_income_stmt if period == "quarterly" else ticker.income_stmt
        elif statement == "balance":
            df = ticker.quarterly_balance_sheet if period == "quarterly" else ticker.balance_sheet
        else:
            df = ticker.quarterly_cashflow if period == "quarterly" else ticker.cashflow
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df.iloc[:, :limit]

    periods = []
    for col in df.columns:
        col_label = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
        data_dict: dict[str, Any] = {}
        for row_label in df.index:
            val = df.at[row_label, col]
            if val is None:
                continue
            try:
                if val != val:  # NaN check
                    continue
            except Exception:
                pass
            try:
                data_dict[str(row_label)] = float(val)
            except (TypeError, ValueError):
                data_dict[str(row_label)] = str(val)
        periods.append({"period_end": col_label, "data": data_dict})

    return {
        "provider": "yfinance",
        "symbol": symbol,
        "statement": statement,
        "period": period,
        "periods": periods,
    }


def _yfinance_earnings(symbol: str, limit: int) -> dict | None:
    ticker = _yfinance_ticker(symbol)
    rows = []
    try:
        df = ticker.earnings_history
        if df is not None and not df.empty:
            df = df.iloc[:limit]
            for idx, row in df.iterrows():
                period_label = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                rows.append({
                    "period": period_label,
                    "actual_eps": _safe_float(row.get("epsActual")),
                    "estimated_eps": _safe_float(row.get("epsEstimate")),
                    "surprise": _safe_float(row.get("epsDifference")),
                    "surprise_percent": _safe_float(row.get("surprisePercent")),
                })
    except Exception:
        pass

    if not rows:
        try:
            df = ticker.quarterly_income_stmt
            if df is not None and not df.empty and "Diluted EPS" in df.index:
                for col in df.columns[:limit]:
                    rows.append({
                        "period": col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col),
                        "actual_eps": _safe_float(df.at["Diluted EPS", col]),
                        "estimated_eps": None,
                        "surprise": None,
                        "surprise_percent": None,
                    })
        except Exception:
            pass

    if not rows:
        return None

    return {"provider": "yfinance", "symbol": symbol, "earnings": rows}


def _yfinance_news(symbol: str, limit: int, days: int | None = None) -> dict | None:
    # `days` is accepted for a uniform section signature with _finnhub_news but
    # unused: yfinance's news endpoint returns only a recent window with no
    # server-side date range to pass through.
    ticker = _yfinance_ticker(symbol)
    try:
        news = ticker.news or []
    except Exception:
        return None
    if not news:
        return None

    articles = []
    for item in news[:limit]:
        content = item.get("content") if isinstance(item, dict) else None
        if content:
            pub_ts = content.get("pubDate") or content.get("displayTime")
            published = pub_ts if isinstance(pub_ts, str) else _ts_to_iso(pub_ts)
            articles.append({
                "headline": content.get("title"),
                "source": (content.get("provider") or {}).get("displayName"),
                "summary": (content.get("summary") or "")[:500],
                "url": (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url"),
                "published": published,
            })
        else:
            articles.append({
                "headline": item.get("title"),
                "source": item.get("publisher"),
                "summary": "",
                "url": item.get("link"),
                "published": _ts_to_iso(item.get("providerPublishTime")),
            })

    return {"provider": "yfinance", "symbol": symbol, "count": len(articles), "articles": articles}


def _yfinance_insider_transactions(symbol: str, weeks: int) -> dict | None:
    ticker = _yfinance_ticker(symbol)
    try:
        df = ticker.insider_transactions
    except Exception:
        return None
    if df is None or df.empty:
        return None

    from datetime import date, timedelta
    today = date.today()
    cutoff = today - timedelta(weeks=weeks)
    from_date = cutoff.isoformat()
    to_date = today.isoformat()

    transactions = []
    shares_bought = shares_sold = 0
    buy_count = sell_count = 0
    for _, row in df.iterrows():
        start = row.get("Start Date")
        tx_date = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else (str(start) if start is not None else None)
        # Filter to the requested lookback window when a date is available.
        if tx_date and tx_date < from_date:
            continue

        text = str(row.get("Text") or row.get("Transaction") or "").lower()
        shares = _safe_int(row.get("Shares")) or 0
        if "sale" in text or "sell" in text:
            direction = "sell"
            sell_count += 1
            shares_sold += shares
        elif "purchase" in text or "buy" in text:
            direction = "buy"
            buy_count += 1
            shares_bought += shares
        else:
            direction = "neutral"

        transactions.append({
            "name": row.get("Insider"),
            "position": row.get("Position"),
            "transaction_date": tx_date,
            "direction": direction,
            "transaction_type": row.get("Transaction") or row.get("Text"),
            "shares": shares,
            "value": _safe_float(row.get("Value")),
        })

    if not transactions:
        return None

    transactions.sort(key=lambda t: t.get("transaction_date") or "", reverse=True)

    return {
        "provider": "yfinance",
        "symbol": symbol,
        "from_date": from_date,
        "to_date": to_date,
        "lookback_weeks": weeks,
        "summary": {
            "total_transactions": len(transactions),
            "buy_transactions": buy_count,
            "sell_transactions": sell_count,
            "shares_bought": shares_bought,
            "shares_sold": shares_sold,
            "net_shares": shares_bought - shares_sold,
        },
        "transactions": transactions,
    }


# Approximate calendar days each interval's bar spans, used to size the fetch
# window so it holds `bars` rows. Daily uses ~5 trading days per 7 calendar days;
# weekly/monthly bars span the obvious calendar spacing. Generous slack (a
# multiplier + a constant) covers holidays/closures so we don't come up short.
_HISTORY_INTERVAL_DAYS = {"1d": 1.5, "1wk": 7, "1mo": 31}


def _yfinance_history(symbol: str, bars: int, interval: str = "1d") -> dict | None:
    """OHLC price history at a daily/weekly/monthly interval (the lone
    non-point-in-time section).

    yfinance's ``.history()`` is the cheap source of a price time series;
    Finnhub/FMP don't offer comparable OHLC bars on their free tiers, so this
    section is yfinance-only. `interval` is "1d", "1wk", or "1mo": the same
    ``bars`` budget then covers a wider span (e.g. 30 weekly or monthly bars =
    ~7 months or ~2.5 years) without enlarging the response. We request a
    calendar window wide enough to hold ``bars`` of the chosen interval (plus
    slack for closures) and keep the most recent ``bars`` rows.
    """
    from datetime import date, timedelta

    if interval not in _HISTORY_INTERVAL_DAYS:
        interval = "1d"
    span_days = int(bars * _HISTORY_INTERVAL_DAYS[interval]) + 10

    ticker = _yfinance_ticker(symbol)
    start = (date.today() - timedelta(days=span_days)).isoformat()
    try:
        df = ticker.history(start=start, interval=interval, auto_adjust=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    df = df.tail(bars)
    rows = []
    for idx, row in df.iterrows():
        bar_date = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
        rows.append({
            "date": bar_date,
            "open": _safe_float(row.get("Open")),
            "high": _safe_float(row.get("High")),
            "low": _safe_float(row.get("Low")),
            "close": _safe_float(row.get("Close")),
            "volume": _safe_int(row.get("Volume")),
        })
    # Most recent first, matching the other history-bearing sections.
    rows.reverse()

    return {
        "provider": "yfinance",
        "symbol": symbol,
        "interval": interval,
        "count": len(rows),
        "bars": rows,
    }


def _yfinance_search(query: str, limit: int) -> list[dict]:
    """Look up tickers by company name via Yahoo's keyless search endpoint.

    Backs the no-key fallback for symbol resolution. Returns the same
    ``{symbol, description, type}`` shape as ``_finnhub_search`` so resolution is
    provider-independent.
    """
    import yfinance as yf

    quotes = yf.Search(query, max_results=limit).quotes or []
    results = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        results.append({
            "symbol": symbol,
            "description": q.get("longname") or q.get("shortname"),
            "type": q.get("quoteType"),
        })
    return results


# ===================================================================
#                       PROVIDER: FMP
# ===================================================================

# FMP retired its legacy `/api/v3` and `/api/v4` endpoints for API keys issued
# after 2025-08-31; `/stable` is the current API. All FMP access goes through
# `_fmp_get` so the base URL and key injection live in one place.
_FMP_STABLE = "https://financialmodelingprep.com/stable"


def _fmp_require_key() -> str:
    if not cfg.fmp_api_key:
        raise RuntimeError("FMP API key not configured.")
    return cfg.fmp_api_key


def _fmp_get(path: str, params: dict | None = None) -> Any:
    """GET an FMP `/stable` endpoint with the API key injected."""
    merged = dict(params or {})
    merged["apikey"] = _fmp_require_key()
    return _http_get_json(f"{_FMP_STABLE}/{path}", merged)


def _fmp_quote(symbol: str) -> dict | None:
    data = _fmp_get("quote", {"symbol": symbol})
    if not data or not isinstance(data, list):
        return None
    q = data[0]
    market_cap = _safe_float(q.get("marketCap"))
    return {
        "provider": "fmp",
        "symbol": symbol,
        "name": q.get("name"),
        "price": _safe_float(q.get("price")),
        "change": _safe_float(q.get("change")),
        "change_percent": _safe_float(q.get("changePercentage")),
        "open": _safe_float(q.get("open")),
        "high": _safe_float(q.get("dayHigh")),
        "low": _safe_float(q.get("dayLow")),
        "previous_close": _safe_float(q.get("previousClose")),
        "volume": _safe_int(q.get("volume")),
        "market_cap": market_cap,
        "market_cap_formatted": _format_large_number(market_cap),
        "50_day_avg": _safe_float(q.get("priceAvg50")),
        "200_day_avg": _safe_float(q.get("priceAvg200")),
        "52_week_high": _safe_float(q.get("yearHigh")),
        "52_week_low": _safe_float(q.get("yearLow")),
        "exchange": q.get("exchange"),
        "timestamp": _ts_to_iso(q.get("timestamp")),
    }


def _fmp_profile(symbol: str) -> dict | None:
    data = _fmp_get("profile", {"symbol": symbol})
    if not data or not isinstance(data, list):
        return None
    p = data[0]
    market_cap = _safe_float(p.get("marketCap"))
    return {
        "provider": "fmp",
        "symbol": symbol,
        "name": p.get("companyName"),
        "exchange": p.get("exchange"),
        "country": p.get("country"),
        "currency": p.get("currency"),
        "sector": p.get("sector"),
        "industry": p.get("industry"),
        "website": p.get("website"),
        "employees": _safe_int(p.get("fullTimeEmployees")),
        "ipo_date": p.get("ipoDate"),
        "summary": (p.get("description") or "")[:1000] or None,
        "ceo": p.get("ceo"),
        "market_cap": market_cap,
        "market_cap_formatted": _format_large_number(market_cap),
        "key_metrics": {
            "price": _safe_float(p.get("price")),
            "beta": _safe_float(p.get("beta")),
            "volume_avg": _safe_int(p.get("averageVolume")),
            "last_dividend": _safe_float(p.get("lastDividend")),
            "range": p.get("range"),
        },
    }


def _fmp_financials(symbol: str, statement: str, period: str, limit: int) -> dict | None:
    endpoint = {
        "income": "income-statement",
        "balance": "balance-sheet-statement",
        "cashflow": "cash-flow-statement",
    }[statement]
    params: dict[str, Any] = {"symbol": symbol, "limit": limit}
    if period == "quarterly":
        params["period"] = "quarter"
    data = _fmp_get(endpoint, params)
    if not data or not isinstance(data, list):
        return None

    # Bookkeeping fields are surfaced as period metadata, not repeated in `data`.
    _meta = {"symbol", "date", "reportedCurrency", "cik", "filingDate",
             "acceptedDate", "fiscalYear", "period"}
    periods = []
    for entry in data[:limit]:
        cleaned = {k: v for k, v in entry.items() if k not in _meta}
        periods.append({
            "period_end": entry.get("date"),
            "fiscal_year": entry.get("fiscalYear"),
            "period_label": entry.get("period"),
            "currency": entry.get("reportedCurrency"),
            "data": cleaned,
        })

    return {
        "provider": "fmp",
        "symbol": symbol,
        "statement": statement,
        "period": period,
        "periods": periods,
    }


def _fmp_earnings(symbol: str, limit: int) -> dict | None:
    data = _fmp_get("earnings", {"symbol": symbol, "limit": limit})
    if not data or not isinstance(data, list):
        return None

    rows = []
    for row in data[:limit]:
        actual = _safe_float(row.get("epsActual"))
        estimate = _safe_float(row.get("epsEstimated"))
        surprise = (actual - estimate) if (actual is not None and estimate is not None) else None
        surprise_pct = (
            (surprise / estimate * 100)
            if (surprise is not None and estimate not in (None, 0))
            else None
        )
        rows.append({
            "period": row.get("date"),
            "actual_eps": actual,
            "estimated_eps": estimate,
            "surprise": round(surprise, 4) if surprise is not None else None,
            "surprise_percent": round(surprise_pct, 4) if surprise_pct is not None else None,
            "revenue_actual": _safe_float(row.get("revenueActual")),
            "revenue_estimated": _safe_float(row.get("revenueEstimated")),
        })

    return {"provider": "fmp", "symbol": symbol, "earnings": rows}


def _fmp_pe_snapshot(kind: str, exchange: str, date: str) -> dict[str, float]:
    """Return ``{label: pe}`` for one exchange's sector- or industry-level P/E
    snapshot. ``kind`` is ``"sector"`` or ``"industry"`` (also the row field
    holding the label)."""
    rows = _fmp_get(f"{kind}-pe-snapshot", {"date": date, "exchange": exchange})
    out: dict[str, float] = {}
    if isinstance(rows, list):
        for r in rows:
            label = r.get(kind)
            pe = _safe_float(r.get("pe"))
            if label and pe is not None:
                out[label] = pe
    return out


def _fmp_pe_dimension(name: str, average_pe: float | None, company_pe: float | None) -> dict:
    """Build one comparison dimension (sector or industry): the group average P/E
    and, when the company's own P/E is known, how it stacks up against it."""
    dim: dict[str, Any] = {"name": name, "average_pe": round(average_pe, 4) if average_pe is not None else None}
    if average_pe is None:
        dim["note"] = "no P/E snapshot available for this exchange"
        return dim
    if company_pe is not None and average_pe:
        diff = company_pe - average_pe
        dim["company_vs_average"] = round(diff, 4)
        dim["premium_percent"] = round((company_pe / average_pe - 1) * 100, 2)
        # 5% band around the average counts as in-line rather than over/under-valued.
        dim["valuation"] = (
            "in_line" if abs(diff) / average_pe < 0.05
            else "above_average" if diff > 0 else "below_average"
        )
    return dim


def _fmp_sector_comparison(symbol: str) -> dict | None:
    """Compare a company's trailing P/E against its sector and industry averages.

    FMP-only: the per-exchange sector/industry P/E snapshots have no free
    yfinance/Finnhub equivalent. Returns ``None`` when the symbol has no FMP
    profile (e.g. a non-equity ticker) or when neither its sector nor its
    industry has a P/E for its exchange (e.g. an exchange the snapshot omits).
    """
    profile = _fmp_get("profile", {"symbol": symbol})
    if not isinstance(profile, list) or not profile:
        return None
    p = profile[0]
    sector = p.get("sector")
    industry = p.get("industry")
    exchange = p.get("exchange")
    if not exchange or not (sector or industry):
        return None

    # The company's own trailing P/E — the value the averages are compared to.
    company_pe = None
    try:
        ratios = _fmp_get("ratios-ttm", {"symbol": symbol})
        if isinstance(ratios, list) and ratios:
            company_pe = _safe_float(ratios[0].get("priceToEarningsRatioTTM"))
    except Exception:
        company_pe = None

    date = datetime.now(tz=timezone.utc).date().isoformat()
    sector_pes = _fmp_pe_snapshot("sector", exchange, date) if sector else {}
    industry_pes = _fmp_pe_snapshot("industry", exchange, date) if industry else {}

    sector_pe = sector_pes.get(sector) if sector else None
    industry_pe = industry_pes.get(industry) if industry else None
    if sector_pe is None and industry_pe is None:
        return None

    result: dict[str, Any] = {
        "provider": "fmp",
        "symbol": symbol,
        "as_of": date,
        "exchange": exchange,
        "metric": "pe_ttm",
        "company_pe": company_pe,
    }
    if sector:
        result["sector"] = _fmp_pe_dimension(sector, sector_pe, company_pe)
    if industry:
        result["industry"] = _fmp_pe_dimension(industry, industry_pe, company_pe)
    return result


# ===================================================================
#                       SYMBOL RESOLUTION
# ===================================================================
#
# `get_company_data` accepts either a ticker ("AAPL") or a company name
# ("Apple"). Resolution folds the old `search_symbol` tool in: a name is looked
# up and turned into a ticker before any section is fetched, so the model makes
# one call instead of two.

# Max search hits to consider when resolving a name (also bounds the
# alternatives surfaced back to the model).
_SEARCH_LIMIT = 10


def _is_special_symbol(s: str) -> bool:
    """True for concrete non-equity ticker forms — crypto ("BTC-USD"), FX
    ("EURUSD=X"), indices ("^GSPC"), exchange-suffixed ("RY.TO"). These carry
    characters a company name never would, and symbol search handles them
    poorly, so they're passed straight through without a lookup."""
    return bool(s) and " " not in s and any(c in s for c in "-=^.")


def _is_plain_ticker(s: str) -> bool:
    """True for a short alphanumeric token (AAPL, MSFT). Used only as a graceful
    fallback: when a name search returns nothing or its providers all fail, an
    input shaped like this is tried directly rather than erroring out."""
    return bool(s) and " " not in s and s.isalnum() and len(s) <= 6


def _search_symbols(query: str, limit: int) -> tuple[list[dict], list[str], str | None]:
    """Run symbol search across providers: Finnhub primary (when keyed), keyless
    yfinance fallback. Returns ``(results, errors, provider)`` where ``provider``
    is ``None`` only when every provider hard-failed (vs. answered with zero
    matches). Each result is ``{symbol, description, type}``."""
    errors: list[str] = []
    results: list[dict] = []
    provider: str | None = None

    if cfg.finnhub_api_key:
        try:
            results = _finnhub_search(query, limit)
            provider = "finnhub"
        except Exception as e:
            errors.append(f"finnhub: {type(e).__name__}: {e}")

    if not results and (provider is None or cfg.prefer_yfinance_fallback):
        try:
            results = _yfinance_search(query, limit)
            provider = "yfinance"
        except Exception as e:
            errors.append(f"yfinance: {type(e).__name__}: {e}")

    return results, errors, provider


def _resolve_symbol(raw: str) -> tuple[str, dict | None]:
    """Resolve a ticker-or-company-name into a concrete ticker symbol.

    Concrete non-equity forms (BTC-USD, ^GSPC, …) pass straight through. Anything
    else is run through symbol search; an exact ticker match (the input was
    already a symbol like "AAPL") short-circuits, otherwise the best-matching hit
    is used. Returns ``(symbol, match)`` where ``match`` is ``None`` for a
    pass-through / exact ticker, or a dict describing the resolved hit (and up to
    four alternatives) so the response can tell the model what a name resolved to.

    Raises ``ToolError`` only when a name search finds nothing (and the input
    isn't itself ticker-shaped) or every search provider hard-fails.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        raise ToolError("A ticker symbol or company name is required.")
    if _is_special_symbol(cleaned):
        return cleaned.upper(), None

    results, errors, provider = _search_symbols(cleaned, _SEARCH_LIMIT)

    if provider is None:
        # Search infrastructure is down. If the input already looks like a
        # ticker, proceed with it directly rather than failing the whole call.
        if _is_plain_ticker(cleaned):
            return cleaned.upper(), None
        raise ToolError(_retrieval_error("symbol search", cleaned, errors))

    if not results:
        if _is_plain_ticker(cleaned):
            return cleaned.upper(), None
        raise ToolError(
            f"No matching ticker found for '{cleaned}'. Try a different company "
            "name, or pass the ticker symbol directly."
        )

    # Prefer an exact symbol match — the input was already a ticker — so a price
    # check on "AAPL" doesn't get re-labelled as having been "resolved".
    upper = cleaned.upper()
    exact = next((r for r in results if (r.get("symbol") or "").upper() == upper), None)
    if exact:
        return upper, None

    chosen = results[0]
    symbol = (chosen.get("symbol") or "").strip().upper()
    if not symbol:
        raise ToolError(f"No usable ticker found for '{cleaned}'.")

    match: dict[str, Any] = {"query": cleaned, "provider": provider, "matched": chosen}
    alternatives = [r for r in results[1:] if r.get("symbol")][:4]
    if alternatives:
        match["alternatives"] = alternatives
    return symbol, match


# ===================================================================
#                       SECTION DISPATCH
# ===================================================================
#
# Each section of company data resolves a provider, calls it, and (when the
# primary provider yields nothing) retries with yfinance. The 1:1 tools this
# replaced each repeated that pattern; it now lives in one helper so the
# consolidated `get_company_data` tool can fetch any mix of sections.

VALID_SECTIONS = ("quote", "profile", "financials", "earnings", "news", "insiders", "price_history", "sector_comparison")
DEFAULT_SECTIONS = ("quote", "profile")


def _fetch_section(provider: str, primary: dict, yf_fn, args: tuple) -> tuple[dict | None, list[str]]:
    """Run the provider's fetcher for one section, then fall back to yfinance.

    ``primary`` maps a concrete provider name to its fetcher. If the resolved
    provider has no entry (e.g. FMP for news/insiders), yfinance handles it —
    mirroring the original tools' ``else: yfinance`` branch. Returns
    ``(result_or_None, errors)``; ``errors`` keeps the per-provider detail.
    """
    errors: list[str] = []
    fn = primary.get(provider)
    if fn is None:
        fn = yf_fn
        provider = "yfinance"

    result: dict | None = None
    try:
        result = fn(*args)
    except Exception as e:
        errors.append(f"{provider}: {type(e).__name__}: {e}")

    if (not result) and cfg.prefer_yfinance_fallback and provider != "yfinance":
        try:
            result = yf_fn(*args)
        except Exception as e:
            errors.append(f"yfinance: {type(e).__name__}: {e}")

    return result, errors


# Section fetchers share a uniform (symbol, opts) signature so the dispatch loop
# can call any of them the same way. `opts` carries the per-call knobs the model
# chose, already clamped to their configured maximums by `_gather_sections`: the
# financials `statement`/`period`, and the resolved amount for each section.

def _section_quote(symbol: str, opts: dict):
    return _fetch_section(
        _resolve_provider(cfg.default_provider),
        {"finnhub": _finnhub_quote, "yfinance": _yfinance_quote, "fmp": _fmp_quote},
        _yfinance_quote, (symbol,),
    )


def _section_profile(symbol: str, opts: dict):
    return _fetch_section(
        _resolve_provider(cfg.default_provider),
        {"finnhub": _finnhub_profile, "yfinance": _yfinance_profile, "fmp": _fmp_profile},
        _yfinance_profile, (symbol,),
    )


def _section_financials(symbol: str, opts: dict):
    return _fetch_section(
        _resolve_provider(cfg.financials_provider, for_financials=True),
        {"fmp": _fmp_financials, "yfinance": _yfinance_financials, "finnhub": _finnhub_financials},
        _yfinance_financials,
        (symbol, opts["statement"], opts["period"], opts["financial_periods"]),
    )


def _section_earnings(symbol: str, opts: dict):
    return _fetch_section(
        _resolve_provider(cfg.default_provider),
        {"finnhub": _finnhub_earnings, "fmp": _fmp_earnings, "yfinance": _yfinance_earnings},
        _yfinance_earnings, (symbol, opts["earnings_periods"]),
    )


def _section_news(symbol: str, opts: dict):
    return _fetch_section(
        _resolve_provider(cfg.default_provider),
        {"finnhub": _finnhub_news, "yfinance": _yfinance_news},
        _yfinance_news, (symbol, opts["news_items"], opts["news_days"]),
    )


def _section_insiders(symbol: str, opts: dict):
    return _fetch_section(
        _resolve_provider(cfg.default_provider),
        {"finnhub": _finnhub_insider_transactions, "yfinance": _yfinance_insider_transactions},
        _yfinance_insider_transactions, (symbol, opts["insider_weeks"]),
    )


def _section_sector_comparison(symbol: str, opts: dict):
    # FMP-only: the sector/industry P/E snapshots have no free yfinance/Finnhub
    # equivalent, so (unlike the other sections) there is no fallback. Without an
    # FMP key configured the section reports the data is unavailable rather than
    # silently returning nothing.
    if not cfg.fmp_api_key:
        return None, ["sector_comparison requires an FMP API key (set STOCK_FMP_API_KEY)"]
    try:
        return _fmp_sector_comparison(symbol), []
    except Exception as e:
        return None, [f"fmp: {type(e).__name__}: {e}"]


def _section_history(symbol: str, opts: dict):
    # Price history is yfinance-only (Finnhub/FMP have no cheap OHLC bars); the
    # empty primary map routes every resolved provider to the yfinance fetcher.
    return _fetch_section(
        _resolve_provider(cfg.default_provider),
        {},
        _yfinance_history, (symbol, opts["history_bars"], opts["history_interval"]),
    )


_SECTION_FETCHERS = {
    "quote": _section_quote,
    "profile": _section_profile,
    "financials": _section_financials,
    "earnings": _section_earnings,
    "news": _section_news,
    "insiders": _section_insiders,
    "price_history": _section_history,
    "sector_comparison": _section_sector_comparison,
}


async def _gather_sections(
    symbol: str,
    sections: list[str],
    statement: str,
    period: str,
    periods: int | None,
    news_items: int | None,
    insider_weeks: int | None,
    history_bars: int | None,
    news_days: int | None,
    history_interval: str,
):
    """Fetch every requested section concurrently.

    The amount knobs (``periods``/``news_items``/``insider_weeks``/``news_days``)
    are the raw values the model requested (or ``None`` for "use the configured
    maximum"). Each is clamped to its env-configured cap here before any fetcher
    runs, so a section can never return more than its maximum. ``periods`` drives
    both the financials and earnings sections, each clamped to its own cap.

    Each section's blocking provider calls run in their own worker thread and the
    sections are awaited together, so a multi-section call pays the slowest
    provider's latency rather than the sum of all of them.

    Returns ``(data, errors)`` where ``data`` maps a section name to its payload
    and ``errors`` maps a section name to why it produced nothing (a provider
    error, or "no data available" for a valid-but-empty result).
    """
    opts = {
        "statement": statement,
        "period": period,
        "financial_periods": _clamp_amount(periods, cfg.max_financial_periods),
        "earnings_periods": _clamp_amount(periods, cfg.max_earnings_periods),
        "news_items": _clamp_amount(news_items, cfg.max_news_items),
        "news_days": _clamp_amount(news_days, cfg.max_news_lookback_days),
        "insider_weeks": _clamp_amount(insider_weeks, cfg.max_insider_lookback_weeks),
        "history_bars": _clamp_amount(history_bars, cfg.max_history_bars),
        "history_interval": history_interval,
    }
    settled = await asyncio.gather(
        *(
            anyio.to_thread.run_sync(_SECTION_FETCHERS[section], symbol, opts)
            for section in sections
        )
    )
    data: dict[str, Any] = {}
    errors: dict[str, list[str]] = {}
    for section, (result, errs) in zip(sections, settled):
        if result:
            data[section] = result
        else:
            errors[section] = errs or ["no data available"]
    return data, errors


# ===================================================================
#                       TOOL REGISTRATION
# ===================================================================

def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def get_company_data(
        symbol: str,
        sections: list[str] | None = None,
        statement: Literal["income", "balance", "cashflow"] = "income",
        period: Literal["annual", "quarterly"] = "annual",
        periods: int | None = None,
        news_items: int | None = None,
        insider_weeks: int | None = None,
        history_bars: int | None = None,
        news_days: int | None = None,
        history_interval: Literal["1d", "1wk", "1mo"] = "1d",
    ) -> str:
        """Get stock/company data. symbol=ticker or name (auto-resolved).

        sections: "quote"(price/day's change)|"profile"(fundamentals/market cap)|
        "financials"(income/balance/cashflow)|"earnings"(EPS vs est)|"news"|
        "insiders"(buy/sell txns)|"price_history"(OHLC bars)|
        "sector_comparison"(company P/E vs its sector & industry average P/E).
        Params: statement(income|balance|cashflow), period(annual|quarterly),
        periods, news_items, insider_weeks, history_bars, news_days (all capped;
        omit=max). history_interval(1d|1wk|1mo): bar size for price_history —
        use 1wk/1mo to cover months/years within the same bar budget.

        Use for: current price, company fundamentals, financial statements,
        earnings reports, recent news, insider trading, price charts, valuation
        vs sector/industry peers. Crypto/FX/indices (BTC-USD, ^GSPC) only
        support quote/price_history. "sector_comparison" is equities-only.

        :param symbol: Ticker or company name.
        :param sections: Sections to fetch (default: quote,profile).
        :param statement: Financials statement type.
        :param period: Annual or quarterly.
        :param periods: Count for financials/earnings.
        :param news_items: Article count.
        :param insider_weeks: Lookback weeks.
        :param history_bars: Price-history bar count.
        :param news_days: News lookback window in days (capped).
        :param history_interval: Price-history bar size: 1d, 1wk, or 1mo.
        :return: JSON {symbol,sections,data:{...},resolved_from?,errors?}.
        """
        log_call(
            log,
            "get_company_data",
            symbol=symbol,
            sections=sections,
            statement=statement,
            period=period,
            periods=periods,
            news_items=news_items,
            insider_weeks=insider_weeks,
            history_bars=history_bars,
            news_days=news_days,
            history_interval=history_interval,
        )
        raw_query = (symbol or "").strip()
        if not raw_query:
            raise ToolError("A ticker symbol or company name is required.")

        requested = sections if sections else list(DEFAULT_SECTIONS)
        normalized: list[str] = []
        for s in requested:
            key = (s or "").strip().lower()
            if key and key not in normalized:
                normalized.append(key)

        if not normalized:
            raise ToolError(
                "At least one section is required. Valid sections: "
                + ", ".join(VALID_SECTIONS) + "."
            )
        invalid = [s for s in normalized if s not in VALID_SECTIONS]
        if invalid:
            raise ToolError(
                f"Invalid section(s): {', '.join(invalid)}. "
                f"Valid sections are: {', '.join(VALID_SECTIONS)}."
            )

        if statement not in ("income", "balance", "cashflow"):
            raise ToolError("statement must be one of: income, balance, cashflow")
        if period not in ("annual", "quarterly"):
            raise ToolError("period must be 'annual' or 'quarterly'")
        if history_interval not in ("1d", "1wk", "1mo"):
            raise ToolError("history_interval must be one of: 1d, 1wk, 1mo")

        # Turn a ticker-or-name into a concrete ticker before fetching anything.
        symbol, match = await anyio.to_thread.run_sync(_resolve_symbol, raw_query)

        data, errors = await _gather_sections(
            symbol, normalized, statement, period,
            periods, news_items, insider_weeks, history_bars,
            news_days, history_interval,
        )

        if not data:
            # Every requested section failed — surface as a ToolError so the
            # failure can't be mistaken for data (see the README error convention).
            flat = [f"{sec}: {'; '.join(errs)}" for sec, errs in errors.items() if errs]
            raise ToolError(_retrieval_error("company data", symbol, flat))

        payload: dict[str, Any] = {"symbol": symbol, "sections": normalized, "data": data}
        if match:
            # The input was a name (or otherwise non-exact); tell the model what
            # it resolved to so it can confirm the right company was used.
            payload["resolved_from"] = match
        if errors:
            # Partial success: report which sections returned nothing and why.
            payload["errors"] = errors
        return log_result(log, "get_company_data", to_json(payload))
