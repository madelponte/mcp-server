"""
Stock Data MCP tool.

Quotes, fundamentals, financials, earnings, and news.
Uses Finnhub (primary, free API key), yfinance (no-key fallback), and optionally
Financial Modeling Prep for deep financial statements. Translated from the Open
WebUI tool; per-user valves and status emitters were removed.
"""

import json
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import anyio
import requests
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from config import stock_settings as cfg
from .cache import TTLCache

# Error convention: every genuine failure raises ToolError, which FastMCP turns
# into a result with `isError: true`, so a model can't mistake the failure for
# market data. See the README "Error handling" section.

# -------------------------- Helpers --------------------------

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _ts_to_iso(ts: Optional[int]) -> Optional[str]:
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _format_large_number(n: Optional[float]) -> Optional[str]:
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


# -------------------------- Cache + HTTP --------------------------

# Unbounded (max_entries=0) to preserve the original behavior; quote/profile
# responses are small and the TTL is short.
_cache = TTLCache(cfg.cache_ttl_seconds)


def _http_get_json(url: str, params: Optional[dict] = None) -> Any:
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


def _finnhub_quote(symbol: str) -> Optional[dict]:
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


def _finnhub_profile(symbol: str) -> Optional[dict]:
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


def _finnhub_financials(symbol: str, statement: str, period: str) -> Optional[dict]:
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
    for entry in data["data"][: cfg.max_financial_periods]:
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


def _finnhub_earnings(symbol: str) -> Optional[dict]:
    token = _finnhub_require_key()
    data = _http_get_json(
        "https://finnhub.io/api/v1/stock/earnings", {"symbol": symbol, "token": token}
    )
    if not data:
        return None

    rows = []
    for row in data[:8]:
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


def _finnhub_news(symbol: str) -> Optional[dict]:
    token = _finnhub_require_key()
    from datetime import date, timedelta
    today = date.today()
    from_date = (today - timedelta(days=7)).isoformat()
    to_date = today.isoformat()
    data = _http_get_json(
        "https://finnhub.io/api/v1/company-news",
        {"symbol": symbol, "from": from_date, "to": to_date, "token": token},
    )
    if not data:
        return None

    articles = []
    for item in data[: cfg.max_news_items]:
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


def _finnhub_insider_transactions(symbol: str, weeks: int) -> Optional[dict]:
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


# ===================================================================
#                       PROVIDER: YFINANCE
# ===================================================================

def _yfinance_ticker(symbol: str):
    import yfinance as yf
    return yf.Ticker(symbol)


def _yfinance_quote(symbol: str) -> Optional[dict]:
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


def _yfinance_profile(symbol: str) -> Optional[dict]:
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


def _yfinance_financials(symbol: str, statement: str, period: str) -> Optional[dict]:
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

    df = df.iloc[:, : cfg.max_financial_periods]

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


def _yfinance_earnings(symbol: str) -> Optional[dict]:
    ticker = _yfinance_ticker(symbol)
    rows = []
    try:
        df = ticker.earnings_history
        if df is not None and not df.empty:
            df = df.iloc[: cfg.max_financial_periods * 2]
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
                for col in df.columns[: cfg.max_financial_periods]:
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


def _yfinance_news(symbol: str) -> Optional[dict]:
    ticker = _yfinance_ticker(symbol)
    try:
        news = ticker.news or []
    except Exception:
        return None
    if not news:
        return None

    articles = []
    for item in news[: cfg.max_news_items]:
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


def _yfinance_insider_transactions(symbol: str, weeks: int) -> Optional[dict]:
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


# ===================================================================
#                       PROVIDER: FMP
# ===================================================================

def _fmp_require_key() -> str:
    if not cfg.fmp_api_key:
        raise RuntimeError("FMP API key not configured.")
    return cfg.fmp_api_key


def _fmp_quote(symbol: str) -> Optional[dict]:
    key = _fmp_require_key()
    data = _http_get_json(
        f"https://financialmodelingprep.com/api/v3/quote/{symbol}", {"apikey": key}
    )
    if not data or not isinstance(data, list):
        return None
    q = data[0]
    return {
        "provider": "fmp",
        "symbol": symbol,
        "name": q.get("name"),
        "price": _safe_float(q.get("price")),
        "change": _safe_float(q.get("change")),
        "change_percent": _safe_float(q.get("changesPercentage")),
        "open": _safe_float(q.get("open")),
        "high": _safe_float(q.get("dayHigh")),
        "low": _safe_float(q.get("dayLow")),
        "previous_close": _safe_float(q.get("previousClose")),
        "volume": _safe_int(q.get("volume")),
        "avg_volume": _safe_int(q.get("avgVolume")),
        "market_cap": _safe_float(q.get("marketCap")),
        "market_cap_formatted": _format_large_number(_safe_float(q.get("marketCap"))),
        "pe": _safe_float(q.get("pe")),
        "eps": _safe_float(q.get("eps")),
        "52_week_high": _safe_float(q.get("yearHigh")),
        "52_week_low": _safe_float(q.get("yearLow")),
        "exchange": q.get("exchange"),
        "timestamp": _ts_to_iso(q.get("timestamp")),
    }


def _fmp_profile(symbol: str) -> Optional[dict]:
    key = _fmp_require_key()
    data = _http_get_json(
        f"https://financialmodelingprep.com/api/v3/profile/{symbol}", {"apikey": key}
    )
    if not data or not isinstance(data, list):
        return None
    p = data[0]
    market_cap = _safe_float(p.get("mktCap"))
    return {
        "provider": "fmp",
        "symbol": symbol,
        "name": p.get("companyName"),
        "exchange": p.get("exchangeShortName"),
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
            "volume_avg": _safe_int(p.get("volAvg")),
            "last_dividend": _safe_float(p.get("lastDiv")),
            "range": p.get("range"),
            "dcf": _safe_float(p.get("dcf")),
            "dcf_diff": _safe_float(p.get("dcfDiff")),
        },
    }


def _fmp_financials(symbol: str, statement: str, period: str) -> Optional[dict]:
    key = _fmp_require_key()
    endpoint = {
        "income": "income-statement",
        "balance": "balance-sheet-statement",
        "cashflow": "cash-flow-statement",
    }[statement]
    params = {"apikey": key, "limit": cfg.max_financial_periods}
    if period == "quarterly":
        params["period"] = "quarter"
    data = _http_get_json(
        f"https://financialmodelingprep.com/api/v3/{endpoint}/{symbol}", params
    )
    if not data or not isinstance(data, list):
        return None

    periods = []
    for entry in data[: cfg.max_financial_periods]:
        cleaned = {
            k: v
            for k, v in entry.items()
            if k not in ("symbol", "reportedCurrency", "cik", "fillingDate",
                         "acceptedDate", "calendarYear", "link", "finalLink")
        }
        periods.append({
            "period_end": entry.get("date"),
            "fiscal_year": entry.get("calendarYear"),
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


def _fmp_earnings(symbol: str) -> Optional[dict]:
    key = _fmp_require_key()
    data = _http_get_json(
        f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}",
        {"apikey": key, "limit": 8},
    )
    if not data or not isinstance(data, list):
        return None

    rows = []
    for row in data[:8]:
        actual = _safe_float(row.get("eps"))
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
            "revenue_actual": _safe_float(row.get("revenue")),
            "revenue_estimated": _safe_float(row.get("revenueEstimated")),
        })

    return {"provider": "fmp", "symbol": symbol, "earnings": rows}


# ===================================================================
#                       TOOL REGISTRATION
# ===================================================================

def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def get_stock_quote(symbol: str) -> str:
        """
        Get the current stock quote for a ticker symbol — including price, day's change,
        open/high/low/previous close, and trading volume.

        :param symbol: The stock ticker symbol (e.g. "AAPL", "MSFT", "TSLA").
        :return: A JSON string with the latest quote data.
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise ToolError("Symbol is required.")

        provider = _resolve_provider(cfg.default_provider)
        result: Optional[dict] = None
        errors: list[str] = []

        try:
            if provider == "finnhub":
                result = await anyio.to_thread.run_sync(_finnhub_quote, symbol)
            elif provider == "yfinance":
                result = await anyio.to_thread.run_sync(_yfinance_quote, symbol)
            elif provider == "fmp":
                result = await anyio.to_thread.run_sync(_fmp_quote, symbol)
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")
            result = None

        if (not result) and cfg.prefer_yfinance_fallback and provider != "yfinance":
            try:
                result = await anyio.to_thread.run_sync(_yfinance_quote, symbol)
            except Exception as e:
                errors.append(f"yfinance: {type(e).__name__}: {e}")

        if not result:
            raise ToolError(_retrieval_error("quote", symbol, errors))
        return json.dumps(result, default=str)

    @mcp.tool()
    async def get_company_profile(symbol: str) -> str:
        """
        Get the company profile and key fundamentals for a ticker — name, sector, industry,
        market cap, employees, exchange, P/E, EPS, dividend yield, 52-week range, and beta.

        :param symbol: The stock ticker symbol (e.g. "AAPL").
        :return: A JSON string with the company profile and key metrics.
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise ToolError("Symbol is required.")

        provider = _resolve_provider(cfg.default_provider)
        result: Optional[dict] = None
        errors: list[str] = []

        try:
            if provider == "finnhub":
                result = await anyio.to_thread.run_sync(_finnhub_profile, symbol)
            elif provider == "yfinance":
                result = await anyio.to_thread.run_sync(_yfinance_profile, symbol)
            elif provider == "fmp":
                result = await anyio.to_thread.run_sync(_fmp_profile, symbol)
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")

        if (not result) and cfg.prefer_yfinance_fallback and provider != "yfinance":
            try:
                result = await anyio.to_thread.run_sync(_yfinance_profile, symbol)
            except Exception as e:
                errors.append(f"yfinance: {type(e).__name__}: {e}")

        if not result:
            raise ToolError(_retrieval_error("profile", symbol, errors))
        return json.dumps(result, default=str)

    @mcp.tool()
    async def get_financials(
        symbol: str,
        statement: Literal["income", "balance", "cashflow"] = "income",
        period: Literal["annual", "quarterly"] = "annual",
    ) -> str:
        """
        Get financial statements for a company — income statement, balance sheet, or cash flow.
        Returns the most recent N periods (configured by STOCK_MAX_FINANCIAL_PERIODS).

        :param symbol: The stock ticker symbol (e.g. "AAPL").
        :param statement: Which statement to fetch — "income", "balance", or "cashflow".
        :param period: "annual" for yearly statements, "quarterly" for quarterly.
        :return: A JSON string with the requested financial statements.
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise ToolError("Symbol is required.")
        if statement not in ("income", "balance", "cashflow"):
            raise ToolError("statement must be one of: income, balance, cashflow")
        if period not in ("annual", "quarterly"):
            raise ToolError("period must be 'annual' or 'quarterly'")

        provider = _resolve_provider(cfg.financials_provider, for_financials=True)
        result: Optional[dict] = None
        errors: list[str] = []

        try:
            if provider == "fmp":
                result = await anyio.to_thread.run_sync(_fmp_financials, symbol, statement, period)
            elif provider == "yfinance":
                result = await anyio.to_thread.run_sync(_yfinance_financials, symbol, statement, period)
            elif provider == "finnhub":
                result = await anyio.to_thread.run_sync(_finnhub_financials, symbol, statement, period)
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")

        if (not result) and cfg.prefer_yfinance_fallback and provider != "yfinance":
            try:
                result = await anyio.to_thread.run_sync(_yfinance_financials, symbol, statement, period)
            except Exception as e:
                errors.append(f"yfinance: {type(e).__name__}: {e}")

        if not result:
            raise ToolError(_retrieval_error("financials", symbol, errors))
        return json.dumps(result, default=str)

    @mcp.tool()
    async def get_earnings(symbol: str) -> str:
        """
        Get historical earnings reports for a company — actual EPS, estimated EPS,
        surprise %, and revenue figures by quarter.

        :param symbol: The stock ticker symbol (e.g. "AAPL").
        :return: A JSON string with historical earnings data.
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise ToolError("Symbol is required.")

        result: Optional[dict] = None
        errors: list[str] = []
        provider = _resolve_provider(cfg.default_provider)
        try:
            if provider == "finnhub":
                result = await anyio.to_thread.run_sync(_finnhub_earnings, symbol)
            elif provider == "fmp":
                result = await anyio.to_thread.run_sync(_fmp_earnings, symbol)
            else:
                result = await anyio.to_thread.run_sync(_yfinance_earnings, symbol)
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")

        if (not result) and cfg.prefer_yfinance_fallback and provider != "yfinance":
            try:
                result = await anyio.to_thread.run_sync(_yfinance_earnings, symbol)
            except Exception as e:
                errors.append(f"yfinance: {type(e).__name__}: {e}")

        if not result:
            raise ToolError(_retrieval_error("earnings", symbol, errors))
        return json.dumps(result, default=str)

    @mcp.tool()
    async def get_company_news(symbol: str) -> str:
        """
        Get recent news articles about a specific company.

        :param symbol: The stock ticker symbol (e.g. "AAPL").
        :return: A JSON string with recent news articles (headline, source, summary, url, published date).
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise ToolError("Symbol is required.")

        result: Optional[dict] = None
        errors: list[str] = []
        provider = _resolve_provider(cfg.default_provider)
        try:
            if provider == "finnhub":
                result = await anyio.to_thread.run_sync(_finnhub_news, symbol)
            else:
                result = await anyio.to_thread.run_sync(_yfinance_news, symbol)
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")

        if (not result) and cfg.prefer_yfinance_fallback and provider != "yfinance":
            try:
                result = await anyio.to_thread.run_sync(_yfinance_news, symbol)
            except Exception as e:
                errors.append(f"yfinance: {type(e).__name__}: {e}")

        if not result:
            raise ToolError(_retrieval_error("news", symbol, errors))
        return json.dumps(result, default=str)

    @mcp.tool()
    async def get_insider_transactions(symbol: str) -> str:
        """
        Get recent insider buying and selling activity for a stock — transactions filed
        by company insiders (officers, directors, and major shareholders) over the last
        N weeks (configured by STOCK_INSIDER_LOOKBACK_WEEKS). Returns a buy/sell summary
        and the individual transactions (insider name, date, share change, and price).

        :param symbol: The stock ticker symbol (e.g. "AAPL").
        :return: A JSON string with insider transaction data and a buy/sell summary.
        """
        symbol = (symbol or "").strip().upper()
        if not symbol:
            raise ToolError("Symbol is required.")

        weeks = cfg.insider_lookback_weeks
        result: Optional[dict] = None
        errors: list[str] = []
        provider = _resolve_provider(cfg.default_provider)
        try:
            if provider == "finnhub":
                result = await anyio.to_thread.run_sync(_finnhub_insider_transactions, symbol, weeks)
            else:
                result = await anyio.to_thread.run_sync(_yfinance_insider_transactions, symbol, weeks)
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}")

        if (not result) and cfg.prefer_yfinance_fallback and provider != "yfinance":
            try:
                result = await anyio.to_thread.run_sync(_yfinance_insider_transactions, symbol, weeks)
            except Exception as e:
                errors.append(f"yfinance: {type(e).__name__}: {e}")

        if not result:
            raise ToolError(_retrieval_error("insider transactions", symbol, errors))
        return json.dumps(result, default=str)

    @mcp.tool()
    async def search_symbol(query: str) -> str:
        """
        Search for a stock ticker symbol by company name or partial symbol.
        Useful when the user names a company but you don't know its ticker.
        Requires a Finnhub API key (STOCK_FINNHUB_API_KEY).

        :param query: The company name or partial ticker to search for (e.g. "apple").
        :return: A JSON string with matching tickers and company names.
        """
        query = (query or "").strip()
        if not query:
            raise ToolError("Query is required.")

        if not cfg.finnhub_api_key:
            raise ToolError(
                "Symbol search requires a Finnhub API key (STOCK_FINNHUB_API_KEY)."
            )

        try:
            data = await anyio.to_thread.run_sync(
                _http_get_json,
                "https://finnhub.io/api/v1/search",
                {"q": query, "token": cfg.finnhub_api_key},
            )
        except Exception as e:
            raise ToolError(f"Symbol search failed for {query!r}: {type(e).__name__}: {e}")

        results = []
        for item in (data.get("result") or [])[:10]:
            results.append({
                "symbol": item.get("symbol"),
                "description": item.get("description"),
                "type": item.get("type"),
            })
        return json.dumps({"query": query, "count": len(results), "results": results})
