"""
market_data_live.py
--------------------
Fetches a LIVE options-chain snapshot (real, current market prices — not
historical) for a real underlying via Yahoo Finance, so this project's
main documented methodological limitation (no free HISTORICAL options
data exists, forcing P-measure-only calibration) can be partly offset: we
DO have free access to a live current snapshot of the real options
market, so we can compare "what our P-measure historical model implies"
against "what the market is actually pricing right now" (Q-measure) —
including the resulting volatility risk premium, a real and widely
traded signal (systematically selling variance/options tends to earn a
premium over realised volatility, on average, across long samples).

This is intentionally kept separate from the historical-calibration
pipeline (data_loader.py / calibration.py): it is a live, point-in-time
snapshot, not something that can be baked into a reproducible historical
backtest — but it is a genuine, real-time capability a practitioner
could use to sanity-check the model against the market on any given day.
"""

import ssl

import numpy as np
import pandas as pd
import urllib3
import yfinance as yf

# Corporate-network SSL fix (see data_loader.py for the same pattern).
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_live_chain(ticker: str = "SPY", target_dte_days: int = 63) -> dict:
    """
    Fetch the live listed-option expiry closest to `target_dte_days`
    calendar days out, with cleaned call/put tables (mid price, moneyness)
    plus the current underlying spot and the chosen expiry's actual DTE.

    Returns
    -------
    dict with keys: ticker, spot, expiry, dte_days, T_years, calls, puts.

    Raises
    ------
    Whatever exception yfinance / the network raises — callers (e.g., the
    dashboard) should catch this and degrade gracefully, since this
    function requires live internet access and a ticker with listed
    options (works for SPY; will fail for BTC-USD, which has no
    Yahoo-Finance-listed options chain).
    """
    tkr = yf.Ticker(ticker)
    expirations = tkr.options
    if not expirations:
        raise ValueError(f"No listed options found for {ticker}.")

    today = pd.Timestamp.today().normalize()
    dtes = [(pd.Timestamp(e) - today).days for e in expirations]
    best_idx = int(np.argmin([abs(d - target_dte_days) for d in dtes]))
    expiry = expirations[best_idx]
    dte = dtes[best_idx]

    chain = tkr.option_chain(expiry)
    spot = float(tkr.fast_info["lastPrice"])

    def _clean(raw: pd.DataFrame, opt_type: str) -> pd.DataFrame:
        out = raw.copy()
        out["mid"] = (out["bid"] + out["ask"]) / 2.0
        out.loc[out["mid"] <= 0, "mid"] = out["lastPrice"]
        out["moneyness"] = out["strike"] / spot
        out["option_type"] = opt_type
        return out[["strike", "moneyness", "bid", "ask", "mid", "lastPrice",
                    "impliedVolatility", "volume", "openInterest", "option_type"]]

    return {
        "ticker": ticker, "spot": spot, "expiry": str(expiry), "dte_days": int(dte),
        "T_years": max(dte, 1) / 365.0,
        "calls": _clean(chain.calls, "call"), "puts": _clean(chain.puts, "put"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI Sanity Check
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Live network check: fetch SPY's live option chain nearest a 3-month
    maturity and print a short summary. Requires internet access.
    Run from repo root: python src/market_data_live.py
    """
    print("Live market data check\n" + "-" * 40)
    result = fetch_live_chain("SPY", target_dte_days=63)
    print(f"Ticker: {result['ticker']}   Spot: {result['spot']:.2f}")
    print(f"Nearest-to-3M expiry: {result['expiry']}  ({result['dte_days']} calendar days out)")
    print(f"\nCalls near the money:")
    calls = result["calls"]
    print(calls[(calls['moneyness'] > 0.95) & (calls['moneyness'] < 1.05)]
          [["strike", "moneyness", "mid", "impliedVolatility"]].to_string(index=False))
