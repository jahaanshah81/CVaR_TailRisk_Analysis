"""
data_loader.py
--------------
Downloads historical underlying price data (SPY and BTC-USD) from Yahoo
Finance for use throughout the Options Market-Making & Delta-Hedging
Simulator project.

We deliberately reuse two assets from very different volatility regimes:
    SPY (calm equity)   — realistic "normal" market-making conditions
    BTC-USD (volatile)  — stress-tests every model's assumptions harder

Sign convention: this project works with RAW PRICES and LOG-RETURNS
(not losses like the companion CVaR project), since we need actual price
levels to price options and simulate hedging P&L.

Run this script locally (requires internet access):
    python src/data_loader.py
"""

import os
import ssl
import time
import numpy as np
import pandas as pd
import urllib3
import yfinance as yf

# Corporate-network SSL fix (see companion CVaR project for the same pattern).
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TICKERS = {
    "spy": "SPY",
    "btc": "BTC-USD",
}

START_DATE = "2018-01-01"
END_DATE   = "2025-12-31"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_prices(ticker: str, start: str, end: str,
                  max_retries: int = 4, retry_delay: float = 30.0) -> pd.DataFrame:
    """
    Download daily OHLC price data from Yahoo Finance.

    Returns
    -------
    pd.DataFrame with columns [close, log_return, realized_vol_21d]
    indexed by date.
    """
    for attempt in range(1, max_retries + 1):
        try:
            tkr = yf.Ticker(ticker)
            raw = tkr.history(start=start, end=end, auto_adjust=True)
            if raw.empty:
                raise ValueError(f"Empty data returned for {ticker}")
            break
        except Exception as exc:
            print(f"  Attempt {attempt}/{max_retries} failed for {ticker}: {exc}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)

    df = pd.DataFrame(index=raw.index)
    df["close"] = raw["Close"].values
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 21-trading-day rolling realized volatility, annualised (sqrt(252)).
    # This is the standard practitioner proxy for "current" volatility used
    # to seed model calibration (Heston v0, Merton diffusive sigma, etc.)
    df["realized_vol_21d"] = df["log_return"].rolling(21).std() * np.sqrt(252)

    df = df.dropna()
    df.index.name = "date"
    return df


def load_prices(asset: str) -> pd.DataFrame:
    """Load pre-fetched price data from data/<asset>.csv."""
    path = os.path.join(DATA_DIR, f"{asset}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No data file found at {path}. Run `python src/data_loader.py` first."
        )
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Fetching underlying price data\n" + "-" * 40)

    for name, ticker in TICKERS.items():
        print(f"\n{name} ({ticker}):")
        df = fetch_prices(ticker, START_DATE, END_DATE)
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path)
        print(f"  {len(df):,} rows saved to {path}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Mean realized vol (21d, annualised): {df['realized_vol_21d'].mean():.2%}")

    print("\nDone.")
