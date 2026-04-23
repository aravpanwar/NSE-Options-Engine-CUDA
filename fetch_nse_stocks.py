"""Fetch NSE single-stock data and emit a contracts CSV for the LSM engine.

Real data from yfinance:
    - Spot (last close)
    - 30-day annualized log-return volatility
    - Underlyings are real NSE tickers

Synthetic but plausible:
    - Strike ladder: generated from spot using NSE's standard strike steps,
      covering roughly +/-20% moneyness. nseindia.com blocks bot traffic
      at the application layer (confirmed empirically), so we skip it and
      document the substitution.
    - Expiries: last Thursday of each of the next 3 calendar months, which
      matches real NSE single-stock expiry convention.
"""

from __future__ import annotations
import argparse
import calendar
import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK"]
RISK_FREE_RATE = 0.065  # RBI repo proxy
VOL_WINDOW_DAYS = 30


def nse_strike_step(spot: float) -> float:
    """NSE single-stock strike step, piecewise by price band."""
    if spot < 100:   return 2.5
    if spot < 250:   return 5.0
    if spot < 500:   return 10.0
    if spot < 1000:  return 20.0
    if spot < 3000:  return 20.0
    return 50.0


def build_strikes(spot: float, lo: float = 0.80, hi: float = 1.20, n_target: int = 15) -> list[float]:
    """Strike ladder from ~lo*spot to ~hi*spot on NSE step intervals, ~n_target strikes."""
    step = nse_strike_step(spot)
    # Widen step if needed to keep the count sane.
    while (hi - lo) * spot / step > 2 * n_target:
        step *= 2
    lo_k = math.floor(spot * lo / step) * step
    hi_k = math.ceil(spot * hi / step) * step
    strikes = []
    k = lo_k
    while k <= hi_k + 1e-9:
        strikes.append(round(k, 2))
        k += step
    return strikes


def last_thursday(year: int, month: int) -> dt.date:
    last_day = calendar.monthrange(year, month)[1]
    d = dt.date(year, month, last_day)
    while d.weekday() != 3:  # 3 = Thursday
        d -= dt.timedelta(days=1)
    return d


def next_n_monthly_expiries(today: dt.date, n: int = 3) -> list[dt.date]:
    out = []
    y, m = today.year, today.month
    while len(out) < n:
        cand = last_thursday(y, m)
        if cand > today:
            out.append(cand)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def fetch_spot_and_vol(ticker: str, window: int = VOL_WINDOW_DAYS) -> tuple[float, float]:
    """Return (spot, annualized log-vol) for an NSE ticker via yfinance."""
    yf_symbol = f"{ticker}.NS"
    t = yf.Ticker(yf_symbol)
    hist = t.history(period=f"{window + 15}d")
    if hist.empty:
        raise RuntimeError(f"yfinance returned no data for {yf_symbol}")
    closes = hist["Close"].dropna()
    if len(closes) < window:
        raise RuntimeError(f"{yf_symbol}: only {len(closes)} closes, need {window}")
    spot = float(closes.iloc[-1])
    log_rets = np.log(closes / closes.shift(1)).dropna().iloc[-window:]
    sigma = float(log_rets.std() * math.sqrt(252))
    return spot, sigma


def build_contracts(tickers: list[str], today: dt.date) -> pd.DataFrame:
    rows = []
    expiries = next_n_monthly_expiries(today, 3)
    for sym in tickers:
        try:
            spot, sigma = fetch_spot_and_vol(sym)
        except Exception as e:
            print(f"  [skip] {sym}: {e}", file=sys.stderr)
            continue
        strikes = build_strikes(spot)
        print(f"  {sym:10s}  spot={spot:>9.2f}  sigma={sigma:.4f}  "
              f"strikes={len(strikes)}  expiries={len(expiries)}")
        for exp in expiries:
            T = (exp - today).days / 365.0
            for K in strikes:
                for typ in ("call", "put"):
                    rows.append({
                        "symbol": sym,
                        "spot":   round(spot, 4),
                        "strike": K,
                        "expiry": exp.isoformat(),
                        "T":      round(T, 6),
                        "r":      RISK_FREE_RATE,
                        "sigma":  round(sigma, 6),
                        "type":   typ,
                    })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                    help="NSE ticker symbols (without .NS suffix)")
    ap.add_argument("--output", default="data/contracts.csv")
    args = ap.parse_args()

    today = dt.date.today()
    print(f"Fetching NSE data for {len(args.tickers)} tickers (today={today})")
    df = build_contracts(args.tickers, today)

    if df.empty:
        print("No contracts built. Aborting.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} contracts to {args.output}")
    print(df.groupby("symbol").size().to_string())


if __name__ == "__main__":
    main()
