from kiteconnect import KiteConnect
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import date

load_dotenv()
kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

def get_spot():
    ticker = yf.Ticker("^NSEI")
    spot = ticker.fast_info["lastPrice"]
    print(f"Spot price (yfinance): {spot}")
    return spot

def bsm_price(S, K, T, r, sigma, option_type):
    from scipy.stats import norm
    import math
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "CE" else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def realistic_iv(strike, spot, base_iv=0.15):
    # Volatility skew � OTM puts have higher IV (typical NSE smile)
    moneyness = (strike - spot) / spot
    skew = 0.08 * moneyness ** 2 - 0.04 * moneyness
    return max(0.08, base_iv + skew)

def fetch_options_chain(symbol="NIFTY"):
    print(f"Fetching {symbol} options chain...")

    # Real strikes and expiries from Kite
    instruments = kite.instruments("NFO")
    df_inst = pd.DataFrame(instruments)
    nifty_opts = df_inst[
        (df_inst["name"] == symbol) &
        (df_inst["instrument_type"].isin(["CE", "PE"]))
    ].copy()

    nearest_expiry = sorted(nifty_opts["expiry"].unique())[0]
    print(f"Using expiry: {nearest_expiry}")

    spot = get_spot()
    r    = 0.065  # RBI repo rate

    weekly = nifty_opts[
        (nifty_opts["expiry"] == nearest_expiry) &
        (nifty_opts["strike"] >= spot * 0.92) &
        (nifty_opts["strike"] <= spot * 1.08)
    ].copy()

    # Time to expiry in years
    today  = date.today()
    expiry = nearest_expiry
    T      = max((expiry - today).days, 1) / 365.0
    print(f"Days to expiry: {int(T * 365)}, T = {T:.4f}")

    rows = []
    for _, inst in weekly.iterrows():
        strike      = inst["strike"]
        option_type = inst["instrument_type"]
        iv          = realistic_iv(strike, spot)
        price       = bsm_price(spot, strike, T, r, iv, option_type)

        rows.append({
            "expiry":     str(expiry),
            "strike":     strike,
            "type":       option_type,
            "last_price": round(price, 2),
            "iv":         round(iv * 100, 2),
            "spot":       spot,
            "T":          round(T, 6),
            "r":          r,
        })

    df = pd.DataFrame(rows)
    df = df[df["last_price"] > 0].reset_index(drop=True)
    df.to_csv("C:/Projects/nse_options_engine/options_data.csv", index=False)

    print(f"\nSaved {len(df)} contracts")
    print(df.head(10).to_string())
    return df, spot

if __name__ == "__main__":
    df, spot = fetch_options_chain("NIFTY")
