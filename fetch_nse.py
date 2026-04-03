# -*- coding: utf-8 -*-
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
import os
from datetime import date
from scipy.stats import norm
import math

load_dotenv()
kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))
kite.set_access_token(os.getenv("KITE_ACCESS_TOKEN"))

def get_spot():
    ticker = yf.Ticker("^NSEI")
    spot = ticker.fast_info["lastPrice"]
    print(f"Spot price: {spot}")
    return spot

def bsm_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "CE" else max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def realistic_iv(strike, spot, T, base_iv=0.15):
    moneyness = (strike - spot) / spot
    # Skew: OTM puts expensive, OTM calls cheaper (NSE typical shape)
    skew = 0.10 * moneyness ** 2 - 0.05 * moneyness
    # Term structure: short-dated options have higher IV
    term_adj = 0.02 * math.exp(-T * 12)
    return max(0.08, base_iv + skew + term_adj)

def fetch_options_chain(symbol="NIFTY", num_expiries=5):
    print(f"Fetching {symbol} options chain ({num_expiries} expiries)...")

    instruments = kite.instruments("NFO")
    df_inst = pd.DataFrame(instruments)
    nifty_opts = df_inst[
        (df_inst["name"] == symbol) &
        (df_inst["instrument_type"].isin(["CE", "PE"]))
    ].copy()

    all_expiries = sorted(nifty_opts["expiry"].unique())[:num_expiries]
    print(f"Expiries: {all_expiries}")

    spot  = get_spot()
    r     = 0.065
    today = date.today()

    rows = []
    for expiry in all_expiries:
        T = max((expiry - today).days, 1) / 365.0
        weekly = nifty_opts[
            (nifty_opts["expiry"] == expiry) &
            (nifty_opts["strike"] >= spot * 0.90) &
            (nifty_opts["strike"] <= spot * 1.10)
        ].copy()

        print(f"  {expiry} — {len(weekly)} contracts, T={T:.4f}")

        for _, inst in weekly.iterrows():
            strike      = inst["strike"]
            option_type = inst["instrument_type"]
            iv          = realistic_iv(strike, spot, T)
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
                "days":       int(T * 365),
            })

    df = pd.DataFrame(rows)
    df = df[df["last_price"] > 0].reset_index(drop=True)
    df.to_csv("C:/Projects/nse_options_engine/options_data.csv", index=False)
    print(f"\nTotal contracts saved: {len(df)}")
    print(df.head(10).to_string())
    return df, spot

if __name__ == "__main__":
    df, spot = fetch_options_chain("NIFTY", num_expiries=5)