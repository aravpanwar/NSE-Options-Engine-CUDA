"""Build an HTML dashboard from the LSM engine's results CSV.

Produces three views:
  - 3D early-exercise premium surface per stock (puts)
  - Heatmap of American price for one stock's full strike x expiry grid
  - Summary bar chart: max premium per stock

Writes plots/dashboard.html (self-contained, viewable offline).
"""

from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def pivot_surface(sub: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = (sub.pivot_table(index="strike", columns="T", values=value, aggfunc="mean")
               .sort_index(axis=0).sort_index(axis=1))
    return grid.columns.to_numpy(), grid.index.to_numpy(), grid.to_numpy()


def premium_surface_figure(df: pd.DataFrame, opt_type: str) -> go.Figure:
    symbols = sorted(df["symbol"].unique())
    ncol = 3
    nrow = (len(symbols) + ncol - 1) // ncol

    fig = make_subplots(
        rows=nrow, cols=ncol,
        specs=[[{"type": "surface"}] * ncol for _ in range(nrow)],
        subplot_titles=symbols,
        horizontal_spacing=0.03, vertical_spacing=0.08,
    )

    for i, sym in enumerate(symbols):
        sub = df[(df["symbol"] == sym) & (df["type"] == opt_type)]
        if sub.empty:
            continue
        x, y, z = pivot_surface(sub, "premium")
        r, c = divmod(i, ncol)
        fig.add_trace(
            go.Surface(
                x=x, y=y, z=z, colorscale="Viridis", showscale=(i == 0),
                colorbar=dict(title="premium", x=1.02) if i == 0 else None,
                hovertemplate="T=%{x:.3f} yr<br>K=%{y:.2f}<br>premium=%{z:.4f}<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(
        title=f"American {opt_type.upper()} early-exercise premium = American price - European BSM",
        height=350 * nrow,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="#111",
        font=dict(color="#eee"),
    )
    for i in range(1, nrow * ncol + 1):
        fig.update_scenes(
            xaxis_title="T (years)", yaxis_title="strike", zaxis_title="premium",
            xaxis=dict(backgroundcolor="#111", gridcolor="#333"),
            yaxis=dict(backgroundcolor="#111", gridcolor="#333"),
            zaxis=dict(backgroundcolor="#111", gridcolor="#333"),
            row=(i - 1) // ncol + 1, col=(i - 1) % ncol + 1,
        )
    return fig


def max_premium_bar(df: pd.DataFrame) -> go.Figure:
    stats = (df.groupby(["symbol", "type"])["premium"]
               .max().reset_index())
    fig = go.Figure()
    for t, colour in [("put", "#f76c6c"), ("call", "#6cc3f7")]:
        sub = stats[stats["type"] == t].sort_values("symbol")
        fig.add_trace(go.Bar(
            name=t.upper(), x=sub["symbol"], y=sub["premium"],
            marker_color=colour,
            hovertemplate="%{x}<br>max " + t + " premium = %{y:.4f}<extra></extra>",
        ))
    fig.update_layout(
        title="Maximum early-exercise premium across strike/expiry grid",
        barmode="group",
        paper_bgcolor="#111", plot_bgcolor="#111",
        font=dict(color="#eee"),
        xaxis_title="symbol",
        yaxis_title="max premium (INR)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def american_heatmap(df: pd.DataFrame, symbol: str, opt_type: str = "put") -> go.Figure:
    sub = df[(df["symbol"] == symbol) & (df["type"] == opt_type)]
    x, y, z = pivot_surface(sub, "american")
    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=z, colorscale="Viridis",
        hovertemplate="T=%{x:.3f}<br>K=%{y:.2f}<br>price=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{symbol}: American {opt_type.upper()} price surface",
        xaxis_title="T (years)",
        yaxis_title="strike",
        paper_bgcolor="#111", plot_bgcolor="#111",
        font=dict(color="#eee"),
    )
    return fig


def summary_table_html(df: pd.DataFrame) -> str:
    by_sym = df.groupby("symbol").agg(
        spot=("spot", "first"),
        sigma=("sigma", "first"),
        contracts=("symbol", "count"),
        max_put_premium=("premium", lambda s: df.loc[s.index].query("type == 'put'")["premium"].max()),
        max_call_premium=("premium", lambda s: df.loc[s.index].query("type == 'call'")["premium"].max()),
        avg_gpu_ms=("gpu_ms", "mean"),
    ).round(4).reset_index()
    return by_sym.to_html(index=False, classes="summary", border=0)


def build_dashboard(df: pd.DataFrame, out_path: str, heatmap_symbol: str | None = None):
    fig_puts = premium_surface_figure(df, "put")
    fig_calls = premium_surface_figure(df, "call")
    fig_bar = max_premium_bar(df)
    if heatmap_symbol is None:
        heatmap_symbol = sorted(df["symbol"].unique())[0]
    fig_heat = american_heatmap(df, heatmap_symbol, "put")

    summary = summary_table_html(df)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>NSE American Option Pricer — LSM CUDA Dashboard</title>
<style>
  body {{ background:#0a0a14; color:#eee; font-family:system-ui,sans-serif; margin:24px; }}
  h1 {{ color:#fff; }}
  h2 {{ color:#9cc; border-bottom:1px solid #333; padding-bottom:4px; margin-top:36px; }}
  table.summary {{ border-collapse:collapse; margin:12px 0; }}
  table.summary th, table.summary td {{ padding:6px 12px; border:1px solid #333; }}
  table.summary th {{ background:#1a1a28; }}
  .caption {{ color:#888; font-size:0.9em; margin-bottom:12px; }}
</style>
</head>
<body>
<h1>NSE American Option Pricer</h1>
<p class="caption">
  Longstaff-Schwartz Monte Carlo on CUDA. Spot and 30-day realized volatility
  sourced from yfinance ({len(df)} contracts priced). Strike ladder is
  generated on NSE's standard strike-step rules.
</p>

<h2>Summary</h2>
{summary}

<h2>Early-exercise premium surfaces (puts)</h2>
<p class="caption">premium = American LSM price - European BSM price.
  Higher values indicate regions where early exercise adds the most value.</p>
{fig_puts.to_html(full_html=False, include_plotlyjs='cdn')}

<h2>Early-exercise premium surfaces (calls)</h2>
<p class="caption">For non-dividend stocks the theoretical American call premium is zero;
  observed non-zero values here are Monte Carlo noise.</p>
{fig_calls.to_html(full_html=False, include_plotlyjs=False)}

<h2>Max premium per stock</h2>
{fig_bar.to_html(full_html=False, include_plotlyjs=False)}

<h2>{heatmap_symbol}: American put price surface</h2>
{fig_heat.to_html(full_html=False, include_plotlyjs=False)}

</body>
</html>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/results.csv")
    ap.add_argument("--output", default="plots/dashboard.html")
    ap.add_argument("--heatmap-symbol", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Missing results CSV: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    if df.empty:
        print("Results CSV is empty.", file=sys.stderr)
        sys.exit(1)

    build_dashboard(df, args.output, args.heatmap_symbol)


if __name__ == "__main__":
    main()
