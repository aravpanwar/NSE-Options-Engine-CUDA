# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go

df = pd.read_csv("C:/Projects/nse_options_engine/options_data.csv")
spot = df["spot"].iloc[0]

# Use calls only for IV surface (standard practice)
calls = df[df["type"] == "CE"].copy()

# Moneyness: strike relative to spot
calls["moneyness"] = calls["strike"] / spot

# Pivot to grid
pivot = calls.pivot_table(index="strike", columns="days", values="iv", aggfunc="mean")
pivot = pivot.dropna()

strikes = pivot.index.values
days    = pivot.columns.values
iv_grid = pivot.values

# Moneyness labels for x axis
moneyness_labels = [f"{s/spot:.3f}" for s in strikes]

fig = go.Figure(data=[go.Surface(
    x=days,
    y=strikes,
    z=iv_grid,
    colorscale="Viridis",
    colorbar=dict(title="IV %"),
    contours=dict(
        z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
    )
)])

# ATM line
atm_strike = calls.iloc[(calls["strike"] - spot).abs().argsort()[:1]]["strike"].values[0]

fig.update_layout(
    title=dict(
        text=f"NIFTY IV Surface — Spot: {spot:,.0f}",
        font=dict(size=20)
    ),
    scene=dict(
        xaxis=dict(title="Days to Expiry"),
        yaxis=dict(title="Strike Price"),
        zaxis=dict(title="Implied Volatility (%)"),
        camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
        bgcolor="rgb(10,10,20)",
        xaxis_backgroundcolor="rgb(10,10,20)",
        yaxis_backgroundcolor="rgb(10,10,20)",
        zaxis_backgroundcolor="rgb(10,10,20)",
    ),
    paper_bgcolor="rgb(10,10,20)",
    plot_bgcolor="rgb(10,10,20)",
    font=dict(color="white"),
    margin=dict(l=0, r=0, t=50, b=0),
    height=700,
)

# Add ATM vertical line annotation
fig.add_trace(go.Scatter3d(
    x=days,
    y=[atm_strike] * len(days),
    z=[pivot.loc[atm_strike, d] for d in days if atm_strike in pivot.index],
    mode="lines",
    line=dict(color="red", width=6),
    name=f"ATM ({atm_strike:,.0f})"
))

output_path = "C:/Projects/nse_options_engine/iv_surface.html"
fig.write_html(output_path)
print(f"IV Surface saved to {output_path}")
print(f"Opening in browser...")

import webbrowser
webbrowser.open(output_path)