"""Impact leaderboards: wins added, and raw vs shrunk skill."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _data import ACCENT, BAD, GOOD, NEUTRAL, load, page_header

page_header("Players",
            "Two numbers per player: wins added (what happened) and shrunk true "
            "rate (how well they actually bat or bowl, once 40 balls stops being "
            "treated as a sample).")

role = st.radio("role", ["Batting", "Bowling"], horizontal=True)
r = "bat" if role == "Batting" else "bowl"

eff = load("player_effects.parquet")
eff = eff[eff.role == r].copy()
wpa = load("player_wpa.parquet")
wpa = wpa[wpa.role == r].groupby(["person_id", "name"], as_index=False).agg(
    balls=("balls", "sum"), wins_added=("wins_added", "sum"))

min_balls = int(st.slider("minimum balls", 100, 2000, 300, step=50))
eff = eff[eff.balls >= min_balls]

tab1, tab2 = st.tabs(["Raw vs shrunk", "Wins added"])

with tab1:
    show_shrunk = st.toggle("show shrunk true rate (off = raw above expectation)", value=True)
    col = "shrunk_per_100" if show_shrunk else "naive_above_expected_per_100"
    label = "shrunk runs/100 above expectation" if show_shrunk else "raw runs/100 above expectation"
    top = eff.reindex(eff[col].sort_values(ascending=False).index)
    disp = top[["name", "balls", "raw_per_100", "naive_above_expected_per_100",
                "shrunk_per_100", "ci_low", "ci_high"]].head(20).round(1)
    disp.columns = ["player", "balls", "raw/100", "raw − xp /100", "shrunk /100", "ci low", "ci high"]
    st.dataframe(disp, hide_index=True, use_container_width=True)

    st.subheader("How much 'above expectation' survives shrinkage")
    fig = go.Figure()
    lim = float(np.abs(np.r_[eff.naive_above_expected_per_100, eff.shrunk_per_100]).max()) * 1.05
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                             line=dict(color=NEUTRAL, dash="dash", width=1), showlegend=False))
    fig.add_trace(go.Scatter(
        x=eff.naive_above_expected_per_100, y=eff.shrunk_per_100, mode="markers",
        marker=dict(size=np.clip(eff.balls / eff.balls.max() * 26, 5, 26),
                    color=ACCENT, opacity=0.55, line=dict(width=0.5, color="white")),
        text=eff.name,
        hovertemplate="%{text}<br>raw %{x:.1f}<br>shrunk %{y:.1f}<extra></extra>"))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="naive: raw − expected (runs/100)",
                      yaxis_title="shrunk true effect (runs/100)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points far below the diagonal are overrated by their raw numbers; "
               "far above, underrated. Bubble size is career balls.")

with tab2:
    w = wpa[wpa.balls >= min_balls].reindex(
        wpa[wpa.balls >= min_balls].wins_added.sort_values(ascending=False).index).head(20)
    w = w[["name", "balls", "wins_added"]].round(2)
    w.columns = ["player", "balls", "wins added (2016–2026)"]
    st.dataframe(w, hide_index=True, use_container_width=True)
    st.caption("Win probability added, summed over every ball the player was "
               "involved in. High-leverage contributions count for more — which "
               "is the point of a context-adjusted statistic, but also means "
               "this rewards opportunity, not just skill.")
