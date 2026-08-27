"""Batter vs bowler-archetype, with the matchup shrunk to its real sample size."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _data import ACCENT, BAD, GOOD, NEUTRAL, load, metrics, page_header

page_header("Matchups",
            "How a batter fares against a *kind* of bowling, with the "
            "batter×archetype interaction penalised toward zero so a thin "
            "matchup falls back to the archetype prior instead of to noise.")

m = load("matchups.parquet")
mu = metrics().get("layer4_matchup", {})
if mu:
    st.markdown(
        f"Across **{mu['cells_ge_40_balls']:,}** batter–archetype cells with ≥40 balls, "
        f"the mean absolute *raw* split is **{mu['mean_abs_raw_split_per_100']:.0f}** "
        f"runs/100 above expectation. After shrinkage it is "
        f"**{mu['mean_abs_shrunk_delta_per_100']:.0f}** — about "
        f"**{mu['shrinkage_ratio']:.0%}** of the raw number. Most of what commentary "
        f"calls a matchup is sampling noise."
    )

players = sorted(m.name.unique())
default = players.index("V Kohli") if "V Kohli" in players else 0
name = st.selectbox("batter", players, index=default)
sub = m[m.name == name].sort_values("expected_runs_per_100", ascending=False)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=sub.archetype, x=sub.expected_runs_per_100, orientation="h",
    marker_color=ACCENT,
    error_x=dict(type="data",
                 array=(sub.ci_high_per_100 - sub.matchup_delta_per_100).clip(lower=0),
                 arrayminus=(sub.matchup_delta_per_100 - sub.ci_low_per_100).clip(lower=0),
                 thickness=1.2, color=NEUTRAL),
    hovertemplate="%{y}<br>%{x:.0f} runs/100  ·  %{customdata[0]} balls<extra></extra>",
    customdata=np.c_[sub.balls]))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                  xaxis_title="expected runs per 100 balls (shrunk, 95% interval)")
st.plotly_chart(fig, use_container_width=True)

disp = sub[["archetype", "balls", "archetype_prior_per_100", "expected_runs_per_100",
            "matchup_delta_per_100", "interaction_per_100"]].round(1)
disp.columns = ["archetype", "balls", "archetype prior /100", "shrunk expected /100",
                "total delta /100", "interaction /100"]
st.dataframe(disp, hide_index=True, use_container_width=True)
st.caption("**archetype prior** = batter's overall level + how the average batter "
           "does against this archetype. **interaction** = what is left that is "
           "specific to this batter against this bowling — the part shrinkage "
           "attacks hardest.")
