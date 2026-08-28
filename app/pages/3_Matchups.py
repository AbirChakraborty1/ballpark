"""Batter vs bowler-archetype, with the matchup shrunk to its real sample size."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _data import ACCENT, BAD, GOOD, NEUTRAL, load, metrics, page_header

page_header("Matchups",
            "How a batter goes against a *type* of bowling — right-arm pace, "
            "the leggie, left-arm orthodox, and so on. Where a batter has faced "
            "a type only a few dozen times, his personal record against it is "
            "regressed hard toward how batters like him generally do, so the "
            "number isn't just noise.")

m = load("matchups.parquet")
mu = metrics().get("layer4_matchup", {})
if mu:
    st.markdown(
        f"Take every batter–bowling-type pairing with at least 40 balls behind "
        f"it — **{mu['cells_ge_40_balls']:,}** of them. The average raw gap from "
        f"what you'd expect is **{mu['mean_abs_raw_split_per_100']:.0f}** runs per "
        f"100 balls. Once you regress each one for how little it's built on, that "
        f"drops to **{mu['mean_abs_shrunk_delta_per_100']:.0f}** — about "
        f"**{mu['shrinkage_ratio']:.0%}** of it. Most of what gets called a "
        f"matchup on TV is a small sample doing the talking."
    )

players = sorted(m.name.unique())
default = players.index("V Kohli") if "V Kohli" in players else 0
name = st.selectbox("Batter", players, index=default)
sub = m[m.name == name].sort_values("expected_runs_per_100", ascending=False)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=sub.archetype, x=sub.expected_runs_per_100, orientation="h",
    marker_color=ACCENT,
    error_x=dict(type="data",
                 array=(sub.ci_high_per_100 - sub.matchup_delta_per_100).clip(lower=0),
                 arrayminus=(sub.matchup_delta_per_100 - sub.ci_low_per_100).clip(lower=0),
                 thickness=1.2, color=NEUTRAL),
    hovertemplate="%{y}<br>%{x:.0f} runs per 100  ·  %{customdata[0]} balls<extra></extra>",
    customdata=np.c_[sub.balls]))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                  xaxis_title="expected runs per 100 balls  (shrunk, with 95% range)")
st.plotly_chart(fig, use_container_width=True)

disp = sub[["archetype", "balls", "archetype_prior_per_100", "expected_runs_per_100",
            "matchup_delta_per_100", "interaction_per_100"]].round(1)
disp.columns = ["bowling type", "balls", "prior /100", "shrunk /100",
                "his edge /100", "just this matchup /100"]
st.dataframe(disp, hide_index=True, use_container_width=True)
st.caption("**prior** is where you'd put him against this type with no personal "
           "record to go on — his overall level, plus how batters generally do "
           "against it. **his edge** is how much better or worse than that the "
           "model has him. **just this matchup** is the slice of that which is "
           "specific to him against this bowling — the bit that gets regressed "
           "hardest, and it's usually small.")
