"""Win-probability replay of a single match."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _data import ACCENT, BAD, GOOD, NEUTRAL, matches, page_header, replay

page_header("Match replay",
            "The win-probability line for a match, ball by ball. Below it, where "
            "the first-innings total was projected to land as the innings went "
            "on, and the deliveries that swung the game most.")

m = matches()
choice = st.selectbox("Match", m.label, index=0)
row = m[m.label == choice].iloc[0]
df = replay(int(row.match_id))
if df.empty:
    st.warning("This match isn't in the modelled set."); st.stop()

bat1 = df[df.innings == 1].batting_team.iloc[0]
bat2 = df[df.innings == 2].batting_team.iloc[0] if (df.innings == 2).any() else None

# --- win-probability ribbon (probability the *first-innings* team wins) ------
df = df.assign(
    ball_no=np.arange(len(df)) + 1,
    wp_team1=lambda d: np.where(d.innings == 1, d.win_prob, 1 - d.win_prob),
)
fig = go.Figure()
fig.add_hline(y=0.5, line=dict(color=NEUTRAL, width=1, dash="dot"))
fig.add_trace(go.Scatter(
    x=df.ball_no, y=df.wp_team1, mode="lines", line=dict(color=ACCENT, width=2),
    name=f"{bat1} to win", hovertemplate="ball %{x}<br>%{y:.0%}<extra></extra>"))
innings_break = int((df.innings == 1).sum())
fig.add_vline(x=innings_break, line=dict(color=NEUTRAL, width=1))
fig.update_layout(
    height=340, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 1], title="win probability"),
    xaxis_title="ball", showlegend=True, legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)

st.caption(f"**{row.result_team}** won. The line tracks {bat1}'s chances through "
           f"the game and finishes at {df.wp_team1.iloc[-1]:.0%}.")

# --- projected first-innings score fan --------------------------------------
first = df[df.innings == 1]
if "proj_q50" in first and first.proj_q50.notna().any():
    st.subheader("Where the first-innings total was heading")
    fig2 = go.Figure()
    x = np.arange(len(first)) + 1
    fig2.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                              y=np.r_[first.proj_q90, first.proj_q10[::-1]],
                              fill="toself", fillcolor="rgba(61,90,128,0.15)",
                              line=dict(width=0), name="10–90%"))
    fig2.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                              y=np.r_[first.proj_q70, first.proj_q30[::-1]],
                              fill="toself", fillcolor="rgba(61,90,128,0.30)",
                              line=dict(width=0), name="30–70%"))
    fig2.add_trace(go.Scatter(x=x, y=first.proj_q50, line=dict(color=ACCENT, width=2),
                              name="projected (median)"))
    fig2.add_trace(go.Scatter(x=x, y=first.score + first.runs_total.fillna(0),
                              line=dict(color="black", width=1.5), name="actual score"))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="ball", yaxis_title="runs",
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig2, use_container_width=True)

# --- biggest swings --------------------------------------------------------
st.subheader("The deliveries that turned it")
sw = df.reindex(df.wpa.abs().sort_values(ascending=False).index).head(8)
sw = sw.assign(
    over_ball=lambda d: d.over.astype(str) + "." + d.ball_in_over.astype(str),
    swing=lambda d: (d.wpa * 100).round(1),
    event=lambda d: np.where(d.wicket_type.fillna("") != "",
                             d.wicket_type + " — " + d.player_dismissed.fillna(""),
                             d.runs_off_bat.astype(str) + " off the bat"),
)
st.dataframe(
    sw[["innings", "over_ball", "batting_team", "bowler", "event", "swing"]]
    .rename(columns={"over_ball": "over", "batting_team": "batting",
                     "swing": "swing to the batting side (win %)"}),
    hide_index=True, use_container_width=True)
