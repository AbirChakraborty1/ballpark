"""Win-probability replay of a single match."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _data import ACCENT, BAD, NEUTRAL, matches, page_header, replay

page_header("Match replay",
            "The win-probability line for a match, ball by ball — a line for "
            "each side, and a marker at every wicket. Below it, where the "
            "first-innings total was projected to land as the innings went on, "
            "and the deliveries that swung the game most.")

m = matches()
choice = st.selectbox("Match", m.label, index=0)
row = m[m.label == choice].iloc[0]
df = replay(int(row.match_id))
if df.empty:
    st.warning("This match isn't in the modelled set."); st.stop()

bat1 = df[df.innings == 1].batting_team.iloc[0]
bat2 = df[df.innings == 2].batting_team.iloc[0] if (df.innings == 2).any() else "the chasing side"


def _clean(x) -> str:
    s = "" if x is None else str(x)
    return "" if s.lower() in ("", "nan", "none") else s


def _dismissal(r: pd.Series) -> str:
    """'c Dhoni b Bumrah', 'lbw b Rashid', 'run out (Kohli)' — from the fields
    Cricsheet gives (fielder only lands for catches, run-outs and stumpings)."""
    wt = _clean(r.wicket_type).lower()
    b, f = _clean(r.bowler), _clean(r.fielder_1)
    if wt == "caught":
        return f"c {f} b {b}" if f else f"c b {b}"
    if wt == "caught and bowled":
        return f"c & b {b}"
    if wt == "bowled":
        return f"b {b}"
    if wt == "lbw":
        return f"lbw b {b}"
    if wt == "stumped":
        return f"st {f} b {b}" if f else f"st b {b}"
    if wt == "run out":
        return f"run out ({f})" if f else "run out"
    if wt == "hit wicket":
        return f"hit wicket b {b}"
    return _clean(r.wicket_type) or "out"


df = df.assign(
    ball_no=np.arange(len(df)) + 1,
    wp_team1=lambda d: np.where(d.innings == 1, d.win_prob, 1 - d.win_prob),
)
wk = df[df.wicket_type.map(_clean) != ""].copy()
wk = wk.assign(
    who=wk.player_dismissed.map(_clean),
    how=wk.apply(_dismissal, axis=1),
    at_score=wk.score + wk.runs_total.fillna(0),
)


def _wicket_markers(frame: pd.DataFrame, y) -> go.Scatter:
    return go.Scatter(
        x=frame.ball_no, y=y, mode="markers", name="wicket",
        marker=dict(symbol="circle", size=11, color="white",
                    line=dict(color=BAD, width=2.5)),
        customdata=np.stack([frame.who, frame.how, frame.batting_team], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                      "<br>%{customdata[2]} · ball %{x}<extra>wicket</extra>")


# --- win-probability lines, one per side, with wickets marked ---------------
fig = go.Figure()
fig.add_hline(y=0.5, line=dict(color=NEUTRAL, width=1, dash="dot"))
fig.add_trace(go.Scatter(
    x=df.ball_no, y=df.wp_team1, mode="lines", line=dict(color=ACCENT, width=2),
    name=f"{bat1} to win",
    hovertemplate=f"{bat1}<br>ball %{{x}}<br>%{{y:.0%}}<extra></extra>"))
fig.add_trace(go.Scatter(
    x=df.ball_no, y=1 - df.wp_team1, mode="lines", line=dict(color=BAD, width=2),
    name=f"{bat2} to win",
    hovertemplate=f"{bat2}<br>ball %{{x}}<br>%{{y:.0%}}<extra></extra>"))
if not wk.empty:
    fig.add_trace(_wicket_markers(wk, wk.wp_team1))
innings_break = int((df.innings == 1).sum())
fig.add_vline(x=innings_break, line=dict(color=NEUTRAL, width=1))
fig.update_layout(
    height=360, margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(range=[0, 1], title="win probability"),
    xaxis_title="ball", showlegend=True, legend=dict(orientation="h", y=1.14),
    hovermode="closest")
st.plotly_chart(fig, use_container_width=True)

st.caption(f"**{row.result_team}** won. Each line is one side's chance ball by "
           f"ball (they add to 100%); {bat1} finishes at "
           f"{df.wp_team1.iloc[-1]:.0%}. Circles mark the fall of a wicket — "
           f"hover for who and how.")

# --- projected first-innings score fan --------------------------------------
first = df[df.innings == 1].reset_index(drop=True)
if "proj_q50" in first and first.proj_q50.notna().any():
    st.subheader("Where the first-innings total was heading")
    fig2 = go.Figure()
    x = np.arange(len(first)) + 1
    fig2.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                              y=np.r_[first.proj_q90, first.proj_q10[::-1]],
                              fill="toself", fillcolor="rgba(61,90,128,0.15)",
                              line=dict(width=0), name="10–90%", hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                              y=np.r_[first.proj_q70, first.proj_q30[::-1]],
                              fill="toself", fillcolor="rgba(61,90,128,0.30)",
                              line=dict(width=0), name="30–70%", hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=x, y=first.proj_q50, line=dict(color=ACCENT, width=2),
                              name="projected (median)"))
    fig2.add_trace(go.Scatter(x=x, y=first.score + first.runs_total.fillna(0),
                              line=dict(color="black", width=1.5), name="actual score"))
    fw = wk[wk.innings == 1]
    if not fw.empty:
        fig2.add_trace(_wicket_markers(fw, fw.at_score))
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="ball", yaxis_title="runs",
                       legend=dict(orientation="h", y=1.14), hovermode="closest")
    st.plotly_chart(fig2, use_container_width=True)

# --- biggest swings --------------------------------------------------------
st.subheader("The deliveries that turned it")
sw = df.reindex(df.wpa.abs().sort_values(ascending=False).index).head(8)
sw = sw.assign(
    over_ball=lambda d: d.over.astype(str) + "." + d.ball_in_over.astype(str),
    swing=lambda d: (d.wpa * 100).round(1),
    event=lambda d: np.where(d.wicket_type.map(_clean) != "",
                             d.player_dismissed.map(_clean) + " — " + d.apply(_dismissal, axis=1),
                             d.runs_off_bat.astype(str) + " off the bat"),
)
st.dataframe(
    sw[["innings", "over_ball", "batting_team", "bowler", "event", "swing"]]
    .rename(columns={"over_ball": "over", "batting_team": "batting",
                     "swing": "swing to the batting side (win %)"}),
    hide_index=True, use_container_width=True)
