"""Bowling-change optimiser on a real match state (second innings)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from _data import load, matches, page_header, replay

page_header("Tactics — bowling-change optimiser",
            "Second innings: pick a match and an over. The optimiser searches "
            "every legal allocation of the remaining overs across the bowlers "
            "who still have quota and reports the one that minimises the "
            "chasing side's win probability — next to what the captain did.")


@st.cache_resource(show_spinner="loading models…")
def _optimiser():
    try:
        from ballpark.models.optimise import BowlingOptimiser
        return BowlingOptimiser(effects=load("player_effects.parquet"))
    except Exception as e:
        st.error(f"Optimiser needs the trained models: {e}")
        st.stop()


m = matches()
# close finishes are where the death-overs allocation actually mattered
close = m[m.winner_runs.fillna(99).le(25) | m.winner_wickets.fillna(99).le(4)]
if len(close) > 20:
    m = close
choice = st.selectbox("match (close finishes only)", m.label, index=0)
row = m[m.label == choice].iloc[0]
inn = replay(int(row.match_id))
inn = inn[inn.innings == 2].sort_values("ball")
if inn.empty or int(inn.over.max()) < 17:
    st.warning("This chase ended too early to optimise the death overs. Pick another match.")
    st.stop()

last_over = int(inn.over.max())
from_over = st.slider("optimise from the start of over", 14, last_over - 1,
                      min(17, last_over - 1))
at = inn[inn.over == from_over].iloc[0]
before = inn[inn.over < from_over]
n_overs = int(inn.over.max()) - from_over + 1

name_to_id = load("player_effects.parquet").set_index("name").person_id.to_dict()
id_to_name = {v: k for k, v in name_to_id.items()}

counts = before.groupby("bowler").over.nunique().to_dict()
quotas = {name_to_id.get(b): 4 - counts.get(b, 0) for b in inn.bowler.unique() if name_to_id.get(b)}
quotas = {k: v for k, v in quotas.items() if v and v > 0}

st.caption(f"State at the start of over {from_over}: **{int(at.score)}/{int(at.wickets_lost)}**, "
           f"chasing {int(at.target)} — {int(at.target - at.score)} needed off {n_overs * 6}.")

if st.button("optimise", type="primary"):
    if sum(quotas.values()) < n_overs:
        st.warning("Not enough bowling quota left to cover the remaining overs from here.")
        st.stop()
    opt = _optimiser()
    start = {"innings": 2, "score": int(at.score), "wickets": int(at.wickets_lost),
             "balls_bowled": (from_over - 1) * 6, "allotted": 120, "venue": row.venue,
             "venue_era": "current", "target": int(at.target),
             "toss": bool(row.toss_winner == inn.batting_team.iloc[0])}
    last = name_to_id.get(before.bowler.iloc[-1]) if len(before) else None
    res = opt.optimise(start, quotas, n_overs, last_bowler=last)
    if res.empty:
        st.warning("No legal allocation from this state."); st.stop()

    actual_order = inn[inn.over >= from_over].groupby("over").bowler.first().tolist()
    actual_ids = [name_to_id.get(b) for b in actual_order]
    best = res.iloc[0]

    c1, c2 = st.columns(2)
    c1.subheader("Optimiser")
    c1.write("  →  ".join(id_to_name.get(b, "?") for b in best["order"]))
    c1.metric("chasing side win probability", f"{best.batting_win_prob:.1%}")

    mr = res[res.order.apply(lambda o: o == actual_ids)]
    if not mr.empty:
        a = float(mr.iloc[0].batting_win_prob)
        c2.subheader("What the captain did")
        c2.write("  →  ".join(actual_order))
        c2.metric("chasing side win probability", f"{a:.1%}",
                  f"{a - best.batting_win_prob:+.1%} vs optimal")

    st.dataframe(
        res.head(10).assign(allocation=lambda d: d.order.apply(
            lambda o: " → ".join(id_to_name.get(b, "?") for b in o)))
        [["allocation", "proj_score", "batting_win_prob"]]
        .rename(columns={"proj_score": "proj. final score", "batting_win_prob": "chase win prob"}),
        hide_index=True, use_container_width=True)

st.caption("Expected-value rollout: each over's runs and wickets are the "
           "Layer-1 expectation for the state, shifted by the bowler's shrunk "
           "Layer-3 effect. Quotas and the no-consecutive-overs rule are "
           "enforced; strike and field settings are not modelled.")
