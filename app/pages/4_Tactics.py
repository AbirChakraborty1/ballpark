"""Bowling-change optimiser — pre-computed for every close finish.

The optimiser itself (src/ballpark/models/optimise.py) runs an expected-value
rollout over every legal allocation of the remaining overs. Running it needs the
Layer-1 and Layer-2 models loaded, which the deployed app doesn't do, so every
close-finish state is solved at build time and written to
data/processed/app/tactics.parquet. Re-run `python scripts/build_app_bundle.py`
to refresh it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from _data import ACCENT, BAD, GOOD, load, page_header

page_header("Tactics — bowling-change optimiser",
            "Second innings, tight finishes. From a given point in the chase, "
            "the optimiser tries every legal way to bowl out the rest of the "
            "innings — who bowls which over, within the quotas — and picks the "
            "one that gives the chasing side the worst chance. Shown next to "
            "what the captain actually did, and what the difference between the "
            "two was worth.")

tac = load("tactics.parquet")
if tac.empty:
    st.info("No solved states in this build.")
    st.stop()

tac = tac.sort_values("delta", ascending=False)

close = (tac.delta.abs() < 0.02).mean()
biggest = tac.delta.abs().max()
st.markdown(
    f"Run on every tight finish in the data, the optimiser's plan and the one "
    f"the captain actually used come out **within two win-probability points of "
    f"each other {close:.0%} of the time**. At the death the gap between a "
    f"front-line over and a fifth bowler's is only three or four runs, and "
    f"spread over the last few overs that usually doesn't add up to much. The "
    f"rows below are where the two disagree most. Even the largest gap is about "
    f"{biggest * 100:.0f} points, and it's the model preferring one attack to "
    f"another — not a single over swinging the game."
)

show = tac.head(40)[["label", "from_over", "needed", "balls_left", "captain",
                     "optimiser", "delta", "result_team"]].rename(columns={
    "from_over": "from over", "needed": "runs needed", "balls_left": "balls left",
    "delta": "the captain's call gave the batting side", "result_team": "won"})
st.dataframe(show, hide_index=True, use_container_width=True,
             column_config={"the captain's call gave the batting side":
                            st.column_config.NumberColumn(format="%.0%")})

st.divider()

opts = (tac.label + "  —  over " + tac.from_over.astype(str)).tolist()
default = next((i for i, o in enumerate(opts)
                if "Kolkata Knight Riders v Sunrisers Hyderabad" in o
                and "Rajiv Gandhi" in o and o.endswith("15")), 0)
pick = st.selectbox("Take a closer look at one", opts, index=default)
row = tac.iloc[opts.index(pick)]

st.caption(f"Start of the {int(row.from_over)}th over: "
           f"**{int(row.score)}/{int(row.wickets)}**, chasing {int(row.target)}, "
           f"{int(row.needed)} needed off {int(row.balls_left)}. {row.result_team} won.")

c1, c2 = st.columns(2)
c1.subheader("The optimiser's call")
c1.write(row.optimiser)
c1.metric("chasing side to win", f"{row.optimiser_wp:.0%}")
c1.caption(f"projects the chase to about {row.optimiser_score:.0f}, "
           f"taken as a spread rather than a fixed number")

c2.subheader("What the captain did")
c2.write(row.captain)
if pd.notna(row.captain_wp):
    lift = row.captain_wp - row.optimiser_wp
    c2.metric("chasing side to win", f"{row.captain_wp:.0%}")
    if abs(lift) < 0.005:
        c2.caption("Same as the optimiser's plan — nothing in it.")
    else:
        c2.caption(f"About **{lift * 100:.0f} points** more than the "
                   f"optimiser's plan — what bowling it this way was worth to "
                   f"the batting side.")

st.write("**The five it rated best:**")
for alt in row.alternatives.split(" | "):
    st.write(f"- {alt}")

st.caption("Scoring an allocation: each remaining over gets the expected runs "
           "and wickets for the match state, nudged by how much better or worse "
           "than average that bowler is — the shrunk rating from the Players "
           "tab — with a small extra cost for spin in the 19th and 20th, which "
           "goes for about a run an over more than pace there in the data. That "
           "gives a projected total, which is then treated as a range rather "
           "than a single number — a chase can land a couple of overs' worth of "
           "runs either side of the projection — and the win probability is "
           "averaged over that range. Quotas and no back-to-back overs are kept "
           "to. It doesn't know who's on strike or where the field is.")
