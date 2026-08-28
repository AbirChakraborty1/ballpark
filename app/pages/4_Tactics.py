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
            "what the captain actually did.")

tac = load("tactics.parquet")
if tac.empty:
    st.info("No solved states in this build.")
    st.stop()

tac = tac.sort_values("delta", ascending=False)

agree = (tac.delta.abs() < 0.005).mean()
st.markdown(
    f"Across every tight finish in the data, the optimiser lands on **the exact "
    f"allocation the captain used {agree:.0%} of the time**. The gap between a "
    f"front-line over and a fifth bowler's is only three or four runs, so most "
    f"of the time there's nothing to argue about. The rows below are the ones "
    f"where it disagrees. The biggest swings are chases balanced on a knife "
    f"edge, where a few projected runs is the difference between a comfortable "
    f"win and a nervy one — so they say more about how much was riding on the "
    f"over than about the model being sure."
)

show = tac.head(40)[["label", "from_over", "needed", "balls_left", "captain",
                     "optimiser", "delta", "result_team"]].rename(columns={
    "from_over": "from over", "needed": "runs needed", "balls_left": "balls left",
    "delta": "win% given to the batting side", "result_team": "won"})
st.dataframe(show, hide_index=True, use_container_width=True,
             column_config={"win% given to the batting side":
                            st.column_config.NumberColumn(format="%.1%")})

st.divider()

opts = (tac.label + "  —  over " + tac.from_over.astype(str)).tolist()
default = next((i for i, o in enumerate(opts)
                if "Sunrisers Hyderabad v Delhi Capitals" in o and "ACA-VDCA" in o
                and o.endswith("17")), 0)
pick = st.selectbox("Take a closer look at one", opts, index=default)
row = tac.iloc[opts.index(pick)]

st.caption(f"Start of the {int(row.from_over)}th over: "
           f"**{int(row.score)}/{int(row.wickets)}**, chasing {int(row.target)}, "
           f"{int(row.needed)} needed off {int(row.balls_left)}. {row.result_team} won.")

c1, c2 = st.columns(2)
c1.subheader("The optimiser's call")
c1.write(row.optimiser)
c1.metric("chasing side to win", f"{row.optimiser_wp:.0%}")
c1.caption(f"projects the chase to reach {row.optimiser_score:.0f}")

c2.subheader("What the captain did")
c2.write(row.captain)
if pd.notna(row.captain_wp):
    c2.metric("chasing side to win", f"{row.captain_wp:.0%}",
              f"{row.captain_wp - row.optimiser_wp:+.0%} vs the optimiser")

st.write("**The five it rated best:**")
for alt in row.alternatives.split(" | "):
    st.write(f"- {alt}")

st.caption("Scoring an allocation: each remaining over gets the expected runs "
           "and wickets for the match state, nudged by how much better or worse "
           "than average that bowler is — the shrunk rating from the Players "
           "tab — and the projected total goes through the win-probability "
           "model. Quotas and no back-to-back overs are kept to. It doesn't "
           "know who's on strike or where the field is.")
