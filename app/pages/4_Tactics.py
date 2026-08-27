"""Bowling-change optimiser — pre-computed for every close finish.

The optimiser itself (src/ballpark/models/optimise.py) runs an expected-value
rollout over every legal allocation of the remaining overs. Running it needs the
Layer-1 and Layer-2 models in memory, which the deployed app deliberately does
not load, so every close-finish state is solved at build time and written to
data/processed/app/tactics.parquet. Re-run `python scripts/build_app_bundle.py`
to refresh it.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from _data import ACCENT, BAD, GOOD, load, page_header

page_header("Tactics — bowling-change optimiser",
            "Second innings, close finishes only. For a live state the optimiser "
            "searches every legal allocation of the remaining overs across the "
            "bowlers with quota left and returns the one that minimises the "
            "chasing side's win probability — shown next to what the captain did.")

tac = load("tactics.parquet")
if tac.empty:
    st.info("No pre-computed tactics states in this build.")
    st.stop()

tac = tac.sort_values("delta", ascending=False)

agree = (tac.delta.abs() < 0.005).mean()
st.markdown(
    f"Run over every close-finish state in the data, the optimiser **agrees with "
    f"the captain exactly {agree:.0%} of the time** — bowler-quality edges are "
    f"small. The rows below are the tail where it doesn't. The very largest "
    f"swings are a knife-edge chase meeting a near-vertical win-probability "
    f"curve: three or four projected runs there read as tens of points, so they "
    f"say more about leverage than about the model's conviction."
)

show = tac.head(40)[["label", "from_over", "needed", "balls_left", "captain",
                     "optimiser", "delta", "result_team"]].rename(columns={
    "from_over": "from over", "needed": "runs needed", "balls_left": "balls left",
    "delta": "Δ win% handed over", "result_team": "won"})
st.dataframe(show, hide_index=True, use_container_width=True,
             column_config={"Δ win% handed over": st.column_config.NumberColumn(format="%.1%")})

st.divider()

opts = (tac.label + "  —  over " + tac.from_over.astype(str)).tolist()
default = next((i for i, o in enumerate(opts)
                if "Sunrisers Hyderabad v Delhi Capitals" in o and "ACA-VDCA" in o
                and o.endswith("17")), 0)
pick = st.selectbox("inspect a state", opts, index=default)
row = tac.iloc[opts.index(pick)]

st.caption(f"Start of over {int(row.from_over)}: **{int(row.score)}/{int(row.wickets)}**, "
           f"chasing {int(row.target)} — {int(row.needed)} needed off {int(row.balls_left)}. "
           f"{row.result_team} won.")

c1, c2 = st.columns(2)
c1.subheader("Optimiser")
c1.write(row.optimiser)
c1.metric("chasing side win probability", f"{row.optimiser_wp:.0%}")
c1.caption(f"projected final score {row.optimiser_score:.0f}")

c2.subheader("What the captain did")
c2.write(row.captain)
if pd.notna(row.captain_wp):
    c2.metric("chasing side win probability", f"{row.captain_wp:.0%}",
              f"{row.captain_wp - row.optimiser_wp:+.0%} vs optimal")

st.write("**Top 5 allocations the optimiser considered:**")
for alt in row.alternatives.split(" | "):
    st.write(f"- {alt}")

st.caption("Expected-value rollout: each over's runs and wickets are the "
           "Layer-1 expectation for the state, shifted by the bowler's shrunk "
           "Layer-3 effect; the projected end state is mapped through Layer 2. "
           "Quotas and the no-consecutive-overs rule are enforced; who is on "
           "strike and the field settings are not modelled.")
