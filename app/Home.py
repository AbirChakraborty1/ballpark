"""ballpark -- context-adjusted IPL valuation. App entry point."""
from __future__ import annotations

import streamlit as st

from _data import metrics

st.set_page_config(page_title="ballpark", page_icon="🏏", layout="wide")

st.title("ballpark")
st.markdown(
    "**Raw IPL statistics are context-blind and small-sample-noisy.** A boundary "
    "in the 3rd over is not worth a boundary in the 19th defending 8-an-over, and "
    "a death-overs strike rate off 40 balls is mostly luck. `ballpark` rebuilds "
    "the conceptual core of a context-adjusted valuation engine from public "
    "ball-by-ball data — an expected-runs model, a calibrated win-probability "
    "model, and partially-pooled player effects — and uses them to answer "
    "tactical questions."
)

m = metrics()
if m:
    c1, c2, c3, c4 = st.columns(4)
    d = m["data"]
    c1.metric("deliveries modelled", f"{d['deliveries']:,}")
    c1.caption(f"{d['matches']:,} matches, {d['seasons'][0]}–{d['seasons'][1]}")

    wp = m["layer2_winprob"]
    c2.metric("win-prob Brier (test)", f"{wp['test_brier']:.3f}",
              f"{wp['test_brier'] - wp['test_base_brier']:+.3f} vs RRR baseline",
              delta_color="inverse")
    c2.caption(f"2nd-innings AUC {wp['innings2_auc']:.2f} · ECE {wp['test_ece']:.3f}")

    x = m["layer1_xruns"]
    c3.metric("xRuns RMSE (walk-forward)", f"{x['walk_forward_rmse']:.3f}",
              f"{x['walk_forward_rmse'] - x['baseline_rmse']:+.3f} vs over×wickets",
              delta_color="inverse")
    c3.caption(f"bias {x['walk_forward_bias']:+.3f} runs/ball")

    mu = m["layer4_matchup"]
    c4.metric("matchup shrinkage ratio", f"{mu['shrinkage_ratio']:.2f}")
    c4.caption(f"raw split {mu['mean_abs_raw_split_per_100']:.0f} → "
               f"shrunk {mu['mean_abs_shrunk_delta_per_100']:.0f} runs/100")

st.divider()

st.markdown(
    """
    ### A 60-second path

    1. **Match replay** — open a recent final, watch the win-probability ribbon
       and the projected-score fan, and see the biggest-swing balls picked out.
    2. **Players** — flip the leaderboard between *raw* and *shrunk* impact and
       watch the small-sample names collapse toward the mean.
    3. **Matchups** — check a famous "he can't play left-arm spin" matchup and
       see how much of it survives shrinkage.
    4. **Tactics** — the bowling-change optimiser on real death overs: where it
       agrees with the captain, and the tail where it doesn't.
    5. **Model card** — where the models are calibrated, where they are not,
       and what a public dataset cannot see.
    """
)

st.info(
    "**On the data.** No ball tracking, no fielding positions, no pitch maps — "
    "this is public cricsheet data. The point is not to compete with a tracking "
    "provider on data; it is to get the modelling judgement right: temporal "
    "validation, calibration, and shrinkage of small-sample effects. The model "
    "card's last section is what I would build first *with* tracking data.",
    icon="📌",
)
