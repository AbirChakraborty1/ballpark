"""Model card: what is validated, what is not, and what the data cannot see."""
from __future__ import annotations

import streamlit as st

from _data import figure, metrics, page_header

page_header("Model card",
            "Publishing where the model is wrong is the point of this page.")

m = metrics()

st.header("Validation protocol")
st.markdown(
    """
    * **Walk-forward** is primary: for each season *t*, train on every prior
      season and score *t*. That is how a model in service is retrained, so it
      is the only honest estimate of deployed accuracy.
    * A **frozen split** (train ≤ 2021, validate 2022–23, test 2024–26) is kept
      as a drift stress-test. It is touched once.
    * **Leakage** is enforced mechanically: a test rebuilds the state table from
      a truncated innings and asserts the surviving rows are bit-identical, and
      another asserts no match ID appears in two splits.
    """
)

if m:
    st.header("Layer 1 — expected runs / wickets")
    x = m["layer1_xruns"]
    c1, c2, c3 = st.columns(3)
    c1.metric("xRuns RMSE", f"{x['walk_forward_rmse']:.3f}",
              f"{x['walk_forward_rmse'] - x['baseline_rmse']:+.3f} vs over×wickets baseline",
              delta_color="inverse")
    c2.metric("xRuns bias", f"{x['walk_forward_bias']:+.3f}", "runs/ball, walk-forward")
    c3.metric("wicket log loss", f"{x['wicket_log_loss']:.3f}",
              f"{x['wicket_log_loss'] - x['wicket_baseline_log_loss']:+.3f} vs baseline",
              delta_color="inverse")
    st.markdown(
        "The margin over a strong conditional-mean baseline is **small** — ball "
        "outcome is mostly irreducible noise. Layer 1 earns its place as a "
        "context baseline, not a sharp point predictor. One known bias: "
        "walk-forward xRuns runs ~4% (0.056 runs/ball) *below* actual, because a "
        "model trained only on prior seasons can't see this year's scoring "
        "inflation. It is a near-constant offset, so it leaves the Layer-3 "
        "*rankings* intact — but it is disclosed, not corrected away."
    )
    if figure("xruns_by_phase.png"):
        st.image(str(figure("xruns_by_phase.png")))

    st.header("Layer 2 — win probability")
    wp = m["layer2_winprob"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Brier, test seasons", f"{wp['test_brier']:.3f}",
              f"{wp['test_brier'] - wp['test_base_brier']:+.3f} vs RRR logistic",
              delta_color="inverse")
    c2.metric("2nd-innings AUC", f"{wp['innings2_auc']:.3f}")
    c3.metric("all-ball ECE", f"{wp['all_ball_ece']:.3f}")
    st.markdown(
        """
        The honest headline: **the second innings is largely a rate problem.**
        A logistic regression on required rate, wickets in hand and balls
        remaining is a genuinely strong model, and on the untouched 2024–26 test
        set it is level with — on log loss, slightly ahead of — the blended
        gradient-boosted model. Recent rule changes (the Impact Player,
        deeper batting orders) have made chases *more* rate-driven, not less.

        The complex model earns its keep on the things the logistic cannot do:
        the **first innings**, the **projected-score fan chart**, **phase-wise
        calibration**, and supplying the per-ball win-probability deltas that
        Layer 3 is built on. It is calibrated out-of-fold (isotonic on
        cross-validated predictions, never on a held-out tail of seasons).
        """
    )
    col1, col2 = st.columns(2)
    if figure("winprob_calibration.png"):
        col1.image(str(figure("winprob_calibration.png")))
    if figure("winprob_reliability.png"):
        col2.image(str(figure("winprob_reliability.png")))

    st.header("Layer 3 — player impact")
    im = m["layer3_impact"]
    st.markdown(
        f"Effects are ridge coefficients on runs above expectation — "
        f"mathematically an empirical-Bayes prior centred on average, with the "
        f"CV-selected penalty *as* the shrinkage. **{im['n_players_scored']}** "
        f"players scored. Intervals are a block bootstrap over whole matches."
    )
    cc = st.columns(2)
    cc[0].caption("Most overrated by raw runs-above-expected")
    cc[0].dataframe(im["overrated_by_raw"], hide_index=True)
    cc[1].caption("Most underrated by raw runs-above-expected")
    cc[1].dataframe(im["underrated_by_raw"], hide_index=True)
    c1, c2 = st.columns(2)
    if figure("scatter_bat.png"):
        c1.image(str(figure("scatter_bat.png")))
    if figure("shrinkage_funnel.png"):
        c2.image(str(figure("shrinkage_funnel.png")))

    st.header("Layer 4 — matchups")
    mu = m["layer4_matchup"]
    st.metric("shrinkage ratio", f"{mu['shrinkage_ratio']:.2f}",
              f"raw {mu['mean_abs_raw_split_per_100']:.0f} → shrunk "
              f"{mu['mean_abs_shrunk_delta_per_100']:.0f} runs/100")
    st.markdown(
        "Roughly **four-fifths** of a raw batter-vs-archetype split is sampling "
        "noise. Real matchup effects exist — off-spin is genuinely economical, "
        "leg-spin genuinely expensive — but batter-specific interactions are "
        "small once penalised."
    )

st.header("Known failure modes & limitations")
st.markdown(
    """
    * **No ball tracking.** No line/length, no pace off the pitch, no spin
      revs, no fielding positions, no pitch map. Every "archetype" here is a
      coarse proxy for what tracking data measures directly.
    * **No batter handedness in the source.** `reference/players_meta.csv` is
      hand-curated for the ~320 highest-volume players (~90% of balls have a
      curated batter, ~87% a curated bowler, ~79% both); the rest fall back to
      the archetype prior in the matchup model.
    * **The simulator** models bowling as league-average unless given a plan,
      does not track who is on strike ball to ball, and ignores byes and wides
      off the simulated bat.
    * **The optimiser** is an expected-value rollout, not a full simulation,
      and cannot see field settings or the specific batters at the crease.
    * **Super overs, abandoned matches and DLS first innings** are excluded
      from training; a handful of seven-ball overs (umpiring miscounts) are
      kept as-is.
    * **IPL only.** League is one config parameter, but the models have not
      been retrained or revalidated on BBL/PSL/international T20s.
    """
)

st.header("What I'd build first with ball-tracking data")
st.markdown(
    """
    1. **Replace archetypes with release-point + trajectory clusters.** The
       matchup model is currently starved of the one thing that makes a matchup
       real: what the ball actually does. Line/length/pace clusters per bowler,
       then batter effects against *those*.
    2. **A shot-quality model.** xRuns conditioned on beaten-edge, false-shot
       and contact-quality would separate a batter riding luck from one
       middling everything — the single biggest source of noise in Layer 3.
    3. **Fielding impact from tracked positions.** `fielder_*` gives catches and
       run-outs; tracking gives range, saved boundaries and pressure. That is a
       genuinely under-measured skill and the fastest win.
    4. **Bowler fatigue and match-ups within a spell** — pace drop-off over-to-
       over, which the current spell-over feature only gestures at.
    """
)
