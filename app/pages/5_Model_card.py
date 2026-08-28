"""Model card: what holds up, what doesn't, and what the data can't see."""
from __future__ import annotations

import streamlit as st

from _data import figure, metrics, page_header

page_header("Model card",
            "The part where I say where the models fall short. Most cricket "
            "models don't, and I think that's a mistake.")

m = metrics()

st.header("How the models were checked")
st.markdown(
    """
    * **Season by season.** For each season, the model is trained on every
      season before it and scored on that one — the way you'd actually retrain
      it if it were in use. Nothing is scored on data it was trained on.
    * A **frozen split** (train up to 2021, test 2024–26) is kept separately and
      looked at once, as a check on how badly a model goes stale if you *don't*
      retrain it.
    * **Leakage** is checked by code, not by reading it back: one test rebuilds
      the per-ball features from a cut-short innings and checks the rows that
      survive are byte-for-byte identical; another checks no match ends up in
      two different splits.
    """
)

if m:
    st.header("Layer 1 — expected runs and wickets")
    x = m["layer1_xruns"]
    c1, c2, c3 = st.columns(3)
    c1.metric("xRuns error (runs/ball)", f"{x['walk_forward_rmse']:.3f}",
              f"{x['walk_forward_rmse'] - x['baseline_rmse']:+.3f} vs an over×wickets average",
              delta_color="inverse")
    c2.metric("runs below actual", f"{x['walk_forward_bias']:+.3f}", "per ball")
    c3.metric("wicket log loss", f"{x['wicket_log_loss']:.3f}",
              f"{x['wicket_log_loss'] - x['wicket_baseline_log_loss']:+.3f} vs a phase average",
              delta_color="inverse")
    st.markdown(
        "It only just beats a decent conditional average, and that's expected — "
        "a single ball is a 0, a 4 or a wicket, so most of the error is just "
        "noise you can't model away. xRuns isn't meant to call the next ball; "
        "it's meant to be an unbiased yardstick everything else is measured "
        "against. One known problem: because it only ever trains on past "
        "seasons, it runs about 0.056 runs a ball below what actually happens — "
        "it can't see this year's scoring going up. That's a roughly flat "
        "offset, so it moves everyone's number by the same amount and doesn't "
        "change the order. But it's there, and I'd rather flag it than quietly "
        "correct it."
    )
    if figure("xruns_by_phase.png"):
        st.image(str(figure("xruns_by_phase.png")))

    st.header("Layer 2 — win probability")
    wp = m["layer2_winprob"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Brier, test seasons", f"{wp['test_brier']:.3f}",
              f"{wp['test_brier'] - wp['test_base_brier']:+.3f} vs required-rate model",
              delta_color="inverse")
    c2.metric("2nd-innings AUC", f"{wp['innings2_auc']:.3f}")
    c3.metric("calibration error, all balls", f"{wp['all_ball_ece']:.3f}")
    st.markdown(
        """
        Put plainly: **a run chase is mostly about the run rate.**
        A plain regression on required rate, wickets in hand and balls left is
        already a strong model, and on the untouched 2024–26 seasons it's about
        level with the fancier gradient-boosted one on second-innings balls — a
        touch ahead on log loss. The Impact Player rule and deeper batting
        line-ups have if anything made chases *more* rate-driven.

        So the tree only gets a fifth of the weight in the blend. It earns that
        on the things the regression can't do: the first innings, the
        projected-total fan on the replay page, keeping the probabilities
        calibrated across the phases, and producing the ball-by-ball swings the
        Players tab adds up. Calibration is done on held-out predictions, never
        on a tail of recent seasons — one season is about 70 games, nowhere near
        enough to fit a calibration curve.
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
        f"The shrunk numbers are ridge-regression coefficients on runs above "
        f"expected — which is the same thing as putting a prior on every player "
        f"centred on average, with the amount of shrinkage picked by "
        f"cross-validation. **{im['n_players_scored']}** players get a number. "
        f"The 95% ranges come from resampling whole matches, not individual "
        f"balls, because balls in the same match aren't independent."
    )
    cc = st.columns(2)
    cc[0].caption("Raw numbers flatter them most")
    cc[0].dataframe(im["overrated_by_raw"], hide_index=True)
    cc[1].caption("Raw numbers sell them shortest")
    cc[1].dataframe(im["underrated_by_raw"], hide_index=True)
    c1, c2 = st.columns(2)
    if figure("scatter_bat.png"):
        c1.image(str(figure("scatter_bat.png")))
    if figure("shrinkage_funnel.png"):
        c2.image(str(figure("shrinkage_funnel.png")))

    st.header("Layer 4 — matchups")
    mu = m["layer4_matchup"]
    st.metric("of a raw matchup split, this much is real", f"{mu['shrinkage_ratio']:.0%}",
              f"raw {mu['mean_abs_raw_split_per_100']:.0f} → shrunk "
              f"{mu['mean_abs_shrunk_delta_per_100']:.0f} runs per 100")
    st.markdown(
        "About four-fifths of a raw batter-vs-type split is small-sample wobble. "
        "The type-level effects are real — off-spin goes for about 5 runs/100 "
        "less than average, the leggie for about 6 more — but the *personal* "
        "matchup, once you account for how thin it is, is usually not much."
    )

st.header("What it can't see, and what I've left out")
st.markdown(
    """
    * **No ball-tracking.** No line, no length, no pace off the pitch, no revs,
      no field settings, no pitch map. "The leggie" here is a stand-in for what
      Hawk-Eye would tell you directly.
    * **Bowling styles aren't in the data.** I typed them in by hand for the
      ~320 most-used players — enough to cover ~90% of balls faced and ~87% of
      balls bowled. The rest fall back to the type-level number.
    * **The simulator** treats bowling as average unless you give it a plan,
      doesn't track who's on strike ball to ball, and skips byes and wides off
      the bat.
    * **The optimiser** rolls forward expected values rather than simulating,
      and can't see the field or which batters are in.
    * **Super overs, no-results and DLS first innings** are left out of
      training. Four overs in IPL history have seven legal balls (umpires
      miscounting) — those are kept as they are.
    * **IPL only.** Adding another league is a one-line config change and a
      download, but I haven't retrained or re-checked anything on the BBL, PSL
      or internationals.
    """
)

st.header("What I'd build first with ball-tracking data")
st.markdown(
    """
    1. **Cluster deliveries by where they pitch and how fast, and rate batters
       against those clusters** — instead of "the leggie", the actual ball. The
       matchup model is starved of the one thing that makes a matchup real.
    2. **A shot-quality model.** xRuns that also knew the false-shot rate,
       whether the batter was beaten, how well he middled it. That would tell a
       batter riding his luck apart from one in complete control — the single
       biggest source of noise left in the player numbers.
    3. **Fielding.** The data here only catches the wicket-ending stuff. Range,
       boundaries saved, the pressure a gun fielder puts on — badly
       under-measured, and probably the quickest thing to get right.
    4. **How a bowler fades within a spell** — pace dropping off over to over,
       which the model barely touches right now.
    """
)
