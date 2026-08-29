"""ballpark -- context-adjusted IPL valuation. App entry point."""
from __future__ import annotations

import streamlit as st

from _data import metrics

st.set_page_config(page_title="ballpark", page_icon="🏏", layout="wide")

st.title("ballpark")
st.markdown(
    "Strike rate and average flatten everything. A hundred off 55 chasing 12 an "
    "over looks the same on the scorecard as a hundred off 55 on a featherbed "
    "with the game already gone. A bowler who goes at eight bowling the 18th and "
    "20th to set batters comes out behind one who goes at eight bowling the 8th "
    "to a new pair.\n\n"
    "I'm a cricket fan who wanted numbers that account for the situation, and "
    "that don't fall apart when the sample is small. `ballpark` is my attempt at "
    "it, built from public ball-by-ball data (Cricsheet, 2008–2026). There is no "
    "ball-tracking here — no line and length, no field settings. What is here is "
    "the modelling: an expected-runs model that prices every ball by the state "
    "it was bowled in, a win-probability model I've checked actually holds up "
    "out of sample, and player ratings that get pulled back toward average when "
    "there isn't enough data to trust them. Roughly the questions Smart Stats "
    "and WinViz answer, from data anyone can download."
)

FEEDBACK_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSdiQHsmJK0twAtgUyxLv_QnEOkaXDrL-nszGXlWiyQKgTYXIA/viewform"
)

st.caption(
    "Abir Chakraborty  ·  mail2abirchakraborty@gmail.com  ·  "
    "[LinkedIn](https://www.linkedin.com/in/abir-chakraborty1/)  ·  "
    "[GitHub](https://github.com/AbirChakraborty1/ballpark)  ·  "
    f"[feedback form]({FEEDBACK_URL})  ·  "
    "a portfolio project, not a product"
)

m = metrics()
if m:
    c1, c2, c3, c4 = st.columns(4)
    d = m["data"]
    c1.metric("deliveries modelled", f"{d['deliveries']:,}")
    c1.caption(f"{d['matches']:,} matches, {d['seasons'][0]}–{d['seasons'][1]}")

    wp = m["layer2_winprob"]
    c2.metric("win-prob Brier (test)", f"{wp['test_brier']:.3f}",
              f"{wp['test_brier'] - wp['test_base_brier']:+.3f} vs required-rate model",
              delta_color="inverse")
    c2.caption(f"2nd-innings AUC {wp['innings2_auc']:.2f} · calibration error {wp['test_ece']:.3f}")

    x = m["layer1_xruns"]
    c3.metric("xRuns error (walk-forward)", f"{x['walk_forward_rmse']:.3f}",
              f"{x['walk_forward_rmse'] - x['baseline_rmse']:+.3f} vs over×wickets average",
              delta_color="inverse")
    c3.caption(f"runs/ball · runs {x['walk_forward_bias']:+.3f} low, on purpose (see model card)")

    mu = m["layer4_matchup"]
    c4.metric("how much of a matchup is real", f"{mu['shrinkage_ratio']:.0%}")
    c4.caption(f"raw split {mu['mean_abs_raw_split_per_100']:.0f} → after shrinkage "
               f"{mu['mean_abs_shrunk_delta_per_100']:.0f} runs/100")

st.divider()

st.markdown(
    """
    ### Where to look first

    1. **Match replay** — pick a game, watch the win-probability line swing, see
       the projected first-innings total narrow ball by ball, and the deliveries
       that turned it.
    2. **Players** — the same leaderboard raw and shrunk, side by side. Watch the
       small-sample names slide back toward the pack when you switch it on.
    3. **Matchups** — pull up a "he can't play the leggie" reputation and see how
       much of it holds up once you account for how few balls it's built on.
    4. **Tactics** — a bowling-change optimiser run on real death overs. Mostly
       it does what the captain did. The interesting part is when it doesn't.
    5. **Model card** — where the models hold up, where they don't, and what a
       public dataset can't see.
    6. **Full-match simulator** — a separate tool. The full IPL ball-by-ball
       history is preloaded (add other leagues' Cricsheet zips if you like);
       set two line-ups and it simulates the game ball by ball in your browser.
    """
)

st.info(
    "**On the data.** No ball-tracking, no fielding positions, no pitch maps — "
    "just public Cricsheet. I'm not trying to out-data a provider that has all "
    "of that. The point was to get the method right: check the models season by "
    "season the way you'd actually retrain them, calibrate the probabilities, "
    "and regress the small samples. The last section of the model card is what "
    "I'd want to build first if I had tracking data to work with.",
    icon="📌",
)

st.divider()
lc, rc = st.columns([3, 2])
lc.markdown(
    "**Found something off, or have an idea?** I'd genuinely like to hear it — "
    "a wrong number, a player the model reads badly, a feature worth adding."
)
rc.link_button("Leave feedback", FEEDBACK_URL, use_container_width=True)
st.caption(
    "Abir Chakraborty  ·  [mail2abirchakraborty@gmail.com](mailto:mail2abirchakraborty@gmail.com)"
    "  ·  [LinkedIn](https://www.linkedin.com/in/abir-chakraborty1/)"
)
