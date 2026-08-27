# ballpark: context-adjusted valuation from public IPL data

**~1,900 words. Live app: https://ballpark-mkmljvquubqdhwezbkgdtg.streamlit.app/ · Code: https://github.com/AbirChakraborty1/ballpark**

## The problem

A batting strike rate of 175 in the death overs means one thing off 800 balls
and something completely different off 40. A boundary in the third over of a
flat chase is not worth the same as a boundary in the nineteenth defending
eight-an-over. Raw T20 statistics are **context-blind** and, for anything
sliced finely enough to be tactically useful, **small-sample-noisy**. Every
number a coach actually wants — "who is the best death bowler", "does this
batter struggle against wrist-spin", "should we have held a bowler back" —
sits exactly where those two problems bite hardest.

`ballpark` rebuilds the conceptual core of a context-adjusted valuation engine
from cricsheet ball-by-ball data (1,243 IPL matches, 2008–2026; 295,732
deliveries). It has no ball tracking, no fielding positions, no pitch maps.
That is deliberate: the project is not an attempt to compete on data. It is an
attempt to get the **modelling judgement** right — temporal validation,
calibration, and shrinkage of small-sample player effects — on data anyone can
download.

## Why raw numbers mislead, concretely

Two mechanisms, and the whole design follows from separating them.

**Context.** The run value of a delivery depends on the match state — phase,
wickets in hand, required rate, how set the batter is. If you do not model that
state, a player who batted more often in easy situations looks more skilful
than one who did not. The fix is an **expected-runs model** conditioned only on
state and deliberately blind to player identity: the neutral baseline that says
what an average batter–bowler pair produces *in exactly this situation*.
Everything downstream is measured against it.

**Sample size.** Once you slice to "this batter, this over range, this kind of
bowling", you have tens of balls, and the observed rate is mostly noise. The
fix is **partial pooling**: estimate each player's effect under a prior that
pulls short careers toward the population mean, with the strength of the pull
chosen by cross-validation.

## The models

**Layer 1 — expected runs and wickets.** A LightGBM multiclass model over runs
off the bat `{0,1,2,3,4,5,6}`, plus binary heads for dismissal and extras,
conditioned on 27 state features. Two features that public IPL work
usually omits matter here: `striker_balls_faced` (the "set batter" effect) and
recency weights — IPL scoring per ball rose 18% between 2021 and 2024–26, so
each ball is down-weighted by `0.5 ** (age_in_seasons / 1.5)`, a half-life
tuned only on pre-2024 walk-forward seasons.

**Layer 2 — win probability.** The second innings is a clean supervised
problem: `P(chasing team wins | state)`. The first innings is modelled directly,
with a separate quantile model projecting the final-score distribution for the
fan chart. The hard part is **sample size again**: a chase contributes ~120
rows that share one outcome, so 85,000 balls carry ~1,000 independent
observations. Left unconstrained a gradient-boosted model memorises match
trajectories — AUC 0.85 and a log loss *worse than guessing 50%*. The shipped
model is a deliberately blunt GBM (15 leaves, 2,000 samples per leaf, no
recency decay because chase dynamics do not drift) blended with a logistic
regression on required rate, wickets and balls (weight 0.2 on the GBM — see
below), then isotonically calibrated on **cross-validated** predictions rather
than a held-out tail of seasons too short to estimate a calibration map.

**Layer 3 — player impact.** Two numbers per player. **Wins added** is
`Σ (P(win | after ball) − P(win | before ball))`, attributed to striker, bowler
and — via cricsheet's new `fielder_*` fields — the fielder on catches and
run-outs. It is descriptive: it rewards high-leverage contributions, which is
the point of a context statistic, but leverage is mostly handed to a player by
circumstance. **Shrunk true rate** is the skill estimate: runs above the
Layer-1 expectation, regressed on one-hot batter and bowler columns under an L2
penalty. Ridge on an offset is not a shortcut around a hierarchical model — an
L2 penalty on player effects *is* a Gaussian prior centred on average, so the
coefficients are posterior means and the CV-selected penalty is the prior
variance. It fits in seconds where a crossed-effects mixed model would not fit
at all. Intervals come from a block bootstrap over whole matches.

**Layer 4 — matchups, simulation, tactics.** A ridge GLM adds batter × bowler-
archetype interaction terms (pace/spin × arm × wrist/finger, six archetypes); a
thin interaction is penalised to zero, at which point the prediction falls back
to the archetype prior. A vectorised Monte Carlo engine plays innings out from
any state. A bowling-change optimiser searches every legal allocation of the
remaining overs and returns the one that minimises the batting side's win
probability.

## Validation and calibration evidence

Everything is **walk-forward**: for each season *t*, train on all prior seasons
and score *t*. A frozen split (train ≤ 2021, test 2024–26, touched once) is kept
only as a drift stress-test. Leakage is enforced by a test that rebuilds the
state table from a truncated innings and asserts the surviving rows are
bit-identical, and by a test that no match appears in two splits.

| model | metric | ballpark | baseline | baseline is |
|---|---|---|---|---|
| Layer 1 xRuns | RMSE (walk-fwd) | **1.682** | 1.694 | runs by over × wickets |
| Layer 1 xRuns | bias | **−0.056** /ball | — | — |
| Layer 1 wicket | log loss | **0.194** | 0.195 | rate by over × wickets |
| Layer 2 WP | Brier, 2024–26 test | **0.182** | 0.190 | logistic on RR / wkts / balls |
| Layer 2 WP | 2nd-innings AUC | **0.88** | — | — |
| Layer 2 WP | ECE, all balls | **0.023** | — | — |
| Layer 4 matchup | mean \|effect\|/100 | **7.5** | 38.9 | unshrunk raw split |

The Layer 1 margin over a strong conditional-mean baseline is small — ball
outcome is mostly irreducible noise, and pretending otherwise would be the
warning sign. There is one known bias: walk-forward xRuns runs ~4% (0.056
runs/ball) *below* actual, because a model trained only on prior seasons cannot
anticipate this year's scoring inflation. It is a near-constant offset — it
shifts every player's "above expectation" by the same amount and so leaves the
Layer 3 *rankings* intact — but it is a real limitation, not something to
paper over.

The Layer 2 story is the one worth being candid about. **The second innings is
largely a rate problem.** A three-number logistic regression is a genuinely
strong model. On the untouched 2024–26 test set the shipped blend beats it on
the pooled metrics (Brier 0.182 vs 0.190, log loss 0.550 vs 0.555), but on
second-innings balls alone the plain logistic is level with it — recent rule
changes (the Impact Player, deeper batting orders) have made chases *more*
rate-driven, not less, so the GBM's weight in the blend is deliberately only
0.2. The blend earns its keep on what the logistic cannot do: the first
innings, the projected-score fan, phase-wise calibration (reliability is within
~3 points of the diagonal in every phase of the second innings), and supplying
the per-ball win-probability deltas Layer 3 depends on.

## Three findings

**1. Most of a "matchup" is noise.** Across 692 batter-vs-archetype cells with
at least 40 balls, the mean absolute *raw* deviation from expectation is 38.9
runs per 100 balls. After shrinkage it is **7.5** — nineteen percent of the
raw figure. Real archetype effects exist: off-spin is genuinely economical
(−5.0 runs/100 vs an average batter), leg-spin genuinely expensive (+6.3). But
the batter-*specific* "he can't play left-arm spin" interaction, once penalised
for its sample size, is about a fifth of what a 40-ball television split
implies.

**2. Raw "runs above expected" massively overstates short careers.** Among
players with 200+ balls, the standard deviation of the naive raw-minus-expected
metric is 12.9 runs/100; the shrunk estimate's is 8.4, and for players under
150 balls the naive spread of 56 collapses to 4. The players who move most
between the two lists — flattered by small hot streaks, or dragged down by
cold ones — are the ones a raw leaderboard gets most wrong. Josh
Fraser-McGurk's naive +65 runs/100 becomes a still-excellent but human +23;
Chris Morris's +23 becomes +1.

**3. Death bowling is the scarcest skill in the auction.** The bowlers whose
shrunk effect survives with the tightest intervals and the largest magnitude —
Bumrah (36 runs saved per 100 balls, 95% interval 30–42), Narine, Rashid Khan —
are almost all death or middle-overs specialists, and the gap between them and
the median bowler is far wider than the equivalent gap among batters. Wins
added tells the same story from the leverage side: the highest per-ball WPA
contributions in the data are concentrated in overs 16–20.

## The tactics engine, on one real over

The optimiser searches every legal allocation of the remaining overs across the
bowlers with quota left — each over scored by expected runs shifted by the
bowler's shrunk Layer-3 effect, the projected total mapped through Layer 2 —
and returns the one that minimises the chasing side's win probability. Run over
all 1,186 close-finish states in the data, **it agrees with the captain exactly
92% of the time**. The value is in the tail.

**2019, Sunrisers Hyderabad v Delhi Capitals, Visakhapatnam.** DC, chasing 163,
needed 42 off 24 starting the 18th over. SRH's captain gave Basil Thampi — a
medium-pacer — the 18th, between Bhuvneshwar Kumar and Khaleel Ahmed. The
optimiser instead alternates Bhuvneshwar and Khaleel through all four remaining
overs: projected DC total 161, and the win probability it hands DC drops from
74% to 29%. DC won by five wickets.

The 45-point swing is mostly a statement about **leverage**, not about the
model's confidence in the change: when a chase is on a knife edge, the
win-probability curve is near-vertical, so the three or four runs a front-line
over saves over a part-timer's reads as a large probability move. Thirty-one of
the 1,186 states show a swing above 15 points, and every one of them is a
genuine last-five-overs coin-flip.

## Limitations

No ball tracking, so no line, length, pace off the pitch or field settings —
every archetype is a coarse proxy for what tracking measures directly. Batter
handedness is not in cricsheet and is hand-curated for the top ~320 players
(~90% of balls have a curated batter, ~87% a curated bowler, ~79% both); the
rest fall back to the archetype prior in the matchup model. The
simulator models bowling as league-average unless given a plan and does not
track strike ball-to-ball. The optimiser is an expected-value rollout. IPL only
— league is one config line, but nothing has been retrained on other
competitions.

## What I'd build first with ball-tracking data

1. **Replace archetypes with release-point and trajectory clusters.** The
   matchup model is starved of the one thing that makes a matchup real: what
   the ball actually does. Cluster deliveries by line / length / pace, then
   estimate batter effects against those clusters.
2. **A shot-quality model** — xRuns conditioned on false-shot rate,
   beaten-edge and contact quality. This would separate a batter riding luck
   from one middling everything, which is the single largest source of residual
   noise in Layer 3.
3. **Fielding impact from tracked positions** — range, boundaries saved,
   pressure created. `fielder_*` only captures the wicket-ending events; the
   rest is a genuinely under-measured skill and the fastest available win.
4. **In-spell bowler fatigue** — over-to-over pace decline, which the current
   `spell_over` feature only gestures at.

*ballpark is a demonstration of method, not a product. The interesting question
is what the same shrinkage and calibration discipline does when pointed at data
that can actually see the ball.*
