# ballpark: context-adjusted valuation from public IPL data

Abir Chakraborty · [LinkedIn](https://www.linkedin.com/in/abir-chakraborty1/) · mail2abirchakraborty@gmail.com

**Live app: https://ballpark-mkmljvquubqdhwezbkgdtg.streamlit.app/ · Code: https://github.com/AbirChakraborty1/ballpark**

## What bugged me

A strike rate of 175 in the death overs means something off 800 balls. Off 40,
it means almost nothing. And a six in the third over of a flat chase is not the
same thing as a six in the nineteenth defending nine an over, but average and
strike rate treat them identically. Almost every question worth asking about a
T20 player — is he actually a death bowler, does he really struggle against
wrist spin, was that the right bowler for the 18th — runs straight into one or
both of those.

I'm a cricket fan, not an analyst by trade, and I wanted to see how far you can
get on those questions with nothing but public ball-by-ball data. `ballpark` is
what came out: an expected-runs model, a calibrated win-probability model, and
player ratings that get regressed toward average when the sample is thin. Built
from Cricsheet — 1,243 IPL matches, 2008 to 2026, 295,732 deliveries. No
ball-tracking, no field settings, no pitch maps. I wasn't trying to compete on
data. I wanted to get the method right and see what it turned up.

## The two problems, separately

**Situation.** What a ball is worth depends on the state of the game: the phase,
wickets in hand, the required rate, whether the batter is set. If you don't
model that, a player who happened to bat more often when it was easy looks
better than one who didn't. The fix is a model of expected runs off the bat that
only knows the match state, nothing about who's involved — an average batter
against an average bowler, right here. Everything else gets measured against
that.

**Sample size.** Slice to "this batter, these overs, this kind of bowling" and
you're down to tens of balls. Whatever number you read off is mostly noise. The
fix is to pull every estimate back toward what players like him generally do,
harder the less data there is behind it, with the strength of the pull chosen by
cross-validation rather than by feel.

## The models

**Layer 1 — expected runs and wickets.** A gradient-boosted model over what
comes off the bat — 0 through 6 — plus separate heads for a dismissal and for
an extra, on 27 features describing the state. Two of those features matter more
than people expect. One is how many balls the striker has faced this innings:
a batter twenty balls in scores a lot faster than one who's just walked out, and
a model that ignores that is going to be wrong. The other is recency. IPL
scoring per ball went up about 18% between 2021 and 2024–26, so each ball is
down-weighted by its age, with the half-life tuned only on seasons up to 2023.

**Layer 2 — win probability.** The second innings is the clean version of the
problem: P(chasing side wins | state). The first innings is done directly, with
a separate model projecting the final total for the fan chart on the replay
page. The catch is sample size again. A chase is one result spread across ~120
balls, so 85,000 rows carry maybe a thousand real observations. Let a boosted
model loose on that and it memorises the shape of individual games: great AUC,
and probabilities so overconfident the log loss is worse than saying 50% every
ball. So the model that ships is a blunt tree — 15 leaves, big leaves, no
recency decay because chase maths doesn't drift — blended four-to-one with a
plain regression on required rate, wickets and balls, and then calibrated on
held-out predictions rather than on a tail of recent seasons — one season is
about 70 games, nowhere near enough to fit a calibration curve.

**Layer 3 — player impact.** Two numbers. *Wins added* is the sum of the
win-probability swing on every ball a player was part of, credited to the
striker, the bowler, and — using Cricsheet's newer fielder fields — the fielder
on catches and run-outs. It rewards doing things when the game's in the balance,
which is the point, but it also rewards batting three in a strong side, so it's
a record of what happened rather than a skill rating. The *shrunk rate* is the
skill read: runs above the Layer-1 expectation, regressed on one-hot batter and
bowler columns with an L2 penalty. That penalty is the same thing as a prior on
every player centred on average — the coefficients are posterior means and
cross-validation picks the prior's tightness. It fits in seconds where a proper
crossed mixed model wouldn't fit at all. The ranges come from resampling whole
matches.

**Layer 4 — matchups, simulation, tactics.** The matchup model adds batter ×
bowling-type columns to the same regression (six types: pace and spin, by arm,
finger and wrist). A type a batter has barely faced gets penalised toward zero,
so the prediction falls back to how batters like him do against it. There's a
vectorised innings simulator, and a bowling-change optimiser that tries every
legal way to bowl out a chase and picks the one that gives the batting side the
worst odds.

## How the models were checked

Everything is scored season by season: for each season, train on every season
before it, score that one. That's how you'd retrain it in use, so it's the only
honest read on how it does live. A frozen split (train to 2021, test 2024–26,
looked at once) is kept as a check on staleness. Leakage is checked by code —
one test rebuilds the feature table from a cut-short innings and checks the
surviving rows are byte-identical; another checks no match ends up in two
splits.

| model | measure | ballpark | to beat | what that is |
|---|---|---|---|---|
| Layer 1 xRuns | error, season-by-season | **1.682** | 1.694 | runs by over × wickets |
| Layer 1 xRuns | bias | **−0.056** /ball | — | — |
| Layer 1 wicket | log loss | **0.194** | 0.195 | rate by over × wickets |
| Layer 2 win prob | Brier, 2024–26 | **0.182** | 0.190 | regression on rate / wickets / balls |
| Layer 2 win prob | 2nd-innings AUC | **0.88** | — | — |
| Layer 2 win prob | calibration error | **0.023** | — | — |
| Layer 4 matchup | avg \|effect\| /100 | **7.5** | 38.9 | the raw split |

Layer 1 only just clears a good conditional average, and that's the right
result — one ball is a 0, a 4 or a wicket, so most of the error isn't
model-able. The one thing to flag is that bias: because it only trains on past
seasons, xRuns comes in about 0.056 runs a ball below reality — it can't see
this year's scoring going up. It's close to a flat offset, so it shifts every
player's "above expected" by the same amount and leaves the order alone. But
it's a real limitation and I'd rather say so.

Layer 2 is the one to be straight about. **A run chase is mostly about the
rate.** A three-number regression is already strong, and on the untouched
2024–26 seasons it's about level with the boosted model on second-innings balls —
slightly ahead on log loss. If anything the Impact Player rule and deeper
line-ups have made chases more rate-driven. So the tree only gets a fifth of the
weight. It earns that on the things the regression can't do: the first innings,
the projected-total fan, calibration that holds across the phases (within about
3 points of the line everywhere in the second innings), and the ball-by-ball
swings that Layer 3 needs.

## Three things that came out of it

**1. Most of a "matchup" is a small sample talking.** Across 692 batter × type
pairings with at least 40 balls behind them, the average gap from expected is
38.9 runs per 100 balls. Regress each one for how thin it is and that drops to
7.5 — nineteen percent. The type-level effects are real: off-spin goes for
about 5 runs/100 less than average, the leggie for about 6 more. It's the
personal "he can't play it" part that mostly isn't there once you account for
the 40 balls it's built on.

**2. Raw "runs above expected" massively overrates short careers.** Among
players with 200+ balls, the spread of the raw number is 12.9 runs/100; the
shrunk one is 8.4. Under 150 balls the raw spread of 56 collapses to 4. The
players who jump most between the two lists — a hot streak here, a cold run
there — are exactly the ones a raw table gets wrong. Fraser-McGurk's raw +65
runs/100 becomes a still-very-good +23; Chris Morris's +23 becomes +1.

**3. Death bowling is the scarcest thing in the auction.** The bowlers whose
shrunk number holds up with the tightest range and the biggest edge — Bumrah
(36 runs saved per 100, 95% range 30–42), Narine, Rashid — are nearly all death
or middle-overs men, and the gap from them to a median bowler is far wider than
the equivalent gap among batters. Wins added says the same from the other end:
the highest per-ball swings in the data sit in overs 16 to 20.

## The optimiser on the death overs

Run over all 1,180 tight finishes in the data, the optimiser's over-by-over
plan and the one the captain actually used land **within two win-probability
points of each other 84% of the time**, and within five points 90% of the time.
Eight finishes out of 1,180 show a gap above ten points; one above fifteen. The
difference between a front-line death over and a fifth bowler's is three or four
runs, and spread over the last five overs that usually doesn't add up to much.

One thing the rollout has to be told, because nothing else in the stack sees
it: spin costs more than pace in the last two overs. Runs conceded against the
model's neutral expectation, 2008–2026, are +0.4 an over for pace at overs
19–20 and +1.6 for spin — +0.6 versus +2.0 at the 20th — and even the best
spinners sit around +1 there. The expected-runs model is blind to bowling type
and the player rating is a career average earned mostly in the middle overs, so
a top spinner was being credited an economy edge at the death that the data
doesn't support. A spinner's over now carries a small extra cost at the 19th
and 20th — half the size of that raw gap, since some of it is desperation spin.

The biggest single disagreement is KKR v SRH at Hyderabad in 2023. SRH needed
48 off 36 with six wickets in hand; KKR's captain used Shardul Thakur and
Vaibhav Arora for the seam overs around Narine and Chakaravarthy. The optimiser
would have gone spin-heavy through the middle of the death — Narine and Varun
bowled out — with a seamer kept back for the 20th. On the model's read of those
bowlers that is worth about ten runs across the remaining overs, which moves
SRH from roughly 79% to 61%. That's the model backing one attack over another —
a real if arguable call, not a single over deciding the game.

An earlier draft had a different example here: SRH v DC at Visakhapatnam in
2019, where the optimiser looked to gain seven points by swapping one Basil
Thampi over for Bhuvneshwar. That gap turned out to be an artefact. Asked for a
single projected total, the win model behaves like a step at the target, so a
sub-run difference in the projection was reading as seven points. The optimiser
now treats the projection as a spread — the total can land a couple of overs'
worth of runs either side of the mean — and integrates the win probability over
it. That 2019 case comes out dead level, and so do most of the ones that used
to look dramatic. It's also why a genuinely open chase like 49 off 30 with
seven wickets in hand now reads as a real contest, not 5%.

## What it can't see

No ball-tracking, so no line, length, pace off the pitch or field. Every
bowling "type" is a rough stand-in for what Hawk-Eye would give you directly.
Bowling styles aren't in Cricsheet — I typed them in for the ~320 most-used
players, covering roughly 90% of balls faced and 87% bowled; the rest fall back
to the type-level number. The simulator treats bowling as average unless given a
plan and doesn't track the strike ball to ball. The optimiser rolls expected
values forward rather than simulating, with a variance term on the projected
total so a knife-edge chase reads near 50/50 instead of snapping to a
near-certain result. IPL only — adding another league is a
config line and a download, but I haven't re-checked anything on the BBL or PSL.

## What I'd build first with ball-tracking data

1. **Cluster deliveries by where they land and how fast, and rate batters
   against those clusters** — the actual ball instead of "the leggie". The
   matchup model is starved of the one thing that makes a matchup real.
2. **A shot-quality model.** xRuns that also knew the false-shot rate, whether
   the batter was beaten, how cleanly he middled it — that separates a batter
   riding his luck from one in control, which is the biggest source of noise
   left in the player numbers.
3. **Fielding.** The data here only catches the wicket-ending events. Range,
   boundaries saved, the pressure a gun fielder creates — badly under-measured,
   and probably the quickest thing to get right.
4. **How a bowler fades within a spell** — pace dropping off over to over, which
   right now the model barely touches.

The whole thing is really a proof of method. What I'd like to know is what the
same approach — checking it season by season, calibrating it, regressing the
small samples — does when it's pointed at data that can actually see the ball.
