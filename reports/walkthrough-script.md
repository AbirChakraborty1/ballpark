# ballpark — walkthrough narration script

Recording notes: ~5 minutes total, conversational pace. The self-playing version
lives at `reports/walkthrough.html` (published as an artifact); this is the plain
script if you want to record a real voiceover over a screen capture of the app.

---

**Scene 1 — Title _(0:00, ~12s)_**
> ballpark. A cricket-analytics project, explained in plain language — what it
> does, and why every number in it is built the way it is.

**Scene 2 — The problem _(0:12, ~25s)_**
> Cricket numbers mislead in two ways. First, context: a boundary in the third
> over isn't worth a boundary in the nineteenth defending eight an over — raw
> stats don't know the difference. Second, sample size: a death-overs strike
> rate off forty balls is mostly luck. Slice the data finely enough to be
> useful, and it turns to noise. Every number a coach actually wants sits
> exactly where these two problems bite hardest.

**Scene 3 — The data _(0:37, ~22s)_**
> The data is just public ball-by-ball records — two hundred and ninety-six
> thousand deliveries, twelve hundred and forty-three IPL matches, 2008 to 2026,
> from cricsheet-dot-org. No ball tracking, no fielding positions, no pitch
> maps. The point isn't to out-data a broadcaster — it's to get the modelling
> judgement right.

**Scene 4 — Layer 1, expected runs _(0:59, ~26s)_**
> The first model is a neutral yardstick. It asks: for this exact situation —
> this over, these wickets in hand, this required rate, this batter has already
> faced twenty balls — what is an average ball worth? That's "expected runs".
> It's deliberately blind to who's batting. Every player number later in the
> project is measured against it.

**Scene 5 — Layer 2, win probability _(1:25, ~28s)_**
> The second model is the odds, recalculated after every ball. The
> win-probability ribbon turns a scorecard into a story — you can see the exact
> over a match turned. And it's honest: across every ball in the data, when it
> says seventy percent, the team goes on to win about seventy percent of the
> time. Its calibration error is around two points.

**Scene 6 — Layer 3, shrinkage _(1:53, ~26s)_**
> Forty balls is not a sample. A raw "runs above average" number for a short
> career is mostly wobble. So every player estimate is pulled toward the league
> mean — hard for small samples, barely at all for long careers. The maths is a
> ridge regression, which turns out to be a Bayesian prior in disguise.

**Scene 7 — Layer 3, the Bumrah example _(2:19, ~24s)_**
> Take Jasprit Bumrah. His raw economy looks ordinary — under seven an over. But
> he bowls the hardest overs to the best batters, and once the model accounts
> for who and when, it rates him the best bowler in the entire dataset — saving
> thirty-six runs per hundred balls, and it's ninety-five percent sure that's
> between thirty and forty-two.

**Scene 8 — Layer 4, matchups _(2:43, ~24s)_**
> "He can't play left-arm spin." Across nearly seven hundred batter-versus-
> bowling-type matchups, the average raw split is thirty-nine runs per hundred
> balls. Shrink it for sample size and it's seven-and-a-half — about a fifth.
> Real bowling-type effects exist. The personal "he can't play it" part is
> mostly a forty-ball illusion.

**Scene 9 — Layer 4, tactics _(3:07, ~24s)_**
> Which bowler, which over? An optimiser plays out every legal way to bowl the
> remaining overs and picks the one that gives the batting side the lowest
> chance of winning. Run over eleven hundred real close finishes, it agrees with
> the captain ninety-four percent of the time. The honest headline is that
> captains are mostly right — the value is in the other six percent.

**Scene 10 — The app _(3:31, ~22s)_**
> All of this is a web app with six tabs you can click through: match replay,
> player leaderboards, the matchup explorer, the bowling-change optimiser — and
> a model card that does the rare thing, and publishes exactly where the models
> are wrong.

**Scene 11 — The simulator _(3:53, ~24s)_**
> The sixth tab is a full-match simulator. Upload any Cricsheet T20 zip, pick
> two line-ups and a venue, and it simulates the match ball by ball — entirely
> in your browser, nothing uploaded. Recency-weighted form, opponent strength,
> a pace-versus-spin matchup, set-batter and momentum effects, and a calibrated
> win probability, all fit from the data you loaded.

**Scene 12 — Close _(4:17, ~15s)_**
> That's ballpark. Public data, honest models. Every number in this tour is
> real, from the 2008-to-2026 cricsheet dump. The app and the code are linked
> below.

---

Links to show on screen for scene 12:
- App — https://ballpark-mkmljvquubqdhwezbkgdtg.streamlit.app/
- Code — https://github.com/AbirChakraborty1/ballpark
