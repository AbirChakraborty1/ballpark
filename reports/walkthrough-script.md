# ballpark — walkthrough narration script

~5 minutes, conversational. The self-playing version is `reports/walkthrough.html`
(published as an artifact). This is the plain script if you'd rather record a
voiceover over a screen capture of the app.

---

**1 — Title _(0:00, ~12s)_**
> ballpark. A cricket project I built with public data. What it does, and why
> the numbers are put together the way they are.

**2 — The problem _(0:12, ~26s)_**
> The scorecard hides two things. The situation: a six in the third over of a
> flat chase isn't a six in the nineteenth defending nine an over, but average
> and strike rate treat them identically. And small samples: a death strike
> rate off forty balls is barely a reading at all. Nearly every question worth
> asking about a T20 player runs into one or both.

**3 — The data _(0:38, ~22s)_**
> The data is just public ball-by-ball records — two hundred and ninety-six
> thousand deliveries, twelve hundred and forty-three IPL matches, 2008 to 2026,
> from Cricsheet. No ball tracking. I wasn't trying to out-data a broadcaster.
> I wanted the method to be right.

**4 — Layer 1, expected runs _(1:00, ~26s)_**
> The first model asks a simple question: given the over, the wickets in hand,
> the required rate, and how set the batter is, what does an average batter
> score off this ball? That's "expected runs". It doesn't know who's actually
> batting, and that's the point — every player number later is measured against
> it.

**5 — Layer 2, win probability _(1:26, ~26s)_**
> Then the odds, recalculated every ball. The win-probability line turns a
> scorecard into a picture of how a match swung. And it holds up: across every
> ball in the data, when the model says seventy percent, the side goes on to
> win about seventy percent of the time. It's off by roughly two points on
> average.

**6 — Layer 3, shrinkage _(1:52, ~24s)_**
> Forty balls is not a sample. A raw "runs above average" number for a short
> career is mostly wobble. So every player number gets pulled back toward the
> pack — hard when there's little data behind it, barely at all for a long
> career. Under the hood it's a ridge regression, which is really just a prior
> on every player.

**7 — Layer 3, the Bumrah example _(2:16, ~24s)_**
> Take Jasprit Bumrah. His raw economy looks ordinary — under seven an over.
> But he bowls the eighteenth and twentieth to set batters. Once the model
> accounts for who he bowls to and when, it rates him the best bowler in the
> whole dataset — saving about thirty-six runs per hundred balls, ninety-five
> percent sure it's between thirty and forty-two.

**8 — Layer 4, matchups _(2:40, ~24s)_**
> "He can't play the leggie." Across nearly seven hundred batter-versus-
> bowling-type pairings, the average raw gap from expected is thirty-nine runs
> per hundred balls. Regress each one for how few balls it's built on and it's
> about a fifth of that. The type-level effects are real. The personal "he
> can't play it" part mostly isn't there once you count the balls.

**9 — Layer 4, tactics _(3:04, ~24s)_**
> Which bowler, which over? An optimiser tries every legal way to bowl out the
> rest of a chase and picks the toughest one for the batting side. Run over
> eleven hundred real tight finishes, its plan lands within a couple of points
> of the captain's more than four times in five. Captains mostly get it right.
> The biggest gap in the whole data set is about eighteen points.

**10 — The app _(3:28, ~22s)_**
> All of this is a web app with six tabs: match replay, player leaderboards,
> the matchup explorer, the bowling-change optimiser — and a model card that
> says where the models fall short, which not many cricket models bother to do.

**11 — The simulator _(3:50, ~24s)_**
> The sixth tab is a full-match simulator. The full IPL history is preloaded —
> add another league's Cricsheet zip if you want — then set two line-ups and a
> venue, and it plays the game out ball by ball, entirely in your browser. Each
> player's recent form, weighted toward strong opposition; a
> pace-versus-spin matchup; who's set; how the last few overs have gone; and a
> calibrated win probability at the end.

**12 — Close _(4:14, ~15s)_**
> That's ballpark. Public data, checked properly. Every number in this tour is
> real, from the 2008-to-2026 Cricsheet data. Links to the app and the code
> are below.

---

Show on screen for scene 12:
- App — https://ballpark-mkmljvquubqdhwezbkgdtg.streamlit.app/
- Code — https://github.com/AbirChakraborty1/ballpark
