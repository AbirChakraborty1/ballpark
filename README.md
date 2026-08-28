# ballpark

**Context-adjusted T20 valuation, built from public ball-by-ball data.**

*Abir Chakraborty — [LinkedIn](https://www.linkedin.com/in/abir-chakraborty1/) · mail2abirchakraborty@gmail.com*

Strike rate and average flatten everything. A six in the third over of a 240
chase and a six in the last over defending 12 an over count the same on the
scorecard. A death strike rate off 40 balls is closer to a coin flip than a
skill reading. `ballpark` is an attempt to price every ball by the situation it
was bowled in, and to stop small samples from lying — roughly the questions
Smart Stats and WinViz answer, from data anyone can download. The models are
checked season by season and the probabilities are calibrated; that checking is
the point, not the data.

| model | what it does | measure | ballpark | to beat |
|---|---|---|---|---|
| **Layer 1** xRuns | expected runs a ball is worth, from the match state alone | error (walk-forward) | **1.682** | 1.694 (runs by over×wickets) |
| **Layer 1** xWicket | chance of a dismissal | log loss | **0.194** | 0.195 |
| **Layer 2** win prob | P(win \| state), calibrated | Brier, 2024–26 test | **0.182** | 0.190 (regression on rate/wickets/balls) |
| | | 2nd-innings AUC | **0.88** | — |
| | | calibration error, all balls | **0.023** | — |
| **Layer 3** impact | context-adjusted player value, regressed for sample size | spread, raw → shrunk (200+ balls) | **12.9 → 8.4** | — |
| **Layer 4** matchup | batter vs bowling type, regressed | average \|effect\| per 100 balls | **7.5** | 38.9 (raw split) |

*1,243 IPL matches · 295,732 deliveries · 2008–2026 · checked season by season; the test seasons are looked at once.*

**[Live app](https://ballpark-mkmljvquubqdhwezbkgdtg.streamlit.app/)** — match
replay with the win-probability line, player leaderboards you can flip between
raw and shrunk, a matchup explorer, a bowling-change optimiser, and a model card
that says where it all falls short.

**[Write-up](reports/report.md)** — the problem, why raw numbers mislead, the
models, how they were checked, three things that came out of it, and what I'd
build first with ball-tracking data.

---

## Run it yourself

```bash
pip install -e .                     # the package + everything in requirements.txt
python scripts/run_all.py            # download → parse → features → models → figures → app data
streamlit run app/Home.py
```

`make all` does the same where `make` is available. Every model seeds from
`config.yaml`; `python -m pytest tests -q` runs the parser and leakage checks.

**Or open [`notebooks/ballpark_end_to_end.ipynb`](notebooks/ballpark_end_to_end.ipynb)** —
point it at a cricsheet `ipl_male_csv2.zip`, run it top to bottom, and it walks
every stage (numbered `1`, `1.1`, `1.2`, …) and ends with dropdowns for matches,
players, matchups and bowling changes. `RUN_MODELS = False` (the default) uses
the trained models in the repo and takes about two minutes; `True` retrains the
lot from your zip.

## Layout

```
config.yaml            paths, season splits, model settings — nothing hardcoded in src/
src/ballpark/
  ingest.py            cricsheet CSVs → deliveries / matches / player-id tables
  normalise.py         19 team spellings → 15 franchises · 60 venue spellings → 36 grounds
  state.py             27 per-ball features, all from balls already bowled
  splits.py            season-by-season splits + recency weights
  archetypes.py        bowling-type vocabulary (from reference/players_meta.csv)
  models/
    outcome.py         Layer 1 — xRuns / xWicket / chance of an extra
    winprob.py         Layer 2 — tree + regression blend, calibrated on held-out predictions
    impact.py          Layer 3 — win-prob added + ridge-regressed player value + bootstrap ranges
    matchup.py         Layer 4a — batter × bowling-type, regressed
    simulate.py        Layer 4b — vectorised ball-by-ball innings sim
    optimise.py        Layer 4c — exhaustive bowling-change search
  evaluate.py          rolls every check into reports/metrics.json + the figures
app/                   6-page Streamlit app
reference/             hand-checked team / venue / player-style maps (committed)
tests/                 parser checks against published scorecards; leakage checks
```

### A few choices I made

- **Players are keyed on cricsheet's 8-character id, never their name.** A lot
  of public IPL work joins on names and quietly splits a player across spellings.
- **Models are scored the way they'd be retrained** — on the next season, having
  seen every one before it. The frozen 2024–26 split is only a drift check.
- **Layer 2 is upfront about sample size.** A chase is one outcome spread over
  120 rows, not 120 data points; the model is regressed hard and blended with a
  three-number regression that, on recent seasons, it only draws with. That's in
  the write-up, not buried.
- **The shrinkage is a ridge penalty on runs above expected** — which is a
  Bayesian prior in disguise, with cross-validation choosing how hard it pulls.
- **The leakage check is a test, not a comment in the code.**

## Data

`data/` is gitignored. `python scripts/download.py` pulls the current
`ipl_male_csv2.zip` from [cricsheet.org](https://cricsheet.org). The app only
reads a small bundle in `data/processed/app/`, so the deployed version trains
nothing.

Cricsheet data is © cricsheet.org, under the
[Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1-0/).
