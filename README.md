# ballpark

**Context-adjusted valuation for T20 cricket, rebuilt from public ball-by-ball data.**

Raw IPL statistics are context-blind (a boundary in over 3 ≠ a boundary in over 19)
and small-sample-noisy (a death strike rate off 40 balls is mostly luck). `ballpark`
rebuilds the conceptual core of a Smart-Stats / WinViz-style engine from cricsheet
data and proves the models are calibrated — the differentiator is the validation and
shrinkage work, not the data.

| model | what it does | metric | ballpark | baseline |
|---|---|---|---|---|
| **Layer 1** xRuns | expected runs per ball from match state alone | RMSE, walk-forward | **1.682** | 1.694 (runs by over×wkts) |
| **Layer 1** xWicket | expected dismissal probability | log loss | **0.194** | 0.195 |
| **Layer 2** win prob | P(win \| state), calibrated | Brier, 2024–26 test | **0.182** | 0.190 (logistic on RR/wkts/balls) |
| | | 2nd-innings AUC | **0.88** | — |
| | | calibration error (ECE), all balls | **0.023** | — |
| **Layer 3** impact | shrunk runs-above-expected + wins added | naive → shrunk SD (200+ balls) | **12.9 → 8.4** | — |
| **Layer 4** matchup | batter × bowler-archetype, shrunk | mean \|effect\| per 100 balls | **7.5** | 38.9 (raw split) |

*1,243 IPL matches · 295,732 deliveries · 2008–2026 · walk-forward validation, test seasons touched once.*

**[Live app](#)** — match replay with win-probability ribbon, impact leaderboards with a
raw↔shrunk toggle, matchup explorer, bowling-change optimiser, and a model card that
publishes where the models are wrong.

**[Written report](reports/report.md)** — the question, why raw stats mislead, the models,
the validation evidence, three findings, and what I'd build first with ball-tracking data.

---

## Reproduce

```bash
pip install -e .                     # installs the package + deps from requirements.txt
python scripts/run_all.py            # download → ingest → features → models → figures → app bundle
streamlit run app/Home.py
```

`make all` does the same on systems with `make`. Every model seeds from `config.yaml`;
`python -m pytest tests -q` runs the parser-reconciliation and leakage tests.

## How it works

```
config.yaml            paths, season splits, model params — nothing hardcoded in src/
src/ballpark/
  ingest.py            cricsheet CSV → deliveries / matches / registry parquet
  normalise.py         19 team strings → 15 franchises · 60 venue strings → 36 grounds
  state.py             27 per-ball features, all strictly backward-looking
  splits.py            temporal splits + walk-forward + recency weights
  archetypes.py        bowler archetype vocabulary (from reference/players_meta.csv)
  models/
    outcome.py         Layer 1 — xRuns / xWickets / P(extra)
    winprob.py         Layer 2 — GBM ⊕ logistic blend, out-of-fold isotonic calibration
    impact.py          Layer 3 — WPA + ridge-shrunk crossed player effects + bootstrap CIs
    matchup.py         Layer 4a — batter × archetype interaction, penalised
    simulate.py        Layer 4b — vectorised Monte Carlo innings engine
    optimise.py        Layer 4c — exhaustive bowling-change search
  evaluate.py          consolidates every backtest into reports/metrics.json + figures
app/                   5-page Streamlit app
reference/             hand-curated team / venue / player-metadata maps (committed)
tests/                 parser reconciliation vs published scorecards; leakage assertions
```

### Design decisions worth calling out

- **Player identity is a stable 8-char cricsheet ID, never a name string.** Public IPL
  projects join on names and silently split a player across spellings.
- **Walk-forward is the primary protocol** — a deployed model is retrained before each
  season, so it is scored that way. The frozen 2024–26 split is a drift stress-test only.
- **Layer 2 is honest about sample size.** A chase is ~1 independent observation, not
  120 balls; the model is shrunk hard and blended with a 3-number logistic that, on the
  most recent seasons, it only draws with. That result is reported, not hidden.
- **Shrinkage = ridge on an offset.** An L2 penalty on player effects is exactly an
  empirical-Bayes Gaussian prior; the CV-selected penalty *is* the shrinkage.
- **Leakage is a test, not a comment.**

## Data

`data/` is gitignored. `python scripts/download.py` fetches the current
`ipl_male_csv2.zip` from [cricsheet.org](https://cricsheet.org). The app reads only the
small committed bundle in `data/processed/app/`, so the deployed build trains nothing.

Cricsheet data is © cricsheet.org, released under the
[Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1-0/).
