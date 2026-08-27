"""Layer 2 -- win probability from any ball of any match.

The hard part of this model is not accuracy, it is honesty about sample size.
A chase contributes ~120 rows that all share a single outcome, so 83,000 balls
carry roughly 1,000 independent observations. Fitted at default settings a
gradient-boosted model exploits that: it memorises individual match
trajectories, scoring AUC 0.86 while returning probabilities so overconfident
its log loss (0.89) is worse than always guessing 50%.

Three things follow, and all three are deliberate:

  lean features     twelve state variables for the chase, not twenty-eight
  heavy shrinkage   15 leaves, 2,000 samples per leaf, 150 trees
  real calibration  isotonic, fitted on three held-out seasons -- one season is
                    ~70 matches, far too few to estimate a calibration map

The benchmark is a logistic regression on required rate, wickets in hand and
balls remaining. It is a genuinely strong model, and the tuned GBM beats it only
narrowly. That margin is reported as it is: on this data the honest headline is
that a three-variable baseline gets most of the way, and the interesting work is
in Layer 3, not here.

The first innings is modelled directly. A two-stage decomposition (project the
total, then run it through the chase model) was implemented and tested first and
came out worse than a constant on log loss, so the quantile score projection it
needed is kept only for the projected-score fan chart, which is what it is
actually good for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import splits
from ..config import load_config, processed
from . import _common as C

# Chase state is genuinely low-dimensional. Everything here is something a
# captain would name unprompted; the rest of the state table adds variance.
CHASE_FEATURES = [
    "required_rate", "runs_required", "balls_remaining", "wickets_in_hand",
    "run_rate", "rate_pressure", "over", "striker_balls_faced",
    "partnership_balls", "target", "venue", "phase",
]
FIRST_FEATURES = [
    "score", "balls_remaining", "wickets_in_hand", "run_rate", "over",
    "striker_balls_faced", "partnership_balls", "runs_l3", "wkts_l3",
    "toss_won_by_batting_team", "venue", "phase",
]

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
CALIBRATION_SEASONS = 3


def _params() -> dict:
    cfg = load_config()["models"]["winprob"]
    return dict(
        objective="binary",
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        random_state=load_config()["project"]["seed"],
        verbose=-1,
    )


class WinProbModel:
    """Per-innings classifiers, each isotonically recalibrated."""

    name = "winprob"

    def fit(self, train: pd.DataFrame) -> "WinProbModel":
        """Fit on `train`, reserving its last seasons to fit the calibration map."""
        cutoff = train.season_year.max() - CALIBRATION_SEASONS + 1
        core, calib = train[train.season_year < cutoff], train[train.season_year >= cutoff]
        if calib.empty:  # very short history: fall back to the last season
            cutoff = train.season_year.max()
            core, calib = train[train.season_year < cutoff], train[train.season_year >= cutoff]

        self.models, self.refs, self.calibrators = {}, {}, {}
        for innings, features in ((1, FIRST_FEATURES), (2, CHASE_FEATURES)):
            fit_rows = core[core.innings == innings]
            X = C.make_matrix(fit_rows, features)
            model = LGBMClassifier(**_params())
            model.fit(
                X, (fit_rows.batting_team_won > 0.5).astype(int),
                sample_weight=splits.recency_weights(
                    fit_rows.season_year, reference=int(train.season_year.max())),
            )
            self.models[innings] = model
            self.refs[innings] = X.head(50)

            held = calib[calib.innings == innings]
            raw = self._raw(held, innings)
            self.calibrators[innings] = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            ).fit(raw, held.batting_team_won)

        self._fit_score_projection(core[core.innings == 1], train)
        return self

    def _features(self, innings: int) -> list[str]:
        return FIRST_FEATURES if innings == 1 else CHASE_FEATURES

    def _raw(self, df: pd.DataFrame, innings: int) -> np.ndarray:
        X = C.align_categories(C.make_matrix(df, self._features(innings)), self.refs[innings])
        return self.models[innings].predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for innings, sub in df.groupby("innings"):
            raw = self._raw(sub, int(innings))
            out.loc[sub.index] = self.calibrators[int(innings)].predict(raw)
        return out

    def predict_uncalibrated(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=df.index, dtype=float)
        for innings, sub in df.groupby("innings"):
            out.loc[sub.index] = self._raw(sub, int(innings))
        return out

    # ------------------------------------------------------------------ #
    # projected score, for the fan chart on the match page
    # ------------------------------------------------------------------ #

    def _fit_score_projection(self, first: pd.DataFrame, train: pd.DataFrame) -> None:
        """Quantile models for runs still to come in the first innings.

        Remaining runs rather than the final total: bounded below by zero, and
        it spares the model from re-learning the current score every ball.
        """
        self.score_ref = C.make_matrix(first, FIRST_FEATURES)
        total = first.groupby(["match_id", "innings"]).runs_total.transform("sum")
        remaining = total - first.score
        w = splits.recency_weights(first.season_year, reference=int(train.season_year.max()))
        self.score_q = {}
        for q in QUANTILES:
            m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=300,
                              learning_rate=0.05, num_leaves=31, min_child_samples=500,
                              random_state=load_config()["project"]["seed"], verbose=-1)
            m.fit(self.score_ref, remaining, sample_weight=w)
            self.score_q[q] = m

    def project_score(self, first: pd.DataFrame) -> pd.DataFrame:
        """Quantiles of the final first-innings total."""
        X = C.align_categories(C.make_matrix(first, FIRST_FEATURES), self.score_ref)
        out = pd.DataFrame(index=first.index)
        for q, m in self.score_q.items():
            out[f"q{int(q * 100)}"] = np.maximum(m.predict(X), 0) + first.score.to_numpy()
        # independently fitted quantiles can cross; sorting each row restores
        # monotonicity without distorting the spread
        out.loc[:, :] = np.sort(out.to_numpy(), axis=1)
        return out


# --------------------------------------------------------------------------- #
# the benchmark
# --------------------------------------------------------------------------- #

def required_rate_baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Logistic regression on the three numbers on the scoreboard.

    Not a straw man. This is close to what a good analyst does in their head,
    and it is the bar the model has to clear to justify its complexity.
    """
    def X(df):
        rr = df.required_rate.fillna(df.run_rate).fillna(8.0).clip(0, 36)
        return np.column_stack([rr, df.wickets_in_hand, df.balls_remaining])

    out = pd.Series(float(train.batting_team_won.mean()), index=test.index)
    tr, te = train[train.innings == 2], test[test.innings == 2]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X(tr), (tr.batting_team_won > 0.5).astype(int))
    out.loc[te.index] = pipe.predict_proba(X(te))[:, 1]
    return out


def main() -> None:
    st = splits.usable(pd.read_parquet(processed("state.parquet")))

    preds, rows = [], []
    for season, train_mask, test_mask in splits.walk_forward(st.season_year):
        train, test = st[train_mask], st[test_mask]
        if train.season_year.nunique() < CALIBRATION_SEASONS + 2:
            continue
        model = WinProbModel().fit(train)
        p = model.predict(test)
        raw = model.predict_uncalibrated(test)
        base = required_rate_baseline(train, test)

        preds.append(test[["match_id", "innings", "ball", "season_year", "batting_team",
                           "batting_team_won"]].assign(win_prob=p.to_numpy()))

        y = test.batting_team_won.to_numpy(float)
        m = C.binary_metrics(y, p.to_numpy())
        mr = C.binary_metrics(y, raw.to_numpy())
        mb = C.binary_metrics(y, base.to_numpy())
        rows.append({"season": season, "n": m["n"],
                     "log_loss": m["log_loss"], "brier": m["brier"], "ece": m["ece"],
                     "auc": m.get("auc", np.nan),
                     "uncal_log_loss": mr["log_loss"], "uncal_ece": mr["ece"],
                     "base_log_loss": mb["log_loss"], "base_brier": mb["brier"],
                     "base_auc": mb.get("auc", np.nan)})
        print(f"  {season}  ll {m['log_loss']:.4f} (base {mb['log_loss']:.4f}, "
              f"uncal {mr['log_loss']:.4f})  auc {m.get('auc', np.nan):.4f} "
              f"(base {mb.get('auc', np.nan):.4f})  ece {m['ece']:.4f}")

    per_season = pd.DataFrame(rows)
    all_preds = pd.concat(preds, ignore_index=True)

    print("\nmean over scored seasons:")
    print(per_season[["log_loss", "base_log_loss", "uncal_log_loss", "brier", "base_brier",
                      "auc", "base_auc", "ece", "uncal_ece"]].mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))

    st_idx = st[["match_id", "innings", "ball", "phase", "runs_required", "balls_remaining"]]
    j = all_preds.merge(st_idx, on=["match_id", "innings", "ball"])
    y = j.batting_team_won.to_numpy(float)

    print("\ncalibration by innings and phase (pooled out-of-sample):")
    for (inn, phase), sub in j.groupby(["innings", "phase"], observed=True):
        mm = C.binary_metrics(sub.batting_team_won.to_numpy(float), sub.win_prob.to_numpy())
        print(f"  innings {inn} {phase:10s} n={mm['n']:>7,}  ece {mm['ece']:.4f}  "
              f"brier {mm['brier']:.4f}  auc {mm.get('auc', float('nan')):.4f}")

    print("\nreliability, all balls:")
    print(C.calibration_table(y, j.win_prob.to_numpy(), bins=10)
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nsanity: final-over chases")
    last = j[(j.innings == 2) & j.balls_remaining.between(1, 6)]
    for lo, hi in [(0, 3), (4, 7), (8, 12), (13, 60)]:
        s = last[last.runs_required.between(lo, hi)]
        if len(s):
            print(f"  need {lo:>2}-{hi:<2} off the last over  n={len(s):>5}  "
                  f"predicted {s.win_prob.mean():.3f}  actual {s.batting_team_won.mean():.3f}")

    production = WinProbModel().fit(st)
    C.Artifact(name=WinProbModel.name, model=production, features=CHASE_FEATURES,
               train_matrix_head=production.refs[2],
               metrics={"walk_forward": per_season.to_dict("records")},
               meta={**C.provenance(st), "quantiles": QUANTILES,
                     "calibration_seasons": CALIBRATION_SEASONS,
                     "benchmark": "logistic on required rate, wickets, balls remaining"}).save()

    all_preds.to_parquet(processed("winprob.parquet"), index=False)
    per_season.to_csv(processed("winprob_walkforward.csv"), index=False)
    print(f"\nsaved {len(all_preds):,} out-of-sample win probabilities")


if __name__ == "__main__":
    main()
