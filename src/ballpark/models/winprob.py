"""Layer 2 -- win probability from any ball of any match.

The hard part of this model is not accuracy, it is honesty about sample size.
A chase contributes ~120 rows that all share a single outcome, so 85,000 balls
carry roughly 1,000 independent observations. Left to its own devices a
gradient-boosted model exploits that: at 31+ leaves it memorises individual
match trajectories, scoring AUC 0.85 while returning probabilities so
overconfident its log loss is *worse than always guessing 50%* (0.72 vs 0.53
for a plain logistic; the numbers are in reports/report.md).

What actually works here, chosen on walk-forward seasons 2019-2023:

  shrink hard        15 leaves, 2,000 samples per leaf, 150 trees -- a tree
                     this blunt cannot fit a single match's shape
  do NOT decay       Layer 1 down-weights old seasons because scoring drifts.
                     Chase dynamics do not: 30 needed off 18 with 3 wickets is
                     the same problem in 2014 and 2025, and the effective
                     sample is far too small to throw seasons away.
  blend with a GLM   final probability is a weighted average (0.2 GBM / 0.8
                     logistic) of the GBM and a logistic on required rate,
                     wickets and balls. The GLM anchors the tail states the tree
                     sees too rarely to learn; the tree adds interaction signal
                     that helps on 2016-2023. On the 2024-26 test seasons the
                     plain logistic is level with the blend -- reported, not
                     hidden -- so the GBM's weight is kept low.
  calibrate honestly out-of-fold: the isotonic map is fitted on cross-validated
                     predictions of the training seasons, not on a held-out
                     tail of ~180 matches that is too small to estimate it.

The first innings is modelled directly. A two-stage decomposition (project the
total, then run it through the chase model) came out worse than a constant on
log loss, so the quantile score projection it needed is kept only for the
projected-score fan chart, which is what it is actually good for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import load_config, processed
from .. import splits
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
CV_FOLDS = 5


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


def _logit_features(df: pd.DataFrame) -> np.ndarray:
    """The three scoreboard numbers, plus the interactions a captain reasons about."""
    rr = df.required_rate.fillna(df.run_rate).fillna(8.0).clip(0, 36).to_numpy(float)
    wih = df.wickets_in_hand.to_numpy(float)
    br = df.balls_remaining.to_numpy(float)
    crr = df.run_rate.fillna(8.0).clip(0, 20).to_numpy(float)
    return np.column_stack([rr, wih, br, rr * wih, rr * br / 120, wih * br / 120, crr])


class WinProbModel:
    """Per-innings GBM + logistic blend, calibrated out-of-fold."""

    name = "winprob"
    # Weight on the GBM; the logistic carries the rest. Chosen on walk-forward
    # 2019-2023: the GBM adds real signal on 2016-2023 but on 2024-26 the plain
    # logistic is level with anything more complex (the Impact Player rule has
    # made chases more rate-driven), so the GBM's weight is deliberately low.
    blend_w = 0.2

    def fit(self, train: pd.DataFrame) -> "WinProbModel":
        self.gbm, self.logit, self.refs, self.calibrators = {}, {}, {}, {}
        seed = load_config()["project"]["seed"]

        for innings, features in ((1, FIRST_FEATURES), (2, CHASE_FEATURES)):
            rows = train[train.innings == innings]
            X = C.make_matrix(rows, features)
            y = (rows.batting_team_won > 0.5).astype(int).to_numpy()
            groups = rows.match_id.to_numpy()
            self.refs[innings] = X.head(50)

            # out-of-fold predictions -> a calibration map fitted on data the
            # component models never saw, without spending a tail of seasons on it
            oof = np.zeros(len(y))
            Xr = X.reset_index(drop=True)
            for tr, te in GroupKFold(CV_FOLDS).split(Xr, y, groups):
                oof[te] = self._blend_fit_predict(
                    Xr.iloc[tr], y[tr], rows.iloc[tr], Xr.iloc[te], rows.iloc[te], innings
                )
            self.calibrators[innings] = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0
            ).fit(oof, rows.batting_team_won.to_numpy(float))

            # production components: refit on everything
            self.gbm[innings] = LGBMClassifier(**_params()).fit(X, y)
            if innings == 2:
                self.logit[innings] = make_pipeline(
                    StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
                ).fit(_logit_features(rows), y)

        self._fit_score_projection(train[train.innings == 1])
        return self

    def _blend_fit_predict(self, Xtr, ytr, rows_tr, Xte, rows_te, innings) -> np.ndarray:
        g = LGBMClassifier(**_params()).fit(Xtr, ytr)
        gp = g.predict_proba(C.align_categories(Xte, Xtr))[:, 1]
        if innings != 2:
            return gp
        lg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            _logit_features(rows_tr), ytr
        )
        lp = lg.predict_proba(_logit_features(rows_te))[:, 1]
        return self.blend_w * gp + (1 - self.blend_w) * lp

    def _features(self, innings: int) -> list[str]:
        return FIRST_FEATURES if innings == 1 else CHASE_FEATURES

    def _raw(self, df: pd.DataFrame, innings: int) -> np.ndarray:
        X = C.align_categories(C.make_matrix(df, self._features(innings)), self.refs[innings])
        gp = self.gbm[innings].predict_proba(X)[:, 1]
        if innings != 2:
            return gp
        lp = self.logit[innings].predict_proba(_logit_features(df))[:, 1]
        return self.blend_w * gp + (1 - self.blend_w) * lp

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

    def _fit_score_projection(self, first: pd.DataFrame) -> None:
        """Quantile models for runs still to come in the first innings.

        Remaining runs rather than the final total: bounded below by zero, and
        it spares the model from re-learning the current score every ball.
        """
        self.score_ref = C.make_matrix(first, FIRST_FEATURES)
        total = first.groupby(["match_id", "innings"]).runs_total.transform("sum")
        remaining = total - first.score
        self.score_q = {}
        for q in QUANTILES:
            m = LGBMRegressor(objective="quantile", alpha=q, n_estimators=300,
                              learning_rate=0.05, num_leaves=31, min_child_samples=500,
                              random_state=load_config()["project"]["seed"], verbose=-1)
            m.fit(self.score_ref, remaining)
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
    """Logistic regression on the numbers on the scoreboard.

    Not a straw man. This is close to what a good analyst does in their head,
    and it is the bar the model has to clear to justify its complexity.
    """
    out = pd.Series(float(train.batting_team_won.mean()), index=test.index)
    tr, te = train[train.innings == 2], test[test.innings == 2]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pipe.fit(_logit_features(tr), (tr.batting_team_won > 0.5).astype(int))
    out.loc[te.index] = pipe.predict_proba(_logit_features(te))[:, 1]
    return out


def main() -> None:
    st = splits.usable(pd.read_parquet(processed("state.parquet")))

    preds, rows = [], []
    for season, train_mask, test_mask in splits.walk_forward(st.season_year):
        train, test = st[train_mask], st[test_mask]
        if train.season_year.nunique() < 4:
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
              f"uncal {mr['log_loss']:.4f})  brier {m['brier']:.4f} (base {mb['brier']:.4f})  "
              f"auc {m.get('auc', np.nan):.4f}  ece {m['ece']:.4f}")

    per_season = pd.DataFrame(rows)
    all_preds = pd.concat(preds, ignore_index=True)

    print("\nmean over scored seasons:")
    print(per_season[["log_loss", "base_log_loss", "uncal_log_loss", "brier", "base_brier",
                      "auc", "base_auc", "ece", "uncal_ece"]].mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))

    first_test = load_config()["splits"]["test_seasons"][0]
    test_only = per_season[per_season.season >= first_test]
    print(f"\ntest seasons {list(test_only.season)}: "
          f"ll {test_only.log_loss.mean():.4f} vs base {test_only.base_log_loss.mean():.4f}  "
          f"brier {test_only.brier.mean():.4f} vs base {test_only.base_brier.mean():.4f}")

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
               meta={**C.provenance(st), "quantiles": QUANTILES, "cv_folds": CV_FOLDS,
                     "blend_weight_gbm": WinProbModel.blend_w,
                     "benchmark": "logistic on required rate, wickets, balls remaining + interactions"}).save()

    all_preds.to_parquet(processed("winprob.parquet"), index=False)
    per_season.to_csv(processed("winprob_walkforward.csv"), index=False)
    print(f"\nsaved {len(all_preds):,} out-of-sample win probabilities")


if __name__ == "__main__":
    # Re-enter through the package path so the pickled WinProbModel carries its
    # real module name rather than "__main__" (which no other process can load).
    from ballpark.models.winprob import main as _main
    _main()
