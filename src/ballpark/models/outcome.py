"""Layer 1 -- what an average IPL delivery is worth in this exact situation.

Three heads, all conditioned on match state and deliberately blind to who is
batting or bowling:

    runs    multiclass over runs off the bat, 0-6   -> xRuns   = sum(p_k * k)
    wicket  binary, dismissal on this ball          -> xWickets
    extra   binary, a wide or no-ball               -> bowler accountability

Player identity is excluded on purpose. That is what makes this the *neutral*
baseline: the run value of a situation, before anyone's skill is applied. Every
player number in Layer 3 is measured against it, which is the whole idea behind
context-adjusted rather than raw batting and bowling statistics.

Evaluation is walk-forward -- fit on every prior season, score season t -- which
mirrors how the model would be retrained in service. The frozen-split numbers
are reported alongside to quantify what era drift costs a model left unmaintained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from .. import splits
from ..config import load_config, processed
from ..state import FEATURES
from . import _common as C


def _lgbm(cfg: dict, objective: str, n_class: int | None = None) -> LGBMClassifier:
    params = dict(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"],
        objective=objective,
        random_state=load_config()["project"]["seed"],
        verbose=-1,
    )
    if n_class:
        params["num_class"] = n_class
    return LGBMClassifier(**params)


# --------------------------------------------------------------------------- #
# baselines the model has to beat
# --------------------------------------------------------------------------- #

def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str,
                         by: list[str] | None) -> np.ndarray:
    """Historical mean of `target`, optionally conditioned on `by` columns."""
    if by is None:
        return np.full(len(test), train[target].mean())
    lookup = train.groupby(by, observed=True)[target].mean()
    idx = pd.MultiIndex.from_frame(test[by]) if len(by) > 1 else pd.Index(test[by[0]])
    return lookup.reindex(idx).fillna(train[target].mean()).to_numpy()


BASELINE_SPECS = {
    "global mean": None,
    "by phase": ["phase"],
    "by over": ["over"],
    "phase x wickets": ["phase", "wickets_in_hand"],
    "over x wickets": ["over", "wickets_in_hand"],
}


def baseline_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """How much signal simple conditioning alone already captures."""
    rows = []
    for name, by in BASELINE_SPECS.items():
        runs = baseline_predictions(train, test, "runs_off_bat", by)
        wkt = baseline_predictions(train, test, "is_dismissal", by)
        rows.append({
            "model": name,
            "xruns_rmse": float(np.sqrt(np.mean((runs - test.runs_off_bat) ** 2))),
            "xruns_bias": float(runs.mean() - test.runs_off_bat.mean()),
            "wicket_log_loss": C.binary_metrics(test.is_dismissal.to_numpy(float), wkt)["log_loss"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

class OutcomeModel:
    """Fitted runs / wicket / extra heads sharing one feature matrix."""

    name = "outcome"

    def __init__(self, features: list[str] | None = None) -> None:
        self.features = list(features or FEATURES)
        self.heads: dict = {}
        self.reference: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, verbose: bool = True) -> "OutcomeModel":
        cfg = load_config()["models"]["outcome"]
        Xtr = C.make_matrix(train, self.features)
        Xva = C.align_categories(C.make_matrix(val, self.features), Xtr)
        self.reference = Xtr.head(50)
        weights = splits.recency_weights(train.season_year, reference=int(val.season_year.max()))

        targets = {
            "runs": (train.runs_off_bat, val.runs_off_bat, "multiclass", 7),
            "wicket": (train.is_dismissal.astype(int), val.is_dismissal.astype(int), "binary", None),
            "extra": (~train.legal_ball, ~val.legal_ball, "binary", None),
        }
        for head, (ytr, yva, objective, n_class) in targets.items():
            model = _lgbm(cfg, objective, n_class)
            model.fit(
                Xtr, ytr.astype(int), sample_weight=weights,
                eval_set=[(Xva, yva.astype(int))],
                eval_metric="multi_logloss" if n_class else "binary_logloss",
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            )
            self.heads[head] = model
            if verbose:
                print(f"  {head:7s} best iteration {model.best_iteration_}")
        return self

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return C.align_categories(C.make_matrix(df, self.features), self.reference)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """xRuns, xWickets and P(extra) for each row."""
        X = self._X(df)
        run_probs = self.heads["runs"].predict_proba(X)
        classes = self.heads["runs"].classes_
        out = pd.DataFrame(
            {
                "x_runs": run_probs @ classes,
                "x_wicket": self.heads["wicket"].predict_proba(X)[:, 1],
                "x_extra": self.heads["extra"].predict_proba(X)[:, 1],
            },
            index=df.index,
        )
        for k, cls in enumerate(classes):
            out[f"p_{cls}"] = run_probs[:, k]
        out["p_boundary"] = out["p_4"] + out["p_6"]
        return out

    def evaluate(self, df: pd.DataFrame) -> dict:
        pred = self.predict(df)
        actual = df.runs_off_bat.to_numpy(float)
        probs = pred[[f"p_{k}" for k in range(7)]].to_numpy()
        picked = probs[np.arange(len(df)), df.runs_off_bat.to_numpy()]
        return {
            "n": int(len(df)),
            "xruns_rmse": float(np.sqrt(np.mean((pred.x_runs - actual) ** 2))),
            "xruns_bias": float(pred.x_runs.mean() - actual.mean()),
            "runs_log_loss": float(-np.mean(np.log(np.clip(picked, 1e-9, 1)))),
            "wicket": C.binary_metrics(df.is_dismissal.to_numpy(float), pred.x_wicket.to_numpy()),
        }


def walk_forward_fit(st: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain before each season and predict it. Returns (predictions, per-season metrics).

    The last training season doubles as the validation set for early stopping,
    so no information from the season being scored reaches the fit.
    """
    preds, rows = [], []
    for season, train_mask, test_mask in splits.walk_forward(st.season_year):
        train, test = st[train_mask], st[test_mask]
        holdout = train.season_year == train.season_year.max()
        model = OutcomeModel().fit(train[~holdout], train[holdout], verbose=False)

        p = model.predict(test)
        preds.append(pd.concat([test[["match_id", "innings", "ball", "season_year"]], p], axis=1))

        m = model.evaluate(test)
        base = baseline_table(train, test).set_index("model")
        rows.append({
            "season": season, "train_n": len(train), "test_n": len(test),
            "xruns_rmse": m["xruns_rmse"], "xruns_bias": m["xruns_bias"],
            "runs_log_loss": m["runs_log_loss"],
            "wicket_log_loss": m["wicket"]["log_loss"], "wicket_ece": m["wicket"]["ece"],
            "base_rmse": base.loc["over x wickets", "xruns_rmse"],
            "base_wicket_log_loss": base.loc["over x wickets", "wicket_log_loss"],
        })
        print(f"  {season}  rmse {m['xruns_rmse']:.4f} (base {rows[-1]['base_rmse']:.4f})"
              f"  bias {m['xruns_bias']:+.4f}"
              f"  wicket ll {m['wicket']['log_loss']:.4f}")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(rows)


def main() -> None:
    st = splits.usable(pd.read_parquet(processed("state.parquet")))

    print("=== walk-forward (primary protocol: retrain before each season) ===")
    preds, per_season = walk_forward_fit(st)
    print("\nmean over seasons:")
    print(per_season[["xruns_rmse", "base_rmse", "xruns_bias", "wicket_log_loss",
                      "base_wicket_log_loss", "wicket_ece"]].mean()
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== frozen split (stress test: what never retraining costs) ===")
    s = splits.add_split(st)
    train, val, test = (s[s.split == x] for x in ("train", "val", "test"))
    frozen = OutcomeModel().fit(train, val)
    frozen_metrics = {x: frozen.evaluate(df) for x, df in
                      (("val", val), ("test", test))}
    print(f"  test rmse {frozen_metrics['test']['xruns_rmse']:.4f}  "
          f"bias {frozen_metrics['test']['xruns_bias']:+.4f}  "
          f"(a model frozen at {train.season_year.max()} under-predicts "
          f"{-frozen_metrics['test']['xruns_bias'] * 6:.2f} runs per over)")

    # The shipped model is fitted on everything, for use on future matches.
    holdout = st.season_year == st.season_year.max()
    production = OutcomeModel().fit(st[~holdout], st[holdout])
    C.Artifact(
        name=OutcomeModel.name, model=production, features=production.features,
        train_matrix_head=production.reference,
        metrics={"walk_forward": per_season.to_dict("records"), "frozen": frozen_metrics},
        meta={**C.provenance(st),
              "protocol": "walk-forward; frozen split reported as a drift stress test",
              "baselines": baseline_table(train, test).to_dict("records")},
    ).save()

    preds.to_parquet(processed("xruns.parquet"), index=False)
    per_season.to_csv(processed("outcome_walkforward.csv"), index=False)
    print(f"\nsaved {len(preds):,} out-of-sample ball predictions "
          f"({preds.season_year.min()}-{preds.season_year.max()})")

    joined = st.merge(preds, on=["match_id", "innings", "ball"], suffixes=("", "_p"))
    print("\nxRuns vs actual by phase, out-of-sample:")
    print(joined.groupby("phase", observed=True)
          .agg(actual=("runs_off_bat", "mean"), expected=("x_runs", "mean"),
               balls=("x_runs", "size"))
          .to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    # Re-enter through the package path so the pickled OutcomeModel carries its
    # real module name rather than "__main__" (which no other process can load).
    from ballpark.models.outcome import main as _main
    _main()
