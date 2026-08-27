"""Choose the recency half-life on walk-forward seasons up to the tuning cutoff.

Selection happens only on seasons <= TUNE_MAX so that the seasons reserved for
the headline numbers stay untouched. Runs the `runs` head alone: the half-life
exists to track scoring drift, which is what that head measures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ballpark import splits  # noqa: E402
from ballpark.config import load_config, processed  # noqa: E402
from ballpark.models._common import align_categories, make_matrix  # noqa: E402
from ballpark.models.outcome import _lgbm  # noqa: E402
from ballpark.state import FEATURES  # noqa: E402

TUNE_MAX = 2023
HALFLIVES = [1.5, 2, 3, 4, 6, 1e6]  # 1e6 == no decay


def run(st: pd.DataFrame, halflife: float) -> pd.DataFrame:
    cfg = load_config()["models"]["outcome"]
    rows = []
    for season, train_mask, test_mask in splits.walk_forward(st.season_year):
        if season > TUNE_MAX:
            continue
        train, test = st[train_mask], st[test_mask]
        Xtr = make_matrix(train, FEATURES)
        Xte = align_categories(make_matrix(test, FEATURES), Xtr)
        age = (train.season_year.max() - train.season_year).clip(lower=0)
        w = 0.5 ** (age / halflife)

        model = _lgbm({**cfg, "n_estimators": 150}, "multiclass", 7)
        model.fit(Xtr, train.runs_off_bat, sample_weight=w)
        x_runs = model.predict_proba(Xte) @ model.classes_
        actual = test.runs_off_bat.to_numpy(float)
        rows.append({
            "season": season,
            "rmse": float(np.sqrt(np.mean((x_runs - actual) ** 2))),
            "bias": float(x_runs.mean() - actual.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    st = splits.usable(pd.read_parquet(processed("state.parquet")))
    out = []
    for hl in HALFLIVES:
        r = run(st, hl)
        out.append({
            "halflife": "none" if hl > 100 else hl,
            "mean_rmse": r.rmse.mean(),
            "mean_bias": r.bias.mean(),
            "mean_abs_bias": r.bias.abs().mean(),
            "worst_bias": r.bias.min(),
        })
        print(f"  halflife {out[-1]['halflife']:>5}  rmse {out[-1]['mean_rmse']:.4f}  "
              f"bias {out[-1]['mean_bias']:+.4f}  |bias| {out[-1]['mean_abs_bias']:.4f}")
    table = pd.DataFrame(out)
    table.to_csv(processed("recency_sweep.csv"), index=False)
    best = table.loc[table.mean_abs_bias.idxmin()]
    print(f"\nselected on seasons <= {TUNE_MAX}: half-life {best.halflife}")


if __name__ == "__main__":
    main()
