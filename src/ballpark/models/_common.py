"""Shared modelling plumbing: matrix building, metrics, calibration, persistence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ..config import load_config, models_dir

CATEGORICAL = ["phase", "venue", "venue_era", "batting_team", "bowling_team"]


def make_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Feature frame with categoricals typed so LightGBM splits on them natively."""
    X = df[features].copy()
    for col in X.columns:
        if col in CATEGORICAL or X[col].dtype == object:
            X[col] = X[col].astype("category")
        elif X[col].dtype == bool:
            X[col] = X[col].astype("int8")
    return X


def align_categories(X: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Give X the same category levels as the training matrix.

    Without this a venue unseen in training silently becomes a different integer
    code at predict time, which is a quiet and very hard-to-spot bug.
    """
    X = X.copy()
    for col in X.columns:
        if isinstance(reference[col].dtype, pd.CategoricalDtype):
            X[col] = pd.Categorical(X[col], categories=reference[col].cat.categories)
    return X


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    """Mean |confidence - accuracy|, weighted by bin population."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "predicted": float(p[m].mean()), "observed": float(y[m].mean())})
    return pd.DataFrame(rows)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Metrics for a probability forecast; y may contain 0.5 for ties."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "base_rate": float(np.mean(y)),
    }
    decided = y != 0.5
    if decided.sum() and 0 < y[decided].mean() < 1:
        out["auc"] = float(roc_auc_score(y[decided], p[decided]))
    return out


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #

@dataclass
class Artifact:
    """A fitted model plus the provenance needed to trust it later."""

    name: str
    model: object
    features: list[str]
    train_matrix_head: pd.DataFrame
    metrics: dict
    meta: dict

    def save(self) -> Path:
        path = models_dir() / f"{self.name}.joblib"
        joblib.dump(self, path)
        (models_dir() / f"{self.name}_metrics.json").write_text(
            json.dumps({"metrics": self.metrics, "meta": self.meta}, indent=2, default=str)
        )
        return path


def load(name: str) -> Artifact:
    return joblib.load(models_dir() / f"{name}.joblib")


def provenance(df: pd.DataFrame) -> dict:
    cfg = load_config()
    return {
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "seasons": [int(df.season_year.min()), int(df.season_year.max())],
        "train_seasons_max": cfg["splits"]["train_seasons_max"],
        "val_seasons": cfg["splits"]["val_seasons"],
        "test_seasons": cfg["splits"]["test_seasons"],
        "seed": cfg["project"]["seed"],
        "league": cfg["data"]["league_slug"],
    }
