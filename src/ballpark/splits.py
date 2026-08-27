"""Temporal splits.

T20 cricket is non-stationary: scoring rates, field restrictions and squad
quality all drift, so a random split would let a model learn from the future
and report a score it cannot reproduce in deployment. Every split here is by
season, and the test seasons are touched once.
"""
from __future__ import annotations

import pandas as pd

from .config import load_config


def assign(season_year: pd.Series) -> pd.Series:
    cfg = load_config()["splits"]
    out = pd.Series("train", index=season_year.index, dtype="object")
    out[season_year.isin(cfg["val_seasons"])] = "val"
    out[season_year.isin(cfg["test_seasons"])] = "test"
    out[season_year > cfg["train_seasons_max"]] = out[season_year > cfg["train_seasons_max"]]
    unassigned = (season_year > cfg["train_seasons_max"]) & ~season_year.isin(
        cfg["val_seasons"] + cfg["test_seasons"]
    )
    if unassigned.any():
        seasons = sorted(season_year[unassigned].unique())
        raise ValueError(f"seasons after the train cutoff with no split assigned: {seasons}")
    return out


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = assign(df["season_year"])
    return df


def walk_forward(season_year: pd.Series, start: int | None = None):
    """Yield (season, train_mask, test_mask), one per season, training on the past.

    This is the primary evaluation protocol. A model in service is retrained
    before each season, so scoring it that way is the only estimate of its real
    accuracy; a single frozen split measures something nobody would deploy.
    """
    start = start or load_config()["splits"]["walk_forward_from"]
    for season in sorted(s for s in season_year.unique() if s >= start):
        yield int(season), season_year < season, season_year == season


def recency_weights(season_year: pd.Series, reference: int | None = None) -> pd.Series:
    """Exponential decay in seasons, so the fit reflects the current game.

    IPL scoring rose 18.3% per ball between 2021 and 2024-26. Weighting every
    season equally fits a game that is no longer being played; dropping old
    seasons outright throws away the sample that makes player effects estimable.
    A half-life splits the difference.
    """
    halflife = load_config()["models"]["recency_halflife_seasons"]
    reference = reference if reference is not None else int(season_year.max())
    age = (reference - season_year).clip(lower=0)
    return 0.5 ** (age / halflife)


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for supervised training.

    A no-result match has no outcome to learn from. Ties are kept: they carry a
    0.5 label and are genuinely informative about level games.
    """
    cfg = load_config()["exclusions"]
    keep = df.innings.isin([1, 2])
    if cfg["drop_no_result"]:
        keep &= ~df.no_result
    return df[keep]
