"""Sanity and leakage checks for Layers 1-4.

Skipped unless the pipeline has been run (the CI job runs `make all` first).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ballpark import splits
from ballpark.archetypes import ARCHETYPES, attach, coverage
from ballpark.config import processed

pytestmark = pytest.mark.skipif(
    not processed("xruns.parquet").exists(), reason="run `make all` first",
)


@pytest.fixture(scope="module")
def xruns() -> pd.DataFrame:
    return pd.read_parquet(processed("xruns.parquet"))


@pytest.fixture(scope="module")
def state() -> pd.DataFrame:
    return pd.read_parquet(processed("state.parquet"))


# --- Layer 1 -------------------------------------------------------------- #

def test_xruns_probabilities_are_valid(xruns: pd.DataFrame) -> None:
    p = xruns[[f"p_{k}" for k in range(7)]].to_numpy()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4)
    assert (p >= -1e-9).all()
    assert xruns.x_wicket.between(0, 1).all()


def test_xruns_bias_is_small_and_negative(xruns: pd.DataFrame, state: pd.DataFrame) -> None:
    """Walk-forward xRuns trails a rising-scoring league by a small, constant
    amount: a model trained only on the past cannot see this season's inflation.
    The offset is ~0.05 runs/ball (~4%), it is documented, and being constant it
    does not affect the *relative* player rankings Layer 3 produces.
    """
    j = xruns.merge(state[["match_id", "innings", "ball", "runs_off_bat"]],
                    on=["match_id", "innings", "ball"])
    bias = j.x_runs.mean() - j.runs_off_bat.mean()
    assert -0.09 < bias < 0.0, f"xRuns bias {bias:+.3f} outside the expected band"


def test_xruns_predictions_are_out_of_sample_only(xruns: pd.DataFrame) -> None:
    """Walk-forward starts in 2016; nothing earlier can have a prediction."""
    assert xruns.season_year.min() >= 2016


# --- Layer 2 ------------------------------------------------------------- #

def test_winprob_is_calibrated_overall() -> None:
    wp = pd.read_parquet(processed("winprob.parquet"))
    from ballpark.models._common import binary_metrics
    m = binary_metrics(wp.batting_team_won.to_numpy(float), wp.win_prob.to_numpy())
    assert m["ece"] < 0.05, f"overall ECE {m['ece']:.3f} too high"


def test_winprob_endgame_sanity() -> None:
    """A team needing < 4 off the last over wins the vast majority of the time."""
    wp = pd.read_parquet(processed("winprob.parquet"))
    st = pd.read_parquet(processed("state.parquet"))[
        ["match_id", "innings", "ball", "balls_remaining", "runs_required"]]
    j = wp.merge(st, on=["match_id", "innings", "ball"])
    easy = j[(j.innings == 2) & j.balls_remaining.between(1, 6) & j.runs_required.between(1, 3)]
    assert easy.win_prob.mean() > 0.8
    assert easy.batting_team_won.mean() > 0.8


def test_winprob_beats_baseline_on_brier_second_innings() -> None:
    """The whole point of Layer 2: better than a constant, and calibrated."""
    wp = pd.read_parquet(processed("winprob.parquet"))
    i2 = wp[wp.innings == 2]
    y = i2.batting_team_won.to_numpy(float)
    model_brier = np.mean((i2.win_prob.to_numpy() - y) ** 2)
    constant_brier = np.mean((y.mean() - y) ** 2)
    assert model_brier < constant_brier * 0.75


# --- Layer 3 ----------------------------------------------------------- #

def test_shrinkage_pulls_small_samples_harder() -> None:
    eff = pd.read_parquet(processed("player_effects.parquet"))
    small = eff[eff.balls < 150]
    large = eff[eff.balls > 1500]
    ratio_small = (small.shrunk_per_100.abs() / small.naive_above_expected_per_100.abs().clip(lower=1)).median()
    ratio_large = (large.shrunk_per_100.abs() / large.naive_above_expected_per_100.abs().clip(lower=1)).median()
    assert ratio_small < ratio_large, "small samples should be shrunk more, not less"


def test_bootstrap_intervals_bracket_the_estimate() -> None:
    eff = pd.read_parquet(processed("player_effects.parquet"))
    ok = (eff.ci_low <= eff.shrunk_per_100 + 1e-6) & (eff.shrunk_per_100 - 1e-6 <= eff.ci_high)
    assert ok.mean() > 0.98


# --- Layer 4 --------------------------------------------------------- #

def test_archetype_vocabulary_is_closed(state: pd.DataFrame) -> None:
    d = attach(state)
    seen = set(d.bowl_archetype.dropna().unique())
    assert seen <= set(ARCHETYPES)


def test_metadata_coverage_is_recorded(state: pd.DataFrame) -> None:
    cov = coverage(state)
    # ~90% of balls have a curated batter, ~87% a curated bowler; both at once
    # (what the matchup model needs) is the product, ~79%. The model card quotes
    # these; the matchup model falls back to the archetype prior for the rest.
    assert cov["bat_hand_pct"] > 0.88, cov
    assert cov["bowl_archetype_pct"] > 0.85, cov
    assert cov["both_pct"] > 0.75, cov


def test_matchup_shrinks_below_raw_splits() -> None:
    m = pd.read_parquet(processed("matchups.parquet"))
    seen = m[m.balls >= 40]
    assert seen.matchup_delta_per_100.abs().mean() < 15
    # the interaction term is the most-shrunk piece
    assert seen.interaction_per_100.abs().mean() < seen.matchup_delta_per_100.abs().mean()


# --- Layer 4c: optimiser ------------------------------------------------ #

def test_optimiser_win_prob_is_not_a_step_at_the_target() -> None:
    """A gettable chase must not read as near-certain either way. The rollout
    projects one total; if the win prob is taken at that point instead of
    integrated over a spread, Layer 2 behaves like a step at the target and a
    two-run change swings the answer 60 points. Guard against that regression:
    49 needed off 30 with 7 wickets in hand is a real contest, ~40-60%.
    """
    from ballpark.models.optimise import BowlingOptimiser

    opt = BowlingOptimiser()
    start = {"innings": 2, "score": 99, "wickets": 3, "balls_bowled": 90,
             "allotted": 120, "venue": "Wankhede Stadium", "venue_era": "current",
             "target": 148, "toss": False}
    # two mid-tier bowlers, three overs each left
    ids = list(opt.eff)[:2]
    res = opt.optimise(start, {ids[0]: 3, ids[1]: 3}, 5)
    assert not res.empty
    assert 0.25 < res.batting_win_prob.min() < 0.75, res.batting_win_prob.min()
    # and the best-to-worst spread over legal orders stays sane, not 0-to-1
    assert res.batting_win_prob.max() - res.batting_win_prob.min() < 0.35
