"""State-feature invariants and leakage tests.

The leakage test is the important one. A feature that peeks at the rest of the
innings will inflate every downstream metric while being invisible in a
correlation matrix, so it is checked mechanically: rebuild the state table from
a truncated innings and require the surviving rows to be bit-identical.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ballpark import splits, state
from ballpark.config import processed

pytestmark = pytest.mark.skipif(
    not processed("state.parquet").exists(),
    reason="run `make features` first",
)


@pytest.fixture(scope="module")
def st() -> pd.DataFrame:
    return pd.read_parquet(processed("state.parquet"))


def test_super_overs_are_excluded(st: pd.DataFrame) -> None:
    assert set(st.innings.unique()) <= {1, 2}


def test_balls_remaining_never_increases(st: pd.DataFrame) -> None:
    rising = (
        st.sort_values(["match_id", "innings", "ball"])
        .groupby(["match_id", "innings"])
        .balls_remaining.apply(lambda s: (s.diff().dropna() > 0).any())
    )
    assert not rising.any()


def test_innings_starts_empty(st: pd.DataFrame) -> None:
    first = st.sort_values(["match_id", "innings", "ball"]).groupby(["match_id", "innings"]).head(1)
    assert (first.score == 0).all()
    assert (first.wickets_in_hand == 10).all()
    assert (first.balls_bowled == 0).all()
    assert (first.partnership_runs == 0).all()


def test_running_score_reconciles_with_innings_total(st: pd.DataFrame) -> None:
    """score is the total *before* the ball, so last score + last ball == total."""
    ordered = st.sort_values(["match_id", "innings", "ball"])
    last = ordered.groupby(["match_id", "innings"]).tail(1).set_index(["match_id", "innings"])
    totals = ordered.groupby(["match_id", "innings"]).runs_total.sum()
    assert ((last.score + last.runs_total) == totals).all()


def test_wickets_never_exceed_ten(st: pd.DataFrame) -> None:
    assert st.wickets_in_hand.between(0, 10).all()


def test_batting_position_is_sane(st: pd.DataFrame) -> None:
    assert st.batting_position.between(1, 11).all()
    openers = st[st.balls_bowled == 0].batting_position
    assert openers.isin([1, 2]).all()


def test_features_are_null_only_where_undefined(st: pd.DataFrame) -> None:
    """Nulls must be explainable, not incidental."""
    assert st.loc[st.innings == 1, "target"].isna().all()
    assert st.loc[st.innings == 1, "required_rate"].isna().all()
    assert st.loc[st.balls_bowled > 0, "run_rate"].notna().all()
    always_present = ["over", "phase", "balls_remaining", "score", "wickets_in_hand",
                      "striker_balls_faced", "batting_position", "venue", "season_year"]
    assert st[always_present].notna().all().all()


def test_ties_are_labelled_as_halves(st: pd.DataFrame) -> None:
    matches = pd.read_parquet(processed("matches.parquet")).set_index("match_id")
    tied = st[st.match_id.map(matches.is_tie).fillna(False)]
    assert len(tied) > 0
    assert (tied.batting_team_won == 0.5).all()


def test_label_agrees_with_the_recorded_winner(st: pd.DataFrame) -> None:
    matches = pd.read_parquet(processed("matches.parquet")).set_index("match_id")
    decided = st[~st.no_result & ~st.match_id.map(matches.is_tie).fillna(False)]
    won = decided.batting_team == decided.match_id.map(matches.result_team)
    assert (decided.batting_team_won == won.astype(float)).all()
    # both sides of a decided match cannot both win
    per_match = decided.groupby(["match_id", "innings"]).batting_team_won.first().unstack()
    assert (per_match.sum(axis=1) == 1).all()


def test_no_lookahead_within_an_innings() -> None:
    """Truncating an innings must not change the state of the balls that remain.

    Everything except the allotted-balls family is required to be identical.
    Those three columns are the single documented exception: a curtailed first
    innings is inferred from its length, which the players also knew at the time
    (see state._allotted_balls).
    """
    deliveries = pd.read_parquet(processed("deliveries.parquet"))
    sample = sorted(deliveries.match_id.unique())[:15]
    subset = deliveries[deliveries.match_id.isin(sample)]

    full = state.build(subset)
    truncated_input = (
        subset.sort_values(["match_id", "innings", "ball"])
        .groupby(["match_id", "innings"])
        .head(60)
    )
    truncated = state.build(truncated_input)

    peeking = {"allotted_balls", "balls_remaining", "overs_remaining",
               "required_rate", "rate_pressure"}
    key = ["match_id", "innings", "ball"]
    cols = [c for c in state.FEATURES if c not in peeking]

    a = full.merge(truncated[key], on=key).sort_values(key).reset_index(drop=True)
    b = truncated.sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)


def test_splits_are_disjoint_and_ordered(st: pd.DataFrame) -> None:
    s = splits.add_split(st)
    seasons = s.groupby("split").season_year.agg(["min", "max"])
    assert seasons.loc["train", "max"] < seasons.loc["val", "min"]
    assert seasons.loc["val", "max"] < seasons.loc["test", "min"]
    # no match may straddle two splits
    assert (s.groupby("match_id").split.nunique() == 1).all()


def test_walk_forward_never_trains_on_the_future(st: pd.DataFrame) -> None:
    for season, train, test in splits.walk_forward(st.season_year):
        assert st.season_year[train].max() < season
        assert (st.season_year[test] == season).all()
