"""Correctness tests for the data spine.

These are not smoke tests. The ball-by-ball file and the info file are two
independent records of the same match, so reconciling one against the other
across all 1,243 matches is a genuine check that the parser, the wicket
accounting and the extras accounting are all right.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ballpark.config import processed, raw_dir
from ballpark.normalise import normalise_team, normalise_venue

pytestmark = pytest.mark.skipif(
    not processed("deliveries.parquet").exists(),
    reason="run `make data` first",
)


@pytest.fixture(scope="module")
def deliveries() -> pd.DataFrame:
    return pd.read_parquet(processed("deliveries.parquet"))


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return pd.read_parquet(processed("matches.parquet")).set_index("match_id")


@pytest.fixture(scope="module")
def innings_totals(deliveries: pd.DataFrame) -> pd.DataFrame:
    t = (
        deliveries[deliveries.innings <= 2]
        .groupby(["match_id", "innings"])
        .agg(runs=("runs_total", "sum"),
             dismissals=("is_dismissal", "sum"),
             legal=("legal_ball", "sum"))
        .unstack("innings")
    )
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    return t


def test_shape(deliveries: pd.DataFrame, matches: pd.DataFrame) -> None:
    assert len(matches) == deliveries.match_id.nunique()
    assert matches.season_year.between(2008, 2100).all()
    assert deliveries.innings.max() <= 6  # 2 innings + super-over innings


def test_every_player_resolves_to_a_person_id(deliveries: pd.DataFrame) -> None:
    for col in ("striker", "non_striker", "bowler"):
        assert deliveries[col + "_id"].notna().all(), f"{col} has unmapped names"
        assert deliveries[col + "_id"].str.len().eq(8).all()


def test_person_ids_are_consistent_within_a_name(deliveries: pd.DataFrame) -> None:
    """A display name may legitimately map to two people, but never at random."""
    pairs = deliveries[["striker", "striker_id"]].drop_duplicates()
    collisions = pairs.striker.value_counts()
    assert (collisions > 1).sum() == 0, f"name/id collisions: {collisions[collisions > 1]}"


def test_players_appear_in_their_own_squad(deliveries: pd.DataFrame) -> None:
    squads = pd.read_parquet(processed("squads.parquet"))
    known = set(zip(squads.match_id, squads.name))
    for col in ("striker", "non_striker", "bowler"):
        seen = set(zip(deliveries.match_id, deliveries[col]))
        assert not (seen - known), f"{col} names absent from the team sheet"


def test_won_by_runs_margin_reconciles(matches, innings_totals) -> None:
    """First-innings total minus second-innings total must equal the margin."""
    j = matches.join(innings_totals)
    won = j[j.winner_runs.notna() & ~j.is_dls & ~j.had_super_over]
    assert len(won) > 500
    margin = won.runs_1 - won.runs_2
    bad = won[margin != won.winner_runs]
    assert bad.empty, f"{len(bad)} run-margin mismatches: {bad.index.tolist()[:5]}"


def test_won_by_wickets_margin_reconciles(matches, innings_totals) -> None:
    """Ten minus dismissals in the chase must equal the margin.

    This is the test that forced the is_wicket / is_dismissal split: seven
    matches contain a 'retired hurt', which cricsheet records as a wicket row
    but which does not cost the batting side one of its ten wickets.
    """
    j = matches.join(innings_totals)
    won = j[j.winner_wickets.notna() & ~j.had_super_over]
    assert len(won) > 600
    bad = won[(10 - won.dismissals_2) != won.winner_wickets]
    assert bad.empty, f"{len(bad)} wicket-margin mismatches: {bad.index.tolist()[:5]}"


def test_extras_decompose_exactly(deliveries: pd.DataFrame) -> None:
    parts = deliveries[["wides", "noballs", "byes", "legbyes", "penalty"]].sum(axis=1)
    assert (parts == deliveries.extras).all()
    assert (deliveries.runs_total == deliveries.runs_off_bat + deliveries.extras).all()


def test_bowler_is_not_charged_for_byes(deliveries: pd.DataFrame) -> None:
    byes = deliveries[(deliveries.byes > 0) | (deliveries.legbyes > 0)]
    assert (byes.runs_conceded == byes.runs_off_bat + byes.wides + byes.noballs).all()


def test_run_outs_are_not_credited_to_the_bowler(deliveries: pd.DataFrame) -> None:
    assert not deliveries.loc[deliveries.wicket_type == "run out", "bowler_wicket"].any()
    assert not deliveries.loc[deliveries.wicket_type.str.startswith("retired"), "bowler_wicket"].any()


def test_legal_balls_per_over(deliveries: pd.DataFrame) -> None:
    """Four overs in IPL history contain seven legal deliveries.

    These are umpiring miscounts that cricsheet records faithfully. They are
    pinned here so that a future parser change cannot quietly invent more, and
    so that downstream code never assumes an innings is exactly 120 balls.
    """
    per_over = (
        deliveries[deliveries.legal_ball]
        .groupby(["match_id", "innings", "over"])
        .size()
    )
    assert per_over.max() == 7
    assert (per_over > 6).sum() == 4


def test_known_scorecard(deliveries: pd.DataFrame, matches: pd.DataFrame) -> None:
    """IPL 2017 opener, Sunrisers Hyderabad v Royal Challengers Bangalore."""
    match = deliveries[deliveries.match_id == 1082591]
    first, second = match[match.innings == 1], match[match.innings == 2]

    assert first.runs_total.sum() == 207
    assert first.is_dismissal.sum() == 4
    assert first.legal_ball.sum() == 120
    assert second.runs_total.sum() == 172
    assert second.is_dismissal.sum() == 10

    assert first.groupby("striker").runs_off_bat.sum()["Yuvraj Singh"] == 62
    row = matches.loc[1082591]
    assert row.winner == "Sunrisers Hyderabad"
    assert row.winner_runs == 35
    assert row.venue == "Rajiv Gandhi International Stadium"  # from the ", Uppal" spelling


def test_reference_maps_are_total(deliveries: pd.DataFrame) -> None:
    """Normalisation must be exhaustive: an unmapped string is an error."""
    raw = pd.read_csv(raw_dir() / "all_matches.csv",
                      usecols=["venue", "batting_team"], low_memory=False)
    assert normalise_venue(raw.venue.drop_duplicates()).venue.notna().all()
    assert normalise_team(raw.batting_team.drop_duplicates()).notna().all()


def test_normalisation_collapses_duplicates(deliveries: pd.DataFrame) -> None:
    """The whole point of the reference maps: fewer canonical values than raw."""
    assert deliveries.venue.nunique() == 36  # from 60 raw strings
    assert deliveries.batting_team.nunique() == 15  # from 19 raw strings
    # the traps that must NOT be collapsed
    teams = set(deliveries.batting_team)
    assert {"Gujarat Lions", "Gujarat Titans"} <= teams
    assert {"Deccan Chargers", "Sunrisers Hyderabad"} <= teams
