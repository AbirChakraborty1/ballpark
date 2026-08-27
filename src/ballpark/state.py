"""Per-ball match state: the feature table every model downstream is built on.

Every feature describes the situation the bowler runs in to -- that is, it is
computed from deliveries *strictly before* the current one. Nothing here may be
derivable from the current ball or from the result of the match; the leakage
tests in tests/test_state.py enforce that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_config, processed

PHASE_ORDER = ["powerplay", "middle", "death"]


def _phase(over: pd.Series, phases: dict) -> pd.Series:
    out = pd.Series(pd.NA, index=over.index, dtype="object")
    for name, (lo, hi) in phases.items():
        out = out.mask(over.between(lo, hi), name)
    return out.fillna("death").astype(
        pd.CategoricalDtype(PHASE_ORDER, ordered=True)
    )


def _shift_cumsum(values: pd.Series, groups: pd.DataFrame) -> pd.Series:
    """Cumulative sum of `values` within `groups`, excluding the current row."""
    grouped = values.groupby([groups[c] for c in groups.columns], sort=False)
    return grouped.cumsum() - values


def _allotted_balls(d: pd.DataFrame, matches: pd.DataFrame, balls_per_over: int,
                    full_overs: int) -> pd.Series:
    """Legal balls this innings is entitled to.

    Innings 2 is authoritative: cricsheet records `target_overs`, which already
    reflects any Duckworth-Lewis reduction. Innings 1 has no such field, so a
    curtailed first innings is inferred -- an innings that ended short of its
    full quota without losing ten wickets was cut, and the players knew the
    revised length at the time, so this is match context rather than leakage.
    """
    full = full_overs * balls_per_over
    actual = d.groupby(["match_id", "innings"]).legal_ball.transform("sum")
    dismissals = d.groupby(["match_id", "innings"]).is_dismissal.transform("sum")

    allotted = pd.Series(full, index=d.index, dtype="float64")

    curtailed_first = (d.innings == 1) & (actual < full) & (dismissals < 10)
    allotted = allotted.mask(curtailed_first, actual)

    target_overs = d.match_id.map(matches.target_overs_2)
    second = (d.innings == 2) & target_overs.notna()
    allotted = allotted.mask(second, target_overs * balls_per_over)
    return allotted


def build(deliveries: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = load_config()
    bpo = cfg["match"]["balls_per_over"]
    full_overs = cfg["match"]["overs"]

    d = deliveries if deliveries is not None else pd.read_parquet(processed("deliveries.parquet"))
    matches = pd.read_parquet(processed("matches.parquet")).set_index("match_id")

    # Super overs are a different game with different incentives.
    d = d[d.innings <= cfg["exclusions"]["max_regulation_innings"]].copy()
    d = d.sort_values(["match_id", "innings", "ball"]).reset_index(drop=True)

    keys = d[["match_id", "innings"]]
    legal = d.legal_ball.astype("int16")

    # --- what has already happened this innings -------------------------------
    d["balls_bowled"] = _shift_cumsum(legal, keys).astype("int16")
    d["score"] = _shift_cumsum(d.runs_total, keys).astype("int16")
    d["wickets_lost"] = _shift_cumsum(d.is_dismissal.astype("int16"), keys).astype("int8")
    d["wickets_in_hand"] = (10 - d.wickets_lost).astype("int8")

    d["allotted_balls"] = _allotted_balls(d, matches, bpo, full_overs)
    d["balls_remaining"] = (d.allotted_balls - d.balls_bowled).clip(lower=0).astype("int16")
    d["overs_remaining"] = d.balls_remaining / bpo
    d["run_rate"] = np.where(d.balls_bowled > 0, d.score / d.balls_bowled * bpo, np.nan)

    d["phase"] = _phase(d["over"], cfg["match"]["phases"])
    d["ball_in_over"] = d.ball_in_over.astype("int8")

    # --- partnership ----------------------------------------------------------
    # A new partnership starts after each dismissal, so the running count of
    # dismissals is itself the partnership id.
    d["partnership_id"] = d.wickets_lost
    pkeys = d[["match_id", "innings", "partnership_id"]]
    d["partnership_runs"] = _shift_cumsum(d.runs_total, pkeys).astype("int16")
    d["partnership_balls"] = _shift_cumsum(legal, pkeys).astype("int16")

    # --- the batter at the crease --------------------------------------------
    # A set batter behaves very differently from one who has just walked in;
    # omitting this is the most common way a ball-outcome model gets misspecified.
    bkeys = d[["match_id", "innings", "striker_id"]]
    d["striker_balls_faced"] = _shift_cumsum(legal, bkeys).astype("int16")
    d["striker_runs"] = _shift_cumsum(d.runs_off_bat, bkeys).astype("int16")
    d["striker_is_new"] = (d.striker_balls_faced < 5)

    # batting position: the order in which each batter first appeared
    first_ball = d.groupby(["match_id", "innings", "striker_id"], sort=False).ball.transform("min")
    order = (
        d.assign(_f=first_ball)
        .drop_duplicates(["match_id", "innings", "striker_id"])
        .sort_values(["match_id", "innings", "_f"])
        .assign(pos=lambda x: x.groupby(["match_id", "innings"]).cumcount() + 1)
        .set_index(["match_id", "innings", "striker_id"])["pos"]
    )
    d["batting_position"] = order.reindex(
        pd.MultiIndex.from_frame(d[["match_id", "innings", "striker_id"]])
    ).to_numpy().astype("int8")

    # --- the bowler -----------------------------------------------------------
    wkeys = d[["match_id", "innings", "bowler_id"]]
    d["bowler_balls_bowled"] = _shift_cumsum(legal, wkeys).astype("int16")
    d["bowler_overs_used"] = (d.bowler_balls_bowled / bpo).astype("float32")
    d["bowler_runs_conceded"] = _shift_cumsum(d.runs_conceded, wkeys).astype("int16")
    # over number within the bowler's spell, 1-indexed
    spell = d.drop_duplicates(["match_id", "innings", "bowler_id", "over"]).copy()
    spell["spell_over"] = spell.groupby(["match_id", "innings", "bowler_id"]).cumcount() + 1
    d = d.merge(
        spell[["match_id", "innings", "bowler_id", "over", "spell_over"]],
        on=["match_id", "innings", "bowler_id", "over"], how="left",
    )
    d["spell_over"] = d.spell_over.astype("int8")

    # --- recent momentum (previous 3 overs, excluding the current one) --------
    over_agg = (
        d.groupby(["match_id", "innings", "over"], as_index=False)
        .agg(over_runs=("runs_total", "sum"), over_wkts=("is_dismissal", "sum"))
    )
    over_agg[["runs_l3", "wkts_l3"]] = (
        over_agg.groupby(["match_id", "innings"])[["over_runs", "over_wkts"]]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
    )
    d = d.merge(over_agg[["match_id", "innings", "over", "runs_l3", "wkts_l3"]],
                on=["match_id", "innings", "over"], how="left")

    # --- the chase ------------------------------------------------------------
    d["target"] = d.match_id.map(matches.target_runs_2).where(d.innings == 2)
    d["runs_required"] = (d.target - d.score).where(d.innings == 2)
    d["required_rate"] = np.where(
        (d.innings == 2) & (d.balls_remaining > 0),
        d.runs_required / d.balls_remaining * bpo,
        np.nan,
    )
    d["rate_pressure"] = d.required_rate - d.run_rate
    d["is_chase"] = d.innings == 2

    # --- match context --------------------------------------------------------
    d["toss_won_by_batting_team"] = d.batting_team == d.match_id.map(matches.toss_winner)
    d["is_dls"] = d.match_id.map(matches.is_dls).fillna(False)

    # --- the label ------------------------------------------------------------
    # A tie is a genuine 0.5, not a loss: labelling super-over ties by their
    # eliminator would teach the model that a level game is already won.
    result = d.match_id.map(matches.result_team)
    d["batting_team_won"] = (d.batting_team == result).astype("float32")
    d.loc[d.match_id.map(matches.is_tie).fillna(False), "batting_team_won"] = 0.5
    d["no_result"] = d.match_id.map(matches.no_result).fillna(False)

    # --- outcome of this ball (targets, not features) -------------------------
    d["outcome_runs"] = d.runs_off_bat
    d["outcome_class"] = np.where(
        d.is_dismissal & (d.wicket_type != "run out"), "wicket",
        np.where(d.runs_off_bat.isin([0, 1, 2, 3, 4, 6]), d.runs_off_bat.astype(str), "other"),
    )
    # a run-out is a wicket, but not one the striker or bowler produced
    d["outcome_class"] = np.where(d.wicket_type == "run out", "wicket_runout", d.outcome_class)

    return d


FEATURES = [
    "innings", "over", "ball_in_over", "phase", "balls_bowled", "balls_remaining",
    "score", "wickets_in_hand", "run_rate", "partnership_runs", "partnership_balls",
    "striker_balls_faced", "striker_runs", "striker_is_new", "batting_position",
    "bowler_balls_bowled", "bowler_runs_conceded", "spell_over", "runs_l3", "wkts_l3",
    "target", "runs_required", "required_rate", "rate_pressure", "is_chase",
    "toss_won_by_batting_team", "venue", "venue_era",
]
# season_year is deliberately NOT a feature. A tree cannot extrapolate past the
# last season it saw, so including it lets the model memorise era instead of
# generalising, and buys nothing when predicting a season that does not yet
# exist. Era is handled by retraining and by recency weights instead.


def main() -> None:
    d = build()
    d.to_parquet(processed("state.parquet"), index=False)
    print("state rows {:>9,}  ({} matches)".format(len(d), d.match_id.nunique()))
    print("features:", len(FEATURES))
    print(d.outcome_class.value_counts().to_string())
    print("\nlabel coverage: {:.1%} of balls have a decided result".format(
        (~d.no_result).mean()))


if __name__ == "__main__":
    main()
