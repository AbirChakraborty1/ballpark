"""Exploratory tables: the structure the models have to capture.

Run after `make features`. Prints the phase x wickets run-rate grid, venue par
scores, the toss/chase effect over time, and the outcome-class distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ballpark.config import processed  # noqa: E402

pd.set_option("display.width", 120)


def main() -> None:
    st = pd.read_parquet(processed("state.parquet"))
    d = pd.read_parquet(processed("deliveries.parquet"))
    m = pd.read_parquet(processed("matches.parquet"))

    print("=== deliveries by innings ===")
    print(d.innings.value_counts().sort_index().to_string())

    print("\n=== run rate by phase x wickets in hand (runs per over) ===")
    grid = (st.groupby(["phase", "wickets_in_hand"], observed=True)
            .runs_total.mean().mul(6).unstack().round(2))
    print(grid.to_string())

    print("\n=== outcome class distribution ===")
    print((st.outcome_class.value_counts(normalize=True) * 100).round(2).to_string())

    print("\n=== venue par: mean 1st-innings total, grounds with 20+ matches ===")
    first = (d[d.innings == 1].groupby(["match_id", "venue"]).runs_total.sum()
             .reset_index().groupby("venue").runs_total.agg(["mean", "count"]))
    print(first[first["count"] >= 20].sort_values("mean", ascending=False).round(1).to_string())

    print("\n=== chasing team win rate by season (decided matches) ===")
    dec = m[m.result_team.notna() & ~m.no_result]
    dec = dec.assign(chase_won=lambda x: x.result_team != x.team_1)  # team_1 bats first? not reliable
    bt = (d[d.innings == 1].groupby("match_id").batting_team.first())
    dec = dec.assign(bat_first=dec.match_id.map(bt),
                     chased_won=lambda x: x.result_team != x.bat_first)
    print(dec.groupby("season_year").chased_won.mean().round(3).to_string())

    print("\n=== set-batter effect: runs/ball by balls faced this innings ===")
    bins = pd.cut(st.striker_balls_faced, [-1, 4, 10, 20, 40, 200])
    print(st.groupby(bins, observed=True).runs_off_bat.mean().round(3).to_string())


if __name__ == "__main__":
    main()
