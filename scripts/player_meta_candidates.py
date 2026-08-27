"""List the players who need a hand/style entry in reference/players_meta.csv.

Ranked by involvement so that a fixed budget of manual curation covers the
largest possible share of deliveries. Prints coverage as the cutoff moves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ballpark.config import processed, reference  # noqa: E402

d = pd.read_parquet(processed("deliveries.parquet"))
d = d[d.innings <= 2]

bat = d.groupby(["striker_id", "striker"]).size().rename("bat_balls")
bowl = d.groupby(["bowler_id", "bowler"]).size().rename("bowl_balls")
bat.index.names = bowl.index.names = ["person_id", "name"]

meta = pd.concat([bat, bowl], axis=1).fillna(0).reset_index()
meta["total"] = meta.bat_balls + meta.bowl_balls
meta = meta.sort_values("total", ascending=False).reset_index(drop=True)

existing = reference("players_meta.csv")
have = set()
if existing.exists():
    have = set(pd.read_csv(existing).person_id)

todo = meta[~meta.person_id.isin(have)]
print(f"{len(meta)} players total, {len(have)} already curated, {len(todo)} to go\n")

for cut in (150, 200, 250, 300, 400, 500):
    top = meta.head(cut)
    bat_cov = top.bat_balls.sum() / meta.bat_balls.sum()
    bowl_cov = top.bowl_balls.sum() / meta.bowl_balls.sum()
    print(f"  top {cut:>3}: {bat_cov:.1%} of batting balls, {bowl_cov:.1%} of bowling balls")

out = meta.head(320)[["person_id", "name", "bat_balls", "bowl_balls"]].copy()
out.to_csv(processed("player_meta_todo.csv"), index=False)
print(f"\nwrote {len(out)} candidates to {processed('player_meta_todo.csv')}")
