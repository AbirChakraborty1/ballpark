"""Raw cricsheet CSVs -> deliveries.parquet + matches.parquet + registry.parquet.

Two inputs per league dump:
  all_matches.csv  - one row per delivery
  <id>_info.csv    - key/value rows of match metadata, one file per match

The info grammar is `info,<key>[,<v1>[,<v2>]]`, where a repeated key means a
repeated value (two `team` rows, eleven `player` rows per side). Two keys carry
an innings number in the first value slot (target_runs, target_overs), and
`registry` carries `people,<name>,<person_id>`.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import normalise as nz
from .config import load_config, processed, raw_dir

# info keys that repeat within a match and must be collected into a list
MULTI = {"team", "umpire", "player_of_match", "match_referee", "date",
         "reserve_umpire", "tv_umpire", "super_over", "declared", "forfeited"}
# info keys whose first value is an innings number
BY_INNINGS = {"target_runs", "target_overs"}


def parse_info(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Return (match_fields, squad_rows, registry_rows) for one _info.csv."""
    info: dict = {}
    squad: list[dict] = []
    registry: list[dict] = []
    match_id = path.name.removesuffix("_info.csv")

    for row in csv.reader(path.open(encoding="utf-8")):
        if not row or row[0] != "info":
            continue
        key, values = row[1], row[2:]
        if key == "registry":
            registry.append({"match_id": match_id, "name": values[1], "person_id": values[2]})
        elif key == "player":
            squad.append({"match_id": match_id, "raw_team": values[0], "name": values[1]})
        elif key in BY_INNINGS:
            info[key + "_" + values[0]] = values[1]
        elif key in MULTI:
            info.setdefault(key, []).append(values[0])
        else:
            info[key] = values[0]

    info["match_id"] = match_id
    return info, squad, registry


def build_matches(files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    infos, squads, registry = [], [], []
    for f in tqdm(files, desc="info files"):
        i, s, r = parse_info(f)
        infos.append(i)
        squads.extend(s)
        registry.extend(r)

    m = pd.DataFrame(infos)
    m["match_id"] = m["match_id"].astype("int64")
    m["start_date"] = pd.to_datetime(m["date"].str[0], format="%Y/%m/%d")
    m["n_days"] = m["date"].str.len()
    m["season_year"] = nz.season_year(m["start_date"])

    teams = pd.DataFrame(m["team"].tolist(), index=m.index, columns=["team_1", "team_2"])
    m["team_1"] = nz.normalise_team(teams["team_1"])
    m["team_2"] = nz.normalise_team(teams["team_2"])
    m["toss_winner"] = nz.normalise_team(m["toss_winner"])
    m["winner"] = nz.normalise_team(m["winner"]) if "winner" in m else pd.NA
    m["eliminator"] = nz.normalise_team(m["eliminator"]) if "eliminator" in m else pd.NA

    venues = nz.normalise_venue(m["venue"])
    for col in venues.columns:
        m[col] = venues[col]

    for col in ("target_runs_2", "target_overs_2", "winner_runs", "winner_wickets",
                "match_number", "overs", "balls_per_over"):
        if col in m:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    m["had_super_over"] = m["super_over"].notna() if "super_over" in m else False
    m["is_dls"] = m["method"].eq("D/L") if "method" in m else False
    if "outcome" not in m:
        m["outcome"] = pd.NA
    m["no_result"] = m["outcome"].eq("no result")
    m["is_tie"] = m["outcome"].eq("tie")
    m["result_team"] = m["winner"].fillna(m["eliminator"])

    keep = ["match_id", "season", "season_year", "start_date", "n_days", "venue", "venue_era",
            "city", "country", "team_1", "team_2", "toss_winner", "toss_decision", "winner",
            "eliminator", "result_team", "outcome", "is_tie", "no_result", "had_super_over",
            "is_dls", "method", "target_runs_2", "target_overs_2", "winner_runs",
            "winner_wickets", "match_number", "event", "gender", "match_type", "overs",
            "balls_per_over"]
    m = m[[c for c in keep if c in m.columns]].sort_values("match_id").reset_index(drop=True)

    sq = pd.DataFrame(squads)
    sq["match_id"] = sq["match_id"].astype("int64")
    sq["team"] = nz.normalise_team(sq["raw_team"])

    reg = pd.DataFrame(registry)
    reg["match_id"] = reg["match_id"].astype("int64")
    return m, sq, reg


NAME_COLS = ["striker", "non_striker", "bowler", "player_dismissed",
             "other_player_dismissed", "fielder_1", "fielder_2", "fielder_3"]


def attach_ids(deliveries: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    """Map every player name to its stable cricsheet person_id.

    Joined on (match_id, name) rather than on name alone: the registry is scoped
    to a match, so two different players sharing a display name stay distinct.
    """
    lookup = reg.drop_duplicates(["match_id", "name"]).set_index(["match_id", "name"])["person_id"]
    for col in NAME_COLS:
        if col not in deliveries:
            continue
        idx = pd.MultiIndex.from_arrays([deliveries["match_id"], deliveries[col]])
        deliveries[col + "_id"] = lookup.reindex(idx).to_numpy()
    return deliveries


def build_deliveries(path: Path, reg: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    d = pd.read_csv(path, low_memory=False)
    d["match_id"] = d["match_id"].astype("int64")

    # `ball` counts every delivery bowled (0.7 exists after a wide); the over is
    # its integer part. actual_delivery is the legal-ball number in the over.
    d["over"] = d["ball"].astype(float).astype(int) + 1  # 1-indexed
    d["ball_in_over"] = (d["actual_delivery"].astype(float) * 10).round().astype(int) % 10

    for col in ("wides", "noballs", "byes", "legbyes", "penalty"):
        d[col] = pd.to_numeric(d.get(col), errors="coerce").fillna(0).astype("int16")

    d["legal_ball"] = (d["wides"] == 0) & (d["noballs"] == 0)
    d["runs_total"] = d["runs_off_bat"] + d["extras"]
    d["wicket_type"] = d["wicket_type"].fillna("")
    d["is_wicket"] = d["wicket_type"].ne("")
    # A 'retired hurt' appears as a wicket row but does not cost the side one of
    # its ten wickets; 'retired out' does. Only is_dismissal decrements wickets.
    d["is_dismissal"] = d["is_wicket"] & ~d["wicket_type"].isin(cfg["non_dismissal_wicket_types"])
    d["bowler_wicket"] = d["wicket_type"].isin(cfg["bowler_wicket_types"])
    # runs the bowler is accountable for: bat + wides + no-balls, never byes/leg-byes
    d["runs_conceded"] = d["runs_off_bat"] + d["wides"] + d["noballs"]

    venues = nz.normalise_venue(d["venue"])
    for col in venues.columns:
        d[col] = venues[col]
    d["batting_team"] = nz.normalise_team(d["batting_team"])
    d["bowling_team"] = nz.normalise_team(d["bowling_team"])
    d["start_date"] = pd.to_datetime(d["start_date"])
    d["season_year"] = nz.season_year(d["start_date"])

    d = attach_ids(d, reg)
    return d.sort_values(["match_id", "innings", "ball"]).reset_index(drop=True)


def main() -> None:
    src = raw_dir()
    info_files = sorted(src.glob("*_info.csv"))
    if not info_files:
        raise FileNotFoundError("no *_info.csv under " + str(src) + "; run `make download` first")

    matches, squads, registry = build_matches(info_files)
    deliveries = build_deliveries(src / "all_matches.csv", registry)

    load_config()["data"]["processed_dir"].mkdir(parents=True, exist_ok=True)
    matches.to_parquet(processed("matches.parquet"), index=False)
    squads.to_parquet(processed("squads.parquet"), index=False)
    registry.drop_duplicates().to_parquet(processed("registry.parquet"), index=False)
    deliveries.to_parquet(processed("deliveries.parquet"), index=False)

    print("matches    {:>8,}  {}-{}".format(len(matches), matches.season_year.min(),
                                            matches.season_year.max()))
    print("deliveries {:>8,}".format(len(deliveries)))
    print("people     {:>8,}".format(registry.person_id.nunique()))
    missing = deliveries[["striker_id", "non_striker_id", "bowler_id"]].isna().sum().sum()
    print("unmapped striker/non-striker/bowler ids:", missing)


if __name__ == "__main__":
    main()
