"""Re-run one real death over through the bowling-change optimiser.

Picks the highest-leverage 18th-over-onward state from a final (or, if
--match is given, that match), then compares the optimiser's allocation for
the remaining overs with what the captain actually did.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ballpark.config import processed  # noqa: E402
from ballpark.models.optimise import BowlingOptimiser  # noqa: E402


def analyse(match_id, from_over, wpa, matches, opt):
    m = matches.loc[match_id]
    inn = wpa[(wpa.match_id == match_id) & (wpa.innings == 2)].sort_values("ball")
    if inn.empty or from_over not in set(inn.over) or inn.over.max() < 19:
        return None
    at = inn[inn.over == from_over].iloc[0]
    if at.wickets_lost >= 8 or pd.isna(at.target):
        return None
    before = inn[inn.over < from_over]
    eff = pd.read_parquet(processed("player_effects.parquet")).set_index("name")
    name_to_id = eff.person_id.to_dict()
    id_to_name = {v: k for k, v in name_to_id.items()}
    counts = before.groupby("bowler").over.nunique().to_dict()
    quotas = {name_to_id.get(b): 4 - counts.get(b, 0) for b in inn.bowler.unique() if name_to_id.get(b)}
    quotas = {k: v for k, v in quotas.items() if v > 0}
    n_overs = 20 - from_over + 1
    if len(quotas) < n_overs:
        return None
    start = {"innings": 2, "score": int(at.score), "wickets": int(at.wickets_lost),
             "balls_bowled": (from_over - 1) * 6, "allotted": 120, "venue": m.venue,
             "venue_era": "current", "target": int(at.target),
             "toss": bool(m.toss_winner == inn.batting_team.iloc[0])}
    last = name_to_id.get(before.bowler.iloc[-1]) if len(before) else None
    res = opt.optimise(start, quotas, n_overs, last_bowler=last)
    if res.empty:
        return None
    actual = inn[inn.over >= from_over].groupby("over").bowler.first().tolist()
    actual_ids = [name_to_id.get(b) for b in actual]
    mr = res[res.order.apply(lambda o: o == actual_ids)]
    if mr.empty:
        return None
    best = res.iloc[0]
    return {"match_id": match_id, "m": m, "from_over": from_over, "inn": inn, "at": at,
            "best": best, "actual": actual, "actual_wp": float(mr.iloc[0].batting_win_prob),
            "delta": float(mr.iloc[0].batting_win_prob - best.batting_win_prob),
            "id_to_name": id_to_name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=None)
    ap.add_argument("--from-over", type=int, default=17)
    args = ap.parse_args()

    wpa = pd.read_parquet(processed("wpa.parquet"))
    matches = pd.read_parquet(processed("matches.parquet")).set_index("match_id")
    opt = BowlingOptimiser()

    if args.match is not None:
        r = analyse(args.match, args.from_over, wpa, matches, opt)
        results = [r] if r else []
    else:
        # scan close finishes: 2nd innings, decided by < 20 runs or in the last over
        close = matches[matches.winner_runs.between(1, 20) | matches.winner_wickets.between(1, 3)]
        pool = [mid for mid in close.index if mid in set(wpa.match_id)]
        results = []
        for mid in pool[:120]:
            for fo in (16, 17, 18):
                r = analyse(mid, fo, wpa, matches, opt)
                if r:
                    results.append(r)
        results.sort(key=lambda r: -r["delta"])

    if not results:
        print("no usable state found"); return
    r = results[0]
    m, at, inn, from_over = r["m"], r["at"], r["inn"], r["from_over"]
    id_to_name = r["id_to_name"]

    print(f"{m.season_year}  {m.team_1} v {m.team_2}  @ {m.venue}")
    print(f"{inn.batting_team.iloc[0]} chasing {int(at.target)}; at the start of over "
          f"{from_over}: {int(at.score)}/{int(at.wickets_lost)}, "
          f"{int(at.target - at.score)} needed off {(20 - from_over + 1) * 6}\n")
    print("optimiser :", " -> ".join(id_to_name.get(b, "?") for b in r["best"]["order"]),
          f"   batting WP {r['best'].batting_win_prob:.1%}")
    print("captain   :", " -> ".join(r["actual"]),
          f"   batting WP {r['actual_wp']:.1%}")
    print(f"\nWin probability the actual choice handed the batting side vs the "
          f"optimiser's: {r['delta']:+.1%}")
    print(f"actual result: {m.result_team} won")

    if len(results) > 1:
        print("\nother large gaps found:")
        for x in results[1:6]:
            print(f"  {x['m'].season_year} {x['m'].team_1[:3]}v{x['m'].team_2[:3]} "
                  f"over {x['from_over']}: {x['delta']:+.1%}")


if __name__ == "__main__":
    main()
