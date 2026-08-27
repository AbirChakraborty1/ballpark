"""Write the compact, committed artifacts the deployed app reads.

Everything here is derived from data/processed/ and models/; the app never
touches the raw dump or trains anything. Output: data/processed/app/.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ballpark.config import load_config, processed  # noqa: E402
from ballpark.models import _common as C  # noqa: E402

OUT = processed("app")
REPLAY_COLS = [
    "match_id", "innings", "ball", "over", "ball_in_over", "phase",
    "batting_team", "bowling_team", "striker", "non_striker", "bowler",
    "runs_off_bat", "runs_total", "extras", "wicket_type", "player_dismissed",
    "is_dismissal", "fielder_1", "score", "wickets_lost", "wickets_in_hand",
    "balls_remaining", "run_rate", "required_rate", "target", "runs_required",
    "win_prob", "win_prob_after", "wpa", "leverage", "batting_team_won",
]


def build_replay() -> pd.DataFrame:
    wpa = pd.read_parquet(processed("wpa.parquet"))
    replay = wpa[REPLAY_COLS].copy()

    # projected first-innings score fan, from the shipped win-prob model
    model = C.load("winprob").model
    first = wpa[wpa.innings == 1]
    proj = model.project_score(first)
    proj.columns = [f"proj_{c}" for c in proj.columns]
    replay = replay.join(proj)
    return replay


def build_tactics() -> pd.DataFrame:
    """Pre-run the bowling-change optimiser on every close finish, so the app
    reads a table instead of loading the models (Streamlit Cloud is memory-tight).
    """
    from ballpark.models.optimise import BowlingOptimiser

    wpa = pd.read_parquet(processed("wpa.parquet"))
    matches = pd.read_parquet(processed("matches.parquet")).set_index("match_id")
    eff = pd.read_parquet(processed("player_effects.parquet"))
    n2i = eff.set_index("name").person_id.to_dict()
    i2n = {v: k for k, v in n2i.items()}
    opt = BowlingOptimiser(effects=eff)

    close = matches[matches.winner_runs.fillna(99).le(25) | matches.winner_wickets.fillna(99).le(4)]
    covered = set(wpa.match_id)
    rows = []
    for mid in [x for x in close.index if x in covered]:
        inn = wpa[(wpa.match_id == mid) & (wpa.innings == 2)].sort_values("ball")
        if inn.empty or int(inn.over.max()) < 17:
            continue
        last_over = int(inn.over.max())
        m = matches.loc[mid]
        for fo in range(15, last_over):
            if fo not in set(inn.over):
                continue
            at = inn[inn.over == fo].iloc[0]
            if pd.isna(at.target) or at.wickets_lost >= 9:
                continue
            before = inn[inn.over < fo]
            counts = before.groupby("bowler").over.nunique().to_dict()
            quotas = {n2i.get(b): 4 - counts.get(b, 0) for b in inn.bowler.unique() if n2i.get(b)}
            quotas = {k: v for k, v in quotas.items() if v and v > 0}
            n_overs = last_over - fo + 1
            if sum(quotas.values()) < n_overs:
                continue
            start = {"innings": 2, "score": int(at.score), "wickets": int(at.wickets_lost),
                     "balls_bowled": (fo - 1) * 6, "allotted": 120, "venue": m.venue,
                     "venue_era": "current", "target": int(at.target),
                     "toss": bool(m.toss_winner == inn.batting_team.iloc[0])}
            last = n2i.get(before.bowler.iloc[-1]) if len(before) else None
            res = opt.optimise(start, quotas, n_overs, last_bowler=last)
            if res.empty:
                continue
            actual = inn[inn.over >= fo].groupby("over").bowler.first().tolist()
            aid = [n2i.get(b) for b in actual]
            mr = res[res.order.apply(lambda o: o == aid)]
            best = res.iloc[0]
            rows.append({
                "match_id": mid, "label": f"{m.season_year}  ·  {m.team_1} v {m.team_2}"
                f"  ·  {m.venue.replace(' Stadium', '')}",
                "from_over": fo, "score": int(at.score), "wickets": int(at.wickets_lost),
                "target": int(at.target), "needed": int(at.target - at.score),
                "balls_left": n_overs * 6, "result_team": m.result_team,
                "optimiser": " → ".join(i2n.get(b, "?") for b in best["order"]),
                "optimiser_wp": float(best.batting_win_prob),
                "optimiser_score": float(best.proj_score),
                "captain": " → ".join(actual),
                "captain_wp": float(mr.iloc[0].batting_win_prob) if not mr.empty else None,
                "alternatives": " | ".join(
                    " → ".join(i2n.get(b, "?") for b in o) + f" ({wp:.0%})"
                    for o, wp in zip(res.order.head(5), res.batting_win_prob.head(5))),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delta"] = df.captain_wp - df.optimiser_wp
    return df


def build_model_meta() -> dict:
    out = {}
    for name in ("outcome", "winprob"):
        p = load_config()["root"] / "models" / f"{name}_metrics.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    build_replay().to_parquet(OUT / "replay.parquet", index=False)
    tac = build_tactics()
    tac.to_parquet(OUT / "tactics.parquet", index=False)
    print(f"tactics: {len(tac)} pre-computed states")

    for name in ("matches.parquet", "player_wpa.parquet", "player_effects.parquet",
                 "matchups.parquet"):
        pd.read_parquet(processed(name)).to_parquet(OUT / name, index=False)

    reports = load_config()["root"] / "reports"
    if (reports / "metrics.json").exists():
        shutil.copy(reports / "metrics.json", OUT / "metrics.json")
    (OUT / "model_meta.json").write_text(json.dumps(build_model_meta(), indent=2, default=str))

    figdir = reports / "figures"
    if figdir.exists():
        for png in figdir.glob("*.png"):
            shutil.copy(png, OUT / "figures" / png.name)

    total = sum(f.stat().st_size for f in OUT.rglob("*")) / 1e6
    print(f"app bundle: {total:.1f} MB in {OUT}")
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  {f.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
