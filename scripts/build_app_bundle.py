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
