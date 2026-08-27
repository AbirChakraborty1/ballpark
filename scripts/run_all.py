"""Reproduce everything from the raw dump to the committed app bundle.

    python scripts/run_all.py [--skip-download]

Equivalent to `make all && python -m ballpark.evaluate && python
scripts/build_app_bundle.py`, for environments without make (Windows).
Deterministic: every model seeds from config.yaml.
"""
from __future__ import annotations

import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_module(mod: str) -> None:
    print(f"\n{'=' * 70}\n  {mod}\n{'=' * 70}", flush=True)
    t0 = time.time()
    sys.path.insert(0, str(SRC))
    runpy.run_module(mod, run_name="__main__")
    print(f"  [{mod} done in {time.time() - t0:.0f}s]", flush=True)


def main() -> None:
    steps = [
        "ballpark.ingest",
        "ballpark.state",
        "ballpark.models.outcome",
        "ballpark.models.winprob",
        "ballpark.models.impact",
        "ballpark.models.matchup",
        "ballpark.evaluate",
    ]
    if "--skip-download" not in sys.argv:
        run_module("scripts.download") if False else subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "download.py")])
    for m in steps:
        run_module(m)
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_app_bundle.py")])
    print("\nAll artifacts rebuilt. `streamlit run app/Home.py` to view.")


if __name__ == "__main__":
    main()
