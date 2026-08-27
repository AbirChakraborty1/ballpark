"""Reproduce everything from the raw dump to the committed app bundle.

    python scripts/run_all.py [--skip-download]

Equivalent to `make all`, for environments without make (Windows). Each stage
runs in its own process, exactly as the Makefile does it, so memory does not
accumulate across the model fits. Deterministic: every model seeds from
config.yaml.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    [sys.executable, "-m", "ballpark.ingest"],
    [sys.executable, "-m", "ballpark.state"],
    [sys.executable, "-m", "ballpark.models.outcome"],
    [sys.executable, "-m", "ballpark.models.winprob"],
    [sys.executable, "-m", "ballpark.models.impact"],
    [sys.executable, "-m", "ballpark.models.matchup"],
    [sys.executable, "-m", "ballpark.evaluate"],
    [sys.executable, str(ROOT / "scripts" / "build_app_bundle.py")],
]


def main() -> None:
    env_src = str(ROOT / "src")
    steps = STEPS
    if "--skip-download" not in sys.argv:
        steps = [[sys.executable, str(ROOT / "scripts" / "download.py")]] + steps

    for cmd in steps:
        label = " ".join(cmd[1:])
        print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, cwd=ROOT, check=True,
                       env={**__import__("os").environ, "PYTHONPATH": env_src})
        print(f"  [done in {time.time() - t0:.0f}s]", flush=True)

    print("\nAll artifacts rebuilt. `streamlit run app/Home.py` to view.")


if __name__ == "__main__":
    main()
