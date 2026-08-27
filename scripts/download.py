"""Fetch and unpack the cricsheet dump named in config.yaml.

Swapping league is a two-line config change, not a code change.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from ballpark.config import load_config, raw_dir  # noqa: E402


def main() -> None:
    cfg = load_config()["data"]
    dest = cfg["raw_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / (cfg["league_slug"] + ".zip")

    print("downloading", cfg["download_url"])
    urllib.request.urlretrieve(cfg["download_url"], archive)
    print("{:.1f} MB".format(archive.stat().st_size / 1e6))

    target = raw_dir()
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(target)
    n = len(list(target.glob("*_info.csv")))
    print("unpacked", n, "matches to", target)
    if not (target / "all_matches.csv").exists():
        raise SystemExit("all_matches.csv missing from the dump")


if __name__ == "__main__":
    main()
