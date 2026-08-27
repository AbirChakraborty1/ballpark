"""Config + path resolution. Nothing else in src/ may hardcode a path."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for key, val in cfg["data"].items():
        if key.endswith("_dir"):
            cfg["data"][key] = ROOT / val
    cfg["root"] = ROOT
    return cfg


def raw_dir() -> Path:
    cfg = load_config()
    return cfg["data"]["raw_dir"] / cfg["data"]["league_slug"]


def interim(name: str) -> Path:
    return load_config()["data"]["interim_dir"] / name


def processed(name: str) -> Path:
    return load_config()["data"]["processed_dir"] / name


def reference(name: str) -> Path:
    return load_config()["data"]["reference_dir"] / name


def models_dir() -> Path:
    d = ROOT / "models"
    d.mkdir(exist_ok=True)
    return d
