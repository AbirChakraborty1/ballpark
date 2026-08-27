"""Cached loaders and shared styling for the app.

The app reads only the compact artifacts in data/processed/app/ (built by
scripts/build_app_bundle.py and committed), so the deployed build needs no
pipeline run. If those are absent it falls back to the full data/processed/
tree for local development.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BUNDLE = ROOT / "data" / "processed" / "app"
FULL = ROOT / "data" / "processed"

# One accent for "better than expected", one for "worse". Everything else is
# neutral grey, so colour only ever carries meaning.
GOOD = "#1f9d76"
BAD = "#d1495b"
NEUTRAL = "#8a8f98"
ACCENT = "#3d5a80"


def _path(name: str) -> Path:
    p = BUNDLE / name
    return p if p.exists() else FULL / name


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    path = _path(name)
    if not path.exists():
        st.error(f"`{name}` is missing. Run `make all && python -m ballpark.evaluate` "
                 f"then `python scripts/build_app_bundle.py`.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    for base in (BUNDLE, ROOT / "reports"):
        p = base / "metrics.json"
        if p.exists():
            return json.loads(p.read_text())
    return {}


@st.cache_data(show_spinner=False)
def matches() -> pd.DataFrame:
    m = load("matches.parquet")
    covered = set(load("replay.parquet").match_id.unique())
    m = m[m.match_id.isin(covered)].copy()
    m["label"] = (
        m.season_year.astype(str) + "  ·  " + m.team_1 + " v " + m.team_2
        + "  ·  " + m.venue.str.replace(" Stadium", "", regex=False)
    )
    return m.sort_values("start_date", ascending=False)


@st.cache_data(show_spinner=False)
def replay(match_id: int) -> pd.DataFrame:
    r = load("replay.parquet")
    return r[r.match_id == match_id].sort_values(["innings", "ball"]).reset_index(drop=True)


def figure(name: str) -> Path | None:
    for base in (BUNDLE / "figures", ROOT / "reports" / "figures"):
        p = base / name
        if p.exists():
            return p
    return None


def page_header(title: str, blurb: str) -> None:
    st.title(title)
    st.caption(blurb)
