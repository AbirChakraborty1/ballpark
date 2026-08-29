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


def _sig(*names: str) -> tuple:
    """(mtime, size) fingerprint of one or more bundle files. Git rewrites both
    when it pulls a changed file, so threading this into every cache key means a
    redeploy that updates a committed parquet actually invalidates Streamlit
    Cloud's @st.cache_data -- which otherwise survives the redeploy and keeps
    serving the pre-push table under freshly-deployed page code."""
    out = []
    for n in names:
        p = _path(n)
        if p.exists():
            s = p.stat()
            out.append((n, s.st_mtime, s.st_size))
    return tuple(out)


@st.cache_data(show_spinner=False)
def _load(name: str, _sig: tuple) -> pd.DataFrame:
    return pd.read_parquet(_path(name))


def load(name: str) -> pd.DataFrame:
    path = _path(name)
    if not path.exists():
        st.error(f"`{name}` is missing. Run `make all && python -m ballpark.evaluate` "
                 f"then `python scripts/build_app_bundle.py`.")
        st.stop()
    return _load(name, _sig(name))


@st.cache_data(show_spinner=False)
def _metrics(_sig: tuple) -> dict:
    for base in (BUNDLE, ROOT / "reports"):
        p = base / "metrics.json"
        if p.exists():
            return json.loads(p.read_text())
    return {}


def metrics() -> dict:
    for base in (BUNDLE, ROOT / "reports"):
        p = base / "metrics.json"
        if p.exists():
            s = p.stat()
            return _metrics((s.st_mtime, s.st_size))
    return {}


@st.cache_data(show_spinner=False)
def _matches(_sig: tuple) -> pd.DataFrame:
    m = load("matches.parquet")
    covered = set(load("replay.parquet").match_id.unique())
    m = m[m.match_id.isin(covered)].copy()
    m["label"] = (
        m.season_year.astype(str) + "  ·  " + m.team_1 + " v " + m.team_2
        + "  ·  " + m.venue.str.replace(" Stadium", "", regex=False)
    )
    return m.sort_values("start_date", ascending=False)


def matches() -> pd.DataFrame:
    return _matches(_sig("matches.parquet", "replay.parquet"))


@st.cache_data(show_spinner=False)
def _replay(match_id: int, _sig: tuple) -> pd.DataFrame:
    r = load("replay.parquet")
    return r[r.match_id == match_id].sort_values(["innings", "ball"]).reset_index(drop=True)


def replay(match_id: int) -> pd.DataFrame:
    return _replay(match_id, _sig("replay.parquet"))


def figure(name: str) -> Path | None:
    for base in (BUNDLE / "figures", ROOT / "reports" / "figures"):
        p = base / name
        if p.exists():
            return p
    return None


def page_header(title: str, blurb: str) -> None:
    st.title(title)
    st.caption(blurb)
