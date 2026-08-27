"""Cached loaders and shared styling for the app."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ballpark.config import processed  # noqa: E402

# One accent for "better than expected", one for "worse". Everything else is
# neutral grey, so colour only ever carries meaning.
GOOD = "#1f9d76"
BAD = "#d1495b"
NEUTRAL = "#8a8f98"
ACCENT = "#3d5a80"


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    path = processed(name)
    if not path.exists():
        st.error(f"`{name}` is missing. Run `make all` to build the pipeline.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def matches() -> pd.DataFrame:
    m = load("matches.parquet")
    m["label"] = (
        m.season_year.astype(str) + " - " + m.team_1 + " v " + m.team_2
        + " (" + m.venue.str.replace(" Stadium", "", regex=False) + ")"
    )
    return m


@st.cache_data(show_spinner=False)
def match_balls(match_id: int) -> pd.DataFrame:
    wpa = load("wpa.parquet")
    return wpa[wpa.match_id == match_id].sort_values(["innings", "ball"])


def available_match_ids() -> set[int]:
    """Matches the models actually cover (walk-forward starts partway through)."""
    return set(load("wpa.parquet").match_id.unique())


def page_header(title: str, blurb: str) -> None:
    st.title(title)
    st.caption(blurb)
