"""Full-match simulator — a standalone, client-side T20 engine.

Unlike the rest of the app (which is IPL 2008-2026 with models trained offline),
this is a self-contained client-side tool: it builds every player/venue profile
and simulates the match live in your browser. The full IPL ball-by-ball set is
shipped alongside it (app/assets/ipl_sample.zip) and injected as base64 so the
page opens ready to run; more Cricsheet zips can be added, and any set removed,
from inside the frame. Nothing is uploaded to a server. The HTML/JS lives at
app/assets/full_match_simulator.html.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from _data import page_header

page_header(
    "Full-match simulator",
    "The full IPL ball-by-ball history is loaded for you; set two line-ups and a "
    "venue and it plays the game out ball by ball. Each player's profile is his "
    "own recent form, weighted toward strong opposition; the ball outcome mixes "
    "the batter and the bowler, adjusts for who's set and how the last few overs "
    "have gone, and the win probability at the end is calibrated. Add another "
    "league's Cricsheet zip, or remove a set, and everything refits.",
)

st.info(
    "A separate tool from the pages above — those use models I trained "
    "offline on the IPL; this one builds everything on the fly from whatever "
    "data is loaded, entirely in your browser. The IPL set comes preloaded; any "
    "other men's T20 zip from [cricsheet.org](https://cricsheet.org/downloads/) "
    "can be added below. It scrolls inside its own frame.",
    icon="📦",
)

_ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _sig(name: str) -> tuple:
    p = _ASSETS / name
    return (p.stat().st_mtime, p.stat().st_size) if p.exists() else ()


@st.cache_data(show_spinner=False)
def _page(_sim_sig: tuple, _zip_sig: tuple) -> str:
    html = (_ASSETS / "full_match_simulator.html").read_text(encoding="utf-8")
    zip_path = _ASSETS / "ipl_sample.zip"
    if zip_path.exists():
        b64 = base64.b64encode(zip_path.read_bytes()).decode("ascii")
        html = html.replace(
            "window.__DEFAULT_DATASET__ = null;",
            'window.__DEFAULT_DATASET__ = {name:"IPL 2008\\u20132026 (Cricsheet)",b64:"'
            + b64 + '"};',
            1,
        )
    return html


components.html(
    _page(_sig("full_match_simulator.html"), _sig("ipl_sample.zip")),
    height=1500,
    scrolling=True,
)
