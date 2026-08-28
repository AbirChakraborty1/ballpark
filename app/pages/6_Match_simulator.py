"""Full-match simulator — a standalone, bring-your-own-data T20 engine.

Unlike the rest of the app (which is IPL 2008-2026 with models trained offline),
this is a self-contained client-side tool: upload your own Cricsheet zip and it
builds every player/venue profile and simulates the match live in your browser.
Nothing is uploaded to a server. The HTML/JS lives at
app/assets/full_match_simulator.html.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from _data import page_header

page_header(
    "Full-match simulator",
    "Load a Cricsheet T20 zip, set two line-ups and a venue, and it plays the "
    "game out ball by ball. Each player's profile is his own recent form, "
    "weighted toward strong opposition; the ball outcome mixes the batter and "
    "the bowler, adjusts for who's set and how the last few overs have gone, "
    "and the win probability at the end is calibrated. All of it fit from the "
    "data you load.",
)

st.info(
    "A separate tool from the pages above — those use models I trained "
    "offline on the IPL; this one builds everything on the fly from whatever "
    "you give it. It runs entirely in your browser, so the zip never leaves "
    "your machine. Any men's T20 set from "
    "[cricsheet.org](https://cricsheet.org/downloads/) works. It scrolls inside "
    "its own frame below.",
    icon="📦",
)

_HTML = (Path(__file__).resolve().parents[1] / "assets" / "full_match_simulator.html").read_text(
    encoding="utf-8"
)

components.html(_HTML, height=1500, scrolling=True)
