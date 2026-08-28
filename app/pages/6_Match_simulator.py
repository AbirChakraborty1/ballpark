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
    "Upload any Cricsheet T20 zip, pick two line-ups and a venue, and simulate "
    "the match ball by ball — recency-weighted profiles, Elo opponent strength, "
    "a pace/spin match-up, set-batter and momentum tilts, and a calibrated "
    "win probability, all fit from the data you load.",
)

st.info(
    "This page is a **self-contained tool**, separate from the trained-model "
    "pages above. Everything runs in your browser — the zip you upload never "
    "leaves your machine. Grab a dump from "
    "[cricsheet.org](https://cricsheet.org/downloads/) (any men's T20 set works). "
    "The tool scrolls inside its own frame.",
    icon="📦",
)

_HTML = (Path(__file__).resolve().parents[1] / "assets" / "full_match_simulator.html").read_text(
    encoding="utf-8"
)

components.html(_HTML, height=1500, scrolling=True)
