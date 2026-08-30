"""Full-match simulator — a standalone, client-side T20 engine.

Unlike the rest of the app (which is IPL 2008-2026 with models trained offline),
this is a self-contained client-side tool: it builds every player/venue profile
and simulates the match live in your browser. The full IPL ball-by-ball set is
shipped alongside it (app/assets/ipl_sample.zip) and injected as base64 so the
page opens ready to run; more Cricsheet zips can be added, and any set removed,
from inside the frame. Nothing is uploaded to a server. The HTML/JS lives at
app/assets/full_match_simulator.html.

The page is rendered as bare as possible — no Streamlit header or intro — so the
component's iframe sits at the top of the view. It sizes its own iframe to its
content (the fit script at the end of the HTML), leaving a single scrollbar (the
page's) and letting its modals use the full height of the screen.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ballpark — match simulator", page_icon="🏏", layout="wide")

# Pull the component close to the top: trim Streamlit's default block padding and
# let it use the full width. The header bar is left in place (it holds the
# sidebar-expand control) but made transparent so it doesn't add visual weight.
st.markdown(
    """
    <style>
      [data-testid="stMainBlockContainer"], .block-container {
        padding: 3rem 1rem 0 1rem; max-width: 100%;
      }
      [data-testid="stHeader"], header[data-testid="stHeader"] { background: transparent; }
      [data-testid="stElementContainer"]:has(> iframe) { line-height: 0; }
    </style>
    """,
    unsafe_allow_html=True,
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
    height=700,          # starting size only; the page grows its own iframe to fit
    scrolling=True,      # fallback for any browser that blocks frameElement access
)
