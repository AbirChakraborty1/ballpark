"""Bowler archetypes -- the unit a matchup is actually estimable at.

"Kohli vs Rashid Khan" is 90 balls; "a right-hander vs right-arm wrist-spin in
the middle overs" is thousands. Commentary talks in the second language and
pretends it has the sample size of the first. The archetype is the coarsest
grouping that still means something tactically:

    right-arm pace | left-arm pace | off-spin | leg-spin |
    left-arm orthodox | left-arm wrist-spin

crossed with the three phases. reference/players_meta.csv carries the raw
hand/style; everything tactical is derived here so the vocabulary lives in one
place.
"""
from __future__ import annotations

import pandas as pd

from .config import reference

PACE = {"RF", "RFM", "RM", "LF", "LFM", "LM"}
SPIN = {"OB", "LB", "SLA", "LWS"}
LEFT_ARM = {"LF", "LFM", "LM", "SLA", "LWS"}
WRIST = {"LB", "LWS"}

ARCHETYPES = [
    "right-arm pace", "left-arm pace", "off-spin", "leg-spin",
    "left-arm orthodox", "left-arm wrist-spin",
]


def _archetype(bowl_type: str) -> str | None:
    if bowl_type in PACE:
        return "left-arm pace" if bowl_type in LEFT_ARM else "right-arm pace"
    if bowl_type in SPIN:
        if bowl_type == "OB":
            return "off-spin"
        if bowl_type == "LB":
            return "leg-spin"
        if bowl_type == "SLA":
            return "left-arm orthodox"
        return "left-arm wrist-spin"
    return None


def load_meta() -> pd.DataFrame:
    m = pd.read_csv(reference("players_meta.csv"), keep_default_na=False)
    m["bowl_archetype"] = m.bowl_type.map(_archetype)
    m["bowl_pace"] = m.bowl_type.where(m.bowl_type.isin(PACE | SPIN)).map(
        lambda t: "pace" if t in PACE else ("spin" if t in SPIN else None)
    )
    return m


def attach(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Add bat_hand and bowl_archetype columns, keyed on person_id."""
    m = load_meta().set_index("person_id")
    d = deliveries.copy()
    d["bat_hand"] = d.striker_id.map(m.bat_hand).replace("", pd.NA)
    d["bowl_archetype"] = d.bowler_id.map(m.bowl_archetype)
    d["bowl_pace"] = d.bowler_id.map(m.bowl_pace)
    return d


def coverage(deliveries: pd.DataFrame) -> dict:
    """Share of deliveries the curated metadata resolves -- for the model card."""
    d = attach(deliveries)
    d = d[d.innings <= 2]
    return {
        "bat_hand_pct": float(d.bat_hand.notna().mean()),
        "bowl_archetype_pct": float(d.bowl_archetype.notna().mean()),
        "both_pct": float((d.bat_hand.notna() & d.bowl_archetype.notna()).mean()),
        "curated_players": int(len(load_meta())),
    }
