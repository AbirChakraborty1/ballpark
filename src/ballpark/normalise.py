"""Team, venue and player-identity normalisation.

Raw cricsheet strings are messy in ways that quietly corrupt analysis:
59 venue strings hide ~40 grounds, and 19 team strings hide 14 franchises.
Every mapping lives in reference/*.csv so it is reviewable, and an unmapped
string is a hard error rather than a silent NaN.
"""
from __future__ import annotations

import pandas as pd

from .config import reference


class UnmappedValueError(ValueError):
    pass


def _load(name: str, key: str) -> pd.DataFrame:
    df = pd.read_csv(reference(name), keep_default_na=False)
    dupes = df[key][df[key].duplicated()]
    if len(dupes):
        raise ValueError(f"{name}: duplicate {key} rows: {sorted(set(dupes))}")
    return df


def team_map() -> pd.DataFrame:
    return _load("teams.csv", "raw_team").set_index("raw_team")


def venue_map() -> pd.DataFrame:
    return _load("venues.csv", "raw_venue").set_index("raw_venue")


def _apply(series: pd.Series, table: pd.DataFrame, column: str, what: str) -> pd.Series:
    missing = sorted(set(series.dropna().unique()) - set(table.index))
    if missing:
        raise UnmappedValueError(
            f"{len(missing)} unmapped {what} value(s); add them to reference/: {missing}"
        )
    return series.map(table[column])


def normalise_team(series: pd.Series) -> pd.Series:
    return _apply(series, team_map(), "team", "team")


def team_short(series: pd.Series) -> pd.Series:
    return _apply(series, team_map(), "team_short", "team")


def normalise_venue(series: pd.Series) -> pd.DataFrame:
    vm = venue_map()
    return pd.DataFrame(
        {
            "venue": _apply(series, vm, "venue", "venue"),
            "venue_era": _apply(series, vm, "venue_era", "venue"),
            "city": _apply(series, vm, "city", "venue"),
            "country": _apply(series, vm, "country", "venue"),
        },
        index=series.index,
    )


def season_year(start_date: pd.Series) -> pd.Series:
    """The IPL edition, taken from the calendar year the match was played in.

    The `season` string cannot be used directly: cricsheet labels IPL 2008 as
    '2007/08', IPL 2010 as '2009/10' and IPL 2020 as '2020/21', so the string's
    trailing year is right for some editions and wrong for others. The year the
    ball was actually bowled is right for all of them.
    """
    return pd.to_datetime(start_date).dt.year.astype("int16")
