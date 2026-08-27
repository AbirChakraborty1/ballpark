"""Layer 4c -- the bowling-change optimiser.

From a live state, with each bowler's remaining quota, choose the over-by-over
allocation for the rest of the innings that minimises the batting side's win
probability. The search is exhaustive: a death-overs decision is 3-5 overs and
4-6 eligible bowlers, and once "no bowler bowls consecutive overs" and
"<= 4 overs each" prune the tree there are only a few hundred legal orderings.

Each candidate over is scored by expected value, not simulation: expected runs
= 6 x xRuns for the over's state, shifted by the bowler's Layer-3 shrunk effect
(runs saved per 100 balls); expected wickets likewise. The projected end state
is mapped to a win probability through Layer 2 -- directly for a chase, and for
a first innings by running the projected total through the chase model from a
neutral opening state. Variance would only add noise to a comparison of means.

The report runs this on one real death over and puts the optimiser's choice
next to the captain's, with the win-probability difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, processed
from . import _common as C


def _phase_of(over: int) -> str:
    ph = load_config()["match"]["phases"]
    for name, (lo, hi) in ph.items():
        if lo <= over <= hi:
            return name
    return "death"


def _bowler_effect(effects: pd.DataFrame) -> dict:
    b = effects[effects.role == "bowl"]
    return dict(zip(b.person_id, b.shrunk_per_100 / 100.0))


def _row(s: dict, balls: int, score: float, wkts: float) -> pd.DataFrame:
    over = min(balls // 6 + 1, 20)
    br = max(s["allotted"] - balls, 0)
    rr = score / max(balls, 1) * 6
    chase = s["innings"] == 2
    req = (s["target"] - score) if chase else np.nan
    return pd.DataFrame([{
        "innings": s["innings"], "over": over, "ball_in_over": 1, "phase": _phase_of(over),
        "balls_bowled": balls, "balls_remaining": br, "score": score,
        "wickets_in_hand": int(round(10 - wkts)), "run_rate": rr,
        "partnership_runs": 12, "partnership_balls": 9, "striker_balls_faced": 12,
        "striker_runs": 15, "striker_is_new": False, "batting_position": min(int(wkts) + 2, 11),
        "bowler_balls_bowled": 0, "bowler_runs_conceded": 0, "spell_over": 1,
        "runs_l3": 28, "wkts_l3": 1, "target": s.get("target", np.nan),
        "runs_required": req,
        "required_rate": (req / max(br, 1) * 6) if chase else np.nan,
        "rate_pressure": (req / max(br, 1) * 6 - rr) if chase else np.nan,
        "is_chase": chase, "toss_won_by_batting_team": s.get("toss", False),
        "venue": s["venue"], "venue_era": s.get("venue_era", "current"),
    }])


class BowlingOptimiser:
    def __init__(self, effects: pd.DataFrame | None = None) -> None:
        self.outcome = C.load("outcome").model
        self.winprob = C.load("winprob").model
        if effects is None:
            effects = pd.read_parquet(processed("player_effects.parquet"))
        self.eff = _bowler_effect(effects)
        self._xr_cache: dict = {}
        self._wp_cache: dict = {}

    def _xruns(self, s: dict, balls: int, score: float, wkts: float) -> tuple[float, float]:
        """xRuns/xWicket for an over. Cached on a rounded state: the many rollout
        branches revisit near-identical states, and this turns thousands of model
        calls into a few dozen."""
        key = (s["innings"], balls, round(score / 2) * 2, int(round(wkts)), s["venue"], s.get("target"))
        hit = self._xr_cache.get(key)
        if hit is None:
            p = self.outcome.predict(_row(s, balls, score, wkts))
            hit = (float(p.x_runs.iloc[0]) * 6, float(p.x_wicket.iloc[0]) * 6)
            self._xr_cache[key] = hit
        return hit

    def _win_prob(self, s: dict, score: float, wkts: float) -> float:
        """Chase win prob at the projected end state. Cached per rounded score --
        it is monotone in score, so 1-run buckets do not change the ranking."""
        key = (round(score), int(round(wkts)), s["venue"], s.get("target"), s["allotted"])
        hit = self._wp_cache.get(key)
        if hit is None:
            hit = float(self.winprob.predict(_row(s, s["allotted"], score, wkts)).iloc[0])
            self._wp_cache[key] = hit
        return hit

    def project(self, start: dict, order: list[str]) -> dict:
        score, wkts, balls = float(start["score"]), float(start["wickets"]), int(start["balls_bowled"])
        path = []
        for bowler in order:
            xr, xw = self._xruns(start, balls, score, wkts)
            xr = xr - self.eff.get(bowler, 0.0) * 6
            score += max(xr, 0); wkts = min(wkts + xw, 10); balls += 6
            path.append((bowler, round(xr, 1)))
        return {"proj_score": score, "proj_wkts": wkts,
                "batting_win_prob": self._win_prob(start, score, wkts), "path": path}

    def _legal_orders(self, quotas: dict, n_overs: int, last_bowler: str | None):
        """Every over-by-over allocation obeying quota and no-consecutive-overs.

        A bowler may appear up to `quota` times (not once) -- `itertools`
        permutations would wrongly forbid a two-over spell in the last five.
        """
        bowlers = [b for b, q in quotas.items() if q > 0]

        def extend(seq, left):
            if left == 0:
                yield tuple(seq)
                return
            for b in bowlers:
                if seq and seq[-1] == b:
                    continue
                if not seq and b == last_bowler:
                    continue
                if seq.count(b) >= quotas[b]:
                    continue
                yield from extend(seq + [b], left - 1)

        return extend([], n_overs)

    def optimise(self, start: dict, quotas: dict, n_overs: int,
                 last_bowler: str | None = None) -> pd.DataFrame:
        # Each call is one live state; clearing keeps caching a within-call win
        # (rollout branches revisit states) with no cross-state leakage.
        self._xr_cache.clear()
        self._wp_cache.clear()
        rows = [{"order": list(o), **self.project(start, list(o))}
                for o in self._legal_orders(quotas, n_overs, last_bowler)]
        cols = ["order", "proj_score", "proj_wkts", "batting_win_prob", "path"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows).sort_values("batting_win_prob").reset_index(drop=True)
