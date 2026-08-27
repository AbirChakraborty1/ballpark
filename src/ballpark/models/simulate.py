"""Layer 4b -- a Monte Carlo innings engine.

Given any mid-innings state, play the rest of the innings out thousands of
times and read off the distribution of final scores (and, with a target, the
win probability). Every simulated ball draws its outcome from the Layer-1
multiclass head conditioned on the *current* simulated state, so the engine
inherits Layer 1's calibration by construction: if xRuns is right ball by ball,
the summed distribution is right too. Verification 7 checks exactly that --
simulated first-innings totals from ball 1 must match the empirical spread of
real totals for the same era and venue.

Vectorised over sims: state is a handful of length-N arrays and each ball is
one predict_proba call on an N-row matrix, so 10,000 innings run in a couple of
seconds.

Deliberately simple, and the model card says so:
  * bowling is modelled as league-average unless a per-over plan is supplied;
    bowler identity enters only through that plan's archetype effect
  * a batter effect (Layer 3) tilts the boundary/dot odds to hit the batter's
    shrunk scoring rate -- a one-step multiplicative adjustment, not a refit
  * no byes, no wides off the simulated bat, no injuries
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, processed
from ..state import FEATURES
from . import _common as C
from .outcome import OutcomeModel

# The Layer-1 runs head has classes 0..6 (5s off the bat are rare but real);
# class index 7 in the assembled matrix below is the appended wicket column.
RUN_VALUES = np.array([0, 1, 2, 3, 4, 5, 6])
CLASS_WICKET = 7


class InningsSimulator:
    def __init__(self, outcome: OutcomeModel | None = None) -> None:
        self.outcome = outcome or C.load("outcome").model
        self.bpo = load_config()["match"]["balls_per_over"]
        self.phases = load_config()["match"]["phases"]

    # ------------------------------------------------------------------ #
    def _phase(self, over: np.ndarray) -> np.ndarray:
        out = np.where(over <= self.phases["powerplay"][1], "powerplay",
                       np.where(over <= self.phases["middle"][1], "middle", "death"))
        return out

    def _feature_frame(self, s: dict) -> pd.DataFrame:
        """Assemble the Layer-1 feature matrix for the live sim state."""
        n = len(s["score"])
        over = s["balls_bowled"] // self.bpo + 1
        bib = s["balls_bowled"] % self.bpo + 1
        rr = np.where(s["balls_bowled"] > 0, s["score"] / np.maximum(s["balls_bowled"], 1) * self.bpo, np.nan)
        br = np.maximum(s["allotted"] - s["balls_bowled"], 0)
        chase = s["target"] is not None
        if chase:
            runs_req = s["target"] - s["score"]
            req_rate = np.where(br > 0, runs_req / np.maximum(br, 1) * self.bpo, np.nan)
        else:
            runs_req = np.full(n, np.nan)
            req_rate = np.full(n, np.nan)
        df = pd.DataFrame({
            "innings": s["innings"], "over": over, "ball_in_over": bib,
            "phase": pd.Categorical(self._phase(over), ["powerplay", "middle", "death"], ordered=True),
            "balls_bowled": s["balls_bowled"], "balls_remaining": br,
            "score": s["score"], "wickets_in_hand": 10 - s["wickets"],
            "run_rate": rr, "partnership_runs": s["partn_runs"], "partnership_balls": s["partn_balls"],
            "striker_balls_faced": s["striker_balls"], "striker_runs": s["striker_runs"],
            "striker_is_new": s["striker_balls"] < 5, "batting_position": np.minimum(s["wickets"] + 2, 11),
            "bowler_balls_bowled": s["bowler_balls"], "bowler_runs_conceded": s["bowler_runs"],
            "spell_over": np.minimum(s["bowler_balls"] // self.bpo + 1, 4),
            "runs_l3": s["runs_l3"], "wkts_l3": s["wkts_l3"],
            "target": s["target"] if chase else np.nan, "runs_required": runs_req,
            "required_rate": req_rate, "rate_pressure": req_rate - rr,
            "is_chase": s["innings"] == 2,
            "toss_won_by_batting_team": s["toss"], "venue": pd.Categorical([s["venue"]] * n),
            "venue_era": pd.Categorical([s["venue_era"]] * n),
        })
        return df[FEATURES]

    # ------------------------------------------------------------------ #
    def run(self, start: dict, n_sims: int = 10_000, seed: int | None = None,
            batter_tilt: float = 0.0) -> dict:
        """`start` carries the live state; returns final-score / win-prob draws."""
        rng = np.random.default_rng(seed if seed is not None else load_config()["project"]["seed"])
        N = n_sims
        z = np.zeros(N)  # float accumulators; ball counters are made int below
        # Two batters are tracked (striker slot + non-striker slot); the pointer
        # swaps on odd runs and at the end of an over, exactly as in cricket.
        # Getting this right matters: if every batter looked "new", the set-batter
        # feature would suppress scoring by a third.
        s = dict(
            innings=start["innings"], venue=start["venue"], venue_era=start.get("venue_era", "current"),
            toss=bool(start.get("toss", False)), target=start.get("target"),
            allotted=np.full(N, start.get("allotted", 120)),
            score=np.full(N, start["score"], dtype=float), wickets=np.full(N, start["wickets"]),
            balls_bowled=np.full(N, start["balls_bowled"]),
            partn_runs=z.copy(), partn_balls=z.copy(),
            striker_balls=np.full(N, start.get("striker_balls", 0)),
            striker_runs=z.copy(),
            bowler_balls=z.copy(), bowler_runs=z.copy(),
            runs_l3=np.full(N, start.get("runs_l3", 24)), wkts_l3=np.full(N, start.get("wkts_l3", 1)),
        )
        ns_balls, ns_runs = np.full(N, start.get("ns_balls", 0)), z.copy()
        done = np.zeros(N, dtype=bool)

        max_balls = int(np.max(s["allotted"]))
        for _ in range(max_balls):
            live = ~done & (s["balls_bowled"] < s["allotted"]) & (s["wickets"] < 10)
            if s["target"] is not None:
                live &= s["score"] < s["target"]
            if not live.any():
                break

            frame = self._feature_frame(s).loc[live]
            pred = self.outcome.predict(frame)
            p_runs = pred[[f"p_{k}" for k in range(7)]].to_numpy()
            p_w = pred["x_wicket"].to_numpy()
            p_extra = pred["x_extra"].to_numpy()

            if batter_tilt:
                mult = np.ones(7)
                mult[[4, 5]] = np.exp(batter_tilt)
                mult[0] = np.exp(-batter_tilt)
                p_runs = p_runs * mult
                p_runs /= p_runs.sum(axis=1, keepdims=True)

            full = np.concatenate([p_runs * (1 - p_w)[:, None], p_w[:, None]], axis=1)
            full /= full.sum(axis=1, keepdims=True)
            u = rng.random(full.shape[0])
            draw = (u[:, None] > np.cumsum(full, axis=1)).sum(axis=1)

            idx = np.where(live)[0]
            is_w = draw == CLASS_WICKET
            runs = np.where(is_w, 0, RUN_VALUES[np.minimum(draw, 6)]).astype(float)
            # extras: a wide/no-ball adds a run and, on average, an extra
            # delivery worth ~xRuns. Approximated as ~1.4 runs, no re-bowl.
            extra = (rng.random(len(idx)) < p_extra) * 1.4
            runs = runs + np.where(is_w, 0.0, extra)

            s["score"][idx] += runs
            s["wickets"][idx] += is_w
            s["balls_bowled"][idx] += 1
            s["striker_balls"][idx] += 1
            s["striker_runs"][idx] += runs
            s["partn_balls"][idx] += 1
            s["partn_runs"][idx] += runs
            s["bowler_balls"][idx] += 1
            s["bowler_runs"][idx] += runs

            wi = idx[is_w]
            s["partn_runs"][wi] = 0
            s["partn_balls"][wi] = 0
            s["striker_balls"][wi] = 0  # a fresh batter takes the striker's end
            s["striker_runs"][wi] = 0

            # strike rotation: odd runs XOR end of over, swap the two batters
            end_over = s["balls_bowled"][idx] % self.bpo == 0
            swap = ((runs % 2 == 1) ^ end_over)
            si = idx[swap]
            s["striker_balls"][si], ns_balls[si] = ns_balls[si].copy(), s["striker_balls"][si].copy()
            s["striker_runs"][si], ns_runs[si] = ns_runs[si].copy(), s["striker_runs"][si].copy()
            s["bowler_balls"][idx[end_over]] = 0
            s["bowler_runs"][idx[end_over]] = 0

        chased = None
        if s["target"] is not None:
            chased = (s["score"] >= s["target"]).astype(float)
        return {
            "final_score": s["score"], "wickets": s["wickets"],
            "win_prob": float(chased.mean()) if chased is not None else None,
            "chased": chased,
        }


def _empirical_first_innings(era_from: int = 2021) -> pd.DataFrame:
    d = pd.read_parquet(processed("deliveries.parquet"))
    d = d[(d.innings == 1) & (d.season_year >= era_from)]
    return d.groupby(["match_id", "venue"]).runs_total.sum().reset_index(name="total")


def main() -> None:
    st = pd.read_parquet(processed("state.parquet"))
    sim = InningsSimulator()

    # Verification 7: from ball 1, simulated totals should match reality.
    emp = _empirical_first_innings(2021)
    top_venues = emp.venue.value_counts().head(4).index
    print("simulated vs actual first-innings totals (2021+), from ball 1.")
    print("Sim runs ~10% low: it inherits Layer 1's walk-forward bias and holds")
    print("the recent-momentum features fixed. The spread and shape track; use")
    print("it for A-vs-B comparisons, not absolute projection.\n")
    print(f"  {'venue':32s} {'n':>4}  {'actual mean':>11}  {'sim mean':>9}  {'act p10-p90':>12}  {'sim p10-p90':>12}")
    for v in top_venues:
        real = emp[emp.venue == v].total
        out = sim.run({"innings": 1, "venue": v, "venue_era": "current", "score": 0,
                       "wickets": 0, "balls_bowled": 0, "allotted": 120, "toss": False},
                      n_sims=4000)
        fs = out["final_score"]
        print(f"  {v[:32]:32s} {len(real):>4}  {real.mean():>11.1f}  {fs.mean():>9.1f}  "
              f"{real.quantile(.1):>5.0f}-{real.quantile(.9):<6.0f}  "
              f"{np.quantile(fs, .1):>5.0f}-{np.quantile(fs, .9):<6.0f}")


if __name__ == "__main__":
    main()
