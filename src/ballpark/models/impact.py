"""Layer 3 -- what each player is actually worth, once context and luck are removed.

Two numbers per player, answering two different questions.

**Wins added (WPA).** How much did this player move the win probability? Summed
over a season it is a descriptive account of what happened: high-leverage runs
count for more than runs in a dead game, which is the entire point of a
context-adjusted statistic. It is not a skill estimate -- leverage is mostly
handed to a player by circumstance.

**True rate (shrunk).** How well does this player bat or bowl, once we stop
pretending 40 balls is a sample? Runs above expectation are regressed on
one-hot batter and bowler columns with the Layer-1 xRuns as an offset, under an
L2 penalty chosen by match-grouped cross-validation.

Ridge on an offset is not a shortcut around a hierarchical model -- it *is* one.
An L2 penalty on player effects is exactly a Gaussian prior centred on average,
so the fitted coefficients are posterior means and the penalty is the prior
variance. Fitting it this way takes seconds rather than hours, and crossed
batter/bowler effects are the case where statsmodels' mixed-model support falls
over. Intervals come from a block bootstrap that resamples whole matches, which
respects the within-match correlation that makes naive standard errors far too
narrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .. import splits
from ..config import load_config, processed

BOOTSTRAP_REPS = 60


# --------------------------------------------------------------------------- #
# (a) win probability added
# --------------------------------------------------------------------------- #

def win_probability_added(state: pd.DataFrame, winprob: pd.DataFrame) -> pd.DataFrame:
    """Change in the batting side's win probability, ball by ball.

    The awkward joins are the two innings boundaries. After the last ball of the
    first innings the game passes to the chasing side, so the first innings'
    win probability is one minus the chase's opening number; after the last ball
    of the second the match is decided, so the closing probability is the result.
    """
    df = state.merge(
        winprob[["match_id", "innings", "ball", "win_prob"]],
        on=["match_id", "innings", "ball"], how="inner",
    ).sort_values(["match_id", "innings", "ball"]).reset_index(drop=True)

    grp = df.groupby(["match_id", "innings"], sort=False)
    after = grp.win_prob.shift(-1)

    # end of the first innings -> the chase begins
    chase_open = (
        df[df.innings == 2].groupby("match_id").win_prob.first().rename("chase_open")
    )
    last_of_innings = grp.cumcount(ascending=False) == 0
    first_end = last_of_innings & (df.innings == 1)
    after = after.mask(first_end, 1.0 - df.match_id.map(chase_open))

    # end of the second innings -> the result
    second_end = last_of_innings & (df.innings == 2)
    after = after.mask(second_end, df.batting_team_won)

    df["win_prob_after"] = after
    df["wpa"] = df.win_prob_after - df.win_prob
    # leverage: how much was riding on this ball, regardless of what happened
    df["leverage"] = grp.win_prob.transform(lambda s: s.diff().abs().rolling(6, min_periods=1).mean())
    return df.dropna(subset=["wpa"])


def player_wpa(wpa: pd.DataFrame) -> pd.DataFrame:
    """Season-level wins added for batters, bowlers and fielders."""
    bat = (wpa.groupby(["season_year", "striker_id", "striker"])
           .agg(balls=("wpa", "size"), wins_added=("wpa", "sum"),
                runs=("runs_off_bat", "sum"))
           .reset_index()
           .rename(columns={"striker_id": "person_id", "striker": "name"}))
    bat["role"] = "bat"

    bowl = (wpa.groupby(["season_year", "bowler_id", "bowler"])
            .agg(balls=("wpa", "size"), wins_added=("wpa", lambda s: -s.sum()),
                 runs=("runs_conceded", "sum"))
            .reset_index()
            .rename(columns={"bowler_id": "person_id", "bowler": "name"}))
    bowl["role"] = "bowl"

    # a fielder's share of the wicket they took part in
    catches = wpa[wpa.fielder_1_id.notna() & wpa.is_dismissal]
    field = (catches.groupby(["season_year", "fielder_1_id", "fielder_1"])
             .agg(balls=("wpa", "size"), wins_added=("wpa", lambda s: -s.sum() * 0.5),
                  runs=("runs_off_bat", "sum"))
             .reset_index()
             .rename(columns={"fielder_1_id": "person_id", "fielder_1": "name"}))
    field["role"] = "field"

    return pd.concat([bat, bowl, field], ignore_index=True)


# --------------------------------------------------------------------------- #
# (b) shrunk player effects
# --------------------------------------------------------------------------- #

class PlayerEffects:
    """Ridge on residual runs, with batter and bowler one-hots.

    The coefficient for a player is their runs per ball above what an average
    player would be expected to score in the same situations -- the L2 penalty
    pulling short careers toward zero exactly as a hierarchical prior would.
    """

    def __init__(self, target: str = "runs") -> None:
        self.target = target

    def _design(self, df: pd.DataFrame) -> sparse.csr_matrix:
        bat = pd.Categorical(df.striker_id, categories=self.batters)
        bowl = pd.Categorical(df.bowler_id, categories=self.bowlers)
        n = len(df)
        rows = np.arange(n)
        bat_m = sparse.csr_matrix(
            (np.ones(n), (rows, bat.codes)), shape=(n, len(self.batters)))
        bowl_m = sparse.csr_matrix(
            (np.ones(n), (rows, bowl.codes)), shape=(n, len(self.bowlers)))
        return sparse.hstack([bat_m, bowl_m], format="csr")

    def _residual(self, df: pd.DataFrame) -> np.ndarray:
        if self.target == "runs":
            return (df.runs_off_bat - df.x_runs).to_numpy(float)
        return (df.is_dismissal.astype(float) - df.x_wicket).to_numpy(float)

    def fit(self, df: pd.DataFrame, alphas: list[float] | None = None) -> "PlayerEffects":
        alphas = alphas or load_config()["models"]["impact"]["ridge_alphas"]
        self.batters = sorted(df.striker_id.unique())
        self.bowlers = sorted(df.bowler_id.unique())

        X = self._design(df)
        y = self._residual(df)
        w = splits.recency_weights(df.season_year).to_numpy()

        # Folds group whole matches: balls inside one match are not independent,
        # so a random split would let the same match sit on both sides and
        # choose a penalty far too weak.
        cv = GroupKFold(n_splits=5)
        scores = {}
        for a in alphas:
            errs = []
            for tr, te in cv.split(X, y, groups=df.match_id):
                m = Ridge(alpha=a, fit_intercept=True, solver="sparse_cg")
                m.fit(X[tr], y[tr], sample_weight=w[tr])
                errs.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
            scores[a] = float(np.mean(errs))
        self.cv_scores = scores
        self.alpha = min(scores, key=scores.get)
        if self.alpha in (min(alphas), max(alphas)) and len(alphas) > 1:
            print(f"  warning: selected alpha {self.alpha} sits on the grid boundary")

        self.model = Ridge(alpha=self.alpha, fit_intercept=True, solver="sparse_cg")
        self.model.fit(X, y, sample_weight=w)
        self.coef_ = self.model.coef_
        return self

    def bootstrap(self, df: pd.DataFrame, reps: int = BOOTSTRAP_REPS) -> np.ndarray:
        """Block bootstrap over matches -- the unit that is actually independent."""
        rng = np.random.default_rng(load_config()["project"]["seed"])
        matches = df.match_id.unique()
        by_match = {m: idx.to_numpy() for m, idx in df.groupby("match_id").indices.items()}
        y_all = self._residual(df)
        w_all = splits.recency_weights(df.season_year).to_numpy()
        X_all = self._design(df)

        draws = np.zeros((reps, X_all.shape[1]))
        for r in range(reps):
            picked = rng.choice(matches, size=len(matches), replace=True)
            idx = np.concatenate([by_match[m] for m in picked])
            m = Ridge(alpha=self.alpha, fit_intercept=True, solver="sparse_cg")
            m.fit(X_all[idx], y_all[idx], sample_weight=w_all[idx])
            draws[r] = m.coef_
        return draws

    def table(self, df: pd.DataFrame, draws: np.ndarray | None = None) -> pd.DataFrame:
        """One row per player-role, with the raw rate beside the shrunk estimate."""
        n_bat = len(self.batters)
        bat_eff = self.coef_[:n_bat]
        bowl_eff = self.coef_[n_bat:]

        raw_bat = df.groupby("striker_id").agg(
            name=("striker", "first"), balls=("runs_off_bat", "size"),
            runs=("runs_off_bat", "sum"), x_runs=("x_runs", "sum"))
        raw_bowl = df.groupby("bowler_id").agg(
            name=("bowler", "first"), balls=("runs_conceded", "size"),
            runs=("runs_conceded", "sum"), x_runs=("x_runs", "sum"))

        out = []
        for role, ids, eff, raw, sign in (
            ("bat", self.batters, bat_eff, raw_bat, 1.0),
            ("bowl", self.bowlers, bowl_eff, raw_bowl, -1.0),
        ):
            t = raw.reindex(ids).reset_index(drop=True)
            t.insert(0, "person_id", ids)
            t["role"] = role
            t["raw_per_100"] = t.runs / t.balls * 100
            t["expected_per_100"] = t.x_runs / t.balls * 100
            t["naive_above_expected_per_100"] = t.raw_per_100 - t.expected_per_100
            # bowler effects read the other way round: conceding less is better
            t["shrunk_per_100"] = eff * 100 * sign
            if draws is not None:
                lo_hi = np.percentile(draws, [2.5, 97.5], axis=0)
                sl = slice(0, len(ids)) if role == "bat" else slice(len(self.batters), None)
                t["ci_low"] = lo_hi[0][sl] * 100 * sign
                t["ci_high"] = lo_hi[1][sl] * 100 * sign
                if sign < 0:
                    t[["ci_low", "ci_high"]] = t[["ci_high", "ci_low"]].to_numpy()
            out.append(t)
        return pd.concat(out, ignore_index=True)


def main() -> None:
    state = splits.usable(pd.read_parquet(processed("state.parquet")))
    xruns = pd.read_parquet(processed("xruns.parquet"))
    winprob = pd.read_parquet(processed("winprob.parquet"))

    df = state.merge(xruns.drop(columns=["season_year"]), on=["match_id", "innings", "ball"])
    print(f"balls with out-of-sample predictions: {len(df):,} "
          f"({df.season_year.min()}-{df.season_year.max()})")

    # --- wins added --------------------------------------------------------
    wpa = win_probability_added(df, winprob)
    wpa.to_parquet(processed("wpa.parquet"), index=False)
    leaders = player_wpa(wpa)
    leaders.to_parquet(processed("player_wpa.parquet"), index=False)

    career = (leaders.groupby(["person_id", "name", "role"])
              .agg(balls=("balls", "sum"), wins_added=("wins_added", "sum"))
              .reset_index())
    print("\nmost wins added, batting (2016-2026):")
    print(career[(career.role == "bat") & (career.balls >= 500)]
          .nlargest(10, "wins_added")[["name", "balls", "wins_added"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nmost wins added, bowling:")
    print(career[(career.role == "bowl") & (career.balls >= 500)]
          .nlargest(10, "wins_added")[["name", "balls", "wins_added"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # --- shrunk skill ------------------------------------------------------
    print("\nfitting player effects...")
    effects = PlayerEffects("runs").fit(df)
    print(f"  selected ridge alpha {effects.alpha} "
          f"(cv mse {effects.cv_scores[effects.alpha]:.5f})")
    draws = effects.bootstrap(df)
    table = effects.table(df, draws)
    table.to_parquet(processed("player_effects.parquet"), index=False)

    min_balls = load_config()["models"]["impact"]["min_balls_display"]
    bat = table[(table.role == "bat") & (table.balls >= min_balls)]
    print(f"\nbest batters by shrunk runs above expectation per 100 balls "
          f"(min {min_balls} balls, n={len(bat)}):")
    print(bat.nlargest(12, "shrunk_per_100")[
        ["name", "balls", "raw_per_100", "naive_above_expected_per_100",
         "shrunk_per_100", "ci_low", "ci_high"]]
        .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    bowl = table[(table.role == "bowl") & (table.balls >= min_balls)]
    print(f"\nbest bowlers by shrunk runs saved per 100 balls (n={len(bowl)}):")
    print(bowl.nlargest(12, "shrunk_per_100")[
        ["name", "balls", "raw_per_100", "naive_above_expected_per_100",
         "shrunk_per_100", "ci_low", "ci_high"]]
        .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    # how much shrinkage actually moved things
    print("\nshrinkage: naive vs shrunk, by sample size")
    bins = pd.cut(table.balls, [0, 100, 300, 1000, 3000, 100000])
    print(table.groupby(bins, observed=True).agg(
        players=("balls", "size"),
        naive_sd=("naive_above_expected_per_100", "std"),
        shrunk_sd=("shrunk_per_100", "std")).to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
