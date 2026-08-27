"""Layer 4a -- batter versus bowler-archetype, with the matchup shrunk to size.

The model is one ridge GLM on runs above expectation, with three blocks of
one-hot columns: batter, archetype, and the batter x archetype interaction.
The interaction coefficient is the matchup effect -- "how much better or worse
than usual does this batter go against this kind of bowling". An L2 penalty
pulls a thin interaction to zero, at which point the prediction falls back to
batter main effect + archetype main effect: the archetype prior. That fallback
is the whole value of the thing. A coach does not need to be told that Kohli
averages 8-an-over against leg-spin off 41 balls; they need a number that does
not lie when the sample is thin.

The headline finding this is built to test: published matchup effects
(the "he can't play left-arm spin" trope) are real but perhaps a third the
size television implies, because television is quoting an unshrunk 40-ball split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .. import splits
from ..archetypes import ARCHETYPES, attach
from ..config import load_config, processed

BOOTSTRAP_REPS = 60
MIN_BALLS_CELL = 40


class MatchupModel:
    """Ridge on residual runs: batter + archetype + batter x archetype."""

    def __init__(self, target: str = "runs") -> None:
        self.target = target

    # ------------------------------------------------------------------ #
    def _index(self, df: pd.DataFrame) -> None:
        self.batters = sorted(df.striker_id.unique())
        self.arches = list(ARCHETYPES)
        self.bat_ix = {b: i for i, b in enumerate(self.batters)}
        self.arch_ix = {a: i for i, a in enumerate(self.arches)}
        self.nb, self.na = len(self.batters), len(self.arches)

    def _design(self, df: pd.DataFrame) -> sparse.csr_matrix:
        n = len(df)
        rows = np.arange(n)
        bi = df.striker_id.map(self.bat_ix).to_numpy()
        ai = df.bowl_archetype.map(self.arch_ix).to_numpy()
        bat = sparse.csr_matrix((np.ones(n), (rows, bi)), shape=(n, self.nb))
        arch = sparse.csr_matrix((np.ones(n), (rows, ai)), shape=(n, self.na))
        inter = sparse.csr_matrix(
            (np.ones(n), (rows, bi * self.na + ai)), shape=(n, self.nb * self.na)
        )
        return sparse.hstack([bat, arch, inter], format="csr")

    def _residual(self, df: pd.DataFrame) -> np.ndarray:
        if self.target == "runs":
            return (df.runs_off_bat - df.x_runs).to_numpy(float)
        return (df.is_dismissal.astype(float) - df.x_wicket).to_numpy(float)

    def _split(self, coef: np.ndarray):
        b = coef[: self.nb]
        a = coef[self.nb : self.nb + self.na]
        i = coef[self.nb + self.na :].reshape(self.nb, self.na)
        return b, a, i

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, alphas: list[float] | None = None) -> "MatchupModel":
        df = df[df.bowl_archetype.notna()].copy()
        self._index(df)
        X, y = self._design(df), self._residual(df)
        w = splits.recency_weights(df.season_year).to_numpy()
        alphas = alphas or load_config()["models"]["impact"]["ridge_alphas"]

        cv = GroupKFold(5)
        self.cv_scores = {}
        for al in alphas:
            errs = []
            for tr, te in cv.split(X, y, groups=df.match_id):
                m = Ridge(alpha=al, solver="sparse_cg").fit(X[tr], y[tr], sample_weight=w[tr])
                errs.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
            self.cv_scores[al] = float(np.mean(errs))
        self.alpha = min(self.cv_scores, key=self.cv_scores.get)

        self.model = Ridge(alpha=self.alpha, solver="sparse_cg").fit(X, y, sample_weight=w)
        self.bat_eff, self.arch_eff, self.inter_eff = self._split(self.model.coef_)

        # context: mean expectation each batter actually faced vs each archetype,
        # so an effect can be expressed as an absolute rate rather than a delta
        ctx = (df.groupby(["striker_id", "bowl_archetype"], observed=True)
               .agg(balls=("x_runs", "size"), x_runs=("x_runs", "mean"),
                    x_wicket=("x_wicket", "mean"), name=("striker", "first")))
        self.context = ctx
        self._df_matches = df.match_id.to_numpy()
        self._X, self._y, self._w = X, y, w
        return self

    def bootstrap(self, reps: int = BOOTSTRAP_REPS) -> np.ndarray:
        rng = np.random.default_rng(load_config()["project"]["seed"])
        matches = np.unique(self._df_matches)
        by_match = {}
        for i, m in enumerate(self._df_matches):
            by_match.setdefault(m, []).append(i)
        by_match = {m: np.array(v) for m, v in by_match.items()}

        draws = np.zeros((reps, self.nb, self.na))
        for r in range(reps):
            pick = rng.choice(matches, size=len(matches), replace=True)
            idx = np.concatenate([by_match[m] for m in pick])
            m = Ridge(alpha=self.alpha, solver="sparse_cg").fit(
                self._X[idx], self._y[idx], sample_weight=self._w[idx])
            b, a, i = self._split(m.coef_)
            draws[r] = b[:, None] + a[None, :] + i
        return draws

    def table(self, draws: np.ndarray | None = None) -> pd.DataFrame:
        total = self.bat_eff[:, None] + self.arch_eff[None, :] + self.inter_eff
        rows = []
        lo = hi = None
        if draws is not None:
            lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        for bi, b in enumerate(self.batters):
            for ai, a in enumerate(self.arches):
                key = (b, a)
                if key not in self.context.index:
                    continue
                c = self.context.loc[key]
                rows.append({
                    "person_id": b, "name": c["name"], "archetype": a,
                    "balls": int(c.balls),
                    "expected_runs_per_100": float((c.x_runs + total[bi, ai]) * 100),
                    "matchup_delta_per_100": float(total[bi, ai] * 100),
                    "interaction_per_100": float(self.inter_eff[bi, ai] * 100),
                    "archetype_prior_per_100": float(
                        (c.x_runs + self.bat_eff[bi] + self.arch_eff[ai]) * 100),
                    "ci_low_per_100": float(lo[bi, ai] * 100) if draws is not None else np.nan,
                    "ci_high_per_100": float(hi[bi, ai] * 100) if draws is not None else np.nan,
                })
        return pd.DataFrame(rows)


def main() -> None:
    state = splits.usable(pd.read_parquet(processed("state.parquet")))
    xruns = pd.read_parquet(processed("xruns.parquet")).drop(columns=["season_year"])
    df = attach(state.merge(xruns, on=["match_id", "innings", "ball"]))

    cov = df[df.bowl_archetype.notna()]
    print(f"{len(cov):,} of {len(df):,} predicted balls have a bowler archetype "
          f"({len(cov) / len(df):.1%})")

    model = MatchupModel("runs").fit(df)
    print(f"selected ridge alpha {model.alpha}")
    draws = model.bootstrap()
    table = model.table(draws)
    table.to_parquet(processed("matchups.parquet"), index=False)

    # archetype main effects: the league-wide read on each kind of bowling
    print("\narchetype main effect on runs per 100 balls (vs an average batter):")
    for a, e in sorted(zip(model.arches, model.arch_eff * 100), key=lambda x: x[1]):
        print(f"  {a:20s} {e:+.1f}")

    # how big are matchup effects really?
    seen = table[table.balls >= MIN_BALLS_CELL]
    print(f"\nmatchup interaction term, |per 100| over {len(seen)} cells with "
          f">= {MIN_BALLS_CELL} balls:")
    print(f"  mean abs interaction   {seen.interaction_per_100.abs().mean():.2f}")
    print(f"  mean abs total delta   {seen.matchup_delta_per_100.abs().mean():.2f}")
    raw = (df.groupby(["striker_id", "bowl_archetype"], observed=True)
           .apply(lambda g: (g.runs_off_bat.mean() - g.x_runs.mean()) * 100, include_groups=False))
    print(f"  mean abs RAW split     {raw.abs().mean():.2f}  "
          f"(shrinkage ratio {seen.matchup_delta_per_100.abs().mean() / raw.abs().mean():.2f})")

    print("\nlargest shrunk matchup effects (min 80 balls):")
    big = table[table.balls >= 80].reindex(
        table[table.balls >= 80].matchup_delta_per_100.abs().sort_values(ascending=False).index)
    print(big.head(14)[["name", "archetype", "balls", "matchup_delta_per_100",
                        "interaction_per_100", "ci_low_per_100", "ci_high_per_100"]]
          .to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
