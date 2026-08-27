"""Layer 5 -- consolidate the evidence into figures and one metrics file.

Each model's `main()` already prints its own backtest. This module is the
single place that turns those into the artifacts the README, the model-card
page and the report all quote, so a number cannot drift between them:

    reports/metrics.json          headline numbers, one source of truth
    reports/figures/*.png         calibration, reliability, the raw-vs-shrunk
                                  scatter, the shrinkage funnel
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, processed
from .models import _common as C

FIG = load_config()["root"] / "reports" / "figures"
GOOD, BAD, NEUT = "#1f9d76", "#d1495b", "#8a8f98"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15, linewidth=0.6)


# --------------------------------------------------------------------------- #

def winprob_evidence() -> dict:
    wp = pd.read_parquet(processed("winprob.parquet"))
    st = pd.read_parquet(processed("state.parquet"))[
        ["match_id", "innings", "ball", "phase", "balls_remaining", "runs_required"]]
    j = wp.merge(st, on=["match_id", "innings", "ball"])
    per_season = pd.read_csv(processed("winprob_walkforward.csv"))
    cfg = load_config()["splits"]
    test = per_season[per_season.season >= cfg["test_seasons"][0]]

    y = j.batting_team_won.to_numpy(float)
    p = j.win_prob.to_numpy()

    # calibration curve, per phase, innings 2
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=NEUT, lw=1, ls="--")
    for phase, colour in zip(["powerplay", "middle", "death"], [NEUT, "#3d5a80", BAD]):
        s = j[(j.innings == 2) & (j.phase == phase)]
        tab = C.calibration_table(s.batting_team_won.to_numpy(float), s.win_prob.to_numpy(), bins=12)
        ax.plot(tab.predicted, tab.observed, "o-", ms=4, color=colour, label=f"{phase} (n={len(s):,})")
    ax.set_xlabel("predicted win probability"); ax.set_ylabel("observed win rate")
    ax.set_title("Win-probability calibration by phase, 2nd innings\nwalk-forward out-of-sample")
    ax.legend(frameon=False, fontsize=8); _style(ax)
    fig.tight_layout(); fig.savefig(FIG / "winprob_calibration.png", dpi=140); plt.close(fig)

    # reliability, all balls
    fig, ax = plt.subplots(figsize=(5.2, 4))
    tab = C.calibration_table(y, p, bins=15)
    ax.bar(tab.predicted, tab.observed - tab.predicted, width=0.04, color=NEUT)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("predicted win probability"); ax.set_ylabel("observed − predicted")
    ax.set_title("Reliability: over/under-confidence by bucket"); _style(ax)
    fig.tight_layout(); fig.savefig(FIG / "winprob_reliability.png", dpi=140); plt.close(fig)

    overall = C.binary_metrics(y, p)
    i2 = j[j.innings == 2]
    m2 = C.binary_metrics(i2.batting_team_won.to_numpy(float), i2.win_prob.to_numpy())
    return {
        "test_seasons": [int(x) for x in test.season],
        "test_log_loss": float(test.log_loss.mean()),
        "test_base_log_loss": float(test.base_log_loss.mean()),
        "test_brier": float(test.brier.mean()),
        "test_base_brier": float(test.base_brier.mean()),
        "test_ece": float(test.ece.mean()),
        "all_ball_ece": overall["ece"],
        "all_ball_auc": overall.get("auc"),
        "innings2_brier": m2["brier"], "innings2_auc": m2.get("auc"),
        "walk_forward_seasons": [int(x) for x in per_season.season],
    }


def xruns_evidence() -> dict:
    per = pd.read_csv(processed("outcome_walkforward.csv"))
    x = pd.read_parquet(processed("xruns.parquet"))
    st = pd.read_parquet(processed("state.parquet"))[
        ["match_id", "innings", "ball", "phase", "runs_off_bat"]]
    j = x.merge(st, on=["match_id", "innings", "ball"])

    fig, ax = plt.subplots(figsize=(5.2, 4))
    by = j.groupby("phase", observed=True).agg(actual=("runs_off_bat", "mean"),
                                               expected=("x_runs", "mean"))
    idx = np.arange(len(by))
    ax.bar(idx - 0.2, by.actual, 0.4, label="actual", color="#3d5a80")
    ax.bar(idx + 0.2, by.expected, 0.4, label="xRuns", color=NEUT)
    ax.set_xticks(idx); ax.set_xticklabels(by.index)
    ax.set_ylabel("runs per ball"); ax.set_title("xRuns vs actual, out-of-sample")
    ax.legend(frameon=False); _style(ax)
    fig.tight_layout(); fig.savefig(FIG / "xruns_by_phase.png", dpi=140); plt.close(fig)

    return {
        "walk_forward_rmse": float(per.xruns_rmse.mean()),
        "baseline_rmse": float(per.base_rmse.mean()),
        "walk_forward_bias": float(per.xruns_bias.mean()),
        "wicket_log_loss": float(per.wicket_log_loss.mean()),
        "wicket_baseline_log_loss": float(per.base_wicket_log_loss.mean()),
        "seasons": [int(s) for s in per.season],
    }


def impact_evidence() -> dict:
    eff = pd.read_parquet(processed("player_effects.parquet"))
    minb = load_config()["models"]["impact"]["min_balls_display"]

    for role, title, fname in (("bat", "Batters", "scatter_bat"), ("bowl", "Bowlers", "scatter_bowl")):
        t = eff[(eff.role == role) & (eff.balls >= minb)]
        fig, ax = plt.subplots(figsize=(5.4, 5))
        lim = np.abs(np.r_[t.naive_above_expected_per_100, t.shrunk_per_100]).max() * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=NEUT, lw=1, ls="--")
        ax.axhline(0, color=NEUT, lw=0.6); ax.axvline(0, color=NEUT, lw=0.6)
        sizes = np.clip(t.balls / t.balls.max() * 220, 12, 220)
        ax.scatter(t.naive_above_expected_per_100, t.shrunk_per_100, s=sizes,
                   alpha=0.5, color="#3d5a80", edgecolor="white", linewidth=0.5)
        gap = (t.naive_above_expected_per_100 - t.shrunk_per_100).abs()
        for _, r in t.loc[gap.nlargest(6).index].iterrows():
            ax.annotate(r["name"], (r.naive_above_expected_per_100, r.shrunk_per_100),
                        fontsize=7, alpha=0.8)
        ax.set_xlabel("naive: raw − expected  (runs per 100 balls)")
        ax.set_ylabel("shrunk true effect  (runs per 100 balls)")
        ax.set_title(f"{title}: how much 'above expectation' survives shrinkage")
        _style(ax)
        fig.tight_layout(); fig.savefig(FIG / f"{fname}.png", dpi=140); plt.close(fig)

    # shrinkage funnel
    fig, ax = plt.subplots(figsize=(5.4, 4))
    ax.scatter(eff.balls.clip(upper=6000), eff.naive_above_expected_per_100,
               s=8, alpha=0.25, color=BAD, label="naive")
    ax.scatter(eff.balls.clip(upper=6000), eff.shrunk_per_100,
               s=8, alpha=0.5, color=GOOD, label="shrunk")
    ax.set_xlabel("balls (career, capped at 6,000)")
    ax.set_ylabel("runs per 100 above expectation")
    ax.set_title("Shrinkage pulls small samples to the mean"); ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(FIG / "shrinkage_funnel.png", dpi=140); plt.close(fig)

    bat = eff[(eff.role == "bat") & (eff.balls >= minb)]
    over = bat.assign(gap=bat.naive_above_expected_per_100 - bat.shrunk_per_100).nlargest(5, "gap")
    under = bat.assign(gap=bat.shrunk_per_100 - bat.naive_above_expected_per_100).nlargest(5, "gap")
    return {
        "n_players_scored": int(len(eff)),
        "overrated_by_raw": over[["name", "naive_above_expected_per_100", "shrunk_per_100"]]
        .round(1).to_dict("records"),
        "underrated_by_raw": under[["name", "naive_above_expected_per_100", "shrunk_per_100"]]
        .round(1).to_dict("records"),
    }


def matchup_evidence() -> dict:
    m = pd.read_parquet(processed("matchups.parquet"))
    seen = m[m.balls >= 40]
    x = pd.read_parquet(processed("xruns.parquet")).drop(columns="season_year")
    st = pd.read_parquet(processed("state.parquet"))
    from .archetypes import attach
    d = attach(st.merge(x, on=["match_id", "innings", "ball"]))
    d = d[d.bowl_archetype.notna()]
    raw = (d.groupby(["striker_id", "bowl_archetype"], observed=True)
           .apply(lambda g: (g.runs_off_bat.mean() - g.x_runs.mean()) * 100, include_groups=False))
    return {
        "cells_ge_40_balls": int(len(seen)),
        "mean_abs_raw_split_per_100": float(raw.abs().mean()),
        "mean_abs_shrunk_delta_per_100": float(seen.matchup_delta_per_100.abs().mean()),
        "shrinkage_ratio": float(seen.matchup_delta_per_100.abs().mean() / raw.abs().mean()),
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = {
        "data": _data_provenance(),
        "layer1_xruns": xruns_evidence(),
        "layer2_winprob": winprob_evidence(),
        "layer3_impact": impact_evidence(),
        "layer4_matchup": matchup_evidence(),
    }
    (load_config()["root"] / "reports" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))
    print("\nfigures written to", FIG)


def _data_provenance() -> dict:
    d = pd.read_parquet(processed("deliveries.parquet"))
    m = pd.read_parquet(processed("matches.parquet"))
    return {
        "deliveries": int(len(d)),
        "matches": int(len(m)),
        "seasons": [int(m.season_year.min()), int(m.season_year.max())],
        "players": int(pd.read_parquet(processed("registry.parquet")).person_id.nunique()),
    }


if __name__ == "__main__":
    main()
