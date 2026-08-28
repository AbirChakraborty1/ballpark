"""Generate notebooks/ballpark_end_to_end.ipynb — a numbered, run-top-to-bottom
walk through the whole pipeline, ending in interactive explorers.

    python scripts/make_notebook.py
"""
from __future__ import annotations

import pathlib

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str):
    cells.append(nbf.v4.new_markdown_cell(src.strip("\n")))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# `ballpark` — end to end

Run this notebook top to bottom. Give it the path to a cricsheet **`ipl_male_csv2.zip`**
in section 0.1 and it will build the whole pipeline — ingest → per-ball state →
the four model layers → the app data — and finish with dropdown explorers for
matches, players, matchups and bowling changes.

* **Fast path (default):** `RUN_MODELS = False` uses the trained models committed
  to the repo. The notebook runs in about two minutes and every explorer works.
* **Full path:** `RUN_MODELS = True` retrains every layer from *your* zip
  (~30–40 minutes) and overwrites `models/` and `data/processed/`.

Sections are numbered `1`, `1.1`, `1.2`, … — run them in order.
""")

# ── 0 ──────────────────────────────────────────────────────────────────────
md("## 0 · Setup")

md(r"""
### 0.1 · Point at your data

Set `ZIP_PATH` to your cricsheet dump. Everything else can stay as-is.
`WORKDIR` is where the code and the build artifacts live; if the `ballpark`
repo is not already there, section 0.2 clones it.
""")
code(r"""
from pathlib import Path

# ---- the only two things you might need to change --------------------------
ZIP_PATH = Path.home() / "Downloads" / "ipl_male_csv2.zip"   # <-- your cricsheet zip
WORKDIR  = Path.cwd()                                        # where the repo lives / will be cloned
# --------------------------------------------------------------------------

RUN_MODELS   = False   # False: use the repo's trained models (fast).  True: retrain from your zip.
REPO_URL     = "https://github.com/AbirChakraborty1/ballpark.git"

print("zip     :", ZIP_PATH, "—", "found" if ZIP_PATH.exists() else "NOT FOUND (fix ZIP_PATH)")
print("workdir :", WORKDIR)
print("mode    :", "retrain everything" if RUN_MODELS else "use committed models")
""")

md(r"""
### 0.2 · Get the code

If a `ballpark` checkout is already in `WORKDIR` (or `WORKDIR` *is* the repo),
this is a no-op. Otherwise it shallow-clones it. Then it installs the package so
`import ballpark` works.
""")
code(r"""
import subprocess, sys

def _has_repo(p: Path) -> bool:
    return (p / "src" / "ballpark").is_dir()

if _has_repo(WORKDIR):
    REPO = WORKDIR
elif _has_repo(WORKDIR / "ballpark"):
    REPO = WORKDIR / "ballpark"
else:
    REPO = WORKDIR / "ballpark"
    print("cloning", REPO_URL, "->", REPO)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)

print("repo:", REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
sys.path.insert(0, str(REPO / "src"))
print("installed. ballpark importable:", __import__("importlib").util.find_spec("ballpark") is not None)
""")

md(r"""
### 0.3 · Imports and configuration

`config.yaml` in the repo is the single source of truth for paths, the season
splits and every model hyper-parameter — nothing is hard-coded in the notebook.
""")
code(r"""
import json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 160)

import os
os.chdir(REPO)                                  # the package resolves paths relative to the repo root

from ballpark.config import load_config, processed, raw_dir, reference, models_dir
CFG = load_config()

GOOD, BAD, ACCENT, NEUTRAL = "#1f9d76", "#d1495b", "#3d5a80", "#8a8f98"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .15, "font.size": 10})

def step(label):
    # context manager: print a timed banner around a pipeline stage
    class _T:
        def __enter__(s): s.t = time.time(); print(f">> {label} ...", flush=True); return s
        def __exit__(s, *a): print(f"   {label} done in {time.time()-s.t:,.0f}s\n", flush=True)
    return _T()

print("splits:", CFG["splits"]["train_seasons_max"], "| val", CFG["splits"]["val_seasons"],
      "| test", CFG["splits"]["test_seasons"])
print("recency half-life (seasons):", CFG["models"]["recency_halflife_seasons"])
""")

# ── 1 ──────────────────────────────────────────────────────────────────────
md("## 1 · Data spine")
md(r"""
Two tidy tables from a pile of cricsheet CSVs: **`deliveries`** (one row per
ball) and **`matches`** (one row per game), plus a **`registry`** that maps every
name to a stable 8-character player id. The parser is the part most public IPL
projects get subtly wrong, so section 1.4 reconciles it against published
scorecards.
""")

md("### 1.1 · Unpack the cricsheet zip")
code(r"""
import zipfile

target = raw_dir()                       # data/raw/ipl_male_csv2/
target.mkdir(parents=True, exist_ok=True)

with step("unzip"):
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(target)

info_files = sorted(target.glob("*_info.csv"))
assert info_files, f"no *_info.csv under {target} — is ZIP_PATH the right file?"
assert (target / "all_matches.csv").exists(), "all_matches.csv missing from the dump"
print(f"{len(info_files):,} matches unpacked to {target}")
""")

md("### 1.2 · Ingest → tidy tables")
code(r"""
from ballpark import ingest

with step("ingest (parse info files + ball-by-ball, attach player ids)"):
    ingest.main()

deliveries = pd.read_parquet(processed("deliveries.parquet"))
matches    = pd.read_parquet(processed("matches.parquet"))
registry   = pd.read_parquet(processed("registry.parquet"))

print(f"deliveries : {len(deliveries):,}")
print(f"matches    : {len(matches):,}   seasons {matches.season_year.min()}–{matches.season_year.max()}")
print(f"players    : {registry.person_id.nunique():,}")
deliveries.head(3)
""")

md("### 1.3 · Team & venue normalisation")
md(r"""
19 raw team strings hide 14 franchises; ~60 venue strings hide ~36 grounds. Every
mapping lives in `reference/*.csv` so it is reviewable, and an unmapped string is
a hard error, never a silent `NaN`.
""")
code(r"""
teams  = pd.read_csv(reference("teams.csv"),  keep_default_na=False)
venues = pd.read_csv(reference("venues.csv"), keep_default_na=False)

print("teams  :", teams.raw_team.nunique(), "raw strings ->", teams.team.nunique(),
      "franchises  (in this dump:", deliveries.batting_team.nunique(), "played)")
print("venues :", venues.raw_venue.nunique(), "raw strings ->", venues.venue.nunique(),
      "grounds  (in this dump:", deliveries.venue.nunique(), "used)")

# the traps that must NOT be merged
for a, b in [("Gujarat Lions", "Gujarat Titans"), ("Deccan Chargers", "Sunrisers Hyderabad")]:
    ok = {a, b} <= set(deliveries.batting_team)
    print(f"  kept separate: {a} / {b}  {'OK' if ok else 'MISSING'}")

venues[venues.venue.duplicated(keep=False)].sort_values("venue").head(12)
""")

md("### 1.4 · Parser check — reconcile five scorecards")
md(r"""
The ball-by-ball file and the info file are two independent records of the same
match. Reconciling one against the other, across every match, is a real
correctness test. This runs the same assertions the CI suite runs.
""")
code(r"""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ingest.py", "-q"],
                   cwd=REPO, capture_output=True, text=True)
print(r.stdout[-2000:] or r.stderr[-2000:])
""")

md("### 1.5 · Per-ball match state")
md(r"""
27 features describing the situation the bowler runs in to — phase, wickets in
hand, required rate, how *set* the striker is, and so on. Every one is computed
from deliveries **strictly before** the current ball; a leakage test enforces it.
""")
code(r"""
from ballpark import state

with step("build per-ball state features"):
    state.main()

st = pd.read_parquet(processed("state.parquet"))
print(f"state rows : {len(st):,}   ({st.match_id.nunique():,} matches, innings 1–2 only)")
print("features   :", ", ".join(state.FEATURES[:8]), "…")
st[["over", "phase", "score", "wickets_in_hand", "run_rate", "striker_balls_faced",
    "required_rate", "outcome_class"]].sample(6, random_state=1)
""")

md("### 1.6 · A first look at the data")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))

g = (st.groupby(["phase", "wickets_in_hand"], observed=True).runs_total.mean().mul(6)
       .unstack().reindex(["powerplay", "middle", "death"]))
im = ax[0].imshow(g.values, aspect="auto", cmap="RdYlGn")
ax[0].set_xticks(range(g.shape[1])); ax[0].set_xticklabels(g.columns)
ax[0].set_yticks(range(3)); ax[0].set_yticklabels(g.index)
ax[0].set_title("run rate by phase × wickets in hand"); ax[0].grid(False)
fig.colorbar(im, ax=ax[0], shrink=.8)

sb = st.groupby(pd.cut(st.striker_balls_faced, [-1, 4, 10, 20, 40, 200]),
                observed=True).runs_off_bat.mean()
ax[1].bar(range(len(sb)), sb.values, color=ACCENT)
ax[1].set_xticks(range(len(sb))); ax[1].set_xticklabels([str(i) for i in sb.index], rotation=30, ha="right")
ax[1].set_title("the set-batter effect (runs/ball by balls faced)")

par = (deliveries[deliveries.innings == 1].groupby(["match_id", "venue"]).runs_total.sum()
       .groupby("venue").mean().sort_values().tail(12))
ax[2].barh(range(len(par)), par.values, color=NEUTRAL)
ax[2].set_yticks(range(len(par))); ax[2].set_yticklabels([v[:22] for v in par.index], fontsize=7)
ax[2].set_title("venue par (mean 1st-innings total)")
plt.tight_layout(); plt.show()
""")

# ── 2 ──────────────────────────────────────────────────────────────────────
md("## 2 · Layer 1 — expected runs & wickets")
md(r"""
A gradient-boosted model over the run outcome of a ball, conditioned **only on
match state** — deliberately blind to who is batting or bowling. That makes it
the neutral yardstick: what an average pair produces in exactly this situation.
`xRuns = Σ P(k) · k`.
""")

md("### 2.1 · Train or load")
code(r"""
from ballpark.models import _common as C
from ballpark.models import outcome

if RUN_MODELS:
    with step("Layer 1 — walk-forward train (this is a long one)"):
        outcome.main()
else:
    print("RUN_MODELS is False — using models/outcome.joblib from the repo")

art = C.load("outcome")
xruns = pd.read_parquet(processed("xruns.parquet"))
print("out-of-sample ball predictions:", f"{len(xruns):,}",
      f"({xruns.season_year.min()}–{xruns.season_year.max()})")
""")

md("### 2.2 · xRuns vs actual, by phase")
code(r"""
j = xruns.merge(st[["match_id", "innings", "ball", "phase", "runs_off_bat"]],
                on=["match_id", "innings", "ball"])
by = j.groupby("phase", observed=True).agg(actual=("runs_off_bat", "mean"),
                                           xRuns=("x_runs", "mean"), balls=("x_runs", "size"))
display(by.round(3))

ax = by[["actual", "xRuns"]].plot.bar(color=[ACCENT, NEUTRAL], figsize=(6, 3.2), rot=0)
ax.set_ylabel("runs per ball"); ax.set_title("xRuns tracks actual within each phase"); plt.show()
print(f"overall xRuns mean {j.x_runs.mean():.3f}  vs actual {j.runs_off_bat.mean():.3f}"
      f"   (bias {j.x_runs.mean()-j.runs_off_bat.mean():+.3f} runs/ball — the documented drift)")
""")

md("### 2.3 · Does it beat the baseline?")
code(r"""
per = pd.read_csv(processed("outcome_walkforward.csv"))
tbl = per[["season", "xruns_rmse", "base_rmse", "xruns_bias",
           "wicket_log_loss", "base_wicket_log_loss"]].round(4)
display(tbl)
print(f"mean walk-forward xRuns RMSE {per.xruns_rmse.mean():.3f}  vs baseline {per.base_rmse.mean():.3f}")
print(f"mean wicket log loss        {per.wicket_log_loss.mean():.3f}  vs baseline "
      f"{per.base_wicket_log_loss.mean():.3f}")
print("\nThe margins are small on purpose — a single ball is almost pure noise. "
      "Layer 1 earns its place by being an *unbiased* context baseline.")
""")

# ── 3 ──────────────────────────────────────────────────────────────────────
md("## 3 · Layer 2 — win probability")
md(r"""
`P(team batting now wins | state)`. The hard part is honesty about sample size: a
chase is ~1 independent observation, not 120 balls. The shipped model is a
heavily-shrunk tree blended 20/80 with a logistic on required rate / wickets /
balls, then calibrated out-of-fold.
""")

md("### 3.1 · Train or load")
code(r"""
from ballpark.models import winprob

if RUN_MODELS:
    with step("Layer 2 — walk-forward train + calibrate"):
        winprob.main()
else:
    print("RUN_MODELS is False — using models/winprob.joblib from the repo")

wp = pd.read_parquet(processed("winprob.parquet"))
wpm = json.loads((models_dir() / "winprob_metrics.json").read_text())["metrics"]["walk_forward"]
print("out-of-sample win probabilities:", f"{len(wp):,}")
""")

md("### 3.2 · Calibration by phase")
code(r"""
from ballpark.models._common import calibration_table, binary_metrics

jj = wp.merge(st[["match_id", "innings", "ball", "phase"]], on=["match_id", "innings", "ball"])
fig, ax = plt.subplots(figsize=(4.8, 4.8))
ax.plot([0, 1], [0, 1], "--", color=NEUTRAL, lw=1)
for ph, col in zip(["powerplay", "middle", "death"], [NEUTRAL, ACCENT, BAD]):
    s = jj[(jj.innings == 2) & (jj.phase == ph)]
    t = calibration_table(s.batting_team_won.to_numpy(float), s.win_prob.to_numpy(), bins=12)
    ax.plot(t.predicted, t.observed, "o-", ms=4, color=col, label=f"{ph} (n={len(s):,})")
ax.set_xlabel("predicted win probability"); ax.set_ylabel("observed win rate")
ax.set_title("2nd-innings calibration, walk-forward"); ax.legend(fontsize=8); plt.show()

m = binary_metrics(wp.batting_team_won.to_numpy(float), wp.win_prob.to_numpy())
print(f"all balls   ECE {m['ece']:.3f}   AUC {m.get('auc', float('nan')):.3f}")
""")

md("### 3.3 · Walk-forward backtest table")
code(r"""
bt = pd.DataFrame(wpm)[["season", "log_loss", "base_log_loss", "brier", "base_brier",
                        "auc", "ece"]].round(4)
display(bt)
test = bt[bt.season >= CFG["splits"]["test_seasons"][0]]
print(f"test seasons {list(test.season)}:  Brier {test.brier.mean():.3f} vs baseline "
      f"{test.base_brier.mean():.3f}   (lower is better)")
""")

# ── 4 ──────────────────────────────────────────────────────────────────────
md("## 4 · Layer 3 — player impact")
md(r"""
Two numbers per player. **Wins added** = summed win-probability swing (what
happened). **Shrunk true rate** = runs above the Layer-1 expectation, regressed
on one-hot batter and bowler columns under an L2 penalty — an empirical-Bayes
prior with the CV-chosen penalty *as* the shrinkage.
""")

md("### 4.1 · Win probability added → wins added")
code(r"""
from ballpark.models import impact

if RUN_MODELS:
    with step("Layer 3 — WPA attribution + ridge + bootstrap"):
        impact.main()
else:
    print("RUN_MODELS is False — reading data/processed/*.parquet from the repo")

wpa       = pd.read_parquet(processed("wpa.parquet"))
player_wa = pd.read_parquet(processed("player_wpa.parquet"))
effects   = pd.read_parquet(processed("player_effects.parquet"))

career = (player_wa.groupby(["name", "role"]).agg(balls=("balls", "sum"),
          wins_added=("wins_added", "sum")).reset_index())
print("most wins added, batting (min 500 balls):")
display(career[(career.role == "bat") & (career.balls >= 500)]
        .nlargest(10, "wins_added").round(2).reset_index(drop=True))
""")

md("### 4.2 · Shrunk true rate")
code(r"""
mb = CFG["models"]["impact"]["min_balls_display"]
bat = effects[(effects.role == "bat") & (effects.balls >= mb)]
print(f"best batters by shrunk runs/100 above expectation (min {mb} balls):")
display(bat.nlargest(12, "shrunk_per_100")[
    ["name", "balls", "raw_per_100", "naive_above_expected_per_100",
     "shrunk_per_100", "ci_low", "ci_high"]].round(1).reset_index(drop=True))

# how much shrinkage moved things, by sample size
bins = pd.cut(effects.balls, [0, 100, 300, 1000, 3000, 100000])
display(effects.groupby(bins, observed=True).agg(
    players=("balls", "size"),
    naive_sd=("naive_above_expected_per_100", "std"),
    shrunk_sd=("shrunk_per_100", "std")).round(2))
""")

md("### 4.3 · Raw vs shrunk")
code(r"""
for role, title in [("bat", "batters"), ("bowl", "bowlers")]:
    t = effects[(effects.role == role) & (effects.balls >= mb)]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = np.abs(np.r_[t.naive_above_expected_per_100, t.shrunk_per_100]).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], "--", color=NEUTRAL, lw=1)
    ax.axhline(0, color=NEUTRAL, lw=.5); ax.axvline(0, color=NEUTRAL, lw=.5)
    ax.scatter(t.naive_above_expected_per_100, t.shrunk_per_100,
               s=np.clip(t.balls / t.balls.max() * 240, 12, 240), alpha=.5, color=ACCENT,
               edgecolor="white", lw=.5)
    gap = (t.naive_above_expected_per_100 - t.shrunk_per_100).abs()
    for _, r in t.loc[gap.nlargest(6).index].iterrows():
        ax.annotate(r["name"], (r.naive_above_expected_per_100, r.shrunk_per_100), fontsize=7)
    ax.set_xlabel("naive: raw − expected  (runs/100)")
    ax.set_ylabel("shrunk true effect  (runs/100)")
    ax.set_title(f"{title}: what survives shrinkage"); plt.show()
""")

# ── 5 ──────────────────────────────────────────────────────────────────────
md("## 5 · Layer 4 — matchups & tactics")

md("### 5.1 · Bowler archetypes & coverage")
md(r"""
`reference/players_meta.csv` is a hand-curated bat-hand / bowl-style map for the
~320 highest-volume players. Everything tactical (six archetypes: pace/spin × arm
× wrist/finger) is derived from it.
""")
code(r"""
from ballpark.archetypes import attach, coverage, ARCHETYPES
cov = coverage(st)
print("archetypes:", ", ".join(ARCHETYPES))
print(f"coverage — batter hand {cov['bat_hand_pct']:.0%}, bowler archetype "
      f"{cov['bowl_archetype_pct']:.0%}, both {cov['both_pct']:.0%}  "
      f"({cov['curated_players']} players curated)")
""")

md("### 5.2 · The matchup model")
code(r"""
from ballpark.models import matchup

if RUN_MODELS:
    with step("Layer 4 — matchup ridge + bootstrap"):
        matchup.main()
else:
    print("RUN_MODELS is False — reading data/processed/matchups.parquet from the repo")

mu = pd.read_parquet(processed("matchups.parquet"))
seen = mu[mu.balls >= 40]

x = pd.read_parquet(processed("xruns.parquet")).drop(columns="season_year")
d = attach(st.merge(x, on=["match_id", "innings", "ball"]))
d = d[d.bowl_archetype.notna()]
raw = (d.groupby(["striker_id", "bowl_archetype"], observed=True)
         .apply(lambda g: (g.runs_off_bat.mean() - g.x_runs.mean()) * 100, include_groups=False))
print(f"{len(seen)} batter×archetype cells with ≥40 balls")
print(f"mean |raw split|          {raw.abs().mean():.1f} runs/100")
print(f"mean |shrunk matchup|     {seen.matchup_delta_per_100.abs().mean():.1f} runs/100"
      f"   → {seen.matchup_delta_per_100.abs().mean()/raw.abs().mean():.0%} of the raw figure")
""")

md("### 5.3 · The bowling-change optimiser")
md(r"""
An expected-value rollout: for each remaining over, `runs = 6·xRuns(state) −
bowler_effect·6`, advance the state. The projected total is then treated as a
spread — `Normal(mean, 2.85·√balls_left)` — and Layer 2 is averaged over it,
so a knife-edge chase reads near 50/50 instead of snapping to a near-certain
result. The search keeps the allocation that minimises the chasing side's win
probability. Section 6.4 makes it interactive; here is one worked state.
""")
code(r"""
tac = pd.read_parquet(processed("app/tactics.parquet")) if processed("app/tactics.parquet").exists() \
      else pd.read_parquet(REPO / "data/processed/app/tactics.parquet")
close = (tac.delta.abs() < 0.02).mean()
print(f"over {len(tac):,} close-finish states, the optimiser's plan is within two "
      f"win-probability points of the captain's {close:.0%} of the time\n")
display(tac.sort_values("delta", ascending=False)
        .head(6)[["label", "from_over", "needed", "balls_left", "captain", "optimiser",
                  "captain_wp", "optimiser_wp", "delta"]]
        .round(3).reset_index(drop=True))
""")

# ── 6 ──────────────────────────────────────────────────────────────────────
md("## 6 · Explore")
md(r"""
Interactive versions of the app's tabs. Change a dropdown and the cell redraws.
They read the compact `data/processed/app/` bundle, so they work whether or not
you retrained.
""")
code(r"""
import ipywidgets as W
from IPython.display import display

APP = processed("app")
def _app(name):
    p = APP / name
    return pd.read_parquet(p if p.exists() else REPO / "data/processed/app" / name)

replay_all  = _app("replay.parquet")
matches_app = _app("matches.parquet")
eff_app     = _app("player_effects.parquet")
wa_app      = _app("player_wpa.parquet")
mu_app      = _app("matchups.parquet")
tac_app     = _app("tactics.parquet")

covered = matches_app[matches_app.match_id.isin(replay_all.match_id.unique())].copy()
covered["label"] = (covered.season_year.astype(str) + "  ·  " + covered.team_1 + " v "
                    + covered.team_2 + "  ·  " + covered.venue.str.replace(" Stadium", "", regex=False))
covered = covered.sort_values("start_date", ascending=False)
print(f"{len(covered)} matches available in the explorers")
""")

md("### 6.1 · Match replay")
md("Pick a match: the win-probability ribbon over both innings, and the balls that moved it most.")
code(r"""
def match_replay(match_label):
    mid = int(covered.loc[covered.label == match_label, "match_id"].iloc[0])
    row = covered[covered.match_id == mid].iloc[0]
    df = replay_all[replay_all.match_id == mid].sort_values(["innings", "ball"]).reset_index(drop=True)
    df["ball_no"] = np.arange(len(df)) + 1
    df["wp_team1"] = np.where(df.innings == 1, df.win_prob, 1 - df.win_prob)
    brk = int((df.innings == 1).sum())

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axhline(.5, color=NEUTRAL, ls=":", lw=1); ax.axvline(brk, color=NEUTRAL, lw=1)
    ax.plot(df.ball_no, df.wp_team1, color=ACCENT, lw=2)
    ax.fill_between(df.ball_no, df.wp_team1, .5, color=ACCENT, alpha=.12)
    ax.set_ylim(0, 1); ax.set_xlabel("ball"); ax.set_ylabel(f"P({row.team_1} win)")
    ax.set_title(f"{row.label}   —   {row.result_team} won")
    plt.show()

    sw = df.reindex(df.wpa.abs().sort_values(ascending=False).index).head(8).copy()
    sw["over"] = sw.over.astype(str) + "." + sw.ball_in_over.astype(str)
    sw["event"] = np.where(sw.wicket_type.fillna("") != "",
                           sw.wicket_type + " — " + sw.player_dismissed.fillna(""),
                           sw.runs_off_bat.astype(str) + " off the bat")
    sw["Δ win% (bat side)"] = (sw.wpa * 100).round(1)
    display(sw[["innings", "over", "batting_team", "bowler", "event", "Δ win% (bat side)"]]
            .reset_index(drop=True))

W.interact(match_replay,
           match_label=W.Dropdown(options=list(covered.label), description="match",
                                  layout=W.Layout(width="640px")));
""")

md("### 6.2 · Players")
md("Pick a role and a player: the raw-vs-shrunk numbers and season-by-season wins added.")
code(r"""
def player_view(role, name):
    r = "bat" if role == "Batting" else "bowl"
    e = eff_app[(eff_app.role == r) & (eff_app.name == name)]
    if e.empty:
        print("no shrunk estimate (below the minimum-balls cutoff)");
    else:
        e = e.iloc[0]
        print(f"{name} — {int(e.balls):,} balls")
        print(f"  raw / 100 balls           {e.raw_per_100:6.1f}")
        print(f"  expected / 100 (xRuns)     {e.expected_per_100:6.1f}")
        print(f"  naive gap                  {e.naive_above_expected_per_100:+6.1f}")
        print(f"  SHRUNK true rate / 100     {e.shrunk_per_100:+6.1f}   95% CI [{e.ci_low:+.1f}, {e.ci_high:+.1f}]")

    w = wa_app[(wa_app.role == r) & (wa_app.name == name)].sort_values("season_year")
    if not w.empty:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.bar(w.season_year.astype(str), w.wins_added,
               color=[GOOD if v >= 0 else BAD for v in w.wins_added])
        ax.axhline(0, color="k", lw=.6); ax.set_title(f"{name} — wins added by season "
                   f"(career {w.wins_added.sum():+.1f})")
        plt.show()

_role = W.RadioButtons(options=["Batting", "Bowling"], description="role")
_name = W.Dropdown(description="player", layout=W.Layout(width="360px"))
def _sync(*_):
    r = "bat" if _role.value == "Batting" else "bowl"
    opts = (eff_app[eff_app.role == r].sort_values("balls", ascending=False).name.tolist())
    _name.options = opts; _name.value = opts[0] if opts else None
_role.observe(_sync, "value"); _sync()
W.interact(player_view, role=_role, name=_name);
""")

md("### 6.3 · Matchups")
md("Pick a batter: expected runs per 100 balls against each bowler archetype, with a 95% interval.")
code(r"""
def matchup_view(batter):
    s = mu_app[mu_app.name == batter].sort_values("expected_runs_per_100")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(s.archetype, s.expected_runs_per_100, color=ACCENT, alpha=.85)
    ax.errorbar(s.expected_runs_per_100, range(len(s)),
                xerr=[s.matchup_delta_per_100 - s.ci_low_per_100,
                      s.ci_high_per_100 - s.matchup_delta_per_100],
                fmt="none", ecolor=NEUTRAL, capsize=3)
    ax.set_xlabel("expected runs per 100 balls (shrunk)"); ax.set_title(batter)
    plt.show()
    display(s[["archetype", "balls", "archetype_prior_per_100", "expected_runs_per_100",
               "matchup_delta_per_100", "interaction_per_100"]].round(1).reset_index(drop=True))

W.interact(matchup_view,
           batter=W.Dropdown(options=sorted(mu_app.name.unique()),
                             value="V Kohli" if "V Kohli" in set(mu_app.name) else None,
                             description="batter", layout=W.Layout(width="360px")));
""")

md("### 6.4 · Tactics")
md(r"""
Pick a close finish and an over. The pre-computed optimiser result: its
over-by-over allocation next to the captain's, and the projected win-probability
gap. (These are read from the bundle; to solve a brand-new state live, use
`ballpark.models.optimise.BowlingOptimiser`.)
""")
code(r"""
tac_app["state"] = tac_app.label + "   —   over " + tac_app.from_over.astype(str) \
                   + "  (" + tac_app.needed.astype(str) + " off " + tac_app.balls_left.astype(str) + ")"
order = tac_app.sort_values("delta", ascending=False)

def tactics_view(state):
    r = tac_app[tac_app.state == state].iloc[0]
    print(r.label)
    print(f"start of over {int(r.from_over)}:  {int(r.score)}/{int(r.wickets)}, chasing "
          f"{int(r.target)}  →  {int(r.needed)} needed off {int(r.balls_left)}.   {r.result_team} won.\n")
    print(f"  OPTIMISER   {r.optimiser:<52s}  chase win prob {r.optimiser_wp:5.0%}   "
          f"(proj {r.optimiser_score:.0f})")
    print(f"  CAPTAIN     {r.captain:<52s}  chase win prob {r.captain_wp:5.0%}"
          if pd.notna(r.captain_wp) else "  CAPTAIN     —")
    if pd.notna(r.captain_wp):
        print(f"\n  Δ handed to the batting side by the actual choice:  {r.captain_wp - r.optimiser_wp:+.0%}")
    print("\n  alternatives the search considered:")
    for a in r.alternatives.split(" | "):
        print("   ", a)

W.interact(tactics_view,
           state=W.Dropdown(options=list(order.state), description="state",
                            layout=W.Layout(width="720px")));
""")

md(r"""
---
Built from [`ballpark`](https://github.com/AbirChakraborty1/ballpark) ·
live app <https://ballpark-mkmljvquubqdhwezbkgdtg.streamlit.app/> ·
the [field guide](https://claude.ai/code/artifact/1b8019e8-a98a-4ff3-a35f-0844304111c6)
explains every number.
""")

# ─────────────────────────────────────────────────────────────────────────────
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "ballpark_end_to_end.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out, "—", len(cells), "cells")
