"""Held-out comparison: TODAY's live CFB method vs the FIXED method
(fix/cfb-live-ratings part C -- the ship evidence).

Scores, on held-out 2023-2025 REG FBS-vs-FBS games carrying a closing spread
(the games the live producer serves; "FCS" pseudo-team games are skipped
there), week by week exactly as the producer would have run before each
week:

  today  -- the pre-fix generate_cfb, reproduced faithfully:
            `legacy_season_to_date_ratings` (frozen copy of the old
            `_season_to_date_ratings`: Elo, SRS, points ratings and
            games_played POOLED over every REG game since 2015 before
            (season, week)) + today's prior (`legacy_priors_for_season`:
            R_pre = 1500 + 1.0 * SAME-season sp_rating, every other weight 0,
            decay half-life 4 / floor 0 -- the committed assets before part B).
  fixed  -- the current generate_cfb: `_season_to_date_ratings` (shared
            walk-forward state) + the leak-free refit prior and decay
            (assets/cfb/priors_weights.json / priors_decay.json).

Both go through the SAME `generate_cfb.build_game_rows` + committed
gameline.json (empty market, so pred = model - bias), i.e. the served
pred_margin / pred_total. Metrics: margin MAE (all weeks, weeks 1-5), total
MAE, ATS vs the closing spread (model side; pushes excluded), n; plus the
paired margin-MAE difference with its standard error.

Usage:
    .venv/bin/python scripts/compare_cfb_live_fix.py [--seasons 2023 2024 2025]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from sportsmodel.cfb.priors import DecayConfig, load_decay_config, load_weights
from sportsmodel.cfb.teams import FCS
from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.points import compute_points_ratings
from sportsmodel.nfl.ratings import BlendConfig
from sportsmodel.nfl.srs import compute_srs


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gc = _load("generate_cfb_cmp", "scripts/generate_cfb.py")
bcp = _load("backtest_cfb_priors_cmp", "scripts/backtest_cfb_priors.py")

HOLDOUT_SEASONS = (2023, 2024, 2025)
EARLY_WEEKS = {1, 2, 3, 4, 5}
LEGACY_DECAY = DecayConfig(half_life_games=4.0, prior_floor=0.0)   # pre-fix priors_decay.json
_LEGACY_TOTAL_SEED = 55.0


# ---------------------------------------------------------------------------
# TODAY's method (frozen copies of the pre-fix live code)
# ---------------------------------------------------------------------------

def legacy_season_to_date_ratings(sched: pd.DataFrame, season: int, week: int,
                                  elo_cfg: EloConfig, blend_cfg: BlendConfig) -> dict:
    """Verbatim logic of the pre-fix generate_cfb._season_to_date_ratings:
    every REG game before (season, week) since 2015, POOLED."""
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched.columns else sched
    reg = reg[reg["home_team"] != reg["away_team"]].copy()
    played = reg[(reg["season"] < season) | ((reg["season"] == season) & (reg["week"] < week))].copy()
    played = played.dropna(subset=["home_score", "away_score"])

    elo_final = run_elo(played, elo_cfg).final if len(played) else {}
    srs_now = compute_srs(played) if len(played) else {}
    if len(played):
        points_ratings, lg_avg = compute_points_ratings(played)
    else:
        points_ratings, lg_avg = {}, _LEGACY_TOTAL_SEED
    games_played: dict[str, int] = {}
    for _, g in played.iterrows():
        games_played[g["home_team"]] = games_played.get(g["home_team"], 0) + 1
        games_played[g["away_team"]] = games_played.get(g["away_team"], 0) + 1
    return {"elo_final": elo_final, "srs_now": srs_now, "points_ratings": points_ratings,
            "lg_avg": lg_avg, "games_played": games_played, "elo_cfg": elo_cfg,
            "blend_cfg": blend_cfg}


def legacy_priors_for_season(priors: pd.DataFrame, season: int) -> dict[str, float]:
    """Pre-fix R_pre with the committed pre-fix weights (sp_scale 1,
    sp_offset 1500, every other weight 0): 1500 + SAME-season sp_rating."""
    rows = priors[priors["season"] == season]
    return {r.team_espn_id: 1500.0 + 1.0 * float(r.sp_rating) for r in rows.itertuples()
            if not pd.isna(r.sp_rating)}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(rows: list[dict], key: str) -> dict:
    """rows carry {week, <key>_margin, <key>_total, actual_margin,
    actual_total, market_spread (home-margin convention)}."""
    def mae(rs, k, a):
        return sum(abs(r[k] - r[a]) for r in rs) / len(rs) if rs else float("nan")
    early = [r for r in rows if r["week"] in EARLY_WEEKS]
    late = [r for r in rows if r["week"] not in EARLY_WEEKS]
    graded = [bcp.grade_vs_market(r[f"{key}_margin"], r["market_spread"], r["actual_margin"])
              ["ats"] for r in rows]
    wins, losses = graded.count("win"), graded.count("loss")
    return {"n": len(rows), "n_early": len(early), "n_late": len(late),
            "margin_mae_wk6p": mae(late, f"{key}_margin", "actual_margin"),
            "margin_mae": mae(rows, f"{key}_margin", "actual_margin"),
            "margin_mae_wk1_5": mae(early, f"{key}_margin", "actual_margin"),
            "total_mae": mae(rows, f"{key}_total", "actual_total"),
            "total_mae_wk1_5": mae(early, f"{key}_total", "actual_total"),
            "ats_w": wins, "ats_l": losses, "ats_push": graded.count("push"),
            "ats_pct": wins / (wins + losses) if wins + losses else float("nan")}


def paired_diff(rows: list[dict], a: str, b: str, target: str = "margin") -> tuple[float, float]:
    """mean(|a - actual| - |b - actual|) and its standard error."""
    d = [abs(r[f"{a}_{target}"] - r[f"actual_{target}"]) - abs(r[f"{b}_{target}"] - r[f"actual_{target}"])
         for r in rows]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return m, math.sqrt(var / n)


# ---------------------------------------------------------------------------
# Walk the held-out weeks
# ---------------------------------------------------------------------------

def build_rows(seasons=HOLDOUT_SEASONS, verbose: bool = True) -> list[dict]:
    elo_cfg, blend_cfg = gc.load_rating()
    gl_cfg = gc.load_gameline()
    sched = pd.read_parquet(ROOT / "assets/cfb/schedules.parquet")
    lines = pd.read_parquet(ROOT / "assets/cfb/lines.parquet")
    lines = lines[lines["home_team"] != lines["away_team"]].drop_duplicates(
        subset=["season", "week", "home_team", "away_team"], keep="first")
    priors = pd.read_parquet(ROOT / "assets/cfb/priors.parquet")
    fixed_w = load_weights(ROOT / "assets/cfb/priors_weights.json")
    fixed_decay = load_decay_config(ROOT / "assets/cfb/priors_decay.json")

    reg = sched[(sched["game_type"] == "REG") & (sched["home_team"] != sched["away_team"])]
    out = []
    for season in seasons:
        r_pre_fixed = gc.load_priors_for_season(season, fixed_w)
        r_pre_today = legacy_priors_for_season(priors, season)
        for week in sorted(reg.loc[reg["season"] == season, "week"].unique()):
            t0 = time.time()
            wk = reg[(reg["season"] == season) & (reg["week"] == week)]
            wk = wk.merge(lines[["season", "week", "home_team", "away_team", "market_spread",
                                 "market_total"]],
                          on=["season", "week", "home_team", "away_team"], how="left")
            games = [{"game_pk": int(g.game_pk), "home_team": g.home_team,
                      "away_team": g.away_team, "home_name": g.home_team,
                      "away_name": g.away_team, "game_date": str(g.start_date)[:10]}
                     for g in wk.itertuples()]
            today_state = legacy_season_to_date_ratings(sched, season, week, elo_cfg, blend_cfg)
            today_state.update({"r_pre": r_pre_today, "decay_cfg": LEGACY_DECAY})
            fixed_state = gc._season_to_date_ratings(sched, season, week, elo_cfg, blend_cfg,
                                                     games)
            fixed_state.update({"r_pre": r_pre_fixed, "decay_cfg": fixed_decay})
            today = {r["game_pk"]: r for r in gc.build_game_rows(games, today_state, week, gl_cfg)}
            fixed = {r["game_pk"]: r for r in gc.build_game_rows(games, fixed_state, week, gl_cfg)}
            for g in wk.itertuples():
                pk = int(g.game_pk)
                if FCS in (g.home_team, g.away_team) or pd.isna(g.market_spread):
                    continue
                out.append({
                    "season": season, "week": int(week), "game_pk": pk,
                    "market_spread": float(g.market_spread),
                    "market_total": None if pd.isna(g.market_total) else float(g.market_total),
                    "actual_margin": float(g.home_score - g.away_score),
                    "actual_total": float(g.home_score + g.away_score),
                    "today_margin": today[pk]["pred_margin"],
                    "today_total": today[pk]["pred_total"],
                    "fixed_margin": fixed[pk]["pred_margin"],
                    "fixed_total": fixed[pk]["pred_total"],
                    "market_margin": float(g.market_spread),
                })
            if verbose:
                print(f"  {season} wk {week:>2}: {len(wk)} games ({time.time() - t0:.1f}s)",
                      flush=True)
    return out


def _fmt(s: dict) -> str:
    return (f"| {s['n']} | {s['margin_mae']:.3f} | {s['margin_mae_wk1_5']:.3f} (n={s['n_early']}) "
            f"| {s['margin_mae_wk6p']:.3f} (n={s['n_late']}) "
            f"| {s['total_mae']:.3f} | {s['total_mae_wk1_5']:.3f} "
            f"| {100 * s['ats_pct']:.1f}% ({s['ats_w']}-{s['ats_l']}-{s['ats_push']}) |")


def report(rows: list[dict]) -> str:
    head = ("| method | n | margin MAE (all) | margin MAE wk 1-5 | margin MAE wk 6+ "
            "| total MAE (all) | total MAE wk 1-5 | ATS vs close (W-L-P) |\n"
            "|---|---|---|---|---|---|---|---|")
    lines = ["### All held-out seasons", "", head]
    for key, label in (("today", "today (pooled + same-season SP+ prior)"),
                       ("fixed", "fixed (walk-forward state + leak-free prior)")):
        lines.append(f"| {label} " + _fmt(score(rows, key)))
    m = score([{**r, "market_total": r["market_total"]} for r in rows], "market")
    lines.append(f"| closing spread (reference) | {m['n']} | {m['margin_mae']:.3f} "
                 f"| {m['margin_mae_wk1_5']:.3f} | {m['margin_mae_wk6p']:.3f} | - | - | - |")
    for tgt in ("margin", "total"):
        d, se = paired_diff(rows, "fixed", "today", tgt)
        lines.append("")
        lines.append(f"Paired {tgt} abs-error difference (fixed - today): {d:+.3f} ± {se:.3f} (SE)")
    early = [r for r in rows if r["week"] in EARLY_WEEKS]
    d, se = paired_diff(early, "fixed", "today", "margin")
    lines.append(f"Paired margin difference, weeks 1-5 only: {d:+.3f} ± {se:.3f} (SE)")
    late = [r for r in rows if r["week"] not in EARLY_WEEKS]
    d, se = paired_diff(late, "fixed", "today", "margin")
    lines.append(f"Paired margin difference, weeks 6+ only: {d:+.3f} ± {se:.3f} (SE)")
    lines += ["", "### By season", "", head.replace("| method |", "| season / method |")]
    for season in sorted({r["season"] for r in rows}):
        sr = [r for r in rows if r["season"] == season]
        for key in ("today", "fixed"):
            lines.append(f"| {season} {key} " + _fmt(score(sr, key)))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=list(HOLDOUT_SEASONS))
    ap.add_argument("--save-rows", type=pathlib.Path, default=None,
                    help="optional JSON dump of the per-game rows")
    args = ap.parse_args(argv)
    rows = build_rows(tuple(args.seasons))
    if args.save_rows:
        args.save_rows.write_text(json.dumps(rows))
    print()
    print(report(rows))


if __name__ == "__main__":
    main()
