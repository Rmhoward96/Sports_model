"""CFB ratings backtest + tuning -- THE SHIP-GATE for CFB P1.

Modeled closely on scripts/backtest_nfl_elo.py, reusing the same
league-agnostic engine (sportsmodel.nfl.elo / .srs / .ratings) untouched.
Two deliberate adaptations for CFB scale (~134 teams incl. the "FCS"
aggregate opponent, ~830 games/season vs NFL's 32 teams / ~267 games/season):

1. `run_backtest` refreshes season-to-date SRS once per WEEK (using only
   games from strictly earlier weeks that season) instead of after every
   single game. Recomputing the full Gauss-Seidel SRS solve after each of
   ~830 games/season (an O(games^2) pattern) took ~87s per single train-span
   evaluation here -- intractable for a coordinate-search tune. Batching to
   once-per-week cuts that to ~2s/eval (see docs/superpowers/reports/... for
   the measured numbers) while, if anything, being slightly MORE
   conservative about leakage: it never lets one same-week game's result
   inform another same-week game's SRS input (backtest_nfl_elo.py's
   per-game refresh does, incidentally, allow that for games processed
   later in iteration order within the same week).
2. `run_backtest` takes an optional `eval_seasons` filter. Elo/SRS state is
   always walked forward across the ENTIRE df passed in (so OOS evaluation
   can carry real rating history in from prior seasons, per the brief's
   "ratings use only prior seasons + season-to-date games"), but margin/
   Brier/win-acc are only accumulated for games whose season is in
   `eval_seasons` (or every season, if None). This lets the TRAIN-only tune
   and the continuous-history OOS evaluation share one function.

Data: assets/cfb/schedules.parquet, 2015-2025, all game_type == "REG".
TRAIN span = 2015-2022 (used for hyperparameter selection by margin MAE).
OOS span = 2023-2024 (two complete, held-out seasons). 2025 is excluded
entirely from both tuning and evaluation -- it is the current/most-recent
season in this asset and the brief calls for keeping thin/partial-season
data out of both steps, so it is dropped rather than risking a distorted
grid search or ship-gate number if the asset is regenerated mid-season.
"""
from __future__ import annotations

import itertools
import json
import pathlib

import pandas as pd

from sportsmodel.nfl.elo import EloConfig, elo_expected_margin, expected_home, run_elo
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.srs import compute_srs

TRAIN_SEASONS = list(range(2015, 2023))  # 2015-2022 inclusive
OOS_SEASONS = {2023, 2024}
EXCLUDED_SEASONS = {2025}  # current/partial season -- kept out of tune + eval


def run_backtest(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                  blend_cfg: BlendConfig, eval_seasons=None) -> dict:
    """Leak-free walk-forward backtest. Elo state (via run_elo) and
    season-to-date SRS are walked forward across every game in
    schedule_df, in season/week order, with SRS refreshed once per
    completed week. Metrics are accumulated only for games whose season is
    in `eval_seasons` (default: every season present)."""
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)               # pre-game elo per game, continuous across seasons
    games = res.games
    n = 0; brier = 0.0; correct = 0; abs_err = 0.0; sq_err = 0.0
    for season, sdf in games.groupby("season"):
        eval_this_season = eval_seasons is None or season in eval_seasons
        sdf = sdf.sort_values("week")
        counts: dict = {}
        srs_cache: dict = {}
        srs_hist = sdf.iloc[0:0]
        for week, wdf in sdf.groupby("week"):
            wdf = wdf.dropna(subset=["home_score", "away_score"])
            if wdf.empty:
                continue
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                gh, ga = counts.get(h, 0), counts.get(a, 0)
                srs_h = srs_cache.get(h); srs_a = srs_cache.get(a)
                em = expected_margin(g["elo_home"], g["elo_away"], srs_h, srs_a,
                                     gh, ga, elo_cfg, blend_cfg)
                e_home = g["e_home"]
                actual_margin = g["home_score"] - g["away_score"]
                result_home = 1.0 if actual_margin > 0 else 0.0
                if eval_this_season:
                    brier += (e_home - result_home) ** 2
                    correct += int((e_home >= 0.5) == (result_home == 1.0))
                    abs_err += abs(em - actual_margin)
                    sq_err += (em - actual_margin) ** 2
                    n += 1
            # after grading the whole week, it joins the played set ->
            # update counts + SRS for the NEXT week (season-to-date, never
            # informed by same-or-future-week results)
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                counts[h] = counts.get(h, 0) + 1
                counts[a] = counts.get(a, 0) + 1
            srs_hist = pd.concat([srs_hist, wdf], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
    return {"brier": brier / n, "win_acc": correct / n,
            "margin_mae": abs_err / n, "margin_rmse": (sq_err / n) ** 0.5, "n": n}


def _coordinate_search(train_df, grid: dict) -> tuple:
    """One parameter at a time over its listed values, holding the others
    at a sensible center, for a few passes, selecting by TRAIN-span margin
    MAE (mirrors scripts/backtest_nfl_elo.py's _coordinate_search -- see
    that file's docstring for why margin_mae, not Brier, is the selection
    metric: Brier/win_acc come from e_home which is pure-Elo and untouched
    by blend_cfg/w_sos, so only margin_mae can discriminate the SoS blend).
    Full Cartesian product over this grid (1500 combos) would still work
    now that run_backtest is weekly-batched (~2s/eval), but coordinate
    search keeps the tune fast and mirrors the reference script's method."""
    order = ["k", "hfa_elo", "carryover", "w_sos", "srs_min_games"]
    current = {p: grid[p][len(grid[p]) // 2] for p in order}
    all_results = []
    seen: dict = {}

    def _eval(params: dict) -> dict:
        key = tuple(params[p] for p in order)
        if key in seen:
            return seen[key]
        ec = EloConfig(k=params["k"], hfa_elo=params["hfa_elo"], carryover=params["carryover"])
        bc = BlendConfig(w_sos=params["w_sos"], srs_min_games=params["srs_min_games"])
        tm = run_backtest(train_df, ec, bc)
        r = {"elo": ec, "blend": bc, "train": tm}
        seen[key] = r
        all_results.append(r)
        return r

    n_passes = 3
    for _ in range(n_passes):
        improved = False
        for p in order:
            best_val = current[p]
            best_mae = None
            for v in grid[p]:
                trial = dict(current); trial[p] = v
                r = _eval(trial)
                mae = r["train"]["margin_mae"]
                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_val = v
            if best_val != current[p]:
                improved = True
            current[p] = best_val
        if not improved:
            break

    best = min(all_results, key=lambda r: r["train"]["margin_mae"])
    return (best["elo"], best["blend"]), all_results


def home_always_baseline(oos_df: pd.DataFrame, hfa_margin: float) -> dict:
    """Predict margin = home-field advantage only (no team strength at
    all): the tuned model's hfa_elo converted to margin points via the
    same /25 scale nfl.elo.elo_expected_margin uses."""
    v = oos_df.dropna(subset=["home_score", "away_score"])
    actual = v["home_score"] - v["away_score"]
    err = actual - hfa_margin
    return {"margin_mae": float(err.abs().mean()),
            "margin_rmse": float((err ** 2).mean() ** 0.5),
            "n": int(len(v)), "hfa_margin": hfa_margin}


def naive_margin_baseline(oos_df: pd.DataFrame) -> dict:
    """Predict margin = 0 for every game."""
    v = oos_df.dropna(subset=["home_score", "away_score"])
    actual = v["home_score"] - v["away_score"]
    return {"margin_mae": float(actual.abs().mean()),
            "margin_rmse": float((actual ** 2).mean() ** 0.5), "n": int(len(v))}


def frozen_prior_season_baseline(games_full: pd.DataFrame, elo_cfg: EloConfig,
                                  oos_seasons) -> dict:
    """'Ratings frozen at prior season's end': for each OOS season, freeze
    every team's rating at the value recorded for its FIRST game that
    season -- i.e. the post-carryover, pre-any-current-season-game rating,
    which run_elo already computed as elo_home/elo_away for that game --
    then predict every game that season with elo_expected_margin on those
    frozen numbers, never updating within the season. Teams with no rating
    yet default to elo_cfg.base, matching run_elo's own convention."""
    abs_err = 0.0; sq_err = 0.0; n = 0
    for season, sdf in games_full.groupby("season"):
        if season not in oos_seasons:
            continue
        sdf = sdf.sort_values("week")
        frozen: dict = {}
        for _, g in sdf.iterrows():
            h, a = g["home_team"], g["away_team"]
            if h not in frozen:
                frozen[h] = g["elo_home"]
            if a not in frozen:
                frozen[a] = g["elo_away"]
        for _, g in sdf.iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            h, a = g["home_team"], g["away_team"]
            fh = frozen.get(h, elo_cfg.base)
            fa = frozen.get(a, elo_cfg.base)
            pred = elo_expected_margin(fh, fa, elo_cfg)
            actual = g["home_score"] - g["away_score"]
            abs_err += abs(pred - actual)
            sq_err += (pred - actual) ** 2
            n += 1
    return {"margin_mae": abs_err / n, "margin_rmse": (sq_err / n) ** 0.5, "n": n}


def main() -> None:
    sched = pd.read_parquet("assets/cfb/schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched

    train = reg[reg["season"].isin(TRAIN_SEASONS)]
    oos_only = reg[reg["season"].isin(OOS_SEASONS)]
    full_span = reg[reg["season"].isin(TRAIN_SEASONS) | reg["season"].isin(OOS_SEASONS)]

    # CFB-appropriate ranges per task-4-brief.md: hfa_elo larger than NFL's
    # 55 (search ~55-110), carryover lower than NFL's 0.6 (search ~0.2-0.6)
    # for transfer-portal/roster-turnover effects, k around NFL's 16
    # (search ~12-28).
    grid = {
        "k": [12, 16, 20, 24, 28],
        "hfa_elo": [55, 70, 85, 100, 110],
        "carryover": [0.2, 0.3, 0.4, 0.5, 0.6],
        "w_sos": [0.0, 0.15, 0.3, 0.45],
        "srs_min_games": [3, 4, 6],
    }

    (best_elo, best_blend), results = _coordinate_search(train, grid)

    # Enforced-fair OOS verdict for the SoS blend itself: pure-Elo
    # counterfactual reuses best_elo's (k, hfa_elo, carryover) exactly,
    # only zeroing w_sos -- same causal-comparison logic as
    # backtest_nfl_elo.py. Evaluated on the CONTINUOUS 2015-2024 span so OOS
    # ratings carry real history in from the train seasons, restricting the
    # metric accumulation to the OOS seasons only.
    pure_elo = best_elo
    pure_blend = BlendConfig(w_sos=0.0, srs_min_games=best_blend.srs_min_games)

    blend_oos = run_backtest(full_span, best_elo, best_blend, eval_seasons=OOS_SEASONS)
    pure_oos = run_backtest(full_span, pure_elo, pure_blend, eval_seasons=OOS_SEASONS)
    blend_wins = blend_oos["margin_mae"] < pure_oos["margin_mae"]  # strict: a tie does not ship the blend

    if blend_wins:
        final_elo, final_blend, final_oos = best_elo, best_blend, blend_oos
    else:
        final_elo, final_blend, final_oos = pure_elo, pure_blend, pure_oos

    out = {"k": final_elo.k, "hfa_elo": final_elo.hfa_elo,
           "carryover": final_elo.carryover, "base": final_elo.base,
           "w_sos": final_blend.w_sos, "srs_min_games": final_blend.srs_min_games}
    pathlib.Path("assets/cfb/rating.json").write_text(json.dumps(out, indent=2) + "\n")

    # Baselines -- all evaluated on the same OOS games (2023-2024), using
    # the SAME continuous 2015-2024 elo trajectory (run under final_elo) so
    # the "prior season frozen rating" baseline reflects real history too.
    full_sorted = full_span.sort_values(["season", "week"]).reset_index(drop=True)
    games_full = run_elo(full_sorted, final_elo).games
    home_always = home_always_baseline(oos_only, final_elo.hfa_elo / 25.0)
    prior_season = frozen_prior_season_baseline(games_full, final_elo, OOS_SEASONS)
    naive_margin = naive_margin_baseline(oos_only)

    print("best blended (train-selected, margin_mae):", {
        "k": best_elo.k, "hfa_elo": best_elo.hfa_elo, "carryover": best_elo.carryover,
        "w_sos": best_blend.w_sos, "srs_min_games": best_blend.srs_min_games})
    print("pure-Elo counterfactual (SAME k/hfa/carryover, w_sos=0):", {
        "k": pure_elo.k, "hfa_elo": pure_elo.hfa_elo, "carryover": pure_elo.carryover})
    print("OOS (2023-2024) blended:", blend_oos)
    print("OOS (2023-2024) pure-Elo:", pure_oos)
    print("SoS blend beat pure Elo OOS on margin MAE (strict <)?", blend_wins)
    print("OOS baseline -- home_always:", home_always)
    print("OOS baseline -- prior_season_rating (frozen):", prior_season)
    print("OOS baseline -- naive_margin (predict 0):", naive_margin)
    print("written rating.json:", out, "| final OOS metrics:", final_oos)
    print("GATE -- tuned final beats naive_margin OOS (margin_mae)?",
          final_oos["margin_mae"] < naive_margin["margin_mae"])
    print("GATE -- tuned final beats prior_season_rating OOS (margin_mae)?",
          final_oos["margin_mae"] < prior_season["margin_mae"])
    print("GATE -- tuned final beats home_always OOS (margin_mae)?",
          final_oos["margin_mae"] < home_always["margin_mae"])


if __name__ == "__main__":
    main()
