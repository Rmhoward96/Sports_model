"""CFB game-line backtest -- fits sigma_margin/sigma_total + the two w(week)
market-shrinkage curves against CFBD's historical closing lines, then reports
the P2 SHIP-GATE: does the CFB model (alone, and blended with the market)
beat the CFB closing line out-of-sample?

Modeled closely on scripts/backtest_nfl_gameline.py, reusing the exact same
league-agnostic engine untouched (sportsmodel.nfl.elo/.srs/.ratings/.points/
.shrink/.gameline). Three deliberate adaptations for CFB:

1. Market lines are a SEPARATE asset (assets/cfb/lines.parquet), unlike NFL's
   nflverse schedules which already embed spread_line/total_line. Task 3
   left-joins schedules -> lines on (season, week, home_team, away_team) and
   maps market_spread/market_total (CFBD's naming) into the "spread_line"/
   "total_line" keys build_gameline expects, sanitizing NaN -> None at that
   boundary exactly as backtest_nfl_gameline's `_clean_market` does. CFBD
   only prices a subset of games (~75% of the FBS schedule -- G5-vs-FCS
   buy games etc. are frequently unpriced), so every metric that references
   the market (blend/market-only MAE, cover/O-U accuracy) is scored on the
   subset of games carrying a valid market number for THAT metric, while
   model-only MAE and the leak-free Elo/SRS/points walk-forward itself use
   every game (matching backtest_cfb_ratings.py's P1 gate).
2. A handful of CFBD rows have home_team == away_team (an aggregation
   artifact: two different unmatched/FCS opponents both collapse to the
   placeholder "FCS" team, or a name fails to resolve on both sides). These
   are dropped from BOTH schedules and lines before anything else runs --
   left in, a team would face itself in run_elo/compute_srs/
   compute_points_ratings, corrupting that team's rating update. Confirmed
   tiny (3 of 9062 schedule rows, 2 of 6825 line rows).
3. Like backtest_cfb_ratings.py, per-game SRS+points-ratings recompute
   (Gauss-Seidel solves over ~137 teams) is intractable at CFB's ~830
   games/season scale -- so `_raw_model_predictions` refreshes both caches
   ONCE PER WEEK (using only strictly-earlier weeks that season) instead of
   after every game, matching backtest_cfb_ratings.py's precedent. The walk-
   forward is run ONCE over the continuous 2015-2024 span (not separately
   per train/OOS span, unlike backtest_nfl_gameline.py) so that OOS (2023-
   2024) ratings carry real multi-season history in, per the brief's "a
   game's ratings use only prior seasons + season-to-date" -- exactly what
   backtest_cfb_ratings.py's `full_span`/`eval_seasons` pattern already
   established for CFB P1. Season 2025 is excluded entirely: lines.parquet
   has no 2025 rows, and it is the current/partial season in schedules.

Data: assets/cfb/schedules.parquet (2015-2025) x assets/cfb/lines.parquet
(2015-2024). TRAIN = 2015-2022 (shrink-curve + sigma fitting). OOS = 2023-
2024 (two complete, held-out seasons with full market coverage) -- the gate.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import pandas as pd

from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.srs import compute_srs
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.points import compute_points_ratings, expected_total
from sportsmodel.nfl.gameline import GameLineConfig, build_gameline
from sportsmodel.nfl.shrink import ShrinkParams

TRAIN_SEASONS = set(range(2015, 2023))   # 2015-2022 inclusive
OOS_SEASONS = {2023, 2024}
EXCLUDED_SEASONS = {2025}                # current/partial season, no market lines
CFB_OFFSET = 110                          # CFB margins run much wider than NFL's
CFB_TOTAL_MAX = 150                       # CFB totals run much higher than NFL's
_DEFAULT_TOTAL_SEED = 55.0                # ~schedule-wide mean/median total, used only
                                          # before any points history exists that season


def _clean_market(value) -> float | None:
    """CFBD market_spread/market_total -> float, or None if missing/NaN.

    Same NaN -> None sanitization as backtest_nfl_gameline._clean_market:
    shrink()/build_gameline only recognize Python `None` as "no market
    line"; NaN would poison the (1-w)*model + w*market blend.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def load_merged_schedule(schedules_path: str, lines_path: str) -> pd.DataFrame:
    """Left-join CFB schedules -> CFB lines on (season, week, home_team,
    away_team), dropping bad CFBD self-matches (home_team == away_team) from
    both sides first so they never corrupt Elo/SRS/points ratings."""
    sched = pd.read_parquet(schedules_path)
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    reg = reg[reg["home_team"] != reg["away_team"]].copy()

    lines = pd.read_parquet(lines_path)
    lines = lines[lines["home_team"] != lines["away_team"]].copy()
    lines = lines.drop_duplicates(subset=["season", "week", "home_team", "away_team"],
                                  keep="first")

    merged = reg.merge(lines, on=["season", "week", "home_team", "away_team"],
                       how="left", validate="one_to_one")
    return merged


def _raw_model_predictions(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                           blend_cfg: BlendConfig) -> list[dict]:
    """Leak-free walk-forward core, WITHOUT any shrink/sigma applied: one
    entry per scored game with the model's own pre-game margin/total (from
    pre-game Elo + season-to-date SRS + season-to-date opponent-adjusted
    points -- refreshed once per completed WEEK, not per game, for CFB-scale
    tractability -- see module docstring point 3), the game's own market
    line (already merged in as market_spread/market_total, NaN -> None),
    and the actuals.

    Mirrors backtest_nfl_gameline._raw_model_predictions's leak-free
    invariant: appending a later game to schedule_df must never change an
    earlier game's entry in the returned list. Intended to be called ONCE
    over the full continuous span; a shrink/sigma search re-scores the
    cached rows via `_apply_gl` instead of re-running this expensive pass.
    """
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)                    # pre-game elo per game, continuous across seasons
    games = res.games
    out = []
    for season, sdf in games.groupby("season"):
        sdf = sdf.sort_values("week")
        counts, srs_cache, pts_cache, lg_cache = {}, {}, {}, 0.0
        srs_hist = sdf.iloc[0:0]
        pts_hist = sdf.iloc[0:0]
        for week, wdf in sdf.groupby("week"):
            wdf = wdf.dropna(subset=["home_score", "away_score"])
            if wdf.empty:
                continue
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                gh, ga = counts.get(h, 0), counts.get(a, 0)
                model_margin = expected_margin(g["elo_home"], g["elo_away"],
                                               srs_cache.get(h), srs_cache.get(a),
                                               gh, ga, elo_cfg, blend_cfg)
                model_total = ((expected_total(pts_cache, lg_cache, h, a) if pts_cache
                               else 2 * lg_cache) if lg_cache else _DEFAULT_TOTAL_SEED)
                out.append({
                    "season": int(season), "week": int(week),
                    "home_team": h, "away_team": a,
                    "model_margin": model_margin, "model_total": model_total,
                    "market_spread": _clean_market(g.get("market_spread")),
                    "market_total": _clean_market(g.get("market_total")),
                    "actual_margin": float(g["home_score"] - g["away_score"]),
                    "actual_total": float(g["home_score"] + g["away_score"]),
                })
            # after grading the whole week, it joins the history -> refresh
            # counts + SRS + points ONCE for next week (season-to-date, never
            # informed by same-or-future-week results)
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                counts[h] = counts.get(h, 0) + 1
                counts[a] = counts.get(a, 0) + 1
            srs_hist = pd.concat([srs_hist, wdf], ignore_index=True)
            pts_hist = pd.concat([pts_hist, wdf], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
            pts_cache, lg_cache = compute_points_ratings(pts_hist, k_points=4.0)
    return out


def _apply_gl(raw: list[dict], gl_cfg: GameLineConfig) -> list[dict]:
    """Cheaply re-score cached raw walk-forward rows: apply market-shrinkage
    + wrap in the Normal dists via build_gameline. O(n), no Elo/SRS/points
    recomputation. Passes market_spread/market_total through unshrunk so
    downstream metrics (cover/O-U accuracy) can compare against them."""
    out = []
    for r in raw:
        market = {"spread_line": r["market_spread"], "total_line": r["market_total"]}
        row = build_gameline(r["model_margin"], r["model_total"], market, r["week"], gl_cfg)
        out.append({"season": r["season"], "week": r["week"],
                    "pred_margin": row["pred_margin"], "pred_total": row["pred_total"],
                    "win_prob": row["home_win_prob"],
                    "market_spread": r["market_spread"], "market_total": r["market_total"],
                    "actual_margin": r["actual_margin"], "actual_total": r["actual_total"]})
    return out


def tune_sigmas(train_preds: list[dict]) -> tuple[float, float]:
    """sigma_margin/sigma_total = RMSE of (pred - actual) on the TRAIN span
    (method-of-moments: the residual SD is the Normal's sigma)."""
    n = len(train_preds)
    if n == 0:
        return (13.2, 10.0)
    sq_m = sum((p["pred_margin"] - p["actual_margin"]) ** 2 for p in train_preds) / n
    sq_t = sum((p["pred_total"] - p["actual_total"]) ** 2 for p in train_preds) / n
    return (math.sqrt(sq_m), math.sqrt(sq_t))


def _tune_shrink_from_raw(raw: list[dict], base_gl_cfg: GameLineConfig,
                          grid: dict) -> tuple[ShrinkParams, ShrinkParams]:
    """Coordinate search over ShrinkParams(start, floor, decay), independently
    for margin and total (minimize train MAE against actuals). Identical
    method to backtest_nfl_gameline._tune_shrink_from_raw."""
    order = ["start", "floor", "decay"]

    def _mae(preds: list[dict], key: str) -> float:
        n = len(preds)
        if n == 0:
            return 0.0
        actual_key = "actual_margin" if key == "pred_margin" else "actual_total"
        return sum(abs(p[key] - p[actual_key]) for p in preds) / n

    def _search(which: str) -> ShrinkParams:
        pred_key = "pred_margin" if which == "margin" else "pred_total"
        current = {p: grid[p][len(grid[p]) // 2] for p in order}
        cache: dict = {}

        def _eval(params: dict) -> float:
            key = tuple(params[p] for p in order)
            if key in cache:
                return cache[key]
            sp = ShrinkParams(start=params["start"], floor=params["floor"], decay=params["decay"])
            if which == "margin":
                gl = GameLineConfig(sigma_margin=base_gl_cfg.sigma_margin,
                                    sigma_total=base_gl_cfg.sigma_total,
                                    offset=base_gl_cfg.offset, total_max=base_gl_cfg.total_max,
                                    w_margin=sp, w_total=base_gl_cfg.w_total)
            else:
                gl = GameLineConfig(sigma_margin=base_gl_cfg.sigma_margin,
                                    sigma_total=base_gl_cfg.sigma_total,
                                    offset=base_gl_cfg.offset, total_max=base_gl_cfg.total_max,
                                    w_margin=base_gl_cfg.w_margin, w_total=sp)
            m = _mae(_apply_gl(raw, gl), pred_key)
            cache[key] = m
            return m

        n_passes = 3
        for _ in range(n_passes):
            improved = False
            for p in order:
                best_val = current[p]
                best_metric = _eval(current)
                for v in grid[p]:
                    trial = dict(current); trial[p] = v
                    m = _eval(trial)
                    if m < best_metric:
                        best_metric = m
                        best_val = v
                if best_val != current[p]:
                    improved = True
                current[p] = best_val
            if not improved:
                break
        return ShrinkParams(start=current["start"], floor=current["floor"], decay=current["decay"])

    best_margin = _search("margin")
    best_total = _search("total")
    return best_margin, best_total


def _mae_se(preds: list[dict], pred_key: str, actual_key: str, valid_key: str) -> float:
    """Standard error of the mean |pred-actual| over games with a valid
    market value for `valid_key` (frames whether an MAE gap between two
    configs is within noise)."""
    vals = [abs(p[pred_key] - p[actual_key]) for p in preds if p[valid_key] is not None]
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)


def _mae(preds: list[dict], pred_key: str, actual_key: str, valid_key: str) -> tuple[float, int]:
    """MAE restricted to games with a valid (non-NaN) market value for
    `valid_key`, so model-only/blend/market-only are compared on the exact
    same game set for margin, and likewise (separately) for total."""
    valid = [p for p in preds if p[valid_key] is not None]
    n = len(valid)
    if n == 0:
        return 0.0, 0
    return sum(abs(p[pred_key] - p[actual_key]) for p in valid) / n, n


def _ats_accuracy(preds: list[dict], pred_key: str, actual_key: str,
                  market_key: str) -> tuple[float, int]:
    """'Beats the closing line' accuracy: fraction of decided games (actual
    != market, i.e. not a push) where the model's side of the market number
    matches the actual side -- sign(pred - market) == sign(actual - market).
    Used for both spread cover accuracy (margin) and over/under accuracy
    (total). Games where the model lands exactly on the market number (no
    pick) are excluded from both numerator and denominator."""
    correct = 0
    n = 0
    for p in preds:
        m = p[market_key]
        if m is None or p[actual_key] == m:      # no market line, or a push
            continue
        model_side = p[pred_key] - m
        if model_side == 0:                       # model has no pick
            continue
        actual_side = p[actual_key] - m
        n += 1
        correct += int((model_side > 0) == (actual_side > 0))
    return (correct / n if n else 0.0), n


def main() -> None:
    t0 = time.time()
    merged = load_merged_schedule("assets/cfb/schedules.parquet", "assets/cfb/lines.parquet")
    full_span = merged[merged["season"].isin(TRAIN_SEASONS) | merged["season"].isin(OOS_SEASONS)].copy()

    rating_path = pathlib.Path("assets/cfb/rating.json")
    rating = json.loads(rating_path.read_text())
    elo_cfg = EloConfig(k=rating["k"], hfa_elo=rating["hfa_elo"],
                        carryover=rating["carryover"], base=rating.get("base", 1500.0))
    blend_cfg = BlendConfig(w_sos=rating["w_sos"], srs_min_games=rating["srs_min_games"])

    # ONE continuous walk-forward pass over 2015-2024 so OOS (2023-2024)
    # ratings carry real multi-season history in (see module docstring
    # point 3); then partition the cached raw rows by season for
    # train-fitting vs OOS-gate scoring. Cheap re-scoring via _apply_gl
    # means the shrink coordinate search never re-runs this expensive pass.
    t_walk0 = time.time()
    raw_full = _raw_model_predictions(full_span, elo_cfg, blend_cfg)
    t_walk = time.time() - t_walk0
    raw_train = [r for r in raw_full if r["season"] in TRAIN_SEASONS]
    raw_valid = [r for r in raw_full if r["season"] in OOS_SEASONS]

    grid = {"start": [0.5, 0.65, 0.75, 0.85, 0.95],
            "floor": [0.05, 0.15, 0.2, 0.3],
            "decay": [0.1, 0.2, 0.25, 0.35, 0.5]}
    base_gl_cfg = GameLineConfig(offset=CFB_OFFSET, total_max=CFB_TOTAL_MAX)
    t_search0 = time.time()
    w_margin, w_total = _tune_shrink_from_raw(raw_train, base_gl_cfg, grid)
    t_search = time.time() - t_search0

    fitted_gl_cfg = GameLineConfig(offset=CFB_OFFSET, total_max=CFB_TOTAL_MAX,
                                   w_margin=w_margin, w_total=w_total)
    train_preds = _apply_gl(raw_train, fitted_gl_cfg)
    sigma_margin, sigma_total = tune_sigmas(train_preds)

    final_gl_cfg = GameLineConfig(sigma_margin=sigma_margin, sigma_total=sigma_total,
                                  offset=CFB_OFFSET, total_max=CFB_TOTAL_MAX,
                                  w_margin=w_margin, w_total=w_total)
    model_only_cfg = GameLineConfig(sigma_margin=sigma_margin, sigma_total=sigma_total,
                                    offset=CFB_OFFSET, total_max=CFB_TOTAL_MAX,
                                    w_margin=ShrinkParams(0.0, 0.0, 0.0),
                                    w_total=ShrinkParams(0.0, 0.0, 0.0))
    market_only_cfg = GameLineConfig(sigma_margin=sigma_margin, sigma_total=sigma_total,
                                     offset=CFB_OFFSET, total_max=CFB_TOTAL_MAX,
                                     w_margin=ShrinkParams(1.0, 1.0, 0.0),
                                     w_total=ShrinkParams(1.0, 1.0, 0.0))

    valid_blend_preds = _apply_gl(raw_valid, final_gl_cfg)
    valid_model_preds = _apply_gl(raw_valid, model_only_cfg)
    valid_market_preds = _apply_gl(raw_valid, market_only_cfg)

    def _agg(preds):
        mae_m, n_m = _mae(preds, "pred_margin", "actual_margin", "market_spread")
        mae_t, n_t = _mae(preds, "pred_total", "actual_total", "market_total")
        return {"margin_mae": mae_m, "n_margin": n_m, "total_mae": mae_t, "n_total": n_t}

    blend_valid = _agg(valid_blend_preds)
    model_valid = _agg(valid_model_preds)
    market_valid = _agg(valid_market_preds)

    se_model_margin = _mae_se(valid_model_preds, "pred_margin", "actual_margin", "market_spread")
    se_blend_margin = _mae_se(valid_blend_preds, "pred_margin", "actual_margin", "market_spread")
    se_market_margin = _mae_se(valid_market_preds, "pred_margin", "actual_margin", "market_spread")
    se_model_total = _mae_se(valid_model_preds, "pred_total", "actual_total", "market_total")
    se_blend_total = _mae_se(valid_blend_preds, "pred_total", "actual_total", "market_total")
    se_market_total = _mae_se(valid_market_preds, "pred_total", "actual_total", "market_total")

    # The whole CFB P2 thesis: does the model's own pick (and the blend's
    # pick) beat the closing line more often than the ~52.4% breakeven,
    # on decided (non-push) games only.
    model_cover, n_cover_model = _ats_accuracy(valid_model_preds, "pred_margin", "actual_margin", "market_spread")
    blend_cover, n_cover_blend = _ats_accuracy(valid_blend_preds, "pred_margin", "actual_margin", "market_spread")
    model_ou, n_ou_model = _ats_accuracy(valid_model_preds, "pred_total", "actual_total", "market_total")
    blend_ou, n_ou_blend = _ats_accuracy(valid_blend_preds, "pred_total", "actual_total", "market_total")

    out = {
        "sigma_margin": sigma_margin,
        "sigma_total": sigma_total,
        "offset": final_gl_cfg.offset,
        "total_max": final_gl_cfg.total_max,
        "w_margin": {"start": w_margin.start, "floor": w_margin.floor, "decay": w_margin.decay},
        "w_total": {"start": w_total.start, "floor": w_total.floor, "decay": w_total.decay},
    }
    pathlib.Path("assets/cfb/gameline.json").write_text(json.dumps(out, indent=2) + "\n")

    print("fitted w_margin:", out["w_margin"], "| fitted w_total:", out["w_total"])
    print("fitted sigma_margin:", round(sigma_margin, 3), "sigma_total:", round(sigma_total, 3))
    print("walk-forward (Elo+SRS+points, full 2015-2024 span) wall-clock (s):", round(t_walk, 1))
    print("shrink coordinate-search wall-clock (s, cheap re-scoring of cached rows):",
          round(t_search, 1))
    print()
    print("=== OOS (2023-2024) MARGIN MAE (games with a valid market_spread) ===")
    print("  model-only:  %.3f  (n=%d, SE=%.4f)" % (model_valid["margin_mae"], model_valid["n_margin"], se_model_margin))
    print("  blend:       %.3f  (n=%d, SE=%.4f)" % (blend_valid["margin_mae"], blend_valid["n_margin"], se_blend_margin))
    print("  market-only: %.3f  (n=%d, SE=%.4f)" % (market_valid["margin_mae"], market_valid["n_margin"], se_market_margin))
    print("=== OOS (2023-2024) TOTAL MAE (games with a valid market_total) ===")
    print("  model-only:  %.3f  (n=%d, SE=%.4f)" % (model_valid["total_mae"], model_valid["n_total"], se_model_total))
    print("  blend:       %.3f  (n=%d, SE=%.4f)" % (blend_valid["total_mae"], blend_valid["n_total"], se_blend_total))
    print("  market-only: %.3f  (n=%d, SE=%.4f)" % (market_valid["total_mae"], market_valid["n_total"], se_market_total))
    print()
    print("=== GATE: beats-the-closing-line accuracy (decided games only, vs 52.4% breakeven) ===")
    print("  spread cover acc -- model-only: %.4f (n=%d) | blend: %.4f (n=%d)"
          % (model_cover, n_cover_model, blend_cover, n_cover_blend))
    print("  over/under acc   -- model-only: %.4f (n=%d) | blend: %.4f (n=%d)"
          % (model_ou, n_ou_model, blend_ou, n_ou_blend))
    print()
    print("NOTE: not declaring pass/fail here -- these numbers are the P2 ship-gate for the",
          "ledger; season-long CLV is the ultimate judge.")
    print("written gameline.json:", out)
    print("total wall-clock (s):", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
