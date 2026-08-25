"""Walk-forward backtest fitting sigma_margin/sigma_total + the two w(week)
market-shrinkage curves against nflverse's historical closing lines.

Wires together P1 (elo/srs/ratings) + Task 3 (points) + Task 4 (shrink) +
Task 5 (gameline) into a leak-free, per-game walk-forward: at each game, the
model margin comes from PRE-game Elo (and, once enough games are played,
season-to-date SRS via ratings.expected_margin's blend); the model total
comes from season-to-date opponent-adjusted points ratings. Both are then
shrunk toward the game's own closing market line via shrink.shrink before
being scored against the game's actual outcome.

NaN handling: nflverse's spread_line/total_line come back as NaN (a float)
when a market line is missing, but shrink() only treats Python `None` as
"no market line" -- passing NaN through would silently NaN-poison the
weighted blend `(1-w)*model + w*NaN == NaN`. per_game_predictions therefore
sanitizes NaN -> None at the market-dict boundary before calling
build_gameline, exactly once, right where the raw schedule value is read.
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


def _clean_market(value) -> float | None:
    """nflverse spread_line/total_line -> float, or None if missing/NaN.

    build_gameline/shrink() only recognize Python `None` as "no market
    line"; NaN would poison the (1-w)*model + w*market blend. This is the
    one place raw schedule values cross into gameline-land, so it is the
    one place that needs the NaN -> None conversion (Ruling 1).
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _raw_model_predictions(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                           blend_cfg: BlendConfig) -> list[dict]:
    """The expensive, leak-free walk-forward core, WITHOUT any shrink/sigma
    applied: one entry per scored game with the model's own pre-game
    margin/total (from pre-game Elo + season-to-date SRS + season-to-date
    opponent-adjusted points -- each computed from games already scored
    that season), the game's own closing market line, and the actuals.

    This is intentionally separated from shrink/sigma application
    (`_apply_gl` / `per_game_predictions`): model_margin/model_total depend
    only on (schedule_df, elo_cfg, blend_cfg), never on GameLineConfig, so a
    shrink/sigma search can call this ONCE per (schedule_df, elo_cfg,
    blend_cfg) and then cheaply re-score many GameLineConfig trials over the
    cached rows instead of re-running Elo+SRS+points for every trial (the
    per-game SRS+points recompute is the entire cost of this walk-forward).

    Appending a later game to schedule_df must never change an earlier
    game's entry in the returned list -- the leak-free invariant the
    accompanying test enforces (via per_game_predictions, which is a thin
    wrapper over this).
    """
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)                    # pre-game elo per game
    games = res.games
    out = []
    for season, sdf in games.groupby("season"):
        srs_hist = sdf.iloc[0:0]                   # empty frame, same cols
        pts_hist = sdf.iloc[0:0]
        counts, srs_cache, pts_cache, lg_cache = {}, {}, {}, 0.0
        for _, g in sdf.iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            h, a = g["home_team"], g["away_team"]
            gh, ga = counts.get(h, 0), counts.get(a, 0)
            model_margin = expected_margin(g["elo_home"], g["elo_away"],
                                           srs_cache.get(h), srs_cache.get(a),
                                           gh, ga, elo_cfg, blend_cfg)
            model_total = (expected_total(pts_cache, lg_cache, h, a)
                           if pts_cache else 2 * lg_cache) if lg_cache else 44.0
            out.append({"model_margin": model_margin, "model_total": model_total,
                        "spread_line": _clean_market(g.get("spread_line")),
                        "total_line": _clean_market(g.get("total_line")),
                        "week": int(g["week"]),
                        "actual_margin": float(g["home_score"] - g["away_score"]),
                        "actual_total": float(g["home_score"] + g["away_score"])})
            # after scoring, this game joins the history -> update caches
            counts[h] = gh + 1; counts[a] = ga + 1
            srs_hist = pd.concat([srs_hist, pd.DataFrame([g])], ignore_index=True)
            pts_hist = pd.concat([pts_hist, pd.DataFrame([g])], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
            pts_cache, lg_cache = compute_points_ratings(pts_hist, k_points=4.0)
    return out


def _apply_gl(raw: list[dict], gl_cfg: GameLineConfig) -> list[dict]:
    """Cheaply re-score cached raw walk-forward rows (see
    `_raw_model_predictions`) under a given GameLineConfig: apply
    market-shrinkage + wrap in the Normal dists via build_gameline. O(n),
    no Elo/SRS/points recomputation."""
    out = []
    for r in raw:
        market = {"spread_line": r["spread_line"], "total_line": r["total_line"]}
        row = build_gameline(r["model_margin"], r["model_total"], market, r["week"], gl_cfg)
        out.append({"pred_margin": row["pred_margin"], "pred_total": row["pred_total"],
                    "win_prob": row["home_win_prob"],
                    "actual_margin": r["actual_margin"], "actual_total": r["actual_total"]})
    return out


def per_game_predictions(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                         blend_cfg: BlendConfig, gl_cfg: GameLineConfig) -> list[dict]:
    """Leak-free walk-forward core: one entry per scored game with its
    pre-game pred_margin/pred_total/win_prob and the actuals. See
    `_raw_model_predictions` for the leak-free walk-forward details; this is
    a thin wrapper that also applies market-shrinkage via `gl_cfg`.
    """
    raw = _raw_model_predictions(schedule_df, elo_cfg, blend_cfg)
    return _apply_gl(raw, gl_cfg)


def run_backtest(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                 blend_cfg: BlendConfig, gl_cfg: GameLineConfig) -> dict:
    preds = per_game_predictions(schedule_df, elo_cfg, blend_cfg, gl_cfg)
    n = len(preds)
    if n == 0:
        return {"margin_mae": 0.0, "total_mae": 0.0, "brier": 0.0,
                "cover_acc": 0.0, "ou_acc": 0.0, "n": 0}
    mae_m = sum(abs(p["pred_margin"] - p["actual_margin"]) for p in preds) / n
    mae_t = sum(abs(p["pred_total"] - p["actual_total"]) for p in preds) / n
    brier = sum((p["win_prob"] - (1.0 if p["actual_margin"] > 0 else 0.0)) ** 2
                for p in preds) / n
    cover = sum(int((p["pred_margin"] > 0) == (p["actual_margin"] > 0)) for p in preds) / n
    # OU accuracy vs the model's own total is trivially 1 (it's tautological);
    # ou_acc instead measures the total's directional bias/calibration --
    # P(actual_total > pred_total). ~0.5 means the total is unbiased high/low;
    # a value far from 0.5 flags a systematic over/under-prediction.
    ou = sum(int(p["actual_total"] > p["pred_total"]) for p in preds) / n
    return {"margin_mae": mae_m, "total_mae": mae_t, "brier": brier,
            "cover_acc": cover, "ou_acc": ou, "n": n}


def tune_sigmas(train_preds: list[dict]) -> tuple[float, float]:
    """sigma_margin/sigma_total = RMSE of (pred - actual) on the TRAIN span
    (method-of-moments: the residual SD is the Normal's sigma)."""
    n = len(train_preds)
    if n == 0:
        return (13.2, 10.0)
    sq_m = sum((p["pred_margin"] - p["actual_margin"]) ** 2 for p in train_preds) / n
    sq_t = sum((p["pred_total"] - p["actual_total"]) ** 2 for p in train_preds) / n
    return (math.sqrt(sq_m), math.sqrt(sq_t))


def tune_shrink(train_df: pd.DataFrame, elo_cfg: EloConfig, blend_cfg: BlendConfig,
                base_gl_cfg: GameLineConfig, grid: dict) -> tuple[ShrinkParams, ShrinkParams]:
    """Coordinate search over ShrinkParams(start, floor, decay) independently
    for margin (minimize train margin MAE) and total (minimize train total
    MAE).

    Performance: model_margin/model_total (the Elo+SRS+points walk-forward)
    do not depend on the shrink curve at all -- only the final shrunk
    value does. So the walk-forward is run via `_raw_model_predictions`
    exactly ONCE for train_df, and every one of the (passes * axes *
    grid-values) trials below just re-applies `_apply_gl` (an O(n) shrink
    pass, no Elo/SRS/points recomputation) to those cached raw rows. This is
    what keeps the coordinate search's wall-clock close to the cost of a
    single walk-forward pass instead of one pass per trial.

    grid: {"start": [...], "floor": [...], "decay": [...]} shared by both
    curves' coordinate search.
    """
    raw = _raw_model_predictions(train_df, elo_cfg, blend_cfg)
    return _tune_shrink_from_raw(raw, base_gl_cfg, grid)


def _tune_shrink_from_raw(raw: list[dict], base_gl_cfg: GameLineConfig,
                          grid: dict) -> tuple[ShrinkParams, ShrinkParams]:
    """Coordinate-search body of tune_shrink, factored out so callers that
    already have cached raw walk-forward rows (e.g. main(), which also needs
    them for sigma-fitting) don't pay for a second Elo+SRS+points pass."""
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


def _mae_se(preds: list[dict], pred_key: str, actual_key: str) -> float:
    """Standard error of the mean of a per-game absolute-error series (the
    per-game |pred-actual| terms averaged into a MAE), used to frame whether
    a metric gap between two configs (e.g. blend vs market-only) is within
    noise rather than a real difference.
    """
    n = len(preds)
    if n < 2:
        return 0.0
    vals = [abs(p[pred_key] - p[actual_key]) for p in preds]
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)


def main() -> None:
    t0 = time.time()
    sched = pd.read_parquet("assets/nfl/schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    train = reg[reg["season"] <= 2019].copy()
    valid = reg[reg["season"] >= 2020].copy()

    rating_path = pathlib.Path("assets/nfl/rating.json")
    rating = json.loads(rating_path.read_text())
    elo_cfg = EloConfig(k=rating["k"], hfa_elo=rating["hfa_elo"],
                        carryover=rating["carryover"], base=rating.get("base", 1500.0))
    blend_cfg = BlendConfig(w_sos=rating["w_sos"], srs_min_games=rating["srs_min_games"])

    # Elo+SRS+points is the expensive part of this walk-forward and does not
    # depend on the shrink curve or sigmas at all -- run it ONCE per span
    # (train, valid) and cheaply re-apply many GameLineConfig trials to the
    # cached raw rows via _apply_gl (an O(n) shrink pass). This keeps a
    # coordinate search's wall-clock close to one walk-forward pass per
    # span, rather than one pass per (config) trial.
    t_walk0 = time.time()
    raw_train = _raw_model_predictions(train, elo_cfg, blend_cfg)
    raw_valid = _raw_model_predictions(valid, elo_cfg, blend_cfg)
    t_walk = time.time() - t_walk0

    grid = {"start": [0.5, 0.65, 0.75, 0.85, 0.95],
            "floor": [0.05, 0.15, 0.2, 0.3],
            "decay": [0.1, 0.2, 0.25, 0.35, 0.5]}
    base_gl_cfg = GameLineConfig()  # sigmas irrelevant to MAE-based shrink search
    t_search0 = time.time()
    w_margin, w_total = _tune_shrink_from_raw(raw_train, base_gl_cfg, grid)
    t_search = time.time() - t_search0

    fitted_gl_cfg = GameLineConfig(w_margin=w_margin, w_total=w_total)
    train_preds = _apply_gl(raw_train, fitted_gl_cfg)
    sigma_margin, sigma_total = tune_sigmas(train_preds)

    final_gl_cfg = GameLineConfig(sigma_margin=sigma_margin, sigma_total=sigma_total,
                                  w_margin=w_margin, w_total=w_total)
    model_only_cfg = GameLineConfig(sigma_margin=sigma_margin, sigma_total=sigma_total,
                                    w_margin=ShrinkParams(0.0, 0.0, 0.0),
                                    w_total=ShrinkParams(0.0, 0.0, 0.0))
    market_only_cfg = GameLineConfig(sigma_margin=sigma_margin, sigma_total=sigma_total,
                                     w_margin=ShrinkParams(1.0, 1.0, 0.0),
                                     w_total=ShrinkParams(1.0, 1.0, 0.0))

    valid_blend_preds = _apply_gl(raw_valid, final_gl_cfg)
    valid_model_preds = _apply_gl(raw_valid, model_only_cfg)
    valid_market_preds = _apply_gl(raw_valid, market_only_cfg)

    def _agg(preds):
        n = len(preds)
        mae_m = sum(abs(p["pred_margin"] - p["actual_margin"]) for p in preds) / n
        mae_t = sum(abs(p["pred_total"] - p["actual_total"]) for p in preds) / n
        brier = sum((p["win_prob"] - (1.0 if p["actual_margin"] > 0 else 0.0)) ** 2
                    for p in preds) / n
        cover = sum(int((p["pred_margin"] > 0) == (p["actual_margin"] > 0)) for p in preds) / n
        ou = sum(int(p["actual_total"] > p["pred_total"]) for p in preds) / n
        return {"margin_mae": mae_m, "total_mae": mae_t, "brier": brier,
                "cover_acc": cover, "ou_acc": ou, "n": n}

    blend_valid = _agg(valid_blend_preds)
    model_valid = _agg(valid_model_preds)
    market_valid = _agg(valid_market_preds)

    se_blend_margin = _mae_se(valid_blend_preds, "pred_margin", "actual_margin")
    se_market_margin = _mae_se(valid_market_preds, "pred_margin", "actual_margin")
    se_model_margin = _mae_se(valid_model_preds, "pred_margin", "actual_margin")

    out = {
        "sigma_margin": sigma_margin,
        "sigma_total": sigma_total,
        "offset": final_gl_cfg.offset,
        "total_max": final_gl_cfg.total_max,
        "w_margin": {"start": w_margin.start, "floor": w_margin.floor, "decay": w_margin.decay},
        "w_total": {"start": w_total.start, "floor": w_total.floor, "decay": w_total.decay},
    }
    pathlib.Path("assets/nfl/gameline.json").write_text(json.dumps(out, indent=2) + "\n")

    print("fitted w_margin:", out["w_margin"], "| fitted w_total:", out["w_total"])
    print("fitted sigma_margin:", sigma_margin, "sigma_total:", sigma_total)
    print("walk-forward (Elo+SRS+points, train+valid) wall-clock (s):", round(t_walk, 1))
    print("shrink coordinate-search wall-clock (s, cheap re-scoring of cached rows):",
          round(t_search, 1))
    print("VALIDATION blend:", blend_valid)
    print("VALIDATION model-only (w=0):", model_valid)
    print("VALIDATION market-only (w=1):", market_valid)
    print("margin MAE SE -- model-only: %.4f | blend: %.4f | market-only: %.4f"
          % (se_model_margin, se_blend_margin, se_market_margin))
    print("NOTE: the blend is expected to land between model-only and market-only;",
          "beating the closing line is NOT a pass condition -- season-long CLV is the real judge.")
    print("written gameline.json:", out)
    print("total wall-clock (s):", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
