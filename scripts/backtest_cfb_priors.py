"""CFB forward-looking priors backtest -- fits PriorWeights + DecayConfig
against assets/cfb/priors.parquet, then reports the accuracy/edge/ablation
tables for the SDD ledger.

Modeled closely on scripts/backtest_cfb_ratings.py (P1 gate) and
scripts/backtest_cfb_gameline.py (P2 gate), reusing the same league-agnostic
walk-forward engine (sportsmodel.nfl.elo/.srs/.ratings/.points) untouched,
plus the pure Task 3/4 prior-blend functions in sportsmodel.cfb.priors
(PriorWeights, DecayConfig, zscore, load_weights, preseason_rating,
season_features_z, prior_weight, blend_rating).

=== SCOPE NOTE (read before running) ===
This script needs assets/cfb/priors.parquet, which is produced by
scripts/build_cfb_priors.py against the live CFBD API (a GitHub Actions
secret, wired up by Task 7). That parquet does not exist in every
environment (e.g. it was absent when this script was authored). `main()`
fails fast with a clear message telling the operator to build it first --
see the `if not priors_path.exists()` check below. Everything else in this
file is written to be correct and ready to run the moment that asset exists;
none of the numbers it *would* print are fabricated anywhere in this repo.

=== DESIGN DECISIONS (documented, since parts of the brief are genuinely
open-ended and this script cannot be executed yet to empirically validate
them) ===

1. **What "in-season rating" the prior blends with.** `priors.preseason_rating`
   produces R_pre "on the model's Elo scale" (its docstring: identity default
   is `1500 + sp_rating`, i.e. it starts at the same base as `EloConfig.base`).
   The natural in-season counterpart on that same scale is each team's
   pre-game Elo rating (`elo_home`/`elo_away` from `run_elo`) -- NOT the SRS
   or points-ratings caches, which live on different scales (margin-of-victory
   SRS, points-per-game). So this script's walk-forward blends R_pre with
   pre-game Elo via `blend_rating(r_pre, elo_pregame, games_played, decay_cfg)`,
   then feeds the BLENDED value into `expected_margin` in place of the raw
   Elo rating, leaving the SRS/points components of `expected_margin`/
   `expected_total` completely untouched. `games_played` is the team's game
   count so far *that season* (0 in week 1), matching `DecayConfig`'s own
   docstring ("At 0 games: weight = 1.0").

2. **Total scoring is not touched by the prior.** priors.parquet carries a
   single overall SP+ rating with no offense/defense split and no points-
   scale signal at all, so there is no principled way to shift
   `expected_total` from it. "prior-seeded" total predictions are therefore
   identical to "current" total predictions in the accuracy table; the table
   labels this explicitly rather than silently duplicating a number that
   looks fitted but isn't.

3. **Market lines are benchmark-only.** `lines.parquet` is loaded ONLY to
   build the edge table (bucket by |model_margin - closing|, ATS win% +
   CLV-proxy per bucket) after weights are already fixed. It is never read
   by `preseason_rating`, `season_features_z`, or the fitting/ablation loop --
   R_pre and the blend are fit purely against ACTUAL MARGINS, never against
   the market, per the brief's constraint.

4. **"CLV" without an opening line.** True closing-line value (the market's
   move from bet-time to close) needs an EARLIER line snapshot; this repo
   only stores CFBD's closing lines (see backtest_cfb_gameline.py's module
   docstring). Lacking that, `edge_table` reports a CLV-PROXY: the signed
   REALIZED margin by which the actual game result beat or missed the
   closing number, from the model's own picked-side perspective
   (`actual_margin - closing_cover_threshold`, sign-flipped for an away
   pick). Positive means the side the model picked actually covered (and by
   how much); negative means it missed. This is distinct from `gap`, which
   is the model's PRE-GAME distance from the market and carries no outcome
   information at all -- `clv_proxy` is the only one of the two that reflects
   whether the model's edge actually paid off. Explicitly labeled "clv_proxy"
   in the printed table -- it is NOT a claim about actual line movement.

5. **Fitting objective + holdout.** `fit_prior_weights` runs a scipy-free
   coordinate search (mirrors `_coordinate_search` in
   backtest_cfb_ratings.py / `_tune_shrink_from_raw` in
   backtest_cfb_gameline.py: one parameter at a time over a documented grid,
   a few passes, select by TRAIN metric) minimizing Weeks 1-5 margin MAE.
   Seasons are split by PARITY (even season -> TRAIN, odd season -> HOLDOUT)
   -- "hold out alternating seasons" per the brief -- rather than a
   contiguous block, so both splits see a mix of eras/rule changes. Elo/SRS/
   points state is still walked forward across the FULL continuous span (all
   seasons in priors.parquet, regardless of split) so a HOLDOUT season's
   in-season ratings carry real multi-season history in, exactly like
   backtest_cfb_ratings.py's `eval_seasons` pattern -- only the *loss
   accumulation* (Weeks 1-5 MAE) is restricted per split.

6. **Two output files, not one.** `sportsmodel.cfb.priors.load_weights(path)` is
   implemented (Task 3, frozen) as `PriorWeights(**json.loads(path.read_text()))`
   -- it passes the ENTIRE parsed JSON object straight into the `PriorWeights`
   constructor, so ANY key beyond `PriorWeights`'s own 8 fields (sp_scale,
   sp_offset, w_portal, w_coach, w_qb, w_starters, w_sos_prior, w_sos_shift)
   raises `TypeError`, whether that extra key is `half_life_games` at the top
   level or nested under a "decay" sub-object (nesting doesn't help --
   `load_weights` never unwraps a sub-key, it always feeds the whole blob to
   `PriorWeights(**...)`). So `DecayConfig`'s `half_life_games`/`prior_floor`
   physically cannot live in the same flat file `load_weights` reads without
   breaking every future caller of `load_weights` (Task 6's producer
   included). This script therefore writes TWO files:
     - `assets/cfb/priors_weights.json` -- exactly the 8 `PriorWeights`
       fields, nothing else, loadable by `load_weights` unchanged.
     - `assets/cfb/priors_decay.json` -- `{"half_life_games": ...,
       "prior_floor": ...}`, a sibling file for `DecayConfig`, loaded via
       `sportsmodel.cfb.priors.load_decay_config` -- same missing-file-safe
       pattern as `load_weights` (falls back to a documented default rather
       than raising). It lives in `sportsmodel.cfb.priors` (next to
       `load_weights`), not in this script, so Task 6's producer -- which
       needs `DecayConfig` at runtime and isn't in the `scripts/` package --
       can import and call it the same way.

7. **Ablation = leave-one-out from the final fit, not a full nested refit.**
   For each of the six adjustment factors (portal, coach, qb, starters,
   sos_prior, sos_shift), the ablation zeroes just that one field of the
   ALREADY-FITTED weight vector and re-evaluates HOLDOUT Weeks 1-5 margin
   MAE (cheap: O(n), no re-search). A factor is KEPT in the written
   priors_weights.json only if the full fit beats its zeroed-out
   counterpart on HOLDOUT MAE (i.e. it demonstrably helps generalize);
   otherwise it is written as 0.0 and the printout says so. This is a
   standard, cheap leave-one-out ablation -- a full nested coordinate search
   per factor would be more thorough but is not required by the brief's
   "report the accuracy/edge delta each contributes" and would multiply the
   already-expensive walk-forward by 6.
"""
from __future__ import annotations

import pathlib
import sys
import time

import pandas as pd

from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.srs import compute_srs
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.points import compute_points_ratings, expected_total
from sportsmodel.cfb.priors import (
    DECAY_PATH as DECAY_OUT_PATH,
    DEFAULT_HALF_LIFE_GAMES as _DEFAULT_HALF_LIFE_GAMES,
    DecayConfig,
    PriorWeights,
    blend_rating,
    load_decay_config,
    preseason_rating,
    season_features_z,
)

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets" / "cfb"
SCHEDULES_PATH = ASSETS / "schedules.parquet"
LINES_PATH = ASSETS / "lines.parquet"
PRIORS_PATH = ASSETS / "priors.parquet"
RATING_PATH = ASSETS / "rating.json"
WEIGHTS_OUT_PATH = ASSETS / "priors_weights.json"

EARLY_WEEKS = {1, 2, 3, 4, 5}  # "Weeks 1-5" per the brief -- where a
                               # forward-looking prior should matter most,
                               # before in-season signal accumulates
ADJUSTMENT_FACTORS = ("w_portal", "w_coach", "w_qb", "w_starters", "w_sos_prior", "w_sos_shift")
EDGE_BUCKETS = [(0.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, float("inf"))]


# --------------------------------------------------------------------------
# Pure metric helpers (unit-tested in tests/cfb/test_backtest_cfb_priors.py)
# --------------------------------------------------------------------------

def ats_result(model_margin: float, closing_home_spread: float, actual_margin: float) -> str:
    """Grade the model's spread pick against the closing line.

    `closing_home_spread` uses standard sportsbook convention: negative means
    the home team is favored (e.g. -3.0 = home favored by 3). The home team
    "covers" if `actual_margin > -closing_home_spread`. The model's pick is
    inferred the same way from `model_margin` vs that same cover threshold:
    home if `model_margin` is greater than the threshold, away otherwise.

    Returns "push" if the game itself pushed (`actual_margin` lands exactly
    on the cover threshold) -- a push is a push regardless of which side
    anyone liked. If the model's own margin lands exactly on the threshold
    (no lean either way, i.e. it agrees with the market to the point), there
    is no pick to grade, so this is also scored "push" (excluded from win%
    denominators by `bucket_winrate`).
    """
    threshold = -closing_home_spread
    if actual_margin == threshold:
        return "push"
    if model_margin == threshold:
        return "push"
    home_covers = actual_margin > threshold
    model_favors_home = model_margin > threshold
    return "win" if home_covers == model_favors_home else "loss"


def bucket_winrate(rows: list[dict], min_gap: float) -> float:
    """ATS win rate over rows whose disagreement `gap` (|model - closing|)
    is at least `min_gap`. Pushes are excluded from both the numerator and
    the denominator (standard ATS win% convention -- a push is a no-decision,
    not a loss). Returns 0.0 if no row qualifies (avoids a ZeroDivisionError
    on an empty/thin bucket rather than raising)."""
    qualifying = [r for r in rows if r["gap"] >= min_gap and r["ats"] != "push"]
    if not qualifying:
        return 0.0
    wins = sum(1 for r in qualifying if r["ats"] == "win")
    return wins / len(qualifying)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_priors_by_season(path) -> dict[int, list[dict]]:
    """assets/cfb/priors.parquet -> {season: [row_dict, ...]}."""
    df = pd.read_parquet(path)
    return {int(season): sdf.to_dict("records") for season, sdf in df.groupby("season")}


def priors_z_by_season(priors_by_season: dict[int, list[dict]]) -> dict[int, dict]:
    """{season: {team_espn_id: {feature: z}}} via the shared Task 3 z-scorer."""
    return {season: season_features_z(rows) for season, rows in priors_by_season.items()}


def _clean_market(value) -> float | None:
    """CFBD market_spread -> float, or None if missing/NaN (same convention
    as backtest_cfb_gameline._clean_market)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def load_merged_schedule(schedules_path=SCHEDULES_PATH, lines_path=LINES_PATH) -> pd.DataFrame:
    """Left-join CFB schedules -> CFB lines, dropping CFBD self-match rows
    from both sides first (identical to backtest_cfb_gameline.load_merged_schedule)."""
    sched = pd.read_parquet(schedules_path)
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    reg = reg[reg["home_team"] != reg["away_team"]].copy()

    lines = pd.read_parquet(lines_path)
    lines = lines[lines["home_team"] != lines["away_team"]].copy()
    lines = lines.drop_duplicates(subset=["season", "week", "home_team", "away_team"], keep="first")

    merged = reg.merge(lines, on=["season", "week", "home_team", "away_team"],
                        how="left", validate="one_to_one")
    return merged


# --------------------------------------------------------------------------
# Leak-free walk-forward: cache the raw per-game engine state ONCE (elo/srs/
# points + games-played counts), independent of PriorWeights/DecayConfig, so
# the coordinate search / ablation only need a cheap re-score pass (mirrors
# backtest_cfb_gameline.py's `_raw_model_predictions` + `_apply_gl` split).
# --------------------------------------------------------------------------

def raw_walk_forward(schedule_df: pd.DataFrame, elo_cfg: EloConfig) -> list[dict]:
    """One entry per scored game: pre-game elo/SRS/points state (all
    strictly leak-free -- refreshed once per completed week, matching
    backtest_cfb_ratings.py/backtest_cfb_gameline.py's precedent for CFB-scale
    tractability), each team's games-played-so-far-this-season count (for the
    decay weight), the market closing spread, and actuals. Weights-independent:
    callers combine this with a given PriorWeights/DecayConfig via
    `predict_margin_with_prior` without re-running Elo/SRS/points."""
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)
    games = res.games
    out = []
    for season, sdf in games.groupby("season"):
        sdf = sdf.sort_values("week")
        counts, srs_cache, pts_cache, lg_cache = {}, {}, {}, 0.0
        pts_hist = sdf.iloc[0:0]
        for week, wdf in sdf.groupby("week"):
            wdf = wdf.dropna(subset=["home_score", "away_score"])
            if wdf.empty:
                continue
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                gh, ga = counts.get(h, 0), counts.get(a, 0)
                model_total = ((expected_total(pts_cache, lg_cache, h, a) if pts_cache
                               else 2 * lg_cache) if lg_cache else 55.0)
                out.append({
                    "season": int(season), "week": int(week),
                    "home_team": h, "away_team": a,
                    "elo_home": g["elo_home"], "elo_away": g["elo_away"],
                    "srs_home": srs_cache.get(h), "srs_away": srs_cache.get(a),
                    "games_home": gh, "games_away": ga,
                    "model_total": model_total,
                    "market_spread": _clean_market(g.get("market_spread")),
                    "actual_margin": float(g["home_score"] - g["away_score"]),
                    "actual_total": float(g["home_score"] + g["away_score"]),
                })
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                counts[h] = counts.get(h, 0) + 1
                counts[a] = counts.get(a, 0) + 1
            pts_hist = pd.concat([pts_hist, wdf], ignore_index=True)
            pts_cache, lg_cache = compute_points_ratings(pts_hist, k_points=4.0)
            # SRS refreshed once per week from the same accumulated history
            srs_cache = compute_srs(pts_hist)
    return out


def _team_r_pre(team_id: str, season: int, priors_by_season: dict, z_by_season: dict,
                 weights: PriorWeights) -> float | None:
    """R_pre for one team-season, or None if the team/season isn't in
    priors.parquet (e.g. the 'FCS' pseudo-team, or a season outside the
    priors asset's coverage) -- callers fall back to pure in-season Elo."""
    rows = priors_by_season.get(season)
    if rows is None:
        return None
    z = z_by_season[season].get(team_id)
    row = next((r for r in rows if r["team_espn_id"] == team_id), None)
    if row is None or z is None:
        return None
    return preseason_rating(row, z, weights)


def current_model_margin(raw_row: dict, elo_cfg: EloConfig, blend_cfg: BlendConfig) -> float:
    """"Current model" (no prior blend): pure pre-game Elo + season-to-date
    SRS, exactly like backtest_cfb_ratings.py/backtest_cfb_gameline.py."""
    return expected_margin(raw_row["elo_home"], raw_row["elo_away"],
                            raw_row["srs_home"], raw_row["srs_away"],
                            raw_row["games_home"], raw_row["games_away"],
                            elo_cfg, blend_cfg)


def prior_seeded_margin(raw_row: dict, priors_by_season: dict, z_by_season: dict,
                         weights: PriorWeights, decay_cfg: DecayConfig,
                         elo_cfg: EloConfig, blend_cfg: BlendConfig) -> float:
    """"Prior-seeded" model: blend each team's pre-game Elo with its R_pre
    (decayed by games played this season) per design decision #1 above, then
    feed the blended ratings into the same expected_margin used everywhere
    else. Teams absent from priors.parquet (see `_team_r_pre`) fall back to
    their raw pre-game Elo unblended (weight-of-prior effectively 0)."""
    season = raw_row["season"]
    h, a = raw_row["home_team"], raw_row["away_team"]
    r_pre_h = _team_r_pre(h, season, priors_by_season, z_by_season, weights)
    r_pre_a = _team_r_pre(a, season, priors_by_season, z_by_season, weights)
    elo_h = raw_row["elo_home"]
    elo_a = raw_row["elo_away"]
    if r_pre_h is not None:
        elo_h = blend_rating(r_pre_h, elo_h, raw_row["games_home"], decay_cfg)
    if r_pre_a is not None:
        elo_a = blend_rating(r_pre_a, elo_a, raw_row["games_away"], decay_cfg)
    return expected_margin(elo_h, elo_a, raw_row["srs_home"], raw_row["srs_away"],
                            raw_row["games_home"], raw_row["games_away"], elo_cfg, blend_cfg)


# --------------------------------------------------------------------------
# Fitting: coordinate search over PriorWeights + DecayConfig.half_life_games,
# minimizing Weeks 1-5 margin MAE on the TRAIN (even-season) split.
# --------------------------------------------------------------------------

def _early_mae(raw_rows: list[dict], priors_by_season: dict, z_by_season: dict,
               weights: PriorWeights, decay_cfg: DecayConfig,
               elo_cfg: EloConfig, blend_cfg: BlendConfig, seasons: set[int]) -> float:
    """Mean |prior-seeded margin - actual margin| over EARLY_WEEKS games in
    `seasons` only. Returns 0.0 if no qualifying game (keeps the search from
    crashing on a tiny/empty split rather than raising)."""
    errs = [
        abs(prior_seeded_margin(r, priors_by_season, z_by_season, weights, decay_cfg,
                                 elo_cfg, blend_cfg) - r["actual_margin"])
        for r in raw_rows if r["season"] in seasons and r["week"] in EARLY_WEEKS
    ]
    return sum(errs) / len(errs) if errs else 0.0


# Coordinate-search grids. sp_offset is fixed at PriorWeights' default
# (1500.0, the same base EloConfig.base uses) -- it is a structural anchor
# for "what elo value = 0 SP+ maps to", not a free fit parameter, matching
# how CFB_OFFSET/CFB_TOTAL_MAX in backtest_cfb_gameline.py are fixed
# constants rather than searched. sp_scale gets a modest range around its
# identity default of 1.0 (SP+ points assumed roughly Elo-point-scale, per
# Task 3's identity-default design) rather than the wide 0-2 an unconstrained
# search might try, to keep the "reduces to the SP+-only identity model"
# character. Adjustment weights (w_*) are symmetric around 0 -- z-scored
# features, so a weight of e.g. 50 means "one std dev of this feature is
# worth 50 Elo points", comparable in spirit to CFB's own hfa_elo (55-110)
# and w_sos-driven margin shifts from backtest_cfb_ratings.py's fitted grid.
_GRID = {
    "sp_scale": [0.8, 0.9, 1.0, 1.1, 1.2],
    "w_portal": [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0],
    "w_coach": [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0],
    "w_qb": [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0],
    "w_starters": [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0],
    "w_sos_prior": [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0],
    "w_sos_shift": [-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0],
    "half_life_games": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
}
_ORDER = ["sp_scale", "w_portal", "w_coach", "w_qb", "w_starters",
          "w_sos_prior", "w_sos_shift", "half_life_games"]


def fit_prior_weights(raw_rows: list[dict], priors_by_season: dict, z_by_season: dict,
                       elo_cfg: EloConfig, blend_cfg: BlendConfig,
                       train_seasons: set[int], grid: dict = _GRID,
                       n_passes: int = 3) -> tuple[PriorWeights, DecayConfig]:
    """One parameter at a time over `grid`, holding the others at the grid's
    center value, for `n_passes` passes, selecting by TRAIN-split Weeks 1-5
    margin MAE (mirrors _coordinate_search in backtest_cfb_ratings.py /
    _tune_shrink_from_raw in backtest_cfb_gameline.py)."""
    current = {p: grid[p][len(grid[p]) // 2] for p in _ORDER}
    cache: dict = {}

    def _params_to_configs(params: dict) -> tuple[PriorWeights, DecayConfig]:
        w = PriorWeights(sp_scale=params["sp_scale"], sp_offset=PriorWeights().sp_offset,
                          w_portal=params["w_portal"], w_coach=params["w_coach"],
                          w_qb=params["w_qb"], w_starters=params["w_starters"],
                          w_sos_prior=params["w_sos_prior"], w_sos_shift=params["w_sos_shift"])
        d = DecayConfig(half_life_games=params["half_life_games"])
        return w, d

    def _eval(params: dict) -> float:
        key = tuple(params[p] for p in _ORDER)
        if key in cache:
            return cache[key]
        w, d = _params_to_configs(params)
        mae = _early_mae(raw_rows, priors_by_season, z_by_season, w, d,
                          elo_cfg, blend_cfg, train_seasons)
        cache[key] = mae
        return mae

    for _ in range(n_passes):
        improved = False
        for p in _ORDER:
            best_val = current[p]
            best_mae = _eval(current)
            for v in grid[p]:
                trial = dict(current); trial[p] = v
                mae = _eval(trial)
                if mae < best_mae:
                    best_mae = mae
                    best_val = v
            if best_val != current[p]:
                improved = True
            current[p] = best_val
        if not improved:
            break

    return _params_to_configs(current)


def ablate_factors(raw_rows: list[dict], priors_by_season: dict, z_by_season: dict,
                    fitted_weights: PriorWeights, fitted_decay: DecayConfig,
                    elo_cfg: EloConfig, blend_cfg: BlendConfig,
                    holdout_seasons: set[int]) -> dict:
    """Leave-one-out ablation (design decision #6): for each adjustment
    factor, zero just that field of the already-fitted weights and compare
    HOLDOUT Weeks 1-5 margin MAE to the full fit's. Returns
    {factor_name: {"full_mae": ..., "zeroed_mae": ..., "keep": bool}}."""
    full_mae = _early_mae(raw_rows, priors_by_season, z_by_season, fitted_weights,
                          fitted_decay, elo_cfg, blend_cfg, holdout_seasons)
    result = {}
    for factor in ADJUSTMENT_FACTORS:
        zeroed = PriorWeights(**{**fitted_weights.__dict__, factor: 0.0})
        zeroed_mae = _early_mae(raw_rows, priors_by_season, z_by_season, zeroed,
                                fitted_decay, elo_cfg, blend_cfg, holdout_seasons)
        result[factor] = {
            "full_mae": full_mae,
            "zeroed_mae": zeroed_mae,
            "keep": full_mae < zeroed_mae,  # dropping it makes holdout MAE worse -> keep it
        }
    return result


# --------------------------------------------------------------------------
# Reporting tables
# --------------------------------------------------------------------------

def accuracy_table(raw_rows: list[dict], priors_by_season: dict, z_by_season: dict,
                    weights: PriorWeights, decay_cfg: DecayConfig,
                    elo_cfg: EloConfig, blend_cfg: BlendConfig) -> list[dict]:
    """Per-week margin MAE, current model vs prior-seeded (total MAE is
    identical for both -- see design decision #2 -- and is included for
    completeness, labeled accordingly)."""
    by_week: dict[int, list[dict]] = {}
    for r in raw_rows:
        by_week.setdefault(r["week"], []).append(r)
    rows = []
    for week in sorted(by_week):
        wrows = by_week[week]
        cur_err = [abs(current_model_margin(r, elo_cfg, blend_cfg) - r["actual_margin"]) for r in wrows]
        prior_err = [abs(prior_seeded_margin(r, priors_by_season, z_by_season, weights, decay_cfg,
                                             elo_cfg, blend_cfg) - r["actual_margin"]) for r in wrows]
        total_err = [abs(r["model_total"] - r["actual_total"]) for r in wrows]
        rows.append({
            "week": week, "n": len(wrows),
            "margin_mae_current": sum(cur_err) / len(cur_err),
            "margin_mae_prior_seeded": sum(prior_err) / len(prior_err),
            "total_mae_unaffected_by_prior": sum(total_err) / len(total_err),
        })
    return rows


def grade_vs_market(model_margin: float, market_spread: float, actual_margin: float) -> dict:
    """Grade one game's model pick against its closing market line.

    `market_spread` (assets/cfb/lines.parquet) is stored in HOME-MARGIN
    convention (positive = home favored), matching `model_margin`/
    `actual_margin` directly -- see build_cfb_lines.py's docstring. `ats_result`
    instead expects `closing_home_spread` in standard SPORTSBOOK convention
    (negative = home favored), so it is negated at this call site only; the
    local `threshold` stays in home-margin convention (== market_spread) so
    it lines up with `model_margin` for `gap`/`clv_proxy`, and with
    `ats_result`'s own internal threshold (`-closing_home_spread ==
    market_spread`) so the pick direction used here matches the one
    `ats_result` grades against.

    Example: home favored by 10 (market_spread=+10), model likes away
    (model_margin=+3, i.e. model expects a smaller home margin than the
    market), home wins by 15 (actual_margin=+15) -> home covers -> the
    model's away pick LOSES.
    """
    threshold = market_spread
    gap = abs(model_margin - threshold)
    ats = ats_result(model_margin, -market_spread, actual_margin)
    model_favors_home = model_margin > threshold
    # clv_proxy is a SIGNED, OUTCOME-based proxy -- not just a repackaging of
    # `gap` (which is pre-game and outcome-independent). It is the realized
    # margin by which the model's picked side beat (+) or missed (-) the
    # closing number: actual_margin - threshold from the home side's
    # perspective, sign-flipped when the model picked away so it always
    # reads positive-good/negative-bad for the SIDE THE MODEL CHOSE. This
    # tracks `ats` (same sign as a "win"/"loss") but keeps the magnitude of
    # the cover/miss instead of collapsing it to a binary result.
    clv_proxy = (actual_margin - threshold) if model_favors_home else (threshold - actual_margin)
    return {"gap": gap, "ats": ats, "clv_proxy": clv_proxy}


def edge_table(raw_rows: list[dict], priors_by_season: dict, z_by_season: dict,
               weights: PriorWeights, decay_cfg: DecayConfig,
               elo_cfg: EloConfig, blend_cfg: BlendConfig) -> list[dict]:
    """Bucket games with a valid closing spread by |model_margin - closing
    threshold|; report ATS win% (via ats_result/bucket_winrate) and mean
    CLV-proxy (design decision #4) per bucket. Per-game grading (market-line
    convention conversion included) is `grade_vs_market`, above."""
    graded = []
    for r in raw_rows:
        if r["market_spread"] is None:
            continue
        model_margin = prior_seeded_margin(r, priors_by_season, z_by_season, weights, decay_cfg,
                                           elo_cfg, blend_cfg)
        graded.append(grade_vs_market(model_margin, r["market_spread"], r["actual_margin"]))

    rows = []
    for lo, hi in EDGE_BUCKETS:
        bucket = [g for g in graded if lo <= g["gap"] < hi]
        win_pct = bucket_winrate(bucket, min_gap=lo)
        mean_clv = sum(g["clv_proxy"] for g in bucket) / len(bucket) if bucket else 0.0
        rows.append({
            "gap_bucket": f"[{lo},{hi})", "n": len(bucket),
            "ats_win_pct": win_pct, "mean_clv_proxy": mean_clv,
        })
    return rows


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    import json

    if not PRIORS_PATH.exists():
        sys.exit(
            "assets/cfb/priors.parquet not found. Run scripts/build_cfb_priors.py "
            "first (needs CFBD_API_KEY) -- e.g.:\n"
            "  CFBD_API_KEY=... PYTHONPATH=src uv run --no-sync python "
            "scripts/build_cfb_priors.py --seasons 2015 2025\n"
            "or trigger the build-cfb-priors GitHub Actions workflow (Task 7), "
            "then re-run this backtest."
        )

    t0 = time.time()
    priors_by_season = load_priors_by_season(PRIORS_PATH)
    z_by_season = priors_z_by_season(priors_by_season)
    priors_seasons = set(priors_by_season)

    merged = load_merged_schedule(SCHEDULES_PATH, LINES_PATH)
    full_span = merged[merged["season"].isin(priors_seasons)].copy()
    if full_span.empty:
        sys.exit("No schedule rows overlap priors.parquet's season coverage -- nothing to backtest.")

    rating = json.loads(RATING_PATH.read_text())
    elo_cfg = EloConfig(k=rating["k"], hfa_elo=rating["hfa_elo"],
                        carryover=rating["carryover"], base=rating.get("base", 1500.0))
    blend_cfg = BlendConfig(w_sos=rating["w_sos"], srs_min_games=rating["srs_min_games"])

    t_walk0 = time.time()
    raw = raw_walk_forward(full_span, elo_cfg)
    t_walk = time.time() - t_walk0

    train_seasons = {s for s in priors_seasons if s % 2 == 0}
    holdout_seasons = {s for s in priors_seasons if s % 2 == 1}
    print(f"TRAIN seasons (even, {len(train_seasons)}): {sorted(train_seasons)}")
    print(f"HOLDOUT seasons (odd, {len(holdout_seasons)}): {sorted(holdout_seasons)}")

    t_fit0 = time.time()
    fitted_weights, fitted_decay = fit_prior_weights(raw, priors_by_season, z_by_season,
                                                      elo_cfg, blend_cfg, train_seasons)
    t_fit = time.time() - t_fit0
    print(f"fitted (pre-ablation): {fitted_weights} half_life_games={fitted_decay.half_life_games}")

    ablation = ablate_factors(raw, priors_by_season, z_by_season, fitted_weights, fitted_decay,
                              elo_cfg, blend_cfg, holdout_seasons)
    final_kwargs = dict(sp_scale=fitted_weights.sp_scale, sp_offset=fitted_weights.sp_offset)
    print("\n=== ABLATION (holdout Weeks 1-5 margin MAE; keep iff dropping the factor hurts) ===")
    for factor, res in ablation.items():
        kept = res["keep"]
        final_kwargs[factor] = getattr(fitted_weights, factor) if kept else 0.0
        print(f"  {factor}: full={res['full_mae']:.4f} zeroed={res['zeroed_mae']:.4f} "
              f"-> {'KEEP' if kept else 'DROP'}")
    final_weights = PriorWeights(**final_kwargs)

    # Two files -- see design note #6 (load_weights raises TypeError on any
    # key beyond PriorWeights' own 8 fields, so DecayConfig cannot share the
    # file load_weights reads).
    weights_out = {
        "sp_scale": final_weights.sp_scale,
        "sp_offset": final_weights.sp_offset,
        "w_portal": final_weights.w_portal,
        "w_coach": final_weights.w_coach,
        "w_qb": final_weights.w_qb,
        "w_starters": final_weights.w_starters,
        "w_sos_prior": final_weights.w_sos_prior,
        "w_sos_shift": final_weights.w_sos_shift,
    }
    decay_out = {
        "half_life_games": fitted_decay.half_life_games,
        "prior_floor": fitted_decay.prior_floor,
    }
    WEIGHTS_OUT_PATH.write_text(json.dumps(weights_out, indent=2) + "\n")
    DECAY_OUT_PATH.write_text(json.dumps(decay_out, indent=2) + "\n")
    print(f"\nwritten {WEIGHTS_OUT_PATH}: {weights_out}")
    print(f"written {DECAY_OUT_PATH}: {decay_out}")

    print("\n=== ACCURACY TABLE (per-week margin MAE, current vs prior-seeded) ===")
    for row in accuracy_table(raw, priors_by_season, z_by_season, final_weights, fitted_decay,
                              elo_cfg, blend_cfg):
        print(f"  week {row['week']:>2} (n={row['n']:>4}): "
              f"current={row['margin_mae_current']:.3f}  "
              f"prior_seeded={row['margin_mae_prior_seeded']:.3f}  "
              f"total(unaffected)={row['total_mae_unaffected_by_prior']:.3f}")

    print("\n=== EDGE TABLE (vs closing spread; CLV is a proxy -- see design note #4) ===")
    for row in edge_table(raw, priors_by_season, z_by_season, final_weights, fitted_decay,
                          elo_cfg, blend_cfg):
        print(f"  gap {row['gap_bucket']:>10} (n={row['n']:>4}): "
              f"ats_win_pct={row['ats_win_pct']:.4f}  mean_clv_proxy={row['mean_clv_proxy']:.3f}")

    print(f"\nwalk-forward wall-clock (s): {round(t_walk, 1)}")
    print(f"fit wall-clock (s): {round(t_fit, 1)}")
    print(f"total wall-clock (s): {round(time.time() - t0, 1)}")


if __name__ == "__main__":
    main()
