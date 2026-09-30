"""CFB preseason-prior backtest -- fits the LEAK-FREE PriorWeights + DecayConfig
against assets/cfb/priors.parquet with the fixed season-to-date engine, then
reports the fit / held-out accuracy / ablation / edge tables.

Rewritten for fix/cfb-live-ratings (part B). What changed and why:

1. **Leak-free prior.** The old fit used priors.parquet's SAME-season
   sp_rating (CFBD /ratings/sp?year=S = END-of-season SP+), coach_first_year
   and forward_sos_shift -- all post-season information. The prior is now
   `sportsmodel.cfb.priors`' leak-free definition: previous season's final
   SP+ (season-1; missing -> league mean) + z-scored preseason features
   (returning_pct, recruiting_points, portal_net, prior_sos) + signed QB flag.
   `prior_inputs_by_season` goes through `season_prior_inputs`, which only
   ever reads the allowed columns (tested with planted values).
2. **Same engine as live and as backtest_cfb_gameline.** The raw walk-forward
   is `sportsmodel.cfb.walkforward.walk` (Elo continuous across seasons; SRS /
   points / games_played season-to-date, refreshed once per completed week)
   -- the exact state generate_cfb serves (see part A's parity tests).
3. **Split.** TRAIN = 2016-2022 (2015 has no previous-season SP+ in the
   asset, so it only warms up Elo), HOLDOUT = 2023-2025, reported only --
   nothing is selected on it (the old leave-one-out ablation used the holdout
   to drop factors; it is now informational only).
4. **Objective.** Weeks 1-5 margin MAE on TRAIN (stage 1; stage 2 re-fits the
   decay on all TRAIN weeks, see 5), FBS-vs-FBS games only (the
   live producer never serves a game against the "FCS" pseudo-team). Raw
   model margin (the gameline bias is a constant applied downstream).
5. **Blend (unchanged design).** R_pre lives on the Elo scale and is blended
   with each team's pre-game Elo via `blend_rating(r_pre, elo, games_played,
   decay)` (games_played = season-to-date), then fed through the usual
   `expected_margin` (SRS/HFA untouched). Totals are not touched by the
   prior. Coordinate search (one parameter at a time over a grid, a few
   passes, select by TRAIN metric) over sp_scale, sp_offset, the five
   feature weights, half_life_games and prior_floor (stage 1, weeks 1-5);
   then (stage 2, fix round 1) the decay -- half_life_games, prior_floor --
   is re-chosen with the weights fixed on ALL TRAIN weeks, because the floor
   governs the whole season.
6. **Market lines are benchmark-only** (edge table after the fit); the
   closing spread is in HOME-MARGIN convention (see `grade_vs_market`).
   "CLV" is a realized-cover proxy, not line movement (only closing lines
   are stored).

Writes assets/cfb/priors_weights.json (PriorWeights fields) and
assets/cfb/priors_decay.json (DecayConfig) unless --dry-run.

Usage:
    .venv/bin/python scripts/backtest_cfb_priors.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel.cfb.priors import (
    DECAY_PATH as DECAY_OUT_PATH,
    DEFAULT_HALF_LIFE_GAMES as _DEFAULT_HALF_LIFE_GAMES,
    DecayConfig,
    PriorWeights,
    blend_rating,
    load_decay_config,  # noqa: F401 -- re-exported (tests pin it lives in priors)
    preseason_rating,
    season_prior_inputs,
)
from sportsmodel.cfb.teams import FCS
from sportsmodel.cfb.walkforward import _clean_market, model_margin_total, walk
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig, expected_margin

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets" / "cfb"
SCHEDULES_PATH = ASSETS / "schedules.parquet"
LINES_PATH = ASSETS / "lines.parquet"
PRIORS_PATH = ASSETS / "priors.parquet"
RATING_PATH = ASSETS / "rating.json"
WEIGHTS_OUT_PATH = ASSETS / "priors_weights.json"

TRAIN_SEASONS = set(range(2016, 2023))      # 2016-2022
HOLDOUT_SEASONS = {2023, 2024, 2025}
WALK_SEASONS = set(range(2015, 2026))       # continuous Elo warm-up from 2015
EARLY_WEEKS = {1, 2, 3, 4, 5}
ADJUSTMENT_FACTORS = ("w_returning", "w_recruiting", "w_portal", "w_qb", "w_sos_prior")
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
# Data loading + leak-free prior inputs
# --------------------------------------------------------------------------

def load_priors_by_season(path) -> dict[int, list[dict]]:
    """assets/cfb/priors.parquet -> {season: [row_dict, ...]}."""
    df = pd.read_parquet(path)
    return {int(season): sdf.to_dict("records") for season, sdf in df.groupby("season")}


def prior_inputs_by_season(priors_by_season: dict[int, list[dict]]) -> dict[int, dict]:
    """{season: {team: season_prior_inputs entry}} -- the ONLY view of
    priors.parquet the fit sees (leak-free by construction)."""
    return {season: season_prior_inputs(priors_by_season, season) for season in priors_by_season}


def r_pre_table(inputs_by_season: dict[int, dict], weights: PriorWeights) -> dict:
    """{(season, team): R_pre} for every team-season with prior inputs."""
    return {(season, team): preseason_rating(inp, weights)
            for season, teams in inputs_by_season.items() for team, inp in teams.items()}


def load_merged_schedule(schedules_path=SCHEDULES_PATH, lines_path=LINES_PATH) -> pd.DataFrame:
    """Left-join CFB schedules -> CFB lines, dropping CFBD self-match rows
    from both sides first (identical to backtest_cfb_gameline.load_merged_schedule)."""
    sched = pd.read_parquet(schedules_path)
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    reg = reg[reg["home_team"] != reg["away_team"]].copy()

    lines = pd.read_parquet(lines_path)
    lines = lines[lines["home_team"] != lines["away_team"]].copy()
    lines = lines.drop_duplicates(subset=["season", "week", "home_team", "away_team"], keep="first")

    return reg.merge(lines, on=["season", "week", "home_team", "away_team"],
                     how="left", validate="one_to_one")


# --------------------------------------------------------------------------
# Raw walk-forward (weights-independent), cached once
# --------------------------------------------------------------------------

def raw_walk_forward(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                     blend_cfg: BlendConfig | None = None) -> list[dict]:
    """One entry per scored game from the shared walk-forward engine
    (`walkforward.walk`): pre-game Elo, season-to-date SRS and games-played
    counts, the model total, closing lines and actuals. Weights-independent:
    the prior is applied on top by `prior_seeded_margin`."""
    blend_cfg = blend_cfg or BlendConfig()
    out = []
    for season, week, g, st in walk(schedule_df, elo_cfg):
        h, a = g["home_team"], g["away_team"]
        _, model_total = model_margin_total(h, a, g["elo_home"], g["elo_away"], st,
                                            elo_cfg, blend_cfg)
        out.append({
            "season": int(season), "week": int(week),
            "home_team": h, "away_team": a, "fcs": FCS in (h, a),
            "elo_home": float(g["elo_home"]), "elo_away": float(g["elo_away"]),
            "srs_home": st.srs.get(h), "srs_away": st.srs.get(a),
            "games_home": st.counts.get(h, 0), "games_away": st.counts.get(a, 0),
            "model_total": model_total,
            "market_spread": _clean_market(g.get("market_spread")),
            "market_total": _clean_market(g.get("market_total")),
            "actual_margin": float(g["home_score"] - g["away_score"]),
            "actual_total": float(g["home_score"] + g["away_score"]),
        })
    return out


def current_model_margin(raw_row: dict, elo_cfg: EloConfig, blend_cfg: BlendConfig) -> float:
    """No prior blend: pure pre-game Elo + season-to-date SRS."""
    return expected_margin(raw_row["elo_home"], raw_row["elo_away"],
                           raw_row["srs_home"], raw_row["srs_away"],
                           raw_row["games_home"], raw_row["games_away"], elo_cfg, blend_cfg)


def prior_seeded_margin(raw_row: dict, r_pre: dict, decay_cfg: DecayConfig,
                        elo_cfg: EloConfig, blend_cfg: BlendConfig) -> float:
    """Blend each side's pre-game Elo with its R_pre (decayed by season-to-date
    games played), then the usual expected_margin -- exactly build_game_rows'
    live blend. Teams without an R_pre (FCS, uncovered) stay unblended."""
    season = raw_row["season"]
    elo_h, elo_a = raw_row["elo_home"], raw_row["elo_away"]
    rh = r_pre.get((season, raw_row["home_team"]))
    ra = r_pre.get((season, raw_row["away_team"]))
    if rh is not None:
        elo_h = blend_rating(rh, elo_h, raw_row["games_home"], decay_cfg)
    if ra is not None:
        elo_a = blend_rating(ra, elo_a, raw_row["games_away"], decay_cfg)
    return expected_margin(elo_h, elo_a, raw_row["srs_home"], raw_row["srs_away"],
                           raw_row["games_home"], raw_row["games_away"], elo_cfg, blend_cfg)


def _scored(raw_rows: list[dict], seasons: set[int], weeks=None) -> list[dict]:
    """FBS-vs-FBS rows in `seasons` (and `weeks`, if given)."""
    return [r for r in raw_rows if r["season"] in seasons and not r["fcs"]
            and (weeks is None or r["week"] in weeks)]


def _mae(rows: list[dict], inputs: dict, weights: PriorWeights, decay_cfg: DecayConfig,
         elo_cfg: EloConfig, blend_cfg: BlendConfig) -> float:
    if not rows:
        return 0.0
    r_pre = r_pre_table(inputs, weights)
    return sum(abs(prior_seeded_margin(r, r_pre, decay_cfg, elo_cfg, blend_cfg)
                   - r["actual_margin"]) for r in rows) / len(rows)


# --------------------------------------------------------------------------
# Fitting: coordinate search, Weeks 1-5 margin MAE on TRAIN
# --------------------------------------------------------------------------

# One eval is ~3 ms (cached walk-forward), so the grids are fine and wide
# enough that no fitted value should sit on an edge (checked in the report).
_W = [float(v) for v in range(-150, 151, 10)]
_GRID = {
    "sp_scale": [2.5 * i for i in range(0, 21)],                     # 0 .. 50
    "sp_offset": [1400.0 + 25.0 * i for i in range(0, 13)],          # 1400 .. 1700
    "w_returning": _W, "w_recruiting": _W, "w_portal": _W, "w_qb": _W, "w_sos_prior": _W,
    "half_life_games": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0],
    "prior_floor": [round(0.05 * i, 2) for i in range(0, 19)],               # 0 .. 0.9
}
_START = {"sp_scale": 20.0, "sp_offset": 1500.0, "w_returning": 0.0, "w_recruiting": 0.0,
          "w_portal": 0.0, "w_qb": 0.0, "w_sos_prior": 0.0, "half_life_games": 4.0,
          "prior_floor": 0.0}
_ORDER = ["sp_scale", "sp_offset", "w_returning", "w_recruiting", "w_portal", "w_qb",
          "w_sos_prior", "half_life_games", "prior_floor"]


def _configs(params: dict) -> tuple[PriorWeights, DecayConfig]:
    w = PriorWeights(**{k: params[k] for k in ("sp_scale", "sp_offset") + ADJUSTMENT_FACTORS})
    return w, DecayConfig(half_life_games=params["half_life_games"],
                          prior_floor=params["prior_floor"])


def fit_prior_weights(raw_rows: list[dict], inputs: dict, elo_cfg: EloConfig,
                      blend_cfg: BlendConfig, train_seasons: set[int] = TRAIN_SEASONS,
                      grid: dict = _GRID, start: dict = _START,
                      n_passes: int = 10) -> tuple[PriorWeights, DecayConfig, float]:
    """Coordinate search (mirrors backtest_cfb_ratings/_gameline): one
    parameter at a time over `grid`, up to `n_passes` passes, selecting by
    TRAIN Weeks 1-5 FBS margin MAE. Returns (weights, decay, train_mae)."""
    rows = _scored(raw_rows, train_seasons, EARLY_WEEKS)
    cache: dict = {}

    def _eval(params: dict) -> float:
        key = tuple(params[p] for p in _ORDER)
        if key not in cache:
            cache[key] = _mae(rows, inputs, *_configs(params), elo_cfg, blend_cfg)
        return cache[key]

    current = dict(start)
    for _ in range(n_passes):
        improved = False
        for p in _ORDER:
            best_val, best_mae = current[p], _eval(current)
            for v in grid[p]:
                mae = _eval({**current, p: v})
                if mae < best_mae - 1e-12:
                    best_val, best_mae = v, mae
            if best_val != current[p]:
                improved = True
                current[p] = best_val
        if not improved:
            break
    w, d = _configs(current)
    return w, d, _eval(current)


_DECAY_ORDER = ["half_life_games", "prior_floor"]


def fit_decay_all_weeks(raw_rows: list[dict], inputs: dict, weights: PriorWeights,
                        elo_cfg: EloConfig, blend_cfg: BlendConfig,
                        train_seasons: set[int] = TRAIN_SEASONS, grid: dict = _GRID,
                        start: DecayConfig | None = None,
                        n_passes: int = 10) -> tuple[DecayConfig, float]:
    """Stage 2 (fix round 1): with the feature weights FIXED (fit on TRAIN
    weeks 1-5), re-choose the decay (half_life_games, prior_floor) by
    coordinate search on ALL TRAIN weeks (full-season FBS margin MAE).
    The floor controls the whole season (half-life 3 hits a 0.65 floor after
    2 games), so a weeks-1-5 objective must not choose it."""
    rows = _scored(raw_rows, train_seasons)
    r_pre = r_pre_table(inputs, weights)
    cache: dict = {}

    def _eval(params: dict) -> float:
        key = (params["half_life_games"], params["prior_floor"])
        if key not in cache:
            d = DecayConfig(half_life_games=key[0], prior_floor=key[1])
            cache[key] = sum(abs(prior_seeded_margin(r, r_pre, d, elo_cfg, blend_cfg)
                                 - r["actual_margin"]) for r in rows) / len(rows)
        return cache[key]

    start = start or DecayConfig(half_life_games=_START["half_life_games"],
                                 prior_floor=_START["prior_floor"])
    current = {"half_life_games": start.half_life_games, "prior_floor": start.prior_floor}
    for _ in range(n_passes):
        improved = False
        for p in _DECAY_ORDER:
            best_val, best_mae = current[p], _eval(current)
            for v in grid[p]:
                mae = _eval({**current, p: v})
                if mae < best_mae - 1e-12:
                    best_val, best_mae = v, mae
            if best_val != current[p]:
                improved = True
                current[p] = best_val
        if not improved:
            break
    return (DecayConfig(half_life_games=current["half_life_games"],
                        prior_floor=current["prior_floor"]), _eval(current))


def ablate_factors(raw_rows: list[dict], inputs: dict, weights: PriorWeights,
                   decay_cfg: DecayConfig, elo_cfg: EloConfig, blend_cfg: BlendConfig,
                   seasons: set[int]) -> dict:
    """INFORMATIONAL leave-one-out: zero one feature weight of the fitted
    vector and re-score Weeks 1-5 margin MAE on `seasons`. Nothing is
    selected on it (the fit is TRAIN-only)."""
    rows = _scored(raw_rows, seasons, EARLY_WEEKS)
    full = _mae(rows, inputs, weights, decay_cfg, elo_cfg, blend_cfg)
    out = {}
    for f in ADJUSTMENT_FACTORS:
        zeroed = PriorWeights(**{**weights.__dict__, f: 0.0})
        out[f] = {"full_mae": full,
                  "zeroed_mae": _mae(rows, inputs, zeroed, decay_cfg, elo_cfg, blend_cfg)}
    return out


# --------------------------------------------------------------------------
# Reporting tables
# --------------------------------------------------------------------------

def accuracy_table(raw_rows: list[dict], inputs: dict, weights: PriorWeights,
                   decay_cfg: DecayConfig, elo_cfg: EloConfig, blend_cfg: BlendConfig,
                   seasons: set[int]) -> list[dict]:
    """Per-week FBS margin MAE on `seasons`: no prior vs prior-seeded (the
    total is unaffected by the prior and reported once)."""
    r_pre = r_pre_table(inputs, weights)
    by_week: dict[int, list[dict]] = {}
    for r in _scored(raw_rows, seasons):
        by_week.setdefault(r["week"], []).append(r)
    rows = []
    for week in sorted(by_week):
        wr = by_week[week]
        n = len(wr)
        rows.append({
            "week": week, "n": n,
            "margin_mae_no_prior": sum(abs(current_model_margin(r, elo_cfg, blend_cfg)
                                           - r["actual_margin"]) for r in wr) / n,
            "margin_mae_prior": sum(abs(prior_seeded_margin(r, r_pre, decay_cfg, elo_cfg,
                                                            blend_cfg) - r["actual_margin"])
                                    for r in wr) / n,
            "total_mae": sum(abs(r["model_total"] - r["actual_total"]) for r in wr) / n,
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


def edge_table(raw_rows: list[dict], inputs: dict, weights: PriorWeights,
               decay_cfg: DecayConfig, elo_cfg: EloConfig, blend_cfg: BlendConfig,
               seasons: set[int]) -> list[dict]:
    """Bucket FBS games in `seasons` with a closing spread by |model -
    closing|; ATS win% (pushes excluded) and mean CLV-proxy per bucket."""
    r_pre = r_pre_table(inputs, weights)
    graded = [grade_vs_market(prior_seeded_margin(r, r_pre, decay_cfg, elo_cfg, blend_cfg),
                              r["market_spread"], r["actual_margin"])
              for r in _scored(raw_rows, seasons) if r["market_spread"] is not None]
    rows = []
    for lo, hi in EDGE_BUCKETS:
        bucket = [g for g in graded if lo <= g["gap"] < hi]
        rows.append({"gap_bucket": f"[{lo},{hi})", "n": len(bucket),
                     "ats_win_pct": bucket_winrate(bucket, min_gap=lo),
                     "mean_clv_proxy": (sum(g["clv_proxy"] for g in bucket) / len(bucket)
                                        if bucket else 0.0)})
    return rows


def load_rating_configs(path=RATING_PATH) -> tuple[EloConfig, BlendConfig]:
    rating = json.loads(pathlib.Path(path).read_text())
    return (EloConfig(k=rating["k"], hfa_elo=rating["hfa_elo"], carryover=rating["carryover"],
                      base=rating.get("base", 1500.0)),
            BlendConfig(w_sos=rating["w_sos"], srs_min_games=rating["srs_min_games"]))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="fit and report without writing assets/cfb/priors_*.json")
    args = ap.parse_args(argv)

    if not PRIORS_PATH.exists():
        sys.exit("assets/cfb/priors.parquet not found. Run scripts/build_cfb_priors.py first "
                 "(needs CFBD_API_KEY), then re-run this backtest.")

    t0 = time.time()
    priors_by_season = load_priors_by_season(PRIORS_PATH)
    inputs = prior_inputs_by_season(priors_by_season)
    elo_cfg, blend_cfg = load_rating_configs()

    merged = load_merged_schedule(SCHEDULES_PATH, LINES_PATH)
    span = merged[merged["season"].isin(WALK_SEASONS)].copy()
    raw = raw_walk_forward(span, elo_cfg, blend_cfg)
    t_walk = time.time() - t0
    print(f"TRAIN seasons: {sorted(TRAIN_SEASONS)}   HOLDOUT seasons: {sorted(HOLDOUT_SEASONS)}")
    print("objective: stage 1 weights+decay on TRAIN wk1-5; stage 2 decay on ALL TRAIN weeks "
          "(FBS-vs-FBS margin MAE)")

    t_fit0 = time.time()
    weights, decay_early, train_mae = fit_prior_weights(raw, inputs, elo_cfg, blend_cfg)
    print(f"stage 1 (TRAIN wk1-5): {weights}  {decay_early}  train wk1-5 MAE={train_mae:.4f}")
    decay, train_all_mae = fit_decay_all_weeks(raw, inputs, weights, elo_cfg, blend_cfg,
                                               start=decay_early)
    t_fit = time.time() - t_fit0
    print(f"stage 2 (decay on ALL TRAIN weeks, weights fixed): {decay}  "
          f"train all-weeks MAE={train_all_mae:.4f}")

    late_weeks = set(range(6, 30))
    for label, seasons in (("TRAIN", TRAIN_SEASONS), ("HOLDOUT", HOLDOUT_SEASONS)):
        for scope, weeks in (("wk1-5", EARLY_WEEKS), ("wk6+", late_weeks), ("all", None)):
            sub = _scored(raw, seasons, weeks)
            no_prior = (sum(abs(current_model_margin(r, elo_cfg, blend_cfg) - r["actual_margin"])
                            for r in sub) / len(sub)) if sub else 0.0
            with_prior = _mae(sub, inputs, weights, decay, elo_cfg, blend_cfg)
            print(f"{label} {scope} FBS margin MAE (n={len(sub)}): no prior={no_prior:.4f}  "
                  f"prior={with_prior:.4f}")

    print("\n=== ABLATION (informational; HOLDOUT wk1-5 MAE with one weight zeroed) ===")
    for f, res in ablate_factors(raw, inputs, weights, decay, elo_cfg, blend_cfg,
                                 HOLDOUT_SEASONS).items():
        print(f"  {f}: full={res['full_mae']:.4f} zeroed={res['zeroed_mae']:.4f}")

    weights_out = {k: float(v) for k, v in weights.__dict__.items()}
    decay_out = {"half_life_games": float(decay.half_life_games),
                 "prior_floor": float(decay.prior_floor)}
    if args.dry_run:
        print(f"\n--dry-run: would write {weights_out} / {decay_out}")
    else:
        WEIGHTS_OUT_PATH.write_text(json.dumps(weights_out, indent=2) + "\n")
        DECAY_OUT_PATH.write_text(json.dumps(decay_out, indent=2) + "\n")
        print(f"\nwritten {WEIGHTS_OUT_PATH}: {weights_out}")
        print(f"written {DECAY_OUT_PATH}: {decay_out}")

    print("\n=== ACCURACY TABLE (HOLDOUT, per-week FBS margin MAE) ===")
    for row in accuracy_table(raw, inputs, weights, decay, elo_cfg, blend_cfg, HOLDOUT_SEASONS):
        print(f"  week {row['week']:>2} (n={row['n']:>4}): no_prior={row['margin_mae_no_prior']:.3f}"
              f"  prior={row['margin_mae_prior']:.3f}  total={row['total_mae']:.3f}")

    print("\n=== EDGE TABLE (HOLDOUT vs closing spread; CLV is a proxy) ===")
    for row in edge_table(raw, inputs, weights, decay, elo_cfg, blend_cfg, HOLDOUT_SEASONS):
        print(f"  gap {row['gap_bucket']:>10} (n={row['n']:>4}): "
              f"ats_win_pct={row['ats_win_pct']:.4f}  mean_clv_proxy={row['mean_clv_proxy']:.3f}")

    print(f"\nwalk-forward (s): {t_walk:.1f}  fit (s): {t_fit:.1f}  "
          f"total (s): {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()
