"""Shared leak-free CFB ratings walk-forward.

Moved from `scripts/backtest_cfb_gameline.py::_raw_model_predictions`
(behavior-identical for every key that function returned; the backtest now
imports it from here) so the CFB profit-model gate and the live bet step share
ONE implementation.

Per scored game: the model's own pre-game margin/total (pre-game Elo via
run_elo, continuous across seasons, + season-to-date SRS + season-to-date
opponent-adjusted points, both refreshed once per completed WEEK -- per-game
Gauss-Seidel re-solves are intractable at CFB's ~830 games/season), the
game's closing market line (NaN -> None), and the actuals.

Extensions over the original (additive keys only):
- pass-through of the schedule/line context columns in PASSTHROUGH_COLS that
  are present on the frame (NaN/NA -> None, numpy scalars -> Python);
- the pre-game ratings fed into the margin: `elo_home`/`elo_away` (run_elo's
  pre-game values) and `srs_home`/`srs_away` (the week's season-to-date SRS
  cache; None when the team has no SRS yet this season).

Leak-free invariant: appending a later game to schedule_df never changes an
earlier game's entry in the returned list.
"""
from __future__ import annotations

import pandas as pd

from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.points import compute_points_ratings, expected_total
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.srs import compute_srs

_DEFAULT_TOTAL_SEED = 55.0   # ~schedule-wide mean/median total, used only
                             # before any points history exists that season

PASSTHROUGH_COLS = ("game_pk", "start_date", "neutral_site", "conference_game",
                    "home_conf", "away_conf", "spread_open", "total_open",
                    "ml_home", "ml_away")


def _clean_market(value) -> float | None:
    """CFBD market_spread/market_total -> float, or None if missing/NaN.

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


def _clean_passthrough(value):
    """NaN/NA -> None; numpy scalars -> plain Python values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def raw_model_predictions(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                          blend_cfg: BlendConfig) -> list[dict]:
    """Leak-free walk-forward core, WITHOUT any shrink/sigma applied (see the
    module docstring). Intended to be called ONCE over a full continuous span;
    callers re-score the cached rows cheaply."""
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)                    # pre-game elo per game, continuous across seasons
    games = res.games
    passthrough = [c for c in PASSTHROUGH_COLS if c in games.columns]
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
                srs_h, srs_a = srs_cache.get(h), srs_cache.get(a)
                model_margin = expected_margin(g["elo_home"], g["elo_away"],
                                               srs_h, srs_a,
                                               gh, ga, elo_cfg, blend_cfg)
                model_total = ((expected_total(pts_cache, lg_cache, h, a) if pts_cache
                               else 2 * lg_cache) if lg_cache else _DEFAULT_TOTAL_SEED)
                row = {
                    "season": int(season), "week": int(week),
                    "home_team": h, "away_team": a,
                    "model_margin": model_margin, "model_total": model_total,
                    "market_spread": _clean_market(g.get("market_spread")),
                    "market_total": _clean_market(g.get("market_total")),
                    "actual_margin": float(g["home_score"] - g["away_score"]),
                    "actual_total": float(g["home_score"] + g["away_score"]),
                }
                for c in passthrough:
                    row[c] = _clean_passthrough(g.get(c))
                row["elo_home"] = _clean_market(g["elo_home"])
                row["elo_away"] = _clean_market(g["elo_away"])
                row["srs_home"] = _clean_market(srs_h)
                row["srs_away"] = _clean_market(srs_a)
                out.append(row)
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
