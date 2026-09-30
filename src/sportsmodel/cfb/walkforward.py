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

from dataclasses import dataclass

import pandas as pd

from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.points import compute_points_ratings, expected_total
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.srs import compute_srs

_DEFAULT_TOTAL_SEED = 55.0   # ~schedule-wide mean/median total, used only
                             # before any points history exists that season

_SENTINEL = "__no_game__"      # live_week_state placeholder team (never rated)

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


@dataclass(frozen=True)
class SeasonState:
    """Season-to-date rating state entering a week (weeks < W of ONE season).

    counts: games played so far this season per team; srs: season-to-date
    SRS (empty before any completed week); pts/lg: season-to-date
    opponent-adjusted points ratings and league-average points per team-game
    (lg == 0.0 before any completed week -> totals fall back to the seed)."""
    counts: dict
    srs: dict
    pts: dict
    lg: float


def iter_week_states(sdf: pd.DataFrame):
    """Walk ONE season's games (already carrying run_elo's elo_home/elo_away
    and ordered as the walk-forward orders them) week by week.

    Yields (week, week_games_df, state) where `state` is the SeasonState
    entering that week, built from the season's strictly-earlier COMPLETED
    weeks only. After each yield, the week's scored games join the history
    and counts/SRS/points are refreshed ONCE (per-game Gauss-Seidel re-solves
    are intractable at CFB scale). A week with no scored games leaves the
    history untouched. This is THE season-to-date state used by both the
    backtest walk-forward and the live producer."""
    counts, srs_cache, pts_cache, lg_cache = {}, {}, {}, 0.0
    hist = sdf.iloc[0:0]
    for week, wdf_all in sdf.groupby("week"):
        yield week, wdf_all, SeasonState(counts, srs_cache, pts_cache, lg_cache)
        wdf = wdf_all.dropna(subset=["home_score", "away_score"])
        if wdf.empty:                 # only unscored games: nothing joins history
            continue
        counts = dict(counts)
        for _, g in wdf.iterrows():
            h, a = g["home_team"], g["away_team"]
            counts[h] = counts.get(h, 0) + 1
            counts[a] = counts.get(a, 0) + 1
        hist = pd.concat([hist, wdf], ignore_index=True)
        srs_cache = compute_srs(hist)
        pts_cache, lg_cache = compute_points_ratings(hist, k_points=4.0)


def model_margin_total(h: str, a: str, elo_h: float, elo_a: float, state: SeasonState,
                       elo_cfg: EloConfig, blend_cfg: BlendConfig) -> tuple[float, float]:
    """The model's pre-game (margin, total) for one game from pre-game Elo
    (possibly prior-blended by the caller) + the season-to-date state,
    including the total seed/fallback: no points history this season ->
    _DEFAULT_TOTAL_SEED."""
    margin = expected_margin(elo_h, elo_a, state.srs.get(h), state.srs.get(a),
                             state.counts.get(h, 0), state.counts.get(a, 0),
                             elo_cfg, blend_cfg)
    total = ((expected_total(state.pts, state.lg, h, a) if state.pts
              else 2 * state.lg) if state.lg else _DEFAULT_TOTAL_SEED)
    return margin, total


def walk(schedule_df: pd.DataFrame, elo_cfg: EloConfig, *,
         include_unscored: bool = False, seasons=None):
    """Leak-free walk-forward generator: yields (season, week, g, state) for
    every emitted game `g` (a run_elo games row carrying pre-game
    elo_home/elo_away), with `state` the SeasonState entering that week.

    Elo runs over the WHOLE frame (continuous across seasons); `seasons`
    (optional) only restricts which seasons' SRS/points walks run and emit
    -- a season's state depends on that season's games alone."""
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)                    # pre-game elo per game, continuous across seasons
    games = res.games
    for season, sdf in games.groupby("season"):
        if seasons is not None and season not in seasons:
            continue
        sdf = sdf.sort_values("week")
        for week, wdf_all, state in iter_week_states(sdf):
            emit = wdf_all if include_unscored else wdf_all.dropna(
                subset=["home_score", "away_score"])
            for _, g in emit.iterrows():
                yield season, week, g, state


def raw_model_predictions(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                          blend_cfg: BlendConfig, *,
                          include_unscored: bool = False) -> list[dict]:
    """Leak-free walk-forward core, WITHOUT any shrink/sigma applied (see the
    module docstring). Intended to be called ONCE over a full continuous span;
    callers re-score the cached rows cheaply.

    include_unscored=False (default) emits only scored games -- exactly the
    original behavior. include_unscored=True also emits each week's games with
    missing scores (live/upcoming games), computed from that week's pre-game
    caches (Elo from run_elo, SRS/points from strictly-earlier weeks) with
    actual_margin/actual_total None; they never join the rating history.
    """
    passthrough = [c for c in PASSTHROUGH_COLS if c in schedule_df.columns]
    out = []
    for season, week, g, state in walk(schedule_df, elo_cfg,
                                       include_unscored=include_unscored):
        h, a = g["home_team"], g["away_team"]
        srs_h, srs_a = state.srs.get(h), state.srs.get(a)
        model_margin, model_total = model_margin_total(h, a, g["elo_home"], g["elo_away"],
                                                       state, elo_cfg, blend_cfg)
        row = {
            "season": int(season), "week": int(week),
            "home_team": h, "away_team": a,
            "model_margin": model_margin, "model_total": model_total,
            "market_spread": _clean_market(g.get("market_spread")),
            "market_total": _clean_market(g.get("market_total")),
            "actual_margin": _clean_market(g["home_score"] - g["away_score"]),
            "actual_total": _clean_market(g["home_score"] + g["away_score"]),
        }
        for c in passthrough:
            row[c] = _clean_passthrough(g.get(c))
        row["elo_home"] = _clean_market(g["elo_home"])
        row["elo_away"] = _clean_market(g["elo_away"])
        row["srs_home"] = _clean_market(srs_h)
        row["srs_away"] = _clean_market(srs_a)
        out.append(row)
    return out


def live_week_state(schedule_df: pd.DataFrame, season: int, week: int,
                    games: list[dict], elo_cfg: EloConfig) -> tuple[dict, SeasonState]:
    """The walk-forward state for the live, unplayed week (season, week).

    Built by running `walk` over exactly the frame the backtest would see if
    this week were the next one: every scored game strictly before (season,
    week) -- nothing at or after it, even if the schedule already carries
    results -- plus this week's `games` (dicts with home_team/away_team)
    appended UNSCORED. Returns ({team: pre-game Elo}, SeasonState entering
    the week): Elo continuous across seasons (with the season-start
    carryover), counts/SRS/points season-to-date. Identical, by construction,
    to what `raw_model_predictions(..., include_unscored=True)` computes for
    those games."""
    sched = schedule_df
    before = sched[(sched["season"] < season)
                   | ((sched["season"] == season) & (sched["week"] < week))]
    before = before.dropna(subset=["home_score", "away_score"])
    # With no games to predict, one unscored sentinel row still yields the
    # state entering the week (an unscored row never changes Elo/SRS/points).
    pairs = ([(g["home_team"], g["away_team"]) for g in games]
             or [(_SENTINEL, _SENTINEL)])
    cols = ["season", "week", "home_team", "away_team", "home_score", "away_score"]
    upcoming = pd.DataFrame([{"season": season, "week": week, "home_team": h,
                              "away_team": a, "home_score": float("nan"),
                              "away_score": float("nan")} for h, a in pairs], columns=cols)
    frame = pd.concat([before[cols].astype({"home_score": "float64", "away_score": "float64"}),
                       upcoming], ignore_index=True)
    elo: dict[str, float] = {}
    state = SeasonState({}, {}, {}, 0.0)
    for _s, w, g, st in walk(frame, elo_cfg, include_unscored=True, seasons={season}):
        if w != week:
            continue
        state = st
        for team, rating in ((g["home_team"], g["elo_home"]), (g["away_team"], g["elo_away"])):
            if team != _SENTINEL:
                elo.setdefault(team, float(rating))
    return elo, state
