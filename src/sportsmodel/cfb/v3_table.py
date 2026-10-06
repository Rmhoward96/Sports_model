"""The v3 feature table: one row per FBS-vs-FBS game with the v2 model's own margin/total, the
decaying prior margin, the efficiency side features and the context features -- all built in ONE
leak-free walk-forward (walkforward.walk), so the fit, the gate and the live producer share a
single implementation.

Row for a game of week W uses only: games before W of that season (Elo is continuous across
seasons), the previous season's final efficiency ratings, schedule/venue/weather facts known
before kickoff. Appending a later game never changes an earlier row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig

from .context import CONTEXT_COLS, ContextAssets, context_features, rest_table
from .efficiency import (POINT_FEATURES, EffConfig, season_prior, side_features, state_before)
from .priors import DecayConfig, PriorWeights, blend_rating, prior_weight, season_priors
from .teams import FCS
from .walkforward import _clean_market, model_margin_total, walk


@dataclass(frozen=True)
class V3Inputs:
    elo_cfg: EloConfig
    blend_cfg: BlendConfig
    prior_weights: PriorWeights
    decay: DecayConfig
    eff_cfg: EffConfig
    eff_games: pd.DataFrame
    priors_rows: dict            # {season: [priors.parquet row dicts]}
    talent: pd.DataFrame | None
    ctx: ContextAssets


SIDE_COLS = [f"{s}_{f}" for s in ("h", "a") for f in POINT_FEATURES]
TABLE_COLUMNS = (["season", "week", "game_pk", "home_team", "away_team", "neutral", "start_date",
                  "actual_margin", "actual_total", "market_spread", "market_total",
                  "margin_v2", "total_v2", "prior_margin", "non_neutral", "games_home", "games_away"]
                 + SIDE_COLS + list(CONTEXT_COLS))


def _r_pre(inp: V3Inputs, season: int) -> dict:
    """{team: R_pre} for `season` via the leak-free priors.season_priors (empty if no rows)."""
    if season not in inp.priors_rows:
        return {}
    rows = {s: r for s, r in inp.priors_rows.items() if s in (season - 1, season)}
    return season_priors(rows, season, inp.prior_weights)


def build_table(frame: pd.DataFrame, inp: V3Inputs, *, seasons=None,
                include_unscored: bool = False) -> pd.DataFrame:
    """Feature table for the games of `frame` (needs season, week, home_team, away_team,
    home_score, away_score, game_pk, neutral_site, start_date; market_spread / market_total
    optional). `seasons` restricts the emitted seasons (Elo still runs over the whole frame);
    include_unscored also emits games with no result yet (the live slate)."""
    rests = rest_table(frame)
    r_pre_cache, prior_cache, state_cache, eff_cache = {}, {}, {}, {}
    rows = []
    for season, week, g, st in walk(frame, inp.elo_cfg, include_unscored=include_unscored,
                                    seasons=None if seasons is None else set(seasons)):
        h, a = g["home_team"], g["away_team"]
        if FCS in (h, a):
            continue
        season, week = int(season), int(week)
        if season not in r_pre_cache:
            r_pre_cache[season] = _r_pre(inp, season)
            prior_cache[season] = season_prior(inp.eff_games, season, inp.priors_rows.get(season, []),
                                               inp.talent, inp.eff_cfg, eff_cache)
        rp = r_pre_cache[season]
        gh, ga = st.counts.get(h, 0), st.counts.get(a, 0)
        eh, ea = g["elo_home"], g["elo_away"]
        if h in rp:
            eh = blend_rating(rp[h], eh, gh, inp.decay)
        if a in rp:
            ea = blend_rating(rp[a], ea, ga, inp.decay)
        margin_v2, total_v2 = model_margin_total(h, a, eh, ea, st, inp.elo_cfg, inp.blend_cfg)
        if h in rp and a in rp:
            w_pair = 0.5 * (prior_weight(gh, inp.decay) + prior_weight(ga, inp.decay))
            prior_margin = w_pair * ((rp[h] + inp.elo_cfg.hfa_elo) - rp[a]) / 25.0
        else:
            prior_margin = 0.0
        if (season, week) not in state_cache:
            state_cache[(season, week)] = state_before(inp.eff_games, season, week, prior_cache[season],
                                                       inp.eff_cfg, eff_cache)
        neutral = bool(g.get("neutral_site"))
        hs = 0.0 if neutral else 1.0
        fh = side_features(state_cache[(season, week)], h, a, hs)
        fa = side_features(state_cache[(season, week)], a, h, -hs)
        ctx = context_features({"season": season, "game_pk": g["game_pk"], "home_team": h, "away_team": a,
                                "start_date": g.get("start_date"), "neutral_site": neutral},
                               inp.ctx, rests, gh, ga)
        scored = not (pd.isna(g["home_score"]) or pd.isna(g["away_score"]))
        rows.append({"season": season, "week": week, "game_pk": int(g["game_pk"]), "home_team": h,
                     "away_team": a, "neutral": neutral, "start_date": str(g.get("start_date") or ""),
                     "actual_margin": float(g["home_score"] - g["away_score"]) if scored else np.nan,
                     "actual_total": float(g["home_score"] + g["away_score"]) if scored else np.nan,
                     "market_spread": _clean_market(g.get("market_spread")),
                     "market_total": _clean_market(g.get("market_total")),
                     "margin_v2": margin_v2, "total_v2": total_v2, "prior_margin": prior_margin,
                     "non_neutral": 0.0 if neutral else 1.0, "games_home": gh, "games_away": ga,
                     **{f"h_{k}": v for k, v in fh.items()}, **{f"a_{k}": v for k, v in fa.items()}, **ctx})
    out = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    for c in ("market_spread", "market_total"):
        out[c] = out[c].astype("float64")
    return out


def live_frame(sched: pd.DataFrame, season: int, week: int, games: list[dict]) -> pd.DataFrame:
    """The frame the walk sees for the live week: every scored REG game strictly before (season,
    week) plus the slate's `games` appended UNSCORED (dicts carry game_pk, home_team, away_team,
    neutral_site, start_date). Mirrors walkforward.live_week_state, so a live row equals the
    backtest row the same game would have had."""
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched.columns else sched
    reg = reg[reg["home_team"] != reg["away_team"]]
    before = reg[(reg["season"] < season) | ((reg["season"] == season) & (reg["week"] < week))]
    before = before.dropna(subset=["home_score", "away_score"])
    cols = ["season", "week", "home_team", "away_team", "home_score", "away_score", "game_pk",
            "neutral_site", "start_date"]
    upcoming = pd.DataFrame([{"season": season, "week": week, "home_team": g["home_team"],
                              "away_team": g["away_team"], "home_score": np.nan, "away_score": np.nan,
                              "game_pk": int(g["game_pk"]), "neutral_site": bool(g.get("neutral_site")),
                              "start_date": g.get("start_date") or g.get("commence_time") or ""}
                             for g in games], columns=cols)
    base = before[cols].astype({"home_score": "float64", "away_score": "float64"})
    return pd.concat([base, upcoming], ignore_index=True)
