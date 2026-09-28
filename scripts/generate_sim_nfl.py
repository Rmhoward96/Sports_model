"""Generate NFL sim-engine slate outputs (current slate -> DB).

The NFL analog of generate_sim.py (MLB): for every upcoming NFL game, build a
drive-based Monte Carlo spec from leakage-free nflverse rates, run the sim
kernel once, and derive both the game-level output (home_win_prob/margin/
total + disagreement vs. the analytic model already in predictions_current)
and every player's prop distributions from that single set of sims. Writes to
Supabase `nfl_sim` + `nfl_player_sim` (db/migration_nfl_sim.sql).

PURE / IO split
----------------
`assemble_sim_rows` is the pure seam: raw sim outputs (NflGameSims) + specs
(NflGameSpec, for player identity) + the analytic home_win_prob already on
file -> the exact row shapes `db.upsert_nfl_sim`/`upsert_nfl_player_sim`
expect. It touches no network/DB and is unit-tested with tiny synthetic
NflGameSims/NflGameSpec objects (tests/sim/nfl/test_generate_sim_nfl.py).

`main()` is the thin IO wrapper: pull the slate from `predictions_current`,
fetch nflverse, build rates/players/injuries, build a spec + simulate per
game (wrapped in try/except so one bad game can't abort the whole slate),
then hand everything to `assemble_sim_rows` in one shot and upsert. Before the
per-game loop it also runs `usage.abbrev_alignment` (shared with
`scripts/backtest_sim_nfl.py`) against the slate's crosswalked team abbrevs
and prints a WARN/OK line, and it counts (`n_empty_active`, printed in the
final summary) games where `active_usage` handed back an empty roster for
either side -- both mirror the backtest's visibility into the same silent
abbrev-mismatch failure mode, so it isn't discovered only after this path
goes live. These are warnings, not hard failures: the game still simulates.

Leakage cutoff (current season/week) -- documented heuristic
--------------------------------------------------------------
`sportsmodel.nfl.injuries_nflverse.nfl_season(now)` already encodes this
project's season-year rule (Sept-Dec = that calendar year's season, Jan/Feb
= the previous year's season) and is reused here for `upto_season` rather
than reimplementing it.

`upto_week` is derived from the freshly-fetched pbp itself: one past the
max week that already has ANY pbp rows for `upto_season` (no rows yet =>
week 1, so a build before Week 1 kicks off excludes the whole new season and
`team_rates_from_pbp`/`usage.active_usage` fall back to full prior-season
data instead of an empty current one).

CONCERN: "has pbp rows for week W" is true as soon as the first game of week
W has been played, even if other week-W games haven't kicked off yet (e.g. a
Thursday-night game done, Sunday slate still ahead). A build run mid-week
would then treat week W as "played" (cutoff = W+1) and include week W's
completed game(s) in the leakage-free rates for teams that HAVEN'T played
that week yet -- which is fine (no leakage for THEM) but is a slightly
different cutoff than "the last fully-completed week." A stricter version
would key off the real NFL schedule instead of pbp presence; left as a v1
heuristic since this only affects rate freshness by a few days, never
correctness (the leakage guard itself, in rates.py, is unaffected).

Props-ML serving (SIM_ML_MODE)
------------------------------
`SIM_ML_MODE` = off (default) | shadow | live. `off` imports/loads nothing
ML. Otherwise, AFTER the current sim is simulated and written, `run_ml`
builds the props-ML feature tables with the training builder
(scripts/build_player_features.py `fetch_sources` + `build_tables`, seasons
2016..current, forecast weather filled for upcoming games), loads the
artifacts in data/props_ml/models (with the tables, so a column mismatch is
rejected at load), and per game: `ml_serving.build_ml_spec` -> simulate with
the backtest's per-game seed -> `ml_player_dists` once -> writes nfl_sim +
nfl_player_sim under `nfl-sim-ml-v1` in ONE transaction
(`db.upsert_nfl_sim_slate`). The ML version is a complete slate: its GAME
rows are the CURRENT sim's (props were gated, game predictions were not --
the ML sims feed player dists only; ML_GAME_LINES=on changes this, see
below), its player rows are the ML dists for
`source == "ml"` markets of the player-markets inside the gated population
(`props_eval.gate_population` on the current sim's dist means) and the
current sim's dists for everything else. A game the ML path can't serve
(postseason, missing feature rows, share fallback, any per-game exception)
gets the current sim's rows under `nfl-sim-ml-v1` and is listed on one
`::warning::props-ml:` line; when EVERY REG game falls back (a
postseason-only slate excepted) that is a GLOBAL failure. A GLOBAL ML
failure (artifacts, market_max mismatch, feature build, DB write, every REG
game falling back) prints `ML: FAILED <reason>` and a `::warning::props-ml:`
Actions annotation; shadow then exits 0, live exits 1 (the current sim is
already written either way). An unknown SIM_ML_MODE value is treated like a
live failure. One `ml_mode=... ml_games=<ML-served games>
ml_players=<their player rows> ml_fallback_games=... ml_status=ok|failed
st_nan_questionable=... st_nan_vacated_tgt=... st_nan_vacated_car=...` line
is printed whenever the mode is not off; st_nan_* = the slate weeks' NaN
share of the serve-time injury features (`na` when the tables were not
built). nflverse's weekly injury report posts Wed-Fri: when it has no rows
for the live report's target week, the live (`current_report`) statuses are
injected into the feature build first (names -> gsis via the target week's
as-of depth chart; unmapped names are counted and printed).

The SERVED version decides what must be written (`nfl_sim_serving`, read
once after the current sim is written -- `db.served_nfl_sim_version`):
  * table missing (None): the *_current views are unfiltered, so anything
    under `nfl-sim-ml-v1` would be served live -- shadow/live are a GLOBAL
    failure ("nfl_sim_serving missing ...") that writes NOTHING under the ML
    version; off is unchanged.
  * `sim-nfl-v1`: shadow/live write the ML slate best-effort; a global
    failure writes nothing under the ML version; off writes nothing ML.
  * `nfl-sim-ml-v1`: an ML-version slate is written EVERY run in every mode
    -- ML where it succeeds, else a full copy of the current sim's rows (a
    global failure writes the copy, then applies the exit semantics); off
    writes the copy and prints a rollback `::warning::`.
  * the read itself fails: shadow/live are a global failure writing nothing;
    off prints a `::warning::` and exits 1 (the served slate may be stale).

ML game lines (ML_GAME_LINES)
-----------------------------
`ML_GAME_LINES` = off (default) | on; anything else is off plus a
`::warning::ml-game-lines:` line. `off` is exactly the behavior above (no
game_predictions writes; the ML version's nfl_sim game rows are the current
sim's). `on` makes the ML sim the NFL game-line source (generate-nfl is
disabled and injury-watch skips generate_nfl.py):
  * the slate is predictions_current PLUS ESPN's target week
    (`merge_espn_slate`; generate_nfl's source): with generate-nfl disabled
    nothing else adds a new week's games. ESPN also supplies commence_time /
    game_date (a listed game that ESPN no longer reports scheduled after now
    is dropped: no post-kickoff re-sim rows) and market_spread/market_total
    (carried from predictions_current when ESPN has no line). A new ESPN
    game has no analytic prediction, so its nfl_sim rows carry `disagreement`
    None; if its sim fails it is a `::warning::ml-game-lines:` line (it has
    no row at all). An ESPN failure warns and keeps the predictions_current
    slate (kickoffs and market lines as served).
  * only while the served version is `nfl-sim-ml-v1`: an ML-served game's
    nfl_sim ML-version game row comes from its ML sims (reverses Props-2 I3
    for the switched state), and every game gets a `game_predictions` row
    under `nfl-sim-ml-v1` in `generate_nfl.build_game_row`'s exact shape
    (`game_prediction_row`) -- from the ML sims when the game was ML-served,
    else from the current sim (per-game fallback, global ML failure, or
    SIM_ML_MODE=off), so the slate stays complete. A failed game_predictions
    write is a `::warning::ml-game-lines:` line and a red run (exit 1).
  * on while the served version is anything else: a `::warning::ml-game-lines:`
    line and no game_predictions writes.

Usage:
    PYTHONPATH=src uv run python scripts/generate_sim_nfl.py
    SIM_ML_MODE=shadow PYTHONPATH=src uv run python scripts/generate_sim_nfl.py

Requires DATABASE_URL (Supabase) and nflverse network access; not run here.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sportsmodel import config
from sportsmodel.model.game_gate import win_prob_pmf
from sportsmodel.model.props_eval import gate_population
from sportsmodel.db import (get_postgres, served_nfl_sim_version, upsert_game_predictions,
                            upsert_nfl_player_sim, upsert_nfl_sim, upsert_nfl_sim_slate)
from sportsmodel.nfl import config as nfl_config
from sportsmodel.nfl.injuries_nflverse import nfl_season
from sportsmodel.nfl.injury_report import current_report, live_injury_rows, resolve_target_week
from sportsmodel.nfl.nflverse import load_release
from sportsmodel.nfl.teams import TEAMS, normalize_team
from sportsmodel.sim.engine import margin_pmf, pred_scores, stat_pmf, total_pmf
from sportsmodel.sim.nfl.aggregate import disagreement, nfl_player_prop_dists
from sportsmodel.sim.nfl.inputs import build_spec_from_usage
from sportsmodel.sim.nfl.kernel import simulate_game
from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.sim.nfl.rates import (fetch_nflverse, ratings_tilt, team_rates_from_pbp,
                                       team_defense_rates_from_pbp)
from sportsmodel.sim.nfl.usage import (
    abbrev_alignment,
    active_usage,
    build_pfr_to_gsis,
    depth_charts_asof,
    fetch_usage_sources,
)

MODEL_VERSION = "sim-nfl-v1"
DEFAULT_N_SIMS = 10_000
SIM_SEED = 42

# Seasons pulled for rates/usage: the current season + this many prior. Set to
# 1 (current + the previous season only) -- NFL rosters/schemes turn over fast,
# so 2+-year-old play-by-play is stale; the previous season stabilizes the
# early-season weeks until the current season accrues games. Combined with
# SEASON_DECAY below, which then leans the blend toward the current season.
FETCH_SEASONS_BACK = 1

# Season recency weighting for team_rates_from_pbp: the previous season is
# down-weighted by this factor (current=1.0, previous=DECAY), so this season's
# play-by-play counts ~2.5x heavier while the previous season still stabilizes
# the early weeks. 1.0 = equal weighting. Default 0.4 is the walk-forward
# optimum: validated on 2025 (prev+current window) it beat 1.0/0.8/0.6/0.2 on
# Brier, margin MAE AND total MAE (total ~2% better, 11.19->10.97). Override via
# SIM_SEASON_DECAY to retune. (Player usage is already recency-weighted per-game
# inside usage.active_usage's last-N-games window.)
SEASON_DECAY = float(os.getenv("SIM_SEASON_DECAY", "0.4"))

# Home-field edge: tilts the home offense's per-drive scoring up by (1+HOME_FIELD)
# and the away offense's down by (1-HOME_FIELD). 0.0 = neutral. Default 0.07 is
# the backtested value (2025 walk-forward): it lands the predicted mean home
# margin on the actual +2.07 while margin/total MAE stay flat-optimal. The old
# sim had NO home edge (predicted home margin ~-0.03). Env-tunable via
# SIM_HOME_FIELD.
HOME_FIELD = float(os.getenv("SIM_HOME_FIELD", "0.07"))

# Power-ranking weight: how strongly the Elo power gap tilts sim scoring on top
# of the bottom-up drive rates (0 = off). Default 0.5 is the 2025 walk-forward
# optimum -- it improved BOTH win-prob Brier (0.2316->0.2248) and margin MAE
# (best of the sweep) while widening strong-vs-weak separation (avg predicted
# margin 3.2->5.2 pts). Higher (1.0+) separates more but costs margin accuracy.
# Env-tunable via SIM_RATINGS_WEIGHT. See sim.nfl.rates.ratings_tilt.
RATINGS_WEIGHT = float(os.getenv("SIM_RATINGS_WEIGHT", "0.5"))
_ELO_BASE = EloConfig().base

# Binning ceilings for nfl_player_prop_dists's pmf markets (anytime_td is
# binary and doesn't need one -- see aggregate.nfl_player_prop_dists).
MARKET_MAX = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 200, "receptions": 15, "pass_tds": 6, "rush_att": 40}

TEAMS_CROSSWALK_PATH = config.PROJECT_ROOT / "assets" / "nfl" / "nfl_teams.json"

# nflverse injury `status` values (case-insensitive) treated as ruled out for
# active_usage's per-team injuries_out_names set.
_OUT_STATUSES = frozenset({"out", "doubtful"})

# "questionable" players are NOT ruled out -- they stay active but have their
# volume scaled by QUESTIONABLE_WEIGHT (freed share redistributes to healthy
# teammates). Historically a Questionable tag suppresses a player's expected
# workload modestly; 1.0 disables the adjustment. Env-tunable + backtested.
_QUESTIONABLE_STATUS = "questionable"
QUESTIONABLE_WEIGHT = float(os.getenv("SIM_QUESTIONABLE_WEIGHT", "0.75"))

# Props-ML serving (SIM_ML_MODE): "off" (default) never imports or loads
# anything ML; "shadow" additionally writes the ML slate under
# ML_MODEL_VERSION and exits 0 even when the ML path fails; "live" does the
# same but exits 1 (red run) when the ML path fails. The current sim is
# always simulated and written first, so its rows never depend on the ML path.
ML_MODES = ("off", "shadow", "live")
ML_MODEL_VERSION = "nfl-sim-ml-v1"
ML_MODEL_DIR = config.PROJECT_ROOT / "data" / "props_ml" / "models"
_SCRIPTS_DIR = Path(__file__).resolve().parent

# ML_GAME_LINES (ML-only NFL): "off" (default) = today's behavior; "on" = the
# ML sim is the NFL game-line source (module docstring). Anything else is
# treated as off with a `::warning::ml-game-lines:` line.
ML_GAME_LINES_VALUES = ("on", "off")
_LINES_WARN = "::warning::ml-game-lines: "


# =============================================================================
# PURE seam
# =============================================================================

def assemble_sim_rows(
    games: list[dict],
    sims_by_game: dict[Any, Any],
    specs_by_game: dict[Any, Any],
    analytic_by_game: dict[Any, float],
    model_version: str = MODEL_VERSION,
    market_max: dict[str, int] | None = None,
    dists_by_game: dict[Any, dict] | None = None,
    allow_missing_analytic: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Turn raw per-game sim outputs into (nfl_sim rows, nfl_player_sim rows). PURE.

    Args:
        games: One dict per game: {"game_pk", "matchup" ("Away @ Home"),
            "commence_time", "home_team", "away_team"}. `home_team`/
            `away_team` are display names (whatever `matchup` was built
            from) -- used ONLY to label player rows' `team` field, never to
            key into `sims_by_game`/`specs_by_game` (those are keyed by
            `game_pk`).
        sims_by_game: {game_pk -> NflGameSims} from `kernel.simulate_game`.
        specs_by_game: {game_pk -> NflGameSpec} -- the spec each game's sims
            were built from. Needed because NflGameSims only carries
            player_id -> stat arrays; player identity (name/pos/which team)
            comes from spec.home_players/away_players.
        analytic_by_game: {game_pk -> home_win_prob} from the analytic model
            already on file in predictions_current, for the disagreement
            metric.
        model_version: Written into every row (default "sim-nfl-v1").
        market_max: Passed to `aggregate.nfl_player_prop_dists`'s pmf
            binning; defaults to module-level MARKET_MAX.
        dists_by_game: Optional {game_pk -> {player_id -> {market -> dist}}}
            used for that game's player rows INSTEAD of
            `nfl_player_prop_dists(sims)` (the props-ML slate: ML markets +
            the current sim's baseline markets -- see `merge_ml_dists`). Game
            rows still come from `sims_by_game`.
        allow_missing_analytic: ML_GAME_LINES=on only -- a game whose
            analytic home_win_prob is None (a new ESPN game with no
            predictions_current row) is written with `disagreement` None
            instead of being skipped. A game absent from `analytic_by_game`
            is still skipped.

    Returns:
        (nfl_sim_rows, nfl_player_sim_rows) matching `db.upsert_nfl_sim` /
        `db.upsert_nfl_player_sim`'s expected columns exactly (`dist` as a
        plain dict -- upsert_nfl_player_sim does the json.dumps).

    A game missing from `sims_by_game`, `specs_by_game`, or
    `analytic_by_game` (or None there, unless `allow_missing_analytic`) is
    silently skipped (documents the contract: `main()`
    only adds a game to those three dicts once its sim has actually
    succeeded, so a game dropped upstream on a per-game try/except never
    reaches here as a partial/crashing entry). A player_id present in a
    game's sims but absent from that game's spec rosters (shouldn't happen --
    simulate_game only ever simulates spec.home_players/away_players -- but
    defensive nonetheless) is likewise skipped rather than written with no
    name/pos/team identity.
    """
    market_max = MARKET_MAX if market_max is None else market_max
    sim_rows: list[dict] = []
    player_rows: list[dict] = []

    for g in games:
        game_pk = g["game_pk"]
        sims = sims_by_game.get(game_pk)
        spec = specs_by_game.get(game_pk)
        analytic_home_win_prob = analytic_by_game.get(game_pk)
        if sims is None or spec is None:
            continue
        if analytic_home_win_prob is None and not (allow_missing_analytic and game_pk in analytic_by_game):
            continue

        scores = pred_scores(sims)
        sim_home_win_prob = scores["home_win_prob"]
        sim_rows.append({
            "game_pk": game_pk,
            "model_version": model_version,
            "matchup": g["matchup"],
            "commence_time": g["commence_time"],
            "sim_home_win_prob": sim_home_win_prob,
            "sim_margin": scores["pred_margin"],
            "sim_total": scores["pred_total"],
            "disagreement": (None if analytic_home_win_prob is None
                             else disagreement(analytic_home_win_prob, sim_home_win_prob)),
            # Full simulated distributions for the game-page histograms
            # (Spread / Total / each team's total). NflGameSims duck-types
            # GameSims (home_score/away_score arrays) for these helpers.
            "margin_dist": margin_pmf(sims, half_range=45),
            "total_dist": {"kind": "pmf", "pmf": total_pmf(sims, max_total=90)},
            "away_score_dist": {"kind": "pmf", "pmf": stat_pmf(sims.away_score, 70)},
            "home_score_dist": {"kind": "pmf", "pmf": stat_pmf(sims.home_score, 70)},
        })

        identity_by_player_id: dict[str, tuple[str, str, str]] = {
            p.player_id: (p.name, p.pos, g["home_team"]) for p in spec.home_players
        }
        identity_by_player_id.update(
            {p.player_id: (p.name, p.pos, g["away_team"]) for p in spec.away_players}
        )

        if dists_by_game is not None and game_pk in dists_by_game:
            dists = dists_by_game[game_pk]
        else:
            dists = nfl_player_prop_dists(sims, market_max)
        for player_id, markets in dists.items():
            identity = identity_by_player_id.get(player_id)
            if identity is None:
                continue
            name, pos, team = identity
            for market, dist in markets.items():
                player_rows.append({
                    "game_pk": game_pk,
                    "player_id": player_id,
                    "model_version": model_version,
                    "name": name,
                    "pos": pos,
                    "team": team,
                    "market": market,
                    "mean": dist["mean"],
                    "dist": dist,
                    "commence_time": g["commence_time"],
                })

    return sim_rows, player_rows


def merge_ml_dists(current: dict[str, dict], ml: dict[str, dict],
                   sources: dict[str, str], gate: set[tuple[str, str]]) -> tuple[dict[str, dict], int]:
    """One game's complete props-ML player slate. PURE.

    `current` / `ml`: {player_id -> {market -> dist}} from the current sim
    (`nfl_player_prop_dists`) and from `ml_serving.ml_player_dists`;
    `sources`: {market -> "ml" | "baseline"} from the artifacts config;
    `gate`: the gated (player_id, market)s -- `props_eval.gate_population` of
    the CURRENT sim's dist means (the population the ML pipeline was gated
    on). A player-market takes the ML dist only when its source is "ml" AND
    it is gated; everything else (source "baseline", not in the config, or
    outside the gate) takes the CURRENT sim's dist -- never the ML sims'
    unmapped draws. A gated "ml" player-market with no ML dist falls back to
    the current dist and is counted (second return value). The slate is
    exactly the current sim's (player, market) set: ML dists for
    player-markets absent from `current` are dropped.
    """
    out: dict[str, dict] = {}
    n_fallback = 0
    for pid, markets in current.items():
        for m, dist in markets.items():
            if sources.get(m) == "ml" and (pid, m) in gate:
                ml_dist = ml.get(pid, {}).get(m)
                if ml_dist is None:
                    n_fallback += 1
                    ml_dist = dist
                out.setdefault(pid, {})[m] = ml_dist
            else:
                out.setdefault(pid, {})[m] = dist
    return out, n_fallback


def gated_player_markets(current: dict[str, dict]) -> set[tuple[str, str]]:
    """{(player_id, market)} of one game inside the props-ML population:
    `props_eval.gate_population` on the CURRENT sim's dist means. PURE."""
    return gate_population(((pid, m), d.get("mean"))
                           for pid, markets in current.items() for m, d in markets.items())


def game_prediction_row(game: dict, sims, gl_cfg) -> dict:
    """One game's sims -> a `game_predictions` row in EXACTLY the shape
    `generate_nfl.build_game_row` / `gameline.build_gameline` emit. PURE.

    `margin_dist` = {"kind": "margin", "offset": gl_cfg.offset, "pmf": [...]}
    with pmf[i] = P(margin == i - offset), length 2*offset+1 -- `engine.margin_pmf`
    already uses gameline's offset convention, so half_range=gl_cfg.offset is
    the whole conversion (the sims' tails beyond +-offset are clipped onto the
    end bins, where the Normal version truncates). `total_dist` =
    {"kind": "pmf", "pmf": [...]} on totals 0..gl_cfg.total_max (NOT the
    engine's default max_total=30, which would pile every NFL total onto the
    last bin). `home_win_prob` = P(margin > 0) + 0.5 P(margin == 0) (the sim
    has no overtime, so ties are split -- the game gate's definition);
    pred_* are the sims' means. The dists are plain dicts: `main()` JSON-encodes
    them at the DB boundary, as generate_nfl does. `game` carries
    game_pk/commence_time/home_team/away_team (display names) and, from
    `merge_espn_slate`, game_date/market_spread/market_total."""
    margin_dist = margin_pmf(sims, half_range=gl_cfg.offset)
    total_dist = {"kind": "pmf", "pmf": total_pmf(sims, max_total=gl_cfg.total_max)}
    scores = pred_scores(sims)
    return {
        "margin_dist": margin_dist,
        "total_dist": total_dist,
        "home_win_prob": win_prob_pmf(margin_dist["pmf"], -margin_dist["offset"]),
        "pred_margin": scores["pred_margin"],
        "pred_total": scores["pred_total"],
        "pred_home_score": scores["pred_home_score"],
        "pred_away_score": scores["pred_away_score"],
        "sport": "nfl",
        "model_version": ML_MODEL_VERSION,
        "game_pk": game["game_pk"],
        "game_date": game.get("game_date"),
        "commence_time": game["commence_time"],
        "market_spread": game.get("market_spread"),
        "market_total": game.get("market_total"),
        "home_team_name": game["home_team"],
        "away_team_name": game["away_team"],
    }


def _as_utc(commence) -> datetime:
    """A kickoff (ESPN ISO string or DB datetime) as an aware datetime."""
    if isinstance(commence, datetime):
        return commence if commence.tzinfo else commence.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _game_date(commence) -> str:
    """US game date of a kickoff: generate_nfl's `_game_date_from_commence`
    rule (UTC - 8h), for strings and DB datetimes alike. PURE."""
    return (_as_utc(commence) - timedelta(hours=8)).date().isoformat()


def merge_espn_slate(games: list[dict], espn_games: list[dict], now: datetime) -> list[dict]:
    """The ML_GAME_LINES=on slate. PURE.

    `games`: `_load_upcoming_games()` (predictions_current, market lines
    included); `espn_games`: `espn.parse_schedule` rows of the target week --
    the source generate_nfl uses. A predictions_current game ESPN lists takes
    ESPN's `commence_time` (and `game_date` from it), as generate_nfl's daily
    rewrite did, and is dropped unless ESPN still reports it STATUS_SCHEDULED
    with a kickoff after `now` (a moved-up / started / postponed game must not
    get a post-kickoff re-sim row). A game ESPN does not list keeps its
    predictions_current kickoff. `market_spread`/`market_total` are ESPN's,
    each carried from predictions_current when ESPN has no row / no line
    (so an ESPN outage keeps the served lines). An ESPN game missing from
    predictions_current that is scheduled after `now` is appended (display
    names, `home_win_prob` None = no analytic prediction): with generate-nfl
    disabled nothing else adds a new week's games to predictions_current, so
    without this the slate would never advance."""
    def upcoming(e):
        return e.get("status") == "STATUS_SCHEDULED" and _as_utc(e["commence_time"]) > now

    def line(e, g, key):
        return e.get(key) if e.get(key) is not None else g.get(key)

    by_pk = {int(e["game_pk"]): e for e in espn_games}
    out = []
    for g in games:
        e = by_pk.get(int(g["game_pk"]))
        if e is not None and not upcoming(e):
            continue
        commence = e["commence_time"] if e is not None else g["commence_time"]
        e = e or {}
        out.append({**g, "commence_time": commence, "game_date": _game_date(commence),
                    "market_spread": line(e, g, "market_spread"),
                    "market_total": line(e, g, "market_total")})
    have = {int(g["game_pk"]) for g in games}
    for e in espn_games:
        home, away = e.get("home_name"), e.get("away_name")
        if int(e["game_pk"]) in have or not home or not away or not upcoming(e):
            continue
        out.append({"game_pk": int(e["game_pk"]), "matchup": f"{away} @ {home}",
                    "commence_time": e["commence_time"], "home_team": home, "away_team": away,
                    "home_win_prob": None, "game_date": _game_date(e["commence_time"]),
                    "market_spread": e.get("market_spread"), "market_total": e.get("market_total")})
    return out


def _ml_game_lines() -> str:
    """ML_GAME_LINES, normalized to on|off; anything else is off + a warning."""
    raw = os.environ.get("ML_GAME_LINES", "off")
    value = raw.strip().lower() or "off"
    if value in ML_GAME_LINES_VALUES:
        return value
    print(f"{_LINES_WARN}unknown ML_GAME_LINES={raw!r} (expected on|off); treated as off", flush=True)
    return "off"


def _norm_team(code) -> str | None:
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def _schedule_week(sched: pd.DataFrame, season: int, home: str, away: str) -> int:
    """The week of `season`'s REG game `away` @ `home` (normalized team
    codes) in the nflverse schedule. PURE. ValueError unless exactly one
    game matches (the props-ML feature tables only cover REG weeks)."""
    g = sched[(sched["season"] == season) & (sched["game_type"] == "REG")]
    hit = g[(g["home_team"].map(_norm_team) == home) & (g["away_team"].map(_norm_team) == away)]
    if len(hit) != 1:
        raise ValueError(f"{len(hit)} REG schedule rows for {season} {away}@{home}")
    return int(hit["week"].iloc[0])


def _is_postseason_game(sched: pd.DataFrame, season: int, home: str, away: str) -> bool:
    """True when `season`'s schedule has `away` @ `home` as a non-REG
    (postseason) game. PURE."""
    g = sched[(sched["season"] == season) & (sched["game_type"] != "REG")]
    hit = g[(g["home_team"].map(_norm_team) == home) & (g["away_team"].map(_norm_team) == away)]
    return len(hit) > 0


def _ml_mode() -> str:
    """SIM_ML_MODE, normalized; ValueError for anything but off/shadow/live."""
    mode = os.environ.get("SIM_ML_MODE", "off").strip().lower() or "off"
    if mode not in ML_MODES:
        raise ValueError(f"unknown SIM_ML_MODE={mode!r} (expected one of {', '.join(ML_MODES)})")
    return mode


# =============================================================================
# IO main()
# =============================================================================

def _load_crosswalk() -> dict[str, str]:
    """{ESPN full team name -> nflverse abbrev}, inverted from nfl_teams.json
    (which maps abbrev -> full name -- see module docstrings in
    injuries_nflverse.py / build_nfl_teams.py)."""
    import json

    raw = json.loads(TEAMS_CROSSWALK_PATH.read_text())
    return {full_name: abbrev for abbrev, full_name in raw.items()}


def _opt_float(x) -> float | None:
    """A nullable numeric column (psycopg may hand back Decimal) as float|None."""
    return None if x is None else float(x)


def _load_upcoming_games() -> list[dict]:
    """Upcoming NFL games from predictions_current. `home_team`/`away_team`
    here are ESPN display names (as stored in predictions_current), NOT
    nflverse abbreviations -- `main()` maps them through `_load_crosswalk()`
    before touching rates/players/injuries."""
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT game_pk, home_team_name, away_team_name, home_win_prob, commence_time,
                   market_spread, market_total
            FROM predictions_current
            WHERE sport = 'nfl' AND commence_time > now()
        """)
        cols = ["game_pk", "home_team_name", "away_team_name", "home_win_prob", "commence_time",
                "market_spread", "market_total"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return [
        {
            "game_pk": r["game_pk"],
            "matchup": f"{r['away_team_name']} @ {r['home_team_name']}",
            "commence_time": r["commence_time"],
            "home_team": r["home_team_name"],
            "away_team": r["away_team_name"],
            "home_win_prob": r["home_win_prob"],
            # carried forward by merge_espn_slate when ESPN has no line (ML_GAME_LINES=on)
            "market_spread": _opt_float(r["market_spread"]),
            "market_total": _opt_float(r["market_total"]),
        }
        for r in rows
    ]


def _load_espn_slate() -> list[dict]:
    """ESPN's target-week NFL schedule (`espn.parse_schedule` rows, market
    lines included) -- what generate_nfl.py prices. IO (network)."""
    from sportsmodel.nfl import espn

    tw = espn.resolve_target_week()
    return espn.fetch_schedule(int(tw["season"]), int(tw["week"]), season_type=int(tw["season_type"]))


def _write_game_lines(games: list[dict], sims_for_game: dict, ml_served: set) -> None:
    """ML_GAME_LINES=on: one `game_predictions` row per game in `games` with
    sims (`game_prediction_row`, dists JSON-encoded at this boundary like
    generate_nfl's main), under ML_MODEL_VERSION, in one upsert. `ml_served`:
    the game_pks whose sims are the ML sims (the rest are the current sim's)."""
    gl_cfg = nfl_config.load_gameline()
    rows = []
    for g in games:
        sims = sims_for_game.get(g["game_pk"])
        if sims is None:
            continue
        row = game_prediction_row(g, sims, gl_cfg)
        rows.append({**row, "margin_dist": json.dumps(row["margin_dist"]),
                     "total_dist": json.dumps(row["total_dist"])})
    upsert_game_predictions(rows)
    n_ml = sum(1 for r in rows if r["game_pk"] in ml_served)
    print(f"game_lines: wrote {len(rows)} game_predictions rows under {ML_MODEL_VERSION} "
          f"(ml_sim={n_ml} current_sim={len(rows) - n_ml})", flush=True)


def _determine_upto_week(pbp_df, upto_season: int) -> int:
    """One past the max week already played in `upto_season`'s freshly-fetched
    pbp (no rows yet for that season => week 1). See module docstring."""
    season_rows = pbp_df[pbp_df["season"] == upto_season]
    if len(season_rows) == 0:
        return 1
    return int(season_rows["week"].max()) + 1


def _out_names_by_team(injuries: dict[str, list[dict]]) -> dict[str, set[str]]:
    """{team_abbrev -> {lowercased player name}} for Out/Doubtful entries,
    the shape `usage.active_usage` expects for `injuries_out_names`."""
    out: dict[str, set[str]] = {}
    for team, entries in injuries.items():
        out[team] = {
            str(entry["player"]).strip().lower()
            for entry in entries
            if str(entry.get("status", "")).strip().lower() in _OUT_STATUSES
        }
    return out


def _questionable_names_by_team(injuries: dict[str, list[dict]]) -> dict[str, set[str]]:
    """{team_abbrev -> {lowercased player name}} for Questionable entries, for
    active_usage's `questionable_names` (down-weighted, not dropped)."""
    out: dict[str, set[str]] = {}
    for team, entries in injuries.items():
        out[team] = {
            str(entry["player"]).strip().lower()
            for entry in entries
            if str(entry.get("status", "")).strip().lower() == _QUESTIONABLE_STATUS
        }
    return out


def _injury_summary(report: dict) -> str:
    """One log line describing the shared injury report's freshness."""
    return (f"injuries: report_week={report.get('report_week')} "
            f"target_week={report.get('target_week')} stale={report.get('stale')} "
            f"espn={'OK' if report.get('espn_available') else 'UNAVAILABLE'} "
            f"conflicts={len(report.get('conflicts') or [])}")


def _load_script(name: str):
    """A sibling script as a module (the repo's importlib pattern for scripts)."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ST_NAN_COLS = ("st_questionable", "st_vacated_tgt", "st_vacated_car")


def _inject_live_injuries(injuries: pd.DataFrame | None, depth: pd.DataFrame, report: dict | None,
                          season: int) -> tuple[pd.DataFrame | None, str]:
    """The builder's nflverse injury frame, with the live report's statuses
    added for (season, report["target_week"]) when nflverse has NO rows for
    that week yet (its weekly report posts Wed-Fri; without this the target
    week's st_questionable / st_vacated_* would be NaN = "no report"). Names
    map to gsis ids through the target week's as-of depth chart
    (`injury_report.live_injury_rows`); unmapped names are counted and named
    in the returned log line. A posted nflverse week (or no report / target
    week) returns `injuries` itself unchanged. PURE."""
    week = None if report is None else report.get("target_week")
    if week is None:
        return injuries, "props-ML injuries: no target week in the live report; nflverse frame as is"
    week = int(week)
    have = (injuries is not None and len(injuries) > 0
            and bool(((injuries["season"] == season) & (injuries["week"] == week)).any()))
    if have:
        return injuries, f"props-ML injuries: nflverse report present for {season} week {week}"
    live, unmapped = live_injury_rows(report.get("by_team") or {}, depth, season, week)
    frame = live if injuries is None or not len(injuries) else pd.concat([injuries, live], ignore_index=True)
    msg = (f"props-ML injuries: nflverse has no {season} week {week} rows: injected {len(live)} live "
           f"statuses (unmapped {len(unmapped)}" + (f": {', '.join(unmapped)}" if unmapped else "") + ")")
    return frame, msg


def _st_nan_shares(feats: pd.DataFrame, season: int, weeks: set[int]) -> dict[str, float | None]:
    """NaN share of each ST_NAN_COLS column over the (season, week in weeks)
    player rows; None when there are no such rows or the column is absent. PURE."""
    rows = feats[(feats["season"] == season) & feats["week"].isin(sorted(weeks))] if weeks else feats.iloc[:0]
    return {c: (float(rows[c].isna().mean()) if c in rows.columns and len(rows) else None)
            for c in ST_NAN_COLS}


def _build_ml_tables(upto_season: int, now: datetime, report: dict | None = None
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(player table, team table, nflverse schedule) for serving, built by the
    SAME builder as training (scripts/build_player_features.py): seasons
    SEASONS[0]..upto_season, active-but-no-snap stubs for every scheduled REG
    team-week (incl. the target week), and upcoming games' NaN weather filled
    from the kickoff-hour forecast. When nflverse has no injury rows for the
    live report's target week yet, the live statuses are injected first
    (`_inject_live_injuries`). Features are strictly pre-game by
    construction (rolling features read only earlier weeks). IO (network)."""
    from sportsmodel.nfl.context import fetch_hourly_forecast, fill_forecast_weather

    bpf = _load_script("build_player_features")
    seasons = list(range(bpf.SEASONS[0], upto_season + 1))
    print(f"props-ML: building feature tables for seasons {seasons[0]}-{seasons[-1]}", flush=True)
    src = bpf.fetch_sources(seasons)
    src["injuries"], msg = _inject_live_injuries(src.get("injuries"), src.get("depth"), report, upto_season)
    print(msg, flush=True)
    ts = pd.Timestamp(now)

    def ctx_fill(ctx, stadiums, sched):
        return fill_forecast_weather(ctx, stadiums, sched, fetch_hourly_forecast, now=ts)

    built = bpf.build_tables(src, ctx_fill=ctx_fill)
    return built["feats"], built["team"], src["sched"]


class _MlGameFallback(Exception):
    """A game the ML path cannot serve (it gets the current sim's rows)."""


class _MlPostseason(_MlGameFallback):
    """A postseason game (the props-ML feature tables cover REG weeks only).
    A slate where only postseason games fall back is not a global failure."""


SERVING_MISSING = "nfl_sim_serving missing — run db/migration_nfl_sim_serving.sql"
OFF_SERVED_ML_WARNING = (
    "::warning::props-ml: SIM_ML_MODE=off but the site serves nfl-sim-ml-v1 — served the current "
    "sim under the ML version; UPDATE nfl_sim_serving back to sim-nfl-v1 to roll back")


def _market_max_mismatches(config: dict) -> list[str]:
    """ML-sourced markets whose artifacts-config `market_max` differs from
    this script's MARKET_MAX (the binning of the current sim's dists, which
    the ML slate mixes in). `anytime_td` is binary (no MARKET_MAX entry) and
    exempt. PURE. Sorted "market: config X vs sim Y" strings."""
    out = []
    for m, v in sorted(config["markets"].items()):
        if v["source"] != "ml" or m == "anytime_td":
            continue
        cfg_max = config["market_max"].get(m)
        if cfg_max is None or MARKET_MAX.get(m) is None or int(cfg_max) != int(MARKET_MAX[m]):
            out.append(f"{m}: config {cfg_max} vs sim {MARKET_MAX.get(m)}")
    return out


def _missing_feature_rows(spec, home: str, away: str, p_rows: pd.DataFrame,
                          t_rows: pd.DataFrame) -> str | None:
    """What the game's feature rows lack, or None when complete: both teams'
    team rows and a player row for every active spec player. PURE."""
    have_teams = set(t_rows["team"].astype(str))
    have_players = set(p_rows["player_id"].astype(str))
    parts = []
    teams = [t for t in (home, away) if t not in have_teams]
    if teams:
        parts.append("team rows: " + ", ".join(teams))
    players = [p.player_id for p in (*spec.home_players, *spec.away_players)
               if str(p.player_id) not in have_players]
    if players:
        parts.append("players: " + ", ".join(players))
    return "; ".join(parts) or None


def _ml_game(game_pk, *, game_keys, specs_by_game, sims_by_game, feats, team, sched, artifacts,
             sources, upto_season, n_sims, ml_serving, game_seed):
    """(one game's merged player dists, the ML sims). The caller decides
    whether the ML sims also feed the game row (ML_GAME_LINES=on with the ML
    version served) or only the player dists. Raises `_MlGameFallback` (or
    anything else) when the game can't be ML-served."""
    home_abbrev, away_abbrev, rtilt = game_keys[game_pk]
    # normalized codes, as the feature tables / schedule / backtest seeds use
    home, away = _norm_team(home_abbrev) or home_abbrev, _norm_team(away_abbrev) or away_abbrev
    try:
        week = _schedule_week(sched, upto_season, home, away)
    except ValueError as exc:
        if _is_postseason_game(sched, upto_season, home, away):
            raise _MlPostseason(f"not a REG game ({exc})") from exc
        raise _MlGameFallback(f"no REG schedule row ({exc})") from exc
    spec = specs_by_game[game_pk]
    teams = (home, away)
    p_rows = feats[(feats["season"] == upto_season) & (feats["week"] == week) & feats["team"].isin(teams)]
    t_rows = team[(team["season"] == upto_season) & (team["week"] == week) & team["team"].isin(teams)]
    missing = _missing_feature_rows(spec, home, away, p_rows, t_rows)
    if missing:
        raise _MlGameFallback(f"missing feature rows ({missing})")
    before = artifacts.learned.share_fallbacks
    spec_ml = ml_serving.build_ml_spec(spec, artifacts, p_rows, t_rows)
    if artifacts.learned.share_fallbacks > before:
        raise _MlGameFallback("share fallback")
    game_rng = np.random.default_rng(game_seed(SIM_SEED, upto_season, week, home, away))
    sims_ml = simulate_game(spec_ml, n_sims, game_rng, home_field=HOME_FIELD, ratings_tilt=rtilt)
    current = nfl_player_prop_dists(sims_by_game[game_pk], MARKET_MAX)
    gate = gated_player_markets(current)
    ml = ml_serving.ml_player_dists(spec_ml, sims_ml, p_rows, t_rows, artifacts, game_rng, gate=gate)
    merged, n_fallback = merge_ml_dists(current, ml, sources, gate)
    if n_fallback:
        print(f"WARN props-ML: game_pk={game_pk}: {n_fallback} ML-sourced player-markets had no "
              f"ML dist; served from the current sim")
    return merged, sims_ml


def run_ml(*, ok_games: list[dict], game_keys: dict[Any, tuple[str, str, float]],
           specs_by_game: dict, sims_by_game: dict, analytic_by_game: dict,
           upto_season: int, n_sims: int, now: datetime, report: dict | None = None,
           diag: dict | None = None, game_lines: bool = False, game_sims: dict | None = None,
           ml_served: set | None = None, allow_missing_analytic: bool = False) -> tuple[int, int, int]:
    """The props-ML slate for the games the current sim produced; writes it
    under ML_MODEL_VERSION in ONE transaction (`upsert_nfl_sim_slate`) and
    returns (ml_games, ml_players, ml_fallback_games): the ML-SERVED games,
    their player rows, and the games served from the current sim.

    GLOBAL failures raise (`main` applies the served-version + SIM_ML_MODE
    failure semantics): artifacts missing/incompatible, a config market_max
    that differs from MARKET_MAX, the feature build, EVERY REG game falling
    back (a postseason-only slate excepted), the DB write. Every game is
    computed before anything is written, so a global failure writes no
    partial ML slate.

    PER-GAME failures fall back: a game that is not a REG game (postseason),
    lacks feature rows (team rows for both teams, a row per active player),
    trips a learned share fallback, or raises anywhere in its ML work gets
    the CURRENT sim's nfl_player_sim rows under ML_MODEL_VERSION (the ML
    slate always covers every current-sim game), is counted, and all such
    games are listed on one `::warning::props-ml:` line.

    Per ML game: `ml_serving.build_ml_spec` on the current spec -> simulate
    with a per-game rng (`backtest_sim_nfl.game_seed`, as the backtest /
    training runs) -> `ml_player_dists` exactly once -> the complete player
    slate from `merge_ml_dists`. Every game's nfl_sim row comes from the
    CURRENT sim (`sims_by_game`): the gate covered props, not game outputs --
    unless `game_lines` (ML_GAME_LINES=on with the ML version served), where an
    ML-served game's nfl_sim row comes from its ML sims (a fallback game's from
    the current sim). `game_sims` / `ml_served` (filled in place only after
    the slate is written, for the game_predictions write): {game_pk -> the
    sims its nfl_sim game row came from} / the game_pks whose game row came
    from ML sims. `allow_missing_analytic`: passed to `assemble_sim_rows`.
    `game_keys`: {game_pk -> (home_abbrev, away_abbrev, ratings_tilt)}.
    `report`: the live injury report (`current_report`), for the feature
    build's target-week injury injection. `diag` (filled in place once the
    tables are built): {"st_nan": `_st_nan_shares` over the slate's REG
    weeks} for the summary line.
    """
    from sportsmodel.model.props_ml.artifacts import CONFIG_FILE
    from sportsmodel.sim.nfl import ml_serving

    if not (ML_MODEL_DIR / CONFIG_FILE).is_file():   # fail before the (minutes-long) feature build
        raise RuntimeError(f"props-ML artifacts missing: no {CONFIG_FILE} in {ML_MODEL_DIR}")
    bsn = _load_script("backtest_sim_nfl")
    feats, team, sched = _build_ml_tables(upto_season, now, report=report)
    if diag is not None:
        weeks = set()
        for g in ok_games:
            home_abbrev, away_abbrev, _ = game_keys[g["game_pk"]]
            try:
                weeks.add(_schedule_week(sched, upto_season, _norm_team(home_abbrev) or home_abbrev,
                                         _norm_team(away_abbrev) or away_abbrev))
            except ValueError:
                pass
        diag["st_nan"] = _st_nan_shares(feats, upto_season, weeks)
    artifacts = ml_serving.load_artifacts(ML_MODEL_DIR, feats, team)
    if artifacts is None:
        raise RuntimeError(f"props-ML artifacts missing or incompatible in {ML_MODEL_DIR}")
    mismatches = _market_max_mismatches(artifacts.config)
    if mismatches:
        raise RuntimeError("props-ML market_max mismatch with the sim's MARKET_MAX: " + "; ".join(mismatches))
    sources = {m: str(v["source"]) for m, v in artifacts.config["markets"].items()}

    dists_by_game: dict = {}
    ml_sims_by_game: dict = {}
    fallbacks: list[tuple[Any, str]] = []
    n_postseason = 0
    for g in ok_games:
        game_pk = g["game_pk"]
        try:
            dists_by_game[game_pk], ml_sims_by_game[game_pk] = _ml_game(
                game_pk, game_keys=game_keys, specs_by_game=specs_by_game, sims_by_game=sims_by_game,
                feats=feats, team=team, sched=sched, artifacts=artifacts, sources=sources,
                upto_season=upto_season, n_sims=n_sims, ml_serving=ml_serving, game_seed=bsn.game_seed)
        except Exception as exc:  # noqa: BLE001 -- per-game: serve this game from the current sim
            reason = str(exc) if isinstance(exc, _MlGameFallback) else f"{type(exc).__name__}: {exc}"
            n_postseason += isinstance(exc, _MlPostseason)
            fallbacks.append((game_pk, " ".join(reason.split())))
            dists_by_game.pop(game_pk, None)   # no dists override -> current sim's dists
            ml_sims_by_game.pop(game_pk, None)
    listed = "; ".join(f"{pk}: {why}" for pk, why in fallbacks)
    if not dists_by_game and len(ok_games) > n_postseason:
        raise RuntimeError(f"every REG game fell back to the current sim "
                           f"({len(fallbacks)} game(s)): {listed}")
    if fallbacks:
        print(f"::warning::props-ml: {len(fallbacks)} game(s) served from the current sim: {listed}",
              flush=True)

    row_sims = {**sims_by_game, **ml_sims_by_game} if game_lines else sims_by_game
    sim_rows, player_rows = assemble_sim_rows(ok_games, row_sims, specs_by_game, analytic_by_game,
                                              model_version=ML_MODEL_VERSION, dists_by_game=dists_by_game,
                                              allow_missing_analytic=allow_missing_analytic)
    upsert_nfl_sim_slate(sim_rows, player_rows)
    if game_sims is not None:
        game_sims.update(row_sims)
    if ml_served is not None and game_lines:
        ml_served.update(ml_sims_by_game)
    ml_players = sum(1 for r in player_rows if r["game_pk"] in dists_by_game)
    return len(dists_by_game), ml_players, len(fallbacks)


def _write_copy_slate(ok_games: list[dict], sims_by_game: dict, specs_by_game: dict,
                      analytic_by_game: dict, allow_missing_analytic: bool = False) -> tuple[int, int]:
    """The current sim's rows re-labelled ML_MODEL_VERSION, in one
    transaction: (game rows, player rows). For when the site serves the ML
    version but the ML path produced nothing (off mode / global failure)."""
    sim_rows, player_rows = assemble_sim_rows(ok_games, sims_by_game, specs_by_game, analytic_by_game,
                                              model_version=ML_MODEL_VERSION,
                                              allow_missing_analytic=allow_missing_analytic)
    upsert_nfl_sim_slate(sim_rows, player_rows)
    return len(sim_rows), len(player_rows)


def _ml_failed(reason: str) -> None:
    """The loud ML-failure lines: a log line + a GitHub Actions annotation
    (one line each; newlines in the reason are flattened)."""
    flat = " ".join(str(reason).split())
    print(f"ML: FAILED {flat}", flush=True)
    print(f"::warning::props-ml: {flat}", flush=True)


def _ml_summary(mode: str, games: int, players: int, fallback_games: int, status: str,
                st_nan: dict | None = None) -> None:
    """The one ml_mode= line; st_nan_* = the target week's NaN share of the
    serve-time injury features (`na` when the tables were not built)."""
    st = st_nan or {}
    nan = " ".join(f"st_nan_{c[3:]}=" + ("na" if st.get(c) is None else f"{st[c]:.2f}")
                   for c in ST_NAN_COLS)
    print(f"ml_mode={mode} ml_games={games} ml_players={players} "
          f"ml_fallback_games={fallback_games} ml_status={status} {nan}", flush=True)


def _utcnow() -> datetime:
    """The run's clock (a seam: the main() tests pin it)."""
    return datetime.now(timezone.utc)


def main() -> None:
    n_sims = int(os.environ.get("DESK_SIM_N", str(DEFAULT_N_SIMS)))
    now = _utcnow()
    try:
        ml_mode, ml_mode_error = _ml_mode(), None
    except ValueError as exc:   # still write the current sim, then fail loudly
        ml_mode, ml_mode_error = "invalid", str(exc)
    lines_on = _ml_game_lines() == "on"

    games = _load_upcoming_games()
    print(f"{len(games)} upcoming NFL games in predictions_current")
    espn_only: set = set()   # game_pks only ESPN's target week supplied (lines on)
    if lines_on:   # the ML sim is the game-line source: ESPN's target week too (merge_espn_slate)
        try:
            espn_games = _load_espn_slate()
        except Exception as exc:  # noqa: BLE001 -- degrade to the predictions_current slate
            espn_games = []
            flat = " ".join(f"{type(exc).__name__}: {exc}".split())
            print(f"{_LINES_WARN}ESPN target-week schedule unavailable ({flat}); "
                  f"slate = predictions_current only, market lines carried from it", flush=True)
        db_pks = {g["game_pk"] for g in games}
        games = merge_espn_slate(games, espn_games, now)
        espn_only = {g["game_pk"] for g in games} - db_pks
        print(f"game_lines: slate {len(games)} games ({len(espn_only)} added from ESPN's target week, "
              f"{len(db_pks) - (len(games) - len(espn_only))} dropped as started/moved/not scheduled)")
    if not games:
        print("games=0 players=0 mean_disagreement=nan")
        if ml_mode_error is not None:
            _ml_failed(ml_mode_error)
            _ml_summary(ml_mode, 0, 0, 0, "failed")
            sys.exit(1)
        if ml_mode != "off":
            _ml_summary(ml_mode, 0, 0, 0, "ok")
        return

    crosswalk = _load_crosswalk()

    upto_season = nfl_season(now)
    seasons = list(range(upto_season - FETCH_SEASONS_BACK, upto_season + 1))
    print(f"fetching nflverse seasons {seasons}")
    nflverse = fetch_nflverse(seasons)

    upto_week = _determine_upto_week(nflverse["pbp"], upto_season)
    print(f"leakage cutoff: upto_season={upto_season} upto_week={upto_week}")

    rates = team_rates_from_pbp(nflverse["pbp"], upto_season, upto_week,
                                season_decay=SEASON_DECAY)
    def_rates = team_defense_rates_from_pbp(nflverse["pbp"], upto_season, upto_week,
                                            season_decay=SEASON_DECAY)
    print(f"team rates: season_decay={SEASON_DECAY} home_field={HOME_FIELD} "
          f"def_rates={len(def_rates)} teams")

    # Current Elo power ratings from the committed schedule (completed games only
    # feed ratings, so this is leakage-free for the upcoming slate). Missing team
    # -> base 1500 (neutral). Used to tilt sim scoring toward the power gap.
    try:
        sched = pd.read_parquet(TEAMS_CROSSWALK_PATH.parent / "schedules.parquet")
        elo = run_elo(sched, EloConfig()).final
    except Exception as exc:  # noqa: BLE001 -- ratings tilt is additive; degrade to none
        print(f"WARN elo unavailable ({exc!r}); ratings_tilt disabled")
        elo = {}
    print(f"power ratings: ratings_weight={RATINGS_WEIGHT} elo={len(elo)} teams")

    # One shared injury report (sportsmodel.nfl.injury_report, also used by the
    # desk): nflverse's official weekly report verified per player against
    # ESPN's live list, and dropped entirely when it's last week's (stale) and
    # ESPN is reachable. Statuses are normalized Out/Doubtful/Questionable.
    # `crosswalk` is already {display name -> abbrev} (= name_to_abbr).
    report = current_report(now, resolve_target_week(now), crosswalk)
    injuries = report["by_team"]
    out_names_by_team = _out_names_by_team(injuries)
    q_names_by_team = _questionable_names_by_team(injuries)
    print(_injury_summary(report))
    print(f"injuries: questionable_weight={QUESTIONABLE_WEIGHT}")

    print(f"fetching usage sources for seasons {seasons}")
    # depth comes from load_release("depth") + depth_charts_asof below; the old
    # nfl_data_py depth import is not downloaded (include_depth=False).
    usage_src = fetch_usage_sources(seasons, include_depth=False)
    pfr2gsis = build_pfr_to_gsis(usage_src["ids"])
    # Per-(season, week, team) depth charts, the same code path as the
    # backtest and the props-ML feature build: old weekly schema passed
    # through; 2025+ snapshots resolved to each team's latest chart at/before
    # its kickoff (usage.depth_charts_asof).
    schedules = load_release("schedules", seasons)
    depth_df = depth_charts_asof(load_release("depth", seasons), schedules)
    print(f"depth chart: {len(depth_df)} rows as of kickoff "
          f"({depth_df['club_code'].nunique() if 'club_code' in depth_df.columns else 0} teams)")

    # Same silent-failure mode `backtest_sim_nfl.py`'s `n_empty_active` guards
    # against: an ESPN->abbrev crosswalk code that doesn't match nflverse
    # club_code/injuries team makes `active_usage` hand back an empty roster
    # with no exception -- see `usage.abbrev_alignment`'s docstring. Resolve
    # the slate's team abbrevs (best-effort; an unresolvable name here will
    # also fail -- and get printed -- in the per-game loop below) and check
    # them, plus the depth/injuries sources, against the known team set.
    game_teams: set[str] = set()
    for g in games:
        for name in (g["home_team"], g["away_team"]):
            abbrev = crosswalk.get(name)
            if abbrev:
                game_teams.add(abbrev)
    injuries_df_like = pd.DataFrame({"team": list(injuries.keys())})
    alignment = abbrev_alignment(depth_df, injuries_df_like, game_teams, TEAMS)
    if alignment["depth_unknown"] or alignment["injuries_unknown"] or alignment["games_unknown"]:
        print(
            f"WARN abbrev mismatch: depth_unknown={alignment['depth_unknown']} "
            f"injuries_unknown={alignment['injuries_unknown']} "
            f"games_unknown={alignment['games_unknown']}"
        )
    else:
        print("abbrev alignment: OK (depth/injuries/schedule codes all known)")

    rng = np.random.default_rng(SIM_SEED)

    sims_by_game: dict = {}
    specs_by_game: dict = {}
    analytic_by_game: dict = {}
    ok_games: list[dict] = []
    game_keys: dict = {}   # game_pk -> (home_abbrev, away_abbrev, ratings_tilt), for the ML path
    n_empty_active = 0

    for g in games:
        game_pk = g["game_pk"]
        try:
            home_abbrev = crosswalk[g["home_team"]]
            away_abbrev = crosswalk[g["away_team"]]
            home_players, home_qb = active_usage(
                home_abbrev,
                upto_season,
                upto_week,
                depth_df,
                nflverse["weekly"],
                usage_src["snaps"],
                pfr2gsis,
                out_names_by_team.get(home_abbrev, set()),
                questionable_names=q_names_by_team.get(home_abbrev, set()),
                questionable_weight=QUESTIONABLE_WEIGHT,
                match_name_keys=True,
            )
            away_players, away_qb = active_usage(
                away_abbrev,
                upto_season,
                upto_week,
                depth_df,
                nflverse["weekly"],
                usage_src["snaps"],
                pfr2gsis,
                out_names_by_team.get(away_abbrev, set()),
                questionable_names=q_names_by_team.get(away_abbrev, set()),
                questionable_weight=QUESTIONABLE_WEIGHT,
                match_name_keys=True,
            )
            if not home_players or home_qb is None or not away_players or away_qb is None:
                # Count it loudly rather than let it silently simulate a
                # player-less team -- see the abbrev_alignment note above.
                n_empty_active += 1
            spec = build_spec_from_usage(
                home_abbrev, away_abbrev, rates, home_players, away_players, home_qb, away_qb,
                def_rates=def_rates,
            )
            rtilt = ratings_tilt(elo.get(home_abbrev, _ELO_BASE),
                                 elo.get(away_abbrev, _ELO_BASE), RATINGS_WEIGHT)
            sims = simulate_game(spec, n_sims, rng, home_field=HOME_FIELD, ratings_tilt=rtilt)
        except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the slate
            if game_pk in espn_only:   # no Elo row either: the game is missing from the site
                flat = " ".join(f"{type(exc).__name__}: {exc}".split())
                print(f"{_LINES_WARN}skipping game_pk={game_pk} ({g['matchup']}), added from ESPN's "
                      f"target week: sim failed ({flat}); it has no game_predictions row", flush=True)
            else:
                print(f"skipping game_pk={game_pk} ({g['matchup']}): {exc}")
            continue
        specs_by_game[game_pk] = spec
        sims_by_game[game_pk] = sims
        analytic_by_game[game_pk] = g["home_win_prob"]
        game_keys[game_pk] = (home_abbrev, away_abbrev, rtilt)
        ok_games.append(g)

    sim_rows, player_rows = assemble_sim_rows(ok_games, sims_by_game, specs_by_game, analytic_by_game,
                                              allow_missing_analytic=lines_on)

    if sim_rows:
        upsert_nfl_sim(sim_rows)
    if player_rows:
        upsert_nfl_player_sim(player_rows)

    disagreements = [r["disagreement"] for r in sim_rows if r["disagreement"] is not None]
    mean_disagreement = float(np.mean(disagreements)) if disagreements else float("nan")
    print(
        f"games={len(sim_rows)} players={len(player_rows)} "
        f"mean_disagreement={mean_disagreement:.4f} n_empty_active={n_empty_active}"
    )

    # --- props-ML (SIM_ML_MODE). The current sim above is already written. ---
    # The SERVED version decides what must be written (module docstring).
    try:
        served, served_error = served_nfl_sim_version(), None
    except Exception as exc:  # noqa: BLE001 -- unknown served version: see below per mode
        served, served_error = None, f"could not read nfl_sim_serving ({type(exc).__name__}: {exc})"
    must_write = served_error is None and served == ML_MODEL_VERSION
    slate = dict(ok_games=ok_games, sims_by_game=sims_by_game, specs_by_game=specs_by_game,
                 analytic_by_game=analytic_by_game, allow_missing_analytic=lines_on)
    # ML_GAME_LINES=on writes game_predictions only while the site serves the
    # ML version (on + served sim-nfl-v1 = a misconfiguration: warn, no writes).
    write_lines = lines_on and must_write
    if lines_on and not must_write:
        shown = "unknown (read failed)" if served_error is not None else served
        print(f"{_LINES_WARN}ML_GAME_LINES=on but the site serves {shown} (not {ML_MODEL_VERSION}); "
              f"no game_predictions written", flush=True)

    def game_lines_ok(sims_for_game: dict, ml_served: set) -> bool:
        """Write this run's game_predictions (write_lines only); False when the write fails."""
        if not write_lines:
            return True
        try:
            _write_game_lines(ok_games, sims_for_game, ml_served)
            return True
        except Exception as exc:  # noqa: BLE001 -- loud: the served game lines are now stale
            flat = " ".join(f"{type(exc).__name__}: {exc}".split())
            print(f"{_LINES_WARN}game_predictions write failed ({flat}); the served NFL game lines "
                  f"were not refreshed", flush=True)
            return False

    if ml_mode == "off":
        if served_error is not None:
            print(f"::warning::props-ml: {' '.join(served_error.split())}; if the site serves "
                  f"{ML_MODEL_VERSION} its slate was not refreshed", flush=True)
            sys.exit(1)
        if must_write:
            n_g, n_p = _write_copy_slate(**slate)
            print(f"props-ML: SIM_ML_MODE=off, served={served}: wrote the current sim under "
                  f"{ML_MODEL_VERSION} (games={n_g} players={n_p})", flush=True)
            print(OFF_SERVED_ML_WARNING, flush=True)
            if not game_lines_ok(sims_by_game, set()):
                sys.exit(1)
        return

    counts = (0, 0, 0)
    diag: dict = {}
    game_sims: dict = {}
    ml_served: set = set()
    if ml_mode_error is not None:
        reason = ml_mode_error
    elif served_error is not None:
        reason = served_error
    elif served is None:
        reason = SERVING_MISSING
    else:
        try:
            counts = run_ml(
                ok_games=ok_games, game_keys=game_keys, specs_by_game=specs_by_game,
                sims_by_game=sims_by_game, analytic_by_game=analytic_by_game,
                upto_season=upto_season, n_sims=n_sims, now=now, report=report, diag=diag,
                game_lines=write_lines, game_sims=game_sims, ml_served=ml_served,
                allow_missing_analytic=lines_on)
            reason = None
        except Exception as exc:  # noqa: BLE001 -- any ML failure -> SIM_ML_MODE failure semantics
            reason = f"{type(exc).__name__}: {exc}"
    if reason is None:
        lines_ok = game_lines_ok(game_sims, ml_served)
        _ml_summary(ml_mode, *counts, "ok", diag.get("st_nan"))
        if not lines_ok:
            sys.exit(1)
        return
    _ml_failed(reason)
    lines_ok = True
    if must_write:   # the site serves the ML version: it must get this run's (current-sim) slate
        try:
            n_g, n_p = _write_copy_slate(**slate)
            counts = (0, n_p, n_g)
            print(f"props-ML: served={served}: wrote the current sim under {ML_MODEL_VERSION} "
                  f"(games={n_g} players={n_p})", flush=True)
        except Exception as exc:  # noqa: BLE001 -- report it; the exit semantics still apply
            flat = " ".join(f"{type(exc).__name__}: {exc}".split())
            print(f"::warning::props-ml: copy slate write failed too ({flat}); the served "
                  f"{ML_MODEL_VERSION} slate was not refreshed", flush=True)
        # game lines follow what nfl_sim now holds for the ML version: the current sim
        lines_ok = game_lines_ok(sims_by_game, set())
    _ml_summary(ml_mode, *counts, "failed", diag.get("st_nan"))
    if ml_mode != "shadow" or not lines_ok:   # live (or an invalid mode) or stale game lines: red run
        sys.exit(1)


if __name__ == "__main__":
    main()
