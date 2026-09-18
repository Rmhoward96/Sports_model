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
then hand everything to `assemble_sim_rows` in one shot and upsert.

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

Usage:
    PYTHONPATH=src uv run python scripts/generate_sim_nfl.py

Requires DATABASE_URL (Supabase) and nflverse network access; not run here.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sportsmodel import config
from sportsmodel.db import get_postgres, upsert_nfl_player_sim, upsert_nfl_sim
from sportsmodel.nfl.injuries_nflverse import current_injuries, nfl_season
from sportsmodel.sim.engine import pred_scores
from sportsmodel.sim.nfl.aggregate import disagreement, nfl_player_prop_dists
from sportsmodel.sim.nfl.inputs import build_spec_from_usage
from sportsmodel.sim.nfl.kernel import simulate_game
from sportsmodel.sim.nfl.rates import fetch_nflverse, team_rates_from_pbp
from sportsmodel.sim.nfl.usage import active_usage, build_pfr_to_gsis, fetch_usage_sources

MODEL_VERSION = "sim-nfl-v1"
DEFAULT_N_SIMS = 10_000
SIM_SEED = 42

# Prior seasons pulled alongside the current one so rates stay stable early
# in a season (when the current season alone has too few games): current +
# 2 prior seasons, per the task brief's "current + prior 1-2" guidance.
FETCH_SEASONS_BACK = 2

# Binning ceilings for nfl_player_prop_dists's pmf markets (anytime_td is
# binary and doesn't need one -- see aggregate.nfl_player_prop_dists).
MARKET_MAX = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 200, "receptions": 15}

TEAMS_CROSSWALK_PATH = config.PROJECT_ROOT / "assets" / "nfl" / "nfl_teams.json"

# nflverse injury `status` values (case-insensitive) treated as ruled out for
# active_usage's per-team injuries_out_names set.
_OUT_STATUSES = frozenset({"out", "doubtful"})


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

    Returns:
        (nfl_sim_rows, nfl_player_sim_rows) matching `db.upsert_nfl_sim` /
        `db.upsert_nfl_player_sim`'s expected columns exactly (`dist` as a
        plain dict -- upsert_nfl_player_sim does the json.dumps).

    A game missing from `sims_by_game`, `specs_by_game`, or
    `analytic_by_game` is silently skipped (documents the contract: `main()`
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
        if sims is None or spec is None or analytic_home_win_prob is None:
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
            "disagreement": disagreement(analytic_home_win_prob, sim_home_win_prob),
        })

        identity_by_player_id: dict[str, tuple[str, str, str]] = {
            p.player_id: (p.name, p.pos, g["home_team"]) for p in spec.home_players
        }
        identity_by_player_id.update(
            {p.player_id: (p.name, p.pos, g["away_team"]) for p in spec.away_players}
        )

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


def _load_upcoming_games() -> list[dict]:
    """Upcoming NFL games from predictions_current. `home_team`/`away_team`
    here are ESPN display names (as stored in predictions_current), NOT
    nflverse abbreviations -- `main()` maps them through `_load_crosswalk()`
    before touching rates/players/injuries."""
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT game_pk, home_team_name, away_team_name, home_win_prob, commence_time
            FROM predictions_current
            WHERE sport = 'nfl' AND commence_time > now()
        """)
        cols = ["game_pk", "home_team_name", "away_team_name", "home_win_prob", "commence_time"]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return [
        {
            "game_pk": r["game_pk"],
            "matchup": f"{r['away_team_name']} @ {r['home_team_name']}",
            "commence_time": r["commence_time"],
            "home_team": r["home_team_name"],
            "away_team": r["away_team_name"],
            "home_win_prob": r["home_win_prob"],
        }
        for r in rows
    ]


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


def main() -> None:
    n_sims = int(os.environ.get("DESK_SIM_N", str(DEFAULT_N_SIMS)))
    now = datetime.now(timezone.utc)

    games = _load_upcoming_games()
    print(f"{len(games)} upcoming NFL games in predictions_current")
    if not games:
        print("games=0 players=0 mean_disagreement=nan")
        return

    crosswalk = _load_crosswalk()

    upto_season = nfl_season(now)
    seasons = list(range(upto_season - FETCH_SEASONS_BACK, upto_season + 1))
    print(f"fetching nflverse seasons {seasons}")
    nflverse = fetch_nflverse(seasons)

    upto_week = _determine_upto_week(nflverse["pbp"], upto_season)
    print(f"leakage cutoff: upto_season={upto_season} upto_week={upto_week}")

    rates = team_rates_from_pbp(nflverse["pbp"], upto_season, upto_week)
    injuries = current_injuries(now)
    out_names_by_team = _out_names_by_team(injuries)

    print(f"fetching usage sources for seasons {seasons}")
    usage_src = fetch_usage_sources(seasons)
    pfr2gsis = build_pfr_to_gsis(usage_src["ids"])

    rng = np.random.default_rng(SIM_SEED)

    sims_by_game: dict = {}
    specs_by_game: dict = {}
    analytic_by_game: dict = {}
    ok_games: list[dict] = []

    for g in games:
        game_pk = g["game_pk"]
        try:
            home_abbrev = crosswalk[g["home_team"]]
            away_abbrev = crosswalk[g["away_team"]]
            home_players, home_qb = active_usage(
                home_abbrev,
                upto_season,
                upto_week,
                usage_src["depth"],
                nflverse["weekly"],
                usage_src["snaps"],
                pfr2gsis,
                out_names_by_team.get(home_abbrev, set()),
            )
            away_players, away_qb = active_usage(
                away_abbrev,
                upto_season,
                upto_week,
                usage_src["depth"],
                nflverse["weekly"],
                usage_src["snaps"],
                pfr2gsis,
                out_names_by_team.get(away_abbrev, set()),
            )
            spec = build_spec_from_usage(
                home_abbrev, away_abbrev, rates, home_players, away_players, home_qb, away_qb
            )
            sims = simulate_game(spec, n_sims, rng)
        except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the slate
            print(f"skipping game_pk={game_pk} ({g['matchup']}): {exc}")
            continue
        specs_by_game[game_pk] = spec
        sims_by_game[game_pk] = sims
        analytic_by_game[game_pk] = g["home_win_prob"]
        ok_games.append(g)

    sim_rows, player_rows = assemble_sim_rows(ok_games, sims_by_game, specs_by_game, analytic_by_game)

    if sim_rows:
        upsert_nfl_sim(sim_rows)
    if player_rows:
        upsert_nfl_player_sim(player_rows)

    mean_disagreement = (
        float(np.mean([r["disagreement"] for r in sim_rows])) if sim_rows else float("nan")
    )
    print(f"games={len(sim_rows)} players={len(player_rows)} mean_disagreement={mean_disagreement:.4f}")


if __name__ == "__main__":
    main()
