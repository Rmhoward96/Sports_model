"""Actual game + player results from ESPN summaries (for grading).

The NFL analog of mlb_results.py: final scores + each player's realized stats in
our prop-market terms, so predictions can be graded against reality. Available
once a game is STATUS_FINAL.

Key mapping for grade_results (Task 7):
  - game lines (moneyline/spread/total): res["home_score"], res["away_score"]
  - props: res["players"][player_id][market] for market in
    {"pass_yds", "pass_tds", "rush_yds", "reception_yds", "receptions",
     "rush_reception_yds", "anytime_td"}
  - player_id is the nflverse/GSIS id (e.g. "00-0034473") -- the SAME id space as
    `rosters.parquet.player_id`, and therefore the same id
    `prop_predictions.player_id` carries (the odds matcher / universe resolve
    prop picks off that roster column, not ESPN's athlete id). This keeps
    grade_results sport-agnostic: `res["players"].get(pick.player_id)` works
    unchanged for both MLB and NFL.

    ESPN's `/summary` boxscore only carries its OWN athlete id per player
    (e.g. "3139477"), not the gsis id -- so `fetch_results` reconciles the two
    via `rosters.parquet`'s `espn_id` column (a committed nflverse/GSIS <->
    ESPN crosswalk) before returning. `parse_results` itself stays pure/ESPN-id
    keyed unless given an explicit `id_map` (see below), so it remains
    fixture-testable without touching the parquet.

  - `parse_results(summary, id_map=None)`: when `id_map` (a dict
    `espn_athlete_id (str) -> gsis_player_id (str)`) is given, the returned
    `players` dict is re-keyed from ESPN id to gsis id via `id_map`. An ESPN id
    with no crosswalk hit keeps its raw ESPN id as the key (never dropped) --
    defensive, so an unmapped athlete doesn't silently vanish from grading.
    With `id_map=None` (the default), `players` stays keyed by raw ESPN ids --
    used by the fixture test, which doesn't need real roster data.

CAVEAT (flagged for Task 9 validation): this parser's box-score shape (boxscore
-> players[] -> statistics[] keyed by category name "passing"/"rushing"/
"receiving", each with parallel "keys"/"stats" arrays per athlete) is modeled
from ESPN's site-api summary endpoint as documented/observed historically. The
committed fixture (tests/fixtures/nfl/espn_summary.json) is a best-effort,
trimmed replica of that shape -- it has NOT been diffed against a live response.
When Task 9 runs this against a real final game, small adjustments (stat key
names, nesting) may be needed.
"""
from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

from .. import config

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
_ROSTERS_PATH = config.PROJECT_ROOT / "assets" / "nfl" / "rosters.parquet"


def _to_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _category_stats(category: dict) -> dict[str, dict[str, str]]:
    """athlete_id -> {stat_key: raw_value} for one boxscore statistics category
    (e.g. "passing"), zipping the category's "keys" against each athlete's
    parallel "stats" array.
    """
    keys = category.get("keys", [])
    out: dict[str, dict[str, str]] = {}
    for ath in category.get("athletes", []):
        pid = ath.get("athlete", {}).get("id")
        if pid is None:
            continue
        out[str(pid)] = dict(zip(keys, ath.get("stats", [])))
    return out


def _remap_players(
    players: dict[str, dict[str, int]], id_map: dict[str, str]
) -> dict[str, dict[str, int]]:
    """Re-key a `players` dict from ESPN athlete id -> gsis player_id via
    `id_map`. An ESPN id absent from `id_map` keeps its original ESPN id as the
    key rather than being dropped, so an unmapped athlete still grades under
    *some* id instead of vanishing silently.
    """
    return {id_map.get(pid, pid): stats for pid, stats in players.items()}


def parse_results(summary: dict, id_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Final scores + per-player prop actuals from an ESPN game-summary payload.

    Returns {"home_score": int, "away_score": int, "final": bool,
             "players": {player_id: {market: value}}}.
    `final` gates on ESPN STATUS_FINAL; players is populated regardless (so a
    caller can still inspect in-progress box scores if it chooses), but callers
    grading picks should check `final` first.

    `players` is keyed by raw ESPN athlete id unless `id_map` (ESPN id -> gsis
    id, e.g. from rosters.parquet's espn_id column) is provided, in which case
    the keys are remapped to gsis ids -- see module docstring.
    """
    header = summary.get("header", {})
    comp = header.get("competitions", [{}])[0]
    status_name = comp.get("status", {}).get("type", {}).get("name")
    competitors = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    home_score = _to_int(competitors.get("home", {}).get("score"))
    away_score = _to_int(competitors.get("away", {}).get("score"))

    players: dict[str, dict[str, int]] = {}
    for team_block in summary.get("boxscore", {}).get("players", []):
        by_category = {
            cat.get("name"): _category_stats(cat)
            for cat in team_block.get("statistics", [])
        }
        passing = by_category.get("passing", {})
        rushing = by_category.get("rushing", {})
        receiving = by_category.get("receiving", {})
        pids = set(passing) | set(rushing) | set(receiving)
        for pid in pids:
            p_stats = passing.get(pid, {})
            ru_stats = rushing.get(pid, {})
            re_stats = receiving.get(pid, {})
            rush_yds = _to_int(ru_stats.get("rushingYards"))
            rec_yds = _to_int(re_stats.get("receivingYards"))
            rush_tds = _to_int(ru_stats.get("rushingTouchdowns"))
            rec_tds = _to_int(re_stats.get("receivingTouchdowns"))
            players[pid] = {
                "pass_yds": _to_int(p_stats.get("passingYards")),
                "pass_tds": _to_int(p_stats.get("passingTouchdowns")),
                "rush_yds": rush_yds,
                "reception_yds": rec_yds,
                "receptions": _to_int(re_stats.get("receptions")),
                "rush_reception_yds": rush_yds + rec_yds,
                "anytime_td": 1 if (rush_tds + rec_tds) >= 1 else 0,
            }

    if id_map:
        players = _remap_players(players, id_map)

    return {
        "home_score": home_score,
        "away_score": away_score,
        "final": status_name == "STATUS_FINAL",
        "players": players,
    }


def _espn_id_crosswalk() -> dict[str, str]:
    """{espn_id (str) -> gsis player_id (str)} from the committed rosters
    snapshot, one entry per espn_id (dedup keeping the LATEST season a given
    espn_id appears in -- a player can recur across seasons; the newest row is
    the freshest link). Rows with a missing/NaN espn_id are skipped.
    """
    rosters = pd.read_parquet(_ROSTERS_PATH, columns=["season", "player_id", "espn_id"])
    rosters = rosters.dropna(subset=["espn_id"])
    rosters = rosters.sort_values("season").drop_duplicates("espn_id", keep="last")
    return {
        str(espn_id): str(pid)
        for espn_id, pid in zip(rosters["espn_id"], rosters["player_id"])
    }


def fetch_results(game_pk: int) -> dict[str, Any] | None:
    """Final scores + per-player actuals for one ESPN event id, or None if the
    game isn't final yet. `players` is keyed by gsis player_id (reconciled from
    ESPN's athlete id via the rosters.parquet espn_id crosswalk) so it lines up
    directly with `prop_predictions.player_id` for grading.
    """
    r = httpx.get(f"{_BASE}/summary", params={"event": game_pk}, timeout=20)
    r.raise_for_status()
    id_map = _espn_id_crosswalk()
    res = parse_results(r.json(), id_map=id_map)
    return res if res["final"] else None


def final_game_pks(start_date: str, end_date: str) -> set[int]:
    """ESPN event ids that are STATUS_FINAL in [start_date, end_date] (YYYY-MM-DD)."""
    r = httpx.get(
        f"{_BASE}/scoreboard",
        params={"dates": f"{start_date.replace('-', '')}-{end_date.replace('-', '')}"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    finals: set[int] = set()
    for ev in data.get("events", []):
        status = ev.get("status", {}).get("type", {}).get("name")
        if status == "STATUS_FINAL":
            finals.add(int(ev["id"]))
    return finals
