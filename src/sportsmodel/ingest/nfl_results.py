"""Actual game + player results from ESPN summaries (for grading).

The NFL analog of mlb_results.py: final scores + each player's realized stats in
our prop-market terms, so predictions can be graded against reality. Available
once a game is STATUS_FINAL.

Key mapping for grade_results (Task 7):
  - game lines (moneyline/spread/total): res["home_score"], res["away_score"]
  - props: res["players"][player_id][market] for market in
    {"pass_yds", "pass_tds", "rush_yds", "reception_yds", "receptions",
     "rush_reception_yds", "anytime_td"}
  - player_id is the int ESPN athlete id (e.g. 3139477) -- NOT the nflverse/GSIS
    id. `prop_predictions`/`board_picks`/`picks`/`prediction_results.player_id`
    are all BIGINT columns, which can't hold the gsis string ("00-0034473"), so
    the NFL producer (`scripts/generate_nfl.py`) writes the int ESPN athlete id
    instead (mapped from its internal gsis id via the rosters gsis->espn_id
    crosswalk at write time). Keying `players` here by that same int ESPN id
    means grading looks it up directly with no remap needed on this side --
    `res["players"].get(pick.player_id)` works unchanged for both MLB (int
    player_id already) and NFL.

    ESPN's `/summary` boxscore is naturally keyed by its own athlete id, so
    this is also the parser's native id space -- `parse_results` just casts it
    to int, no crosswalk lookup or roster data needed to stay pure/fixture-
    tested.

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

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


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


def parse_results(summary: dict) -> dict[str, Any]:
    """Final scores + per-player prop actuals from an ESPN game-summary payload.

    Returns {"home_score": int, "away_score": int, "final": bool,
             "players": {player_id: {market: value}}}.
    `final` gates on ESPN STATUS_FINAL; players is populated regardless (so a
    caller can still inspect in-progress box scores if it chooses), but callers
    grading picks should check `final` first.

    `players` is keyed by the int ESPN athlete id -- see module docstring for
    why (matches the int id the NFL producer writes as `player_id`).
    """
    header = summary.get("header", {})
    comp = header.get("competitions", [{}])[0]
    status_name = comp.get("status", {}).get("type", {}).get("name")
    competitors = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    home_score = _to_int(competitors.get("home", {}).get("score"))
    away_score = _to_int(competitors.get("away", {}).get("score"))

    players: dict[int, dict[str, int]] = {}
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
            players[int(pid)] = {
                "pass_yds": _to_int(p_stats.get("passingYards")),
                "pass_tds": _to_int(p_stats.get("passingTouchdowns")),
                "rush_yds": rush_yds,
                "reception_yds": rec_yds,
                "receptions": _to_int(re_stats.get("receptions")),
                "rush_reception_yds": rush_yds + rec_yds,
                "anytime_td": 1 if (rush_tds + rec_tds) >= 1 else 0,
            }

    return {
        "home_score": home_score,
        "away_score": away_score,
        "final": status_name == "STATUS_FINAL",
        "players": players,
    }


def fetch_results(game_pk: int) -> dict[str, Any] | None:
    """Final scores + per-player actuals for one ESPN event id, or None if the
    game isn't final yet. `players` is keyed by the int ESPN athlete id (see
    `parse_results`/module docstring), matching the int `player_id` the NFL
    producer writes to `prop_predictions` -- no crosswalk needed here.
    """
    r = httpx.get(f"{_BASE}/summary", params={"event": game_pk}, timeout=20)
    r.raise_for_status()
    res = parse_results(r.json())
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
