"""Grade `ev_prop_picks` FORWARD against the player's actual weekly stat and
the closing Pinnacle prop price, tracking closing-line value (CLV) ->
`ev_prop_results`.

Sibling of scripts/grade_ev.py (mirrors its structure closely) but for the
NFL player-prop +EV board (sub-project C, Task 5) instead of the game-level
+EV pilot: "did the actual stat clear the picked line, and did the price
taken beat what Pinnacle closed at for that same player/market/side/line."

NFL and MLB (matches props_ev.py / build_ev_props_board.py scope). MLB actuals
come from the StatsAPI boxscore (`ingest.mlb_results`) once the game is final; a
player who never appears in the box (DNP) can't be graded and is skipped. No
Odds API calls -- reads Supabase (`ev_prop_picks`, `odds_snapshot`) and
nflverse (weekly stats, schedules) only, same "no live network odds fetch"
posture as grade_ev.py.

Actual-stat resolution: the player's realized stat for the pick's market
that week, from the live nflverse weekly player-stats release, joined by
`player_id == gsis_id`. The game's WEEK is resolved by matching
`ev_prop_picks.game_pk` (an ESPN event id, same convention as grade_ev.py's
FINAL_PROVIDERS) against nflverse `import_schedules`' own `espn` column for
the pick's season (`sportsmodel.nfl.injuries_nflverse.nfl_season` derives the
season from `commence_time`) -- an exact join, no date-proximity heuristic
needed since nflverse schedules already carry the ESPN id per game.

Closing price: the last `odds_snapshot` row at/before `commence_time` for
(game_pk, odds-side market, side, line), restricted to book='pinnacle' and
matched to the pick's player via `normalize_player_name` (odds_snapshot's
`player_name` is the Odds API's own formatting, not nflverse's).

Any pick whose actual can't be resolved (schedule/weekly fetch fails, no
matching week, player not in that week's weekly data, etc.) is SKIPPED, not
crashed on -- same one-bad-row-must-not-abort-the-batch posture as
grade_ev.main's per-pick try/except.

Runs on a rolling window (like grade_ev.py); idempotent -- re-running just
re-upserts the same rows via (game_pk, player_id, market, line,
model_version) (db.upsert_ev_prop_results).

Usage:
    uv run python scripts/grade_ev_props.py [--days 7]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel import config
from sportsmodel.db import get_postgres, upsert_ev_prop_results
from sportsmodel.ingest import mlb_results, mlb_statsapi
from sportsmodel.nfl.data import load_schedules
from sportsmodel.nfl.injuries_nflverse import nfl_season
from sportsmodel.nfl.nflverse import load_release
from sportsmodel.serving.props_ev import (
    MLB_SIM_TO_ODDS_MARKET,
    SIM_MARKET_TO_WEEKLY,
    grade_prop_pick,
    normalize_player_name,
    odds_market_for,
)

PICK_COLS = [
    "sport", "game_pk", "player_id", "player_name", "market", "side", "line",
    "model_version", "commence_time", "model_prob", "best_price", "pinnacle_price",
]


def _window_start(days: int, today: date | None = None) -> str:
    d = today or date.today()
    return (d - timedelta(days=days)).isoformat()


def _pending_prop_picks(cur, start: str, sport: str = "nfl") -> list[dict]:
    """`ev_prop_picks` rows for `sport` -- one per (game_pk, player_id, market,
    line) -- whose game has already started (commence_time <= now(), so a
    final stat line should exist) and that haven't been graded yet.

    Filtered to is_pick = true: ev_prop_picks also carries rows for markets
    the board didn't take (no +EV side) -- those aren't real bets. The
    NOT EXISTS against ev_prop_results makes re-runs idempotent and cheap.

    `pinnacle_price` is COALESCE(open_pinnacle_price, pinnacle_price) -- the
    FROZEN pick-time price when available (same convention grade_ev.py uses
    for ev_picks), else the mutable pinnacle_price column for rows written
    before open_pinnacle_price existed. This is the honest pick-time anchor
    CLV is measured from; the raw (mutable) pinnacle_price column drifts
    toward the close on every board rebuild and would collapse CLV to ~0.
    """
    cur.execute("""
        SELECT ep.sport, ep.game_pk, ep.player_id, ep.player_name, ep.market,
               ep.side, ep.line, ep.model_version, ep.commence_time, ep.model_prob,
               ep.best_price,
               COALESCE(ep.open_pinnacle_price, ep.pinnacle_price) AS pinnacle_price
        FROM ev_prop_picks ep
        WHERE ep.sport = %(sport)s AND ep.commence_time >= %(start)s
          AND ep.commence_time <= now()
          AND ep.is_pick = true
          AND NOT EXISTS (
              SELECT 1 FROM ev_prop_results er
              WHERE er.game_pk = ep.game_pk AND er.player_id = ep.player_id
                AND er.market = ep.market AND er.line = ep.line
                AND er.model_version = ep.model_version
          )
        ORDER BY ep.game_pk, ep.player_id, ep.market
    """, {"start": start, "sport": sport})
    return [dict(zip(PICK_COLS, row)) for row in cur.fetchall()]


# Per-run caches so a slate of many picks sharing a season doesn't refetch
# nflverse schedules/weekly data once per pick.
_schedule_cache: dict[int, pd.DataFrame | None] = {}
_weekly_cache: dict[int, pd.DataFrame | None] = {}


def _schedule_for(season: int) -> pd.DataFrame | None:
    if season not in _schedule_cache:
        try:
            _schedule_cache[season] = load_schedules([season])
        except Exception as exc:  # noqa: BLE001 -- schedule fetch is best-effort
            print(f"  schedule fetch failed for season {season}: {exc}")
            _schedule_cache[season] = None
    return _schedule_cache[season]


def _weekly_for(season: int) -> pd.DataFrame | None:
    if season not in _weekly_cache:
        # nfl_data_py's import_weekly_data points at a retired nflverse URL
        # (silently empty); read the canonical stats_player_week release.
        _weekly_cache[season] = load_release("weekly", [season], required=False)
    return _weekly_cache[season]


def _week_for_game(game_pk, season: int) -> int | None:
    """The NFL week `game_pk` (an ESPN event id) was played in `season`, via
    an exact join against nflverse `import_schedules`' own `espn` column."""
    sched = _schedule_for(season)
    if sched is None or sched.empty or "espn" not in sched.columns:
        return None
    matches = sched[sched["espn"].astype(str) == str(game_pk)]
    if matches.empty:
        return None
    week = matches.iloc[0]["week"]
    return int(week) if pd.notna(week) else None


def _actual_for(pick: dict) -> float | None:
    """The player's realized stat for `pick["market"]` that week, from
    nflverse weekly data joined by gsis player_id. None (skip -- don't
    crash) if any step of the resolution can't be completed."""
    weekly_col = SIM_MARKET_TO_WEEKLY.get(pick.get("market"))
    if weekly_col is None:
        return None
    commence = pick.get("commence_time")
    if commence is None:
        return None
    season = nfl_season(commence)
    week = _week_for_game(pick.get("game_pk"), season)
    if week is None:
        return None
    weekly = _weekly_for(season)
    if weekly is None or weekly.empty or weekly_col not in weekly.columns:
        return None
    rows = weekly[(weekly["player_id"] == pick.get("player_id")) & (weekly["week"] == week)]
    if rows.empty:
        return None
    val = rows.iloc[0][weekly_col]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return float(val)


_mlb_results_cache: dict[int, dict | None] = {}


def _mlb_actual_for(pick: dict) -> float | None:
    """The player's realized MLB stat for `pick["market"]`, from the final boxscore
    (None until the game is a completed one, or if the player never played)."""
    gp = pick.get("game_pk")
    if gp not in _mlb_results_cache:
        final = mlb_statsapi.fetch_final(gp)  # not final / postponed -> None
        _mlb_results_cache[gp] = mlb_results.fetch_results(gp) if final is not None else None
    return mlb_results.actual_for_pick(_mlb_results_cache[gp], pick.get("player_id"), pick.get("market"))


def _closing_price(cur, pick: dict, market_map: dict[str, str] | None = None) -> int | None:
    """The CLOSING Pinnacle price for (game_pk, market, side, line, player):
    the last odds_snapshot row at/before commence_time, matched to the
    pick's player via normalize_player_name (odds_snapshot stores the Odds
    API's own player_name formatting, not nflverse's)."""
    odds_market = (market_map.get(pick.get("market")) if market_map is not None
                   else odds_market_for(pick.get("market")))
    if odds_market is None:
        return None
    cur.execute("""
        SELECT player_name, price FROM odds_snapshot
        WHERE game_pk = %s AND market = %s AND side = %s AND line = %s
          AND book = 'pinnacle' AND captured_at <= %s
        ORDER BY captured_at DESC
    """, (pick["game_pk"], odds_market, pick["side"], pick["line"], pick["commence_time"]))
    target = normalize_player_name(pick.get("player_name"))
    for player_name, price in cur.fetchall():
        if price is not None and normalize_player_name(player_name) == target:
            return int(price)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (grading reads/writes Supabase).")

    start = _window_start(args.days)
    graded_rows: list[dict] = []

    # sport -> (actual-stat resolver, odds-side market map for the closing-price lookup;
    # None = the NFL map in props_ev.odds_market_for)
    legs = {"nfl": (_actual_for, None), "mlb": (_mlb_actual_for, MLB_SIM_TO_ODDS_MARKET)}
    with get_postgres() as conn, conn.cursor() as cur:
        for sport, (actual_of, market_map) in legs.items():
            pending = _pending_prop_picks(cur, start, sport)
            n = 0
            for pick in pending:
                try:
                    actual = actual_of(pick)
                    if actual is None:
                        continue  # can't resolve the actual -- skip, don't crash
                    closing_price = _closing_price(cur, pick, market_map)
                    graded_rows.append(grade_prop_pick(pick, actual, closing_price))
                    n += 1
                except Exception as exc:  # noqa: BLE001 -- one bad pick must not abort the batch
                    print(f"  {sport} {pick.get('game_pk')} {pick.get('player_name')} "
                          f"{pick.get('market')}/{pick.get('side')}: grading failed ({exc}); skipping")
                    continue
            print(f"{sport}: {len(pending)} pending, {n} graded")

    if graded_rows:
        written = upsert_ev_prop_results(graded_rows)
        wins = sum(1 for r in graded_rows if r["result"] == "win")
        losses = sum(1 for r in graded_rows if r["result"] == "loss")
        pushes = sum(1 for r in graded_rows if r["result"] == "push")
        clvs = [r["clv"] for r in graded_rows if r["clv"] is not None]
        mean_clv = sum(clvs) / len(clvs) if clvs else None
        mean_clv_str = f"{mean_clv:.4f}" if mean_clv is not None else "n/a"
        print(f"Upserted {written} ev_prop_results rows. "
              f"{wins}W-{losses}L-{pushes}P, mean_clv={mean_clv_str} "
              f"({len(clvs)}/{len(graded_rows)} with CLV).")


if __name__ == "__main__":
    main()
