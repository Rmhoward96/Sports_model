"""Daily game-info job (NFL + CFB): venue + weather (and the CFB final line score) for every game from
2 days ago to 10 days ahead -> Supabase `game_info` (db/migration_site_panels.sql). Descriptive only.

* CFB: ESPN's FBS scoreboard for the current week +- 1 gives the games, kickoffs and final quarter scores;
  CFBD `/games/weather` (one call per distinct week inside the window) gives venue id, indoor flag and the
  forecast / observed weather; assets/cfb/venues.parquet gives city / state. Needs CFBD_API_KEY.
* NFL: ESPN scoreboard for the current week +- 1, then one free `/summary` per game in the window.
* `game_info` upserts keep an earlier reading when a later run has none (see db._site_panel_sql).
* `--dry-run` prints counts and a sample and writes nothing (CFB still calls CFBD, read-only).

Usage:
    uv run python scripts/build_game_info.py --sport all [--dry-run] [--now ISO]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sportsmodel import config, db, game_info  # noqa: E402
from sportsmodel.cfb import cfbd_games, espn as cfb_espn  # noqa: E402
from sportsmodel.cfb.cfbd import CfbdClient  # noqa: E402
from sportsmodel.nfl import espn as nfl_espn  # noqa: E402

VENUES_PATH = ROOT / "assets" / "cfb" / "venues.parquet"


def warn(msg: str) -> None:
    print(f"::warning::game-info: {msg}", flush=True)


def window_weeks(games: list[dict], now) -> list[tuple[int, int]]:
    """Sorted distinct (season, week) of the games whose kickoff is inside the window."""
    return sorted({(int(g["season"]), int(g["week"])) for g in games
                   if g.get("season") is not None and g.get("week") is not None
                   and game_info.in_window(g.get("commence_time"), now)})


def build_cfb(now, client, espn=cfb_espn, venues_path: Path = VENUES_PATH) -> list[dict]:
    cur = espn.fetch_current_week()
    season, week, st = int(cur["season"]), int(cur["week"]), int(cur["season_type"])
    if st not in (2, 3):
        print("cfb: ESPN is between seasons; no rows", flush=True)
        return []
    games, line_scores = [], {}
    for w in (week - 1, week, week + 1):
        if w < 1:
            continue
        try:
            payload = espn.fetch_scoreboard(season, w, st)
        except Exception as exc:  # noqa: BLE001 -- one week down must not lose the rest
            warn(f"cfb: ESPN week {w} unavailable ({type(exc).__name__})")
            continue
        games += espn.parse_schedule(payload)
        line_scores.update(espn.parse_line_scores(payload))
    stype = "postseason" if st == 3 else "regular"
    frames = []
    for s, w in window_weeks(games, now):
        try:
            frames.append(cfbd_games.parse_weather_window(
                client.get("/games/weather", {"year": s, "week": w, "seasonType": stype})))
        except Exception as exc:  # noqa: BLE001
            warn(f"cfb: CFBD weather week {w} unavailable ({type(exc).__name__})")
    weather = pd.concat(frames, ignore_index=True) if frames else cfbd_games.parse_weather_window([])
    venues = pd.read_parquet(venues_path) if venues_path.exists() else None
    if venues is None or "city" not in venues.columns:
        warn(f"cfb: {venues_path.name} has no city / state (run the venues backfill); venue names only")
    return game_info.cfb_game_info(games, line_scores, weather, venues, now)


def build_nfl(now, espn=nfl_espn) -> list[dict]:
    tw = espn.target_week(espn.fetch_current_week())
    season, week, st = int(tw["season"]), int(tw["week"]), int(tw["season_type"])
    games = []
    for w in (week - 1, week, week + 1):
        if w < 1 or (st == 2 and w > 18):
            continue
        try:
            games += espn.fetch_schedule(season, w, season_type=st)
        except Exception as exc:  # noqa: BLE001
            warn(f"nfl: ESPN week {w} unavailable ({type(exc).__name__})")
    return game_info.nfl_game_info(games, espn.fetch_game_info, now, warn)


def run_sport(sport: str, now, dry_run: bool) -> int:
    rows = build_cfb(now, CfbdClient.from_env()) if sport == "cfb" else build_nfl(now)
    withwx = sum(1 for r in rows if r["weather_kind"])
    print(f"[{sport}] game_info rows={len(rows)} (with weather {withwx}, indoor "
          f"{sum(1 for r in rows if r['indoor'])}, with line score {sum(1 for r in rows if r['line_score'])})",
          flush=True)
    if dry_run:
        for r in rows[:3]:
            print("  ", {k: v for k, v in r.items() if v is not None})
        return len(rows)
    print(f"[{sport}] upserted {db.upsert_game_info(rows)} rows", flush=True)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sport", choices=["nfl", "cfb", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="compute and print; no DB writes")
    ap.add_argument("--now", default=None, help="as-of time (ISO, default: now UTC)")
    args = ap.parse_args(argv)
    if not args.dry_run and not config.DATABASE_URL:
        sys.exit("DATABASE_URL is not set (use --dry-run to compute without writing)")
    now = game_info.utc(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    failed = []
    for sport in (["nfl", "cfb"] if args.sport == "all" else [args.sport]):
        try:   # one sport failing must not block the other
            run_sport(sport, now, args.dry_run)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- SystemExit: CfbdClient.from_env without a key
            print(f"::error::game-info: {sport} failed: {type(exc).__name__}: {exc}", flush=True)
            failed.append(sport)
    if failed:
        sys.exit(f"game-info failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
