"""Day-before injury watch: detect Out/Doubtful changes since the last desk run.

    --check   games kicking off within the next 30h: fingerprint each game's
              Out/Doubtful designations, compare to `injury_snapshots`, print
              each changed game's matchup + per-player diff, and append
              `changed=true|false` to $GITHUB_OUTPUT (when set). Always exits 0.
    --record  games kicking off within the next 7 days: upsert the current
              fingerprint/statuses into `injury_snapshots`.

A game with no stored snapshot counts as changed. Questionable designations
never count (see sportsmodel.serving.injury_watch).

Injuries are keyed by the team display names in `predictions_current`:
  - NFL: the shared nflverse+ESPN report (`injury_report.current_report`),
    abbr -> name via assets/nfl/nfl_teams.json.
  - CFB: SportsDataIO via scripts/desk_inputs.py's `_cfb_injuries_by_name` +
    `_rekey_by_espn_name`, exactly as desk_inputs.main does. Needs
    SPORTSDATA_API_KEY; when unset this warns and exits 0 (changed=false for
    --check, nothing recorded for --record).

The pure seams (`plan_check`, `snapshot_rows`, `write_github_output`) are
unit-tested; main()'s DB/injury IO is thin.

Usage:
    DATABASE_URL=... uv run python scripts/injury_watch.py --sport nfl --check
    DATABASE_URL=... SPORTSDATA_API_KEY=... uv run python scripts/injury_watch.py --sport cfb --record
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config  # noqa: E402
from sportsmodel.serving.injury_watch import (  # noqa: E402
    changed_games,
    diff_statuses,
    fingerprint,
    game_statuses,
)

CHECK_WINDOW = timedelta(hours=30)
RECORD_WINDOW = timedelta(days=7)


# =============================================================================
# pure seams
# =============================================================================

def _matchup(g: dict) -> str:
    return f"{g['away_team']} @ {g['home_team']} (game_pk {g['game_pk']})"


def plan_check(games: list[dict], injuries_by_team: dict, stored: dict[int, dict]) -> tuple[list[int], list[str]]:
    """PURE. games = [{game_pk, home_team, away_team}], stored = {game_pk ->
    {"fingerprint", "statuses"}}. Returns (changed game_pks, summary lines):
    per changed game its matchup, then indented `diff_statuses` lines (or
    "no stored snapshot")."""
    by_pk = {g["game_pk"]: g for g in games}
    current = {pk: game_statuses(g["home_team"], g["away_team"], injuries_by_team) for pk, g in by_pk.items()}
    changed = changed_games({pk: fingerprint(st) for pk, st in current.items()},
                            {pk: s.get("fingerprint") for pk, s in (stored or {}).items()})
    lines: list[str] = []
    for pk in changed:
        lines.append(_matchup(by_pk[pk]))
        if pk not in (stored or {}):
            lines.append("  no stored snapshot")
        else:
            lines.extend(f"  {d}" for d in diff_statuses(stored[pk].get("statuses") or [], current[pk]))
    return changed, lines


def snapshot_rows(sport: str, games: list[dict], injuries_by_team: dict) -> list[dict]:
    """PURE. One `injury_snapshots` row per game."""
    rows = []
    for g in games:
        st = game_statuses(g["home_team"], g["away_team"], injuries_by_team)
        rows.append({"sport": sport, "game_pk": g["game_pk"], "fingerprint": fingerprint(st), "statuses": st})
    return rows


def write_github_output(changed: bool) -> None:
    """Append `changed=true|false` to $GITHUB_OUTPUT when it is set."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        fh.write(f"changed={'true' if changed else 'false'}\n")


# =============================================================================
# IO
# =============================================================================

def fetch_games(sport: str, now: datetime, horizon: datetime) -> list[dict]:
    from sportsmodel.db import get_postgres

    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT game_pk, home_team_name, away_team_name, commence_time
            FROM predictions_current
            WHERE sport = %s AND commence_time > %s AND commence_time <= %s
            ORDER BY commence_time, game_pk
        """, (sport, now, horizon))
        return [{"game_pk": int(pk), "home_team": h, "away_team": a, "commence_time": ct}
                for pk, h, a, ct in cur.fetchall()]


def _load_desk_inputs():
    p = Path(__file__).resolve().parent / "desk_inputs.py"
    spec = importlib.util.spec_from_file_location("desk_inputs", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nfl_injuries(now: datetime) -> dict[str, list[dict]]:
    from sportsmodel.nfl import injury_report

    crosswalk = json.loads((config.PROJECT_ROOT / "assets" / "nfl" / "nfl_teams.json").read_text())
    name_to_abbr = {name: abbr for abbr, name in crosswalk.items()}
    report = injury_report.current_report(now, injury_report.resolve_target_week(now), name_to_abbr)
    if report.get("stale"):
        print(f"WARN nfl injury report stale (report week {report.get('report_week')}, "
              f"target week {report.get('target_week')}, espn_available={report.get('espn_available')})")
    return {crosswalk[abbr]: rows for abbr, rows in report["by_team"].items() if abbr in crosswalk}


def cfb_injuries(now: datetime, api_key: str, espn_names: list[str]) -> dict[str, list[dict]]:
    desk = _load_desk_inputs()
    cfg = desk._sport_config("cfb")
    cw_path = cfg["crosswalk_path"]
    crosswalk = json.loads(cw_path.read_text()) if cw_path.exists() else {}
    by_name, _meta = desk._cfb_injuries_by_name(cfg["adapter"], api_key, crosswalk, now)
    return desk._rekey_by_espn_name(by_name, espn_names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Day-before injury watch (check / record snapshots).")
    ap.add_argument("--sport", choices=["nfl", "cfb"], required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--record", action="store_true")
    args = ap.parse_args(argv)

    api_key = os.environ.get("SPORTSDATA_API_KEY")
    if args.sport == "cfb" and not api_key:
        print("WARN SPORTSDATA_API_KEY not set; cfb injuries unavailable -- "
              + ("reporting changed=false" if args.check else "recording nothing"))
        if args.check:
            write_github_output(False)
        return 0

    now = datetime.now(timezone.utc)
    games = fetch_games(args.sport, now, now + (CHECK_WINDOW if args.check else RECORD_WINDOW))
    if not games:
        print(f"No upcoming {args.sport} games in the window.")
        if args.check:
            write_github_output(False)
        return 0

    if args.sport == "nfl":
        injuries = nfl_injuries(now)
    else:
        espn_names = sorted({g["home_team"] for g in games} | {g["away_team"] for g in games})
        injuries = cfb_injuries(now, api_key, espn_names)

    from sportsmodel import db

    if args.record:
        n = db.upsert_injury_snapshots(snapshot_rows(args.sport, games, injuries))
        print(f"Recorded {n} {args.sport} injury snapshot(s).")
        return 0

    stored = db.load_injury_snapshots(args.sport, [g["game_pk"] for g in games])
    changed, lines = plan_check(games, injuries, stored)
    print(f"{args.sport}: {len(changed)} of {len(games)} game(s) within 30h have injury changes.")
    for line in lines:
        print(line)
    write_github_output(bool(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
