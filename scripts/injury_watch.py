"""Day-before injury watch: detect Out/Doubtful changes since the last desk run.

    --check   games kicking off within the next 30h: fingerprint each game's
              Out/Doubtful designations, compare to `injury_snapshots`, print
              each changed game's matchup + per-player diff, and append
              `changed=true|false` to $GITHUB_OUTPUT (when set). Exits non-zero
              on DB/SportsDataIO errors, so the run is visibly red and the gated
              re-run is skipped; exits 0 with changed=false when ESPN is degraded
              (NFL) or the CFB key is missing.
    --record  snapshot into `injury_snapshots`. With `--bundle PATH` it records
              exactly what the desk used (the desk_inputs bundle JSON: per game
              `game_pk`, `matchup` "Away @ Home", `news.injuries.home/away`);
              without it, it re-fetches injuries for games in the next 7 days.
              Never fails the calling workflow: any error is a WARN + exit 0 (a
              missing snapshot is safe -- the next check treats it as changed).

A game with no stored snapshot counts as changed. Questionable designations
never count (see sportsmodel.serving.injury_watch).

Injuries are fetched exactly as desk_inputs.main does (one code path for both
sports): the sport's `injury_source(adapter, api_key, crosswalk, now)` from
`_sport_config`, then `_rekey_by_espn_name` over the team names of the FULL
7-day game list (the desk's list -- prefix-ambiguous keys like Iowa / Iowa
State are dropped the same way the desk drops them); only then is `--check`
narrowed to the 30h games.
  - CFB needs SPORTSDATA_API_KEY; when unset (and no --bundle) this warns and
    exits 0 (changed=false for --check, nothing recorded for --record).
  - NFL: if ESPN was unreachable (`espn_available` False) this warns, and
    `--check` treats the run as unknown: changed=false.

The pure seams (`plan_check`, `check_inputs`, `snapshot_rows`,
`bundle_snapshot_rows`, `write_github_output`) are unit-tested; main()'s
DB/injury IO is thin.

Usage:
    DATABASE_URL=... uv run python scripts/injury_watch.py --sport nfl --check
    DATABASE_URL=... uv run python scripts/injury_watch.py --sport cfb --record --bundle data/cfb/desk_bundle.json
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


def _espn_names(games: list[dict]) -> list[str]:
    return sorted({g["home_team"] for g in games} | {g["away_team"] for g in games})


def check_inputs(games_7d: list[dict], by_name: dict, rekey, now: datetime) -> tuple[list[dict], dict]:
    """PURE (given a pure `rekey`). Rekey `by_name` over the FULL 7-day game
    list's names (as the desk does), THEN narrow to games with
    commence_time <= now + 30h. Returns (games_30h, injuries_by_team)."""
    injuries = rekey(by_name, _espn_names(games_7d))
    horizon = now + CHECK_WINDOW
    return [g for g in games_7d if now < g["commence_time"] <= horizon], injuries


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


def bundle_snapshot_rows(sport: str, bundle: list[dict]) -> list[dict]:
    """PURE. `injury_snapshots` rows from a desk_inputs bundle -- exactly the
    injuries the desk saw. Team names come from `matchup` ("Away @ Home");
    entries whose matchup doesn't split are skipped."""
    rows = []
    for entry in bundle or []:
        parts = str(entry.get("matchup") or "").split(" @ ")
        if len(parts) != 2:
            continue
        away, home = parts
        inj = ((entry.get("news") or {}).get("injuries") or {})
        game = {"game_pk": int(entry["game_pk"]), "home_team": home, "away_team": away}
        rows.extend(snapshot_rows(sport, [game], {home: inj.get("home") or [], away: inj.get("away") or []}))
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


def fetch_injuries(sport: str, now: datetime, api_key: str | None) -> tuple[dict, dict]:
    """({full team name -> rows}, meta) from the sport's desk injury source,
    exactly as desk_inputs.main calls it (not yet rekeyed)."""
    desk = _load_desk_inputs()
    cfg = desk._sport_config(sport)
    cw_path = cfg["crosswalk_path"]
    crosswalk = json.loads(cw_path.read_text()) if cw_path.exists() else {}
    return cfg["injury_source"](cfg["adapter"], api_key, crosswalk, now)


def _rekey(by_name: dict, espn_names: list[str]) -> dict:
    return _load_desk_inputs()._rekey_by_espn_name(by_name, espn_names)


def _record_from_bundle(sport: str, bundle_path: str) -> None:
    from sportsmodel import db

    bundle = json.loads(Path(bundle_path).read_text())
    n = db.upsert_injury_snapshots(bundle_snapshot_rows(sport, bundle))
    print(f"Recorded {n} {sport} injury snapshot(s) from {bundle_path}.")


def _record_from_fetch(sport: str, api_key: str | None) -> None:
    from sportsmodel import db

    now = datetime.now(timezone.utc)
    games = fetch_games(sport, now, now + RECORD_WINDOW)
    if not games:
        print(f"No upcoming {sport} games in the 7-day window; nothing recorded.")
        return
    by_name, meta = fetch_injuries(sport, now, api_key)
    if (meta or {}).get("espn_available") is False:
        print(f"WARN {sport} ESPN injuries unavailable; recording the degraded report")
    n = db.upsert_injury_snapshots(snapshot_rows(sport, games, _rekey(by_name, _espn_names(games))))
    print(f"Recorded {n} {sport} injury snapshot(s).")


def _check(sport: str, api_key: str | None) -> None:
    from sportsmodel import db

    now = datetime.now(timezone.utc)
    games_7d = fetch_games(sport, now, now + RECORD_WINDOW)
    by_name, meta = fetch_injuries(sport, now, api_key) if games_7d else ({}, {})
    if (meta or {}).get("espn_available") is False:
        print(f"WARN {sport} ESPN injuries unavailable; treating this check as unknown (changed=false)")
        write_github_output(False)
        return
    games, injuries = check_inputs(games_7d, by_name, _rekey, now)
    if not games:
        print(f"No {sport} games within 30h.")
        write_github_output(False)
        return
    stored = db.load_injury_snapshots(sport, [g["game_pk"] for g in games])
    changed, lines = plan_check(games, injuries, stored)
    print(f"{sport}: {len(changed)} of {len(games)} game(s) within 30h have injury changes.")
    for line in lines:
        print(line)
    write_github_output(bool(changed))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Day-before injury watch (check / record snapshots).")
    ap.add_argument("--sport", choices=["nfl", "cfb"], required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--record", action="store_true")
    ap.add_argument("--bundle", default=None,
                    help="--record only: snapshot the desk bundle JSON the desk actually used.")
    args = ap.parse_args(argv)
    if args.bundle and not args.record:
        ap.error("--bundle requires --record")

    if args.record and args.bundle:
        try:
            _record_from_bundle(args.sport, args.bundle)
        except Exception as exc:  # noqa: BLE001 -- never fail the calling workflow
            print(f"WARN injury snapshot record failed ({type(exc).__name__}); nothing recorded")
        return 0

    from sportsmodel import config  # noqa: F401 -- loads .env before reading the key

    api_key = os.environ.get("SPORTSDATA_API_KEY")
    if args.sport == "cfb" and not api_key:
        print("WARN SPORTSDATA_API_KEY not set; cfb injuries unavailable -- "
              + ("reporting changed=false" if args.check else "recording nothing"))
        if args.check:
            write_github_output(False)
        return 0

    if args.record:
        try:
            _record_from_fetch(args.sport, api_key)
        except Exception as exc:  # noqa: BLE001 -- never fail the calling workflow
            print(f"WARN injury snapshot record failed ({type(exc).__name__}); nothing recorded")
        return 0

    _check(args.sport, api_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
