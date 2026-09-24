"""PURE helpers for the day-before injury watch (scripts/injury_watch.py).

A game's injury "fingerprint" covers only Out/Doubtful designations for its two
teams -- Questionable churn never triggers a re-run. Statuses are normalized
via `injury_report.normalize_status`, so IR/PUP/NFI/suspended count as Out.
Players are matched on `injury_report.name_key` (so ESPN's "Ja’Marr Chase" and
nflverse's "Ja'Marr Chase" are one player); the stored rows keep display names.
"""
from __future__ import annotations

import hashlib

from sportsmodel.nfl.injury_report import name_key, normalize_status

_WATCHED = ("Out", "Doubtful")
_NONE = "(none)"


def game_statuses(home: str, away: str, injuries_by_team: dict) -> list[dict]:
    """[{team, player, status}] for the game's Out/Doubtful players, sorted by
    (team, player) and de-duplicated on (team, name_key(player)); a player
    listed twice keeps the more severe status (Out over Doubtful) and the
    lexically-smallest display name, so input order never matters."""
    seen: dict[tuple[str, str], dict] = {}
    for team in (home, away):
        for r in (injuries_by_team or {}).get(team) or []:
            st = normalize_status(r.get("status"))
            player = str(r.get("player") or "").strip()
            k = name_key(player)
            if st not in _WATCHED or not k:
                continue
            prev = seen.get((team, k))
            if prev is None:
                seen[(team, k)] = {"team": team, "player": player, "status": st}
            else:
                prev["player"] = min(prev["player"], player)
                if st == "Out":
                    prev["status"] = "Out"
    return sorted(seen.values(), key=lambda s: (s["team"], s["player"]))


def fingerprint(statuses: list[dict]) -> str:
    """sha1 hex of the sorted `team|name_key(player)|status` lines
    (order-insensitive; spelling variants of one name hash the same)."""
    lines = sorted(f"{s['team']}|{name_key(s['player'])}|{s['status']}" for s in statuses)
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def diff_statuses(old: list[dict], new: list[dict]) -> list[str]:
    """Human lines for each player whose watched status differs, e.g.
    "Atlanta Falcons: Michael Penix Jr. Out -> (none)". Players are matched on
    name_key (a spelling-only change is no diff); the newest display name is
    shown. Sorted by (team, player)."""
    o = {(s["team"], name_key(s["player"])): s for s in old or []}
    n = {(s["team"], name_key(s["player"])): s for s in new or []}
    rows = []
    for key in set(o) | set(n):
        before = o[key]["status"] if key in o else _NONE
        after = n[key]["status"] if key in n else _NONE
        if before != after:
            player = (n.get(key) or o[key])["player"]
            rows.append((key[0], player, before, after))
    return [f"{team}: {player} {before} -> {after}" for team, player, before, after in sorted(rows)]


def changed_games(current: dict[int, str], stored: dict[int, str]) -> list[int]:
    """game_pks whose current fingerprint differs from the stored one; a game
    with no stored fingerprint counts as changed. Sorted."""
    return sorted(pk for pk, fp in current.items() if stored.get(pk) != fp)
