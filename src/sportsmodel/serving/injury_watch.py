"""PURE helpers for the day-before injury watch (scripts/injury_watch.py).

A game's injury "fingerprint" covers only Out/Doubtful designations for its two
teams -- Questionable churn never triggers a re-run. Statuses are normalized
via `injury_report.normalize_status`, so IR/PUP/NFI/suspended count as Out.
"""
from __future__ import annotations

import hashlib

from sportsmodel.nfl.injury_report import normalize_status

_WATCHED = ("Out", "Doubtful")
_NONE = "(none)"


def game_statuses(home: str, away: str, injuries_by_team: dict) -> list[dict]:
    """[{team, player, status}] for the game's Out/Doubtful players, sorted by
    (team, player) and de-duplicated; a player listed twice keeps the more
    severe status (Out over Doubtful), so input order never matters."""
    seen: dict[tuple[str, str], str] = {}
    for team in (home, away):
        for r in (injuries_by_team or {}).get(team) or []:
            st = normalize_status(r.get("status"))
            player = str(r.get("player") or "").strip()
            if st in _WATCHED and player and seen.get((team, player)) != "Out":
                seen[(team, player)] = st
    return [{"team": t, "player": p, "status": s} for (t, p), s in sorted(seen.items())]


def fingerprint(statuses: list[dict]) -> str:
    """sha1 hex of the sorted `team|player|status` lines (order-insensitive)."""
    lines = sorted(f"{s['team']}|{s['player']}|{s['status']}" for s in statuses)
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def diff_statuses(old: list[dict], new: list[dict]) -> list[str]:
    """Human lines for each player whose watched status differs, e.g.
    "Atlanta Falcons: Michael Penix Jr. Out -> (none)". Sorted by (team, player)."""
    o = {(s["team"], s["player"]): s["status"] for s in old or []}
    n = {(s["team"], s["player"]): s["status"] for s in new or []}
    lines = []
    for team, player in sorted(set(o) | set(n)):
        before, after = o.get((team, player), _NONE), n.get((team, player), _NONE)
        if before != after:
            lines.append(f"{team}: {player} {before} -> {after}")
    return lines


def changed_games(current: dict[int, str], stored: dict[int, str]) -> list[int]:
    """game_pks whose current fingerprint differs from the stored one; a game
    with no stored fingerprint counts as changed. Sorted."""
    return sorted(pk for pk, fp in current.items() if stored.get(pk) != fp)
