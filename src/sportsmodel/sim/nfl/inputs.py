"""Assemble a per-game NflGameSpec, zeroing out injured players.

Pure function: injuries are passed in as a dict (no IO here). Callers are
responsible for sourcing the injuries dict, e.g. from
sportsmodel.nfl.injuries_nflverse.current_injuries.
"""
from __future__ import annotations

from dataclasses import replace

from sportsmodel.sim.nfl.spec import NflGameSpec, PlayerInput, TeamRates

_SHARE_FIELDS = ("target_share", "carry_share", "td_share")


def _renormalize(players: list[PlayerInput]) -> list[PlayerInput]:
    """Renormalize target_share, carry_share, and td_share to sum to 1.

    If a share's sum across the surviving players is 0, that share is left
    unchanged (avoids divide-by-zero; nothing meaningful to renormalize).
    """
    if not players:
        return players

    sums = {field: sum(getattr(p, field) for p in players) for field in _SHARE_FIELDS}

    out = []
    for p in players:
        updates = {}
        for field in _SHARE_FIELDS:
            total = sums[field]
            if total > 0:
                updates[field] = getattr(p, field) / total
        out.append(replace(p, **updates) if updates else p)
    return out


def _team_out_names(injuries: dict[str, list[dict]], team: str, out_statuses: frozenset[str]) -> set[str]:
    entries = injuries.get(team, [])
    out_lower = {s.lower() for s in out_statuses}
    return {
        entry["player"].strip().lower()
        for entry in entries
        if str(entry.get("status", "")).strip().lower() in out_lower
    }


def _build_team_players(
    team: str,
    players: dict[str, list[PlayerInput]],
    injuries: dict[str, list[dict]],
    out_statuses: frozenset[str],
) -> list[PlayerInput]:
    if team not in players:
        raise KeyError(f"No player inputs found for team {team!r}")

    out_names = _team_out_names(injuries, team, out_statuses)
    surviving = [p for p in players[team] if p.name.strip().lower() not in out_names]
    return _renormalize(surviving)


def build_spec(
    home_team: str,
    away_team: str,
    rates: dict[str, TeamRates],
    players: dict[str, list[PlayerInput]],
    injuries: dict[str, list[dict]],
    out_statuses: frozenset[str] = frozenset({"Out", "Doubtful"}),
) -> NflGameSpec:
    """Assemble an NflGameSpec for a game, dropping injured players.

    For each of home_team/away_team, players whose name matches an injury
    entry for that team with a status in `out_statuses` (case-insensitive
    name match; "Doubtful" is treated as "Out" by default, "Questionable" is
    kept by default) are dropped. Remaining players' target_share,
    carry_share, and td_share are each renormalized to sum to 1 (a share left
    unchanged if its sum is 0).

    Args:
        home_team: Home team identifier, used as key into rates/players/injuries.
        away_team: Away team identifier, used as key into rates/players/injuries.
        rates: Dict mapping team -> TeamRates.
        players: Dict mapping team -> list[PlayerInput].
        injuries: Dict mapping team -> list of {"player","position","status","note"}.
        out_statuses: Statuses (case-insensitive) treated as ruled out.

    Returns:
        NflGameSpec with surviving, renormalized players for both teams.

    Raises:
        KeyError: If home_team/away_team is missing from `rates` or `players`.
    """
    if home_team not in rates:
        raise KeyError(f"No rates found for home team {home_team!r}")
    if away_team not in rates:
        raise KeyError(f"No rates found for away team {away_team!r}")

    home_players = _build_team_players(home_team, players, injuries, out_statuses)
    away_players = _build_team_players(away_team, players, injuries, out_statuses)

    return NflGameSpec(
        home_team=home_team,
        away_team=away_team,
        home=rates[home_team],
        away=rates[away_team],
        home_players=home_players,
        away_players=away_players,
    )
