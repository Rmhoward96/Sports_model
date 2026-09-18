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


def _assert_qb_attribution(players: list[PlayerInput], qb_gsis: str | None, team: str) -> None:
    """Ruling C1: if `qb_gsis` is given, `players` must contain EXACTLY ONE
    `pos=="QB"` and its `player_id` must equal `qb_gsis`. If `qb_gsis` is
    None, no assertion is made (the no-QB edge case is allowed)."""
    if qb_gsis is None:
        return
    qbs = [p for p in players if p.pos == "QB"]
    if len(qbs) != 1 or qbs[0].player_id != qb_gsis:
        found = [(p.player_id, p.pos) for p in qbs]
        raise ValueError(
            f"{team}: expected exactly one QB with player_id == {qb_gsis!r}, "
            f"found QB(s) {found!r}"
        )


def build_spec_from_usage(
    home_team: str,
    away_team: str,
    rates: dict[str, TeamRates],
    home_players: list[PlayerInput],
    away_players: list[PlayerInput],
    home_qb_gsis: str | None,
    away_qb_gsis: str | None,
) -> NflGameSpec:
    """Assemble an NflGameSpec from active-usage PlayerInputs (usage.active_usage).

    Ruling B1: `home_players`/`away_players` are ALREADY active + injury-
    filtered + renormalized-over-the-active-set by `active_usage`. This
    function does NOT re-drop injuries and does NOT re-renormalize -- it
    assembles the spec directly from the passed players.

    Ruling C1: if `home_qb_gsis`/`away_qb_gsis` is not None, asserts the
    corresponding players list contains exactly one `pos=="QB"` player whose
    `player_id` equals that gsis id (raises ValueError otherwise), making the
    kernel's pass_yds-attribution guarantee explicit at spec-build time. A
    None qb_gsis allows zero QBs (no-QB edge case) with no assertion.

    Args:
        home_team: Home team identifier, used as key into rates.
        away_team: Away team identifier, used as key into rates.
        rates: Dict mapping team -> TeamRates.
        home_players: Home team's active PlayerInputs (already filtered/renormalized).
        away_players: Away team's active PlayerInputs (already filtered/renormalized).
        home_qb_gsis: Home team's starting QB gsis_id, or None.
        away_qb_gsis: Away team's starting QB gsis_id, or None.

    Returns:
        NflGameSpec built directly from the passed players.

    Raises:
        KeyError: If home_team/away_team is missing from `rates`.
        ValueError: If a non-None qb_gsis doesn't match exactly one QB in
            that side's players.
    """
    if home_team not in rates:
        raise KeyError(f"No rates found for home team {home_team!r}")
    if away_team not in rates:
        raise KeyError(f"No rates found for away team {away_team!r}")

    _assert_qb_attribution(home_players, home_qb_gsis, home_team)
    _assert_qb_attribution(away_players, away_qb_gsis, away_team)

    return NflGameSpec(
        home_team=home_team,
        away_team=away_team,
        home=rates[home_team],
        away=rates[away_team],
        home_players=home_players,
        away_players=away_players,
    )
