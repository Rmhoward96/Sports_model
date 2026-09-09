"""Pure, schedule-derived feature functions (no IO, no leakage).

These functions compute per-team features from an **Elo-augmented per-game
schedule** — the kind of frame produced by ``run_elo(...).games`` (see
``sportsmodel.nfl.elo``). Each row of the consumed frame represents one
game and carries (at minimum):

    season      : int   — season year
    week        : int   — week number within the season
    home_team   : str   — home team code
    away_team   : str   — away team code
    home_score  : float — home team's final score (NaN/None if not yet played)
    away_score  : float — away team's final score (NaN/None if not yet played)
    elo_home    : float — home team's PRE-GAME Elo rating for this game
    elo_away    : float — away team's PRE-GAME Elo rating for this game
    gameday     : str   — ISO date ('YYYY-MM-DD') the game was/will be played
                          (NFL only; not required for CFB-only callers)

All functions here are pure: DataFrame/rows in, value out. They never read
files, hit the network, or mutate their inputs.

No-leakage contract
--------------------
Callers MUST pass ``prior_games`` as a slice containing only games strictly
before the target game (e.g. ``schedule[schedule.index < target_idx]``, or
an equivalent chronological cut). As a second line of defense:

  * ``last10_win_pct``, ``sos``, and ``sov`` only ever use games with both
    scores present (i.e. already completed) — an accidentally-included
    current/future game has NaN scores and is silently excluded from win/
    loss and Elo aggregation (though it WOULD still count as a "game
    played" for ``sos``/``sov`` mean-Elo purposes if it had scores, so
    correct slicing by the caller is still required).
  * ``rest_days_nfl`` only considers prior games whose ``gameday`` is
    strictly earlier than ``this_gameday``.
  * ``rest_weeks_cfb`` only considers prior games whose ``week`` is
    strictly less than ``this_week``.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def _team_rows(prior_games: pd.DataFrame, team: str) -> list[dict[str, Any]]:
    """Iterate prior_games for `team` (home or away), yielding per-game facts.

    Internal helper — not part of the public API. For each row where `team`
    played (as either home or away), returns a dict with the team's own
    score, the opponent's score, the opponent's PRE-GAME Elo (elo_away when
    `team` was home, elo_home when `team` was away), plus `week`, `season`,
    and `gameday` carried through for the rest/bye calculations.
    """
    rows: list[dict[str, Any]] = []
    if prior_games is None or len(prior_games) == 0:
        return rows
    for _, g in prior_games.iterrows():
        home = g["home_team"]
        away = g["away_team"]
        if home == team:
            own_score = g["home_score"]
            opp_score = g["away_score"]
            opp_elo = g["elo_away"]
        elif away == team:
            own_score = g["away_score"]
            opp_score = g["home_score"]
            opp_elo = g["elo_home"]
        else:
            continue
        rows.append(
            {
                "own_score": own_score,
                "opp_score": opp_score,
                "opp_elo": opp_elo,
                "week": g.get("week"),
                "season": g.get("season"),
                "gameday": g.get("gameday"),
            }
        )
    return rows


def _is_completed(row: dict[str, Any]) -> bool:
    return pd.notna(row["own_score"]) and pd.notna(row["opp_score"])


def _is_win(row: dict[str, Any]) -> bool:
    return _is_completed(row) and row["own_score"] > row["opp_score"]


def last10_win_pct(prior_games: pd.DataFrame, team: str) -> float | None:
    """Win fraction over the team's last <=10 COMPLETED prior games.

    A game counts only if both scores are present. None if there is no
    completed prior game.
    """
    rows = _team_rows(prior_games, team)
    completed = [r for r in rows if _is_completed(r)]
    if not completed:
        return None
    completed.sort(key=lambda r: (r["season"], r["week"]))
    last10 = completed[-10:]
    wins = sum(1 for r in last10 if _is_win(r))
    return wins / len(last10)


def sos(prior_games: pd.DataFrame, team: str) -> float | None:
    """Mean of the opponent's pre-game Elo across the team's prior games.

    None if there is no prior game.
    """
    rows = _team_rows(prior_games, team)
    if not rows:
        return None
    return sum(r["opp_elo"] for r in rows) / len(rows)


def sov(prior_games: pd.DataFrame, team: str) -> float | None:
    """Mean of the opponent's pre-game Elo, restricted to games `team` WON.

    None if there are no wins among the prior games.
    """
    rows = _team_rows(prior_games, team)
    won = [r for r in rows if _is_win(r)]
    if not won:
        return None
    return sum(r["opp_elo"] for r in won) / len(won)


def rest_days_nfl(
    prior_games: pd.DataFrame, team: str, this_gameday: str
) -> int | None:
    """Days between `this_gameday` and the team's most-recent prior game.

    Dates are ISO strings ('YYYY-MM-DD'), parsed with date.fromisoformat.
    Only prior games strictly before `this_gameday` are considered (a
    leakage guard). None if there is no such prior game.
    """
    this_date = _to_date(this_gameday)
    rows = _team_rows(prior_games, team)
    prior_dates = []
    for r in rows:
        gd = r["gameday"]
        if gd is None or (isinstance(gd, float) and pd.isna(gd)):
            continue
        d = _to_date(gd)
        if d < this_date:
            prior_dates.append(d)
    if not prior_dates:
        return None
    most_recent = max(prior_dates)
    return (this_date - most_recent).days


def rest_weeks_cfb(
    prior_games: pd.DataFrame, team: str, this_week: int
) -> dict[str, Any]:
    """Weeks since the team's most-recent prior game (within the season slice
    the caller passes), and whether that gap indicates a bye.

    Returns {"weeks_since_prev": int | None, "off_bye": bool}. Only prior
    games with week strictly less than `this_week` are considered (a
    leakage guard). weeks_since_prev is None (and off_bye False) if there
    is no such prior game; off_bye is True iff weeks_since_prev > 1.
    """
    rows = _team_rows(prior_games, team)
    weeks = [
        r["week"]
        for r in rows
        if r["week"] is not None
        and not (isinstance(r["week"], float) and pd.isna(r["week"]))
        and r["week"] < this_week
    ]
    if not weeks:
        return {"weeks_since_prev": None, "off_bye": False}
    most_recent_week = max(weeks)
    weeks_since_prev = this_week - most_recent_week
    return {"weeks_since_prev": weeks_since_prev, "off_bye": weeks_since_prev > 1}


def _to_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
