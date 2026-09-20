"""NFL simulation spec dataclasses and result container.

Defines the input specification for the NFL drive-based Monte Carlo engine
and the output container for simulation results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TeamRates:
    """Offensive metrics for a team in a game.

    Attributes:
        drive_outcomes: Dict mapping drive end outcomes to probabilities.
            Keys: "td", "fg", "punt", "turnover", "downs", "end" (sum ≈ 1.0).
        pass_rate: Fraction of plays that are passing plays.
        drives_per_game: Expected number of drives in a game.
        rz_td_rate: Touchdown rate when in the red zone.
        pass_att_pg: Real pass attempts per game (excludes sacks).
        rush_att_pg: Rush attempts per game.
        sack_rate: Sacks / pass plays.
        completion_pct: Completions / attempts.
    """
    drive_outcomes: dict[str, float]
    pass_rate: float
    drives_per_game: float
    rz_td_rate: float
    pass_att_pg: float = 0.0
    rush_att_pg: float = 0.0
    sack_rate: float = 0.0
    completion_pct: float = 0.0
    pass_td_share: float = 0.0  # fraction of the team's offensive TDs that are passing (receiving)


@dataclass(frozen=True)
class PlayerInput:
    """NFL player statistics for simulation input.

    Attributes:
        player_id: Unique player identifier.
        name: Player's name.
        pos: Player position.
        target_share: Fraction of team targets allocated to this player.
        carry_share: Fraction of team carries allocated to this player.
        ypt: Yards per target (kept for reference/back-compat; the kernel
            attributes reception yardage using `ypr`, not `ypt`).
        ypc: Yards per carry.
        ypr: Yards per reception (receiving_yards / receptions). This is
            what the kernel uses to draw a completed reception's yardage --
            using ypt there would understate receiver output by roughly a
            factor of catch_rate, since ypt = catch_rate * ypr.
        catch_rate: Fraction of targets caught.
        td_share: Fraction of team touchdowns allocated to this player.
    """
    player_id: str
    name: str
    pos: str
    target_share: float
    carry_share: float
    ypt: float
    ypc: float
    ypr: float
    catch_rate: float
    td_share: float
    rec_td_share: float = 0.0   # share of the team's PASSING (receiving) TDs
    rush_td_share: float = 0.0  # share of the team's RUSHING TDs


@dataclass(frozen=True)
class NflGameSpec:
    """Full specification for an NFL game simulation.

    Attributes:
        home_team: Home team name/identifier.
        away_team: Away team name/identifier.
        home: Home team rates and offensive metrics.
        away: Away team rates and offensive metrics.
        home_players: List of home team players.
        away_players: List of away team players.
    """
    home_team: str
    away_team: str
    home: TeamRates
    away: TeamRates
    home_players: list[PlayerInput]
    away_players: list[PlayerInput]


@dataclass
class NflGameSims:
    """Container for NFL game simulation results.

    Duck-types with the engine's score helper functions (home_win_prob,
    margin_pmf, etc.) via home_score and away_score arrays.

    Attributes:
        home_score: Per-simulation home team final scores (numpy array).
        away_score: Per-simulation away team final scores (numpy array).
        player_stats: Per-player market statistics. Structure:
            {player_id: {market_name: np.ndarray of values}}.
    """
    home_score: np.ndarray
    away_score: np.ndarray
    player_stats: dict[str, dict[str, np.ndarray]]
