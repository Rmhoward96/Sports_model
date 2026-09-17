"""Drive-based Monte Carlo kernel for NFL simulation.

Provides pure kernel helpers for sampling drive outcomes and computing
game results.
"""
from __future__ import annotations

import numpy as np

from sportsmodel.sim.nfl.spec import PlayerInput, TeamRates

# Per-play yardage variance and floor. Yards on a single target/carry are
# drawn from a Normal distribution centered on the player's rate stat and
# floored so a single play can't produce an absurd negative outcome.
_PASS_PLAY_YDS_SD = 6.0
_RUSH_PLAY_YDS_SD = 4.0
_MIN_PLAY_YDS = -5.0


def sample_drive(off: TeamRates, deff: TeamRates, rng) -> tuple[str, int]:
    """Sample a single drive outcome combining offense and defense rates.

    Combines the offense and defense drive-outcome probabilities by averaging
    them element-wise, then draws a single outcome from the resulting
    multinomial distribution.

    Args:
        off: Offensive team rates.
        deff: Defensive team rates.
        rng: numpy random Generator (e.g., np.random.default_rng()).

    Returns:
        Tuple of (outcome_key, points) where outcome_key is one of
        'td', 'fg', 'punt', 'turnover', 'downs', 'end' and points is
        7 for TD, 3 for FG, 0 otherwise.
    """
    # Average the drive outcome rates element-wise
    combined = {}
    for key in off.drive_outcomes:
        combined[key] = (off.drive_outcomes[key] + deff.drive_outcomes[key]) / 2.0

    # Renormalize to sum to 1
    total = sum(combined.values())
    combined = {k: v / total for k, v in combined.items()}

    # Prepare for multinomial draw
    outcome_keys = list(combined.keys())
    probs = np.array([combined[k] for k in outcome_keys])

    # Create cumulative probabilities
    cumsum = np.cumsum(probs)

    # Draw uniform random value and find corresponding outcome
    rand_val = rng.random()
    outcome_idx = np.searchsorted(cumsum, rand_val)

    # Cap to valid range (handles edge case where rand_val is very close to 1.0)
    outcome_idx = min(outcome_idx, len(outcome_keys) - 1)
    outcome_key = outcome_keys[outcome_idx]

    # Map outcome to points
    points_map = {"td": 7, "fg": 3}
    points = points_map.get(outcome_key, 0)

    return (outcome_key, points)


def _normalized_probs(shares: list[float]) -> np.ndarray:
    """Clip negative shares to 0 and normalize to sum to 1.

    Falls back to a uniform distribution when all shares are non-positive,
    so a multinomial draw never receives an all-zero pvals vector.
    """
    arr = np.clip(np.array(shares, dtype=float), 0.0, None)
    total = arr.sum()
    if total <= 0:
        return np.full(len(arr), 1.0 / len(arr)) if len(arr) else arr
    return arr / total


def attribute_offense(
    players: list[PlayerInput],
    n_pass: int,
    n_rush: int,
    n_off_tds: int,
    rng,
) -> dict[str, dict[str, int]]:
    """Attribute a drive-based offensive box score to individual players.

    Allocates `n_pass` targets across `players` by `target_share` and
    `n_rush` carries by `carry_share` (both via multinomial draws), then
    simulates each individual target/carry: yards per target are drawn
    around the player's `ypt`, gated by `catch_rate` to determine whether
    the target becomes a reception (only receptions accrue rec_yds); yards
    per carry are drawn around the player's `ypc`. Touchdowns (`n_off_tds`)
    are allocated across players by `td_share` via a separate multinomial
    draw. The team's passing yards are attributed to the QB (the player
    whose `pos == "QB"`; if zero or multiple players have that position,
    the player with the highest `target_share` is used instead) as the sum
    of the whole team's receiving yards.

    Args:
        players: Offensive players to attribute stats to.
        n_pass: Number of pass attempts (targets) to allocate this game.
        n_rush: Number of rush attempts (carries) to allocate this game.
        n_off_tds: Number of offensive touchdowns to allocate this game.
        rng: numpy random Generator (e.g., np.random.default_rng()).

    Returns:
        Dict mapping player_id -> {"pass_yds", "rush_yds", "rec_yds",
        "receptions", "td"}, all ints.
    """
    if not players:
        return {}

    stats: dict[str, dict[str, int]] = {
        p.player_id: {"pass_yds": 0, "rush_yds": 0, "rec_yds": 0, "receptions": 0, "td": 0}
        for p in players
    }

    target_probs = _normalized_probs([p.target_share for p in players])
    carry_probs = _normalized_probs([p.carry_share for p in players])
    td_probs = _normalized_probs([p.td_share for p in players])

    target_counts = rng.multinomial(n_pass, target_probs)
    carry_counts = rng.multinomial(n_rush, carry_probs)
    td_counts = rng.multinomial(n_off_tds, td_probs)

    for player, n_targets in zip(players, target_counts):
        pdata = stats[player.player_id]
        for _ in range(int(n_targets)):
            if rng.random() < player.catch_rate:
                yds = max(rng.normal(player.ypt, _PASS_PLAY_YDS_SD), _MIN_PLAY_YDS)
                pdata["rec_yds"] += int(round(yds))
                pdata["receptions"] += 1

    for player, n_carries in zip(players, carry_counts):
        pdata = stats[player.player_id]
        for _ in range(int(n_carries)):
            yds = max(rng.normal(player.ypc, _RUSH_PLAY_YDS_SD), _MIN_PLAY_YDS)
            pdata["rush_yds"] += int(round(yds))

    for player, n_td in zip(players, td_counts):
        stats[player.player_id]["td"] += int(n_td)

    qb_candidates = [p for p in players if p.pos == "QB"]
    qb = qb_candidates[0] if len(qb_candidates) == 1 else max(players, key=lambda p: p.target_share)
    team_rec_yds = sum(pdata["rec_yds"] for pdata in stats.values())
    stats[qb.player_id]["pass_yds"] = team_rec_yds

    return stats
