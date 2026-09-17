"""Drive-based Monte Carlo kernel for NFL simulation.

Provides pure kernel helpers for sampling drive outcomes and computing
game results.
"""
from __future__ import annotations

import numpy as np

from sportsmodel.sim.nfl.spec import TeamRates


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
