"""American-odds helpers for no-vig conversion and CLV (docs/methodology.md §B.4).

Odds are captured for backtesting only — the model never trains on them.
"""
from __future__ import annotations

from collections.abc import Sequence


def american_to_prob(odds: int) -> float:
    """Implied probability from American odds (includes the vig)."""
    if odds < 0:
        return -odds / (-odds + 100)
    return 100 / (odds + 100)


def prob_to_american(p: float) -> int:
    """Fair American odds from a probability (no vig)."""
    if not 0 < p < 1:
        raise ValueError("probability must be in (0, 1)")
    if p >= 0.5:
        return round(-100 * p / (1 - p))
    return round(100 * (1 - p) / p)


def remove_vig(probs: Sequence[float]) -> list[float]:
    """No-vig probabilities: normalize a set of implied probs to sum to 1."""
    total = sum(probs)
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return [p / total for p in probs]


def no_vig_two_way(odds_a: int, odds_b: int) -> tuple[float, float]:
    """No-vig probabilities for a two-way market from both sides' American odds."""
    a, b = american_to_prob(odds_a), american_to_prob(odds_b)
    return tuple(remove_vig([a, b]))  # type: ignore[return-value]
