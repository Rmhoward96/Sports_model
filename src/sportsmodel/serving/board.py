"""Pure pick-math for the serving board: best-book selection, EV, no-vig, and
per-market row builders. Shared by scripts/generate_board.py and grade_results.py
so the live board and the graded track record can never drift."""
from __future__ import annotations

from ..model.calibration import calibrate
from ..model.distributions import apply_affine, prob_cover, prob_over_dist


def decimal_odds(american: int) -> float:
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied_prob(american: int) -> float:
    return 1.0 / decimal_odds(american)


def novig(price_side: int, price_other: int) -> float:
    """No-vig implied probability of `price_side` given the two-way market."""
    io, iu = implied_prob(price_side), implied_prob(price_other)
    return io / (io + iu)


def best_price(entries):
    """(book, american) with the highest decimal odds (best for the bettor); None if empty."""
    entries = [(bk, p) for bk, p in (entries or []) if p]
    if not entries:
        return None
    return max(entries, key=lambda e: decimal_odds(e[1]))


def ev(prob: float, american: int) -> float:
    return prob * decimal_odds(american) - 1
