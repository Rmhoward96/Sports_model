"""What one monitor poll asks the Odds API for, what it costs, and when the credit floor stops it. PURE -- no IO.

The call uses the `bookmakers` parameter instead of `regions`: the 9 US books plus Pinnacle are 10 bookmakers, which the
Odds API bills as one region, so a poll costs one credit per market: 2 for CFB (spreads, totals), 1 for NFL (spreads).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import API_MARKETS, BOOKMAKERS, MONITOR_FLOOR

BOOKMAKERS_PER_REGION = 10


def bookmakers_param() -> str:
    """The `bookmakers` query value: the 10 keys, comma separated."""
    return ",".join(BOOKMAKERS)


def api_markets(sport: str) -> list[str]:
    return list(API_MARKETS[sport])


def regions_billed(n_books: int = len(BOOKMAKERS)) -> int:
    return max(1, math.ceil(n_books / BOOKMAKERS_PER_REGION))


def poll_cost(sport: str) -> int:
    """Credits one poll of `sport` is billed: markets x regions."""
    return len(API_MARKETS[sport]) * regions_billed()


def parse_credits(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class BudgetDecision:
    ok: bool
    warning: str | None = None


def budget_decision(remaining) -> BudgetDecision:
    """May the monitor make its next paid call, given `x-requests-remaining` from the last response?

    Below MONITOR_FLOOR (1,500) it may not and says why. The value is unknown before the first response of a run (or
    when the header is missing or unreadable), and an unknown value runs: the first call always runs."""
    credits = parse_credits(remaining)
    if credits is None:
        return BudgetDecision(True)
    if credits < MONITOR_FLOOR:
        return BudgetDecision(False, f"{credits} Odds API credits left (< {MONITOR_FLOOR}): "
                                     "the early-week monitor is skipping its remaining calls")
    return BudgetDecision(True)
