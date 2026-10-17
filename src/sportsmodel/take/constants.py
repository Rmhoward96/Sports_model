"""Constants for the 'take these' list. Values are fixed by the spec / global constraints."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ..early.polling import parse_credits

# league -> Odds API sport key
LEAGUES: dict[str, str] = {
    "nfl": "americanfootball_nfl",
    "cfb": "americanfootball_ncaaf",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "nba": "basketball_nba",
    "wnba": "basketball_wnba",
    "epl": "soccer_epl",
    "mls": "soccer_usa_mls",
}
SOCCER = ("epl", "mls")                  # 3-way moneyline (home, draw, away)
MODEL_LEAGUES = ("nfl", "cfb", "mlb")

# williamhill_us is Caesars. No bet365 key exists for US leagues (only bet365_au), so it is not sent.
TAKE_BOOKS: tuple[str, ...] = ("draftkings", "fanduel", "betmgm", "williamhill_us",
                               "fanatics", "hardrockbet", "ballybet")
REFERENCE = "pinnacle"
BOOKMAKERS: tuple[str, ...] = TAKE_BOOKS + (REFERENCE,)

MARKETS = ("ml", "spread", "total")
API_MARKET = {"ml": "h2h", "spread": "spreads", "total": "totals"}

EV_MIN = 0.015
EPS = 1e-12
PRICE_MIN = -250
PRICE_MAX = 500
PIN_MAX_AGE_MIN = 30

# Credit floors: skip below these remaining balances.
PULL_FLOOR = 1500
CLOSE_FLOOR = 300
SCORES_FLOOR = 300

CLOSE_LO_MIN = 5
CLOSE_HI_MIN = 20
CLOSE_VALID_MAX_MIN = 30
EVENTS_HORIZON_DAYS = 7
PRICE_RETENTION_DAYS = 7

# The CLV test (judged once; the verdict row is append-only).
TEST_START = datetime(2026, 11, 1, tzinfo=UTC)
N_REQUIRED = 300
BEAT_MIN = 0.55
RESAMPLES = 10_000
SEED = 20261017
LOWER_PERCENTILE = 2.5
DIRECTION_STEP = 0.5


@dataclass(frozen=True)
class League:
    """Duck-typed for the `odds.fetch_*` functions, which read `.key` and `.odds_sport`."""
    key: str
    odds_sport: str


def league(key: str) -> League:
    return League(key=key, odds_sport=LEAGUES[key])


def budget_ok(remaining, floor: int) -> tuple[bool, str | None]:
    """May a paid call run given `x-requests-remaining`? Unknown balance runs."""
    credits = parse_credits(remaining)
    if credits is None or credits >= floor:
        return True, None
    return False, f"{credits} Odds API credits left (< {floor}): skipping"
