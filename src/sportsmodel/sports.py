"""Sport configuration registry.

Captures the per-sport values that were previously hardcoded to MLB (odds
API sport key, game markets, prop market name mapping, commence-time
shift), so downstream producers can be parameterized by sport instead of
assuming MLB.
"""
from __future__ import annotations

from dataclasses import dataclass

from sportsmodel.ingest.odds import GAME_MARKETS, PROP_MARKET_MAP


@dataclass(frozen=True)
class SportConfig:
    key: str
    odds_sport: str
    game_markets: list[str]
    prop_market_map: dict[str, str]
    commence_shift_hours: int


SPORTS: dict[str, SportConfig] = {
    "mlb": SportConfig(
        key="mlb",
        odds_sport="baseball_mlb",
        game_markets=GAME_MARKETS,
        prop_market_map=PROP_MARKET_MAP,
        commence_shift_hours=10,
    ),
    "nfl": SportConfig(
        key="nfl",
        odds_sport="americanfootball_nfl",
        game_markets=GAME_MARKETS,
        prop_market_map={
            "pass_yds": "player_pass_yds",
            "pass_tds": "player_pass_tds",
            "reception_yds": "player_reception_yds",
            "receptions": "player_receptions",
            "rush_yds": "player_rush_yds",
            "rush_reception_yds": "player_rush_reception_yds",
            "anytime_td": "player_anytime_td",
        },
        commence_shift_hours=0,
    ),
}


def get(sport: str) -> SportConfig:
    """Look up a sport's config. Raises KeyError for unknown sports."""
    return SPORTS[sport]
