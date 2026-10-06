"""Match a The Odds API MLB event to our MLB StatsAPI game_pk.

Sibling of sportsmodel.nfl.matcher / sportsmodel.cfb.matcher, with two MLB specifics:

  * The date is the US game day (`odds.resolved_game_date`: UTC commence shifted -10h),
    compared with the schedule's `game_date` (StatsAPI officialDate). MLB night games
    commence after 00:00 UTC, so the raw UTC date (what NFL/CFB compare) would pair them
    with the wrong day.
  * A doubleheader has two games with the same teams on the same day. The downstream
    (home, away, date) -> game_pk lookup cannot tell them apart, so an event whose
    (home, away, date) matches more than one scheduled game is left UNMATCHED rather than
    risk filing one game's prices under the other's game_pk.

`games` are dicts with game_pk, home_name, away_name and game_date (YYYY-MM-DD).
"""
from __future__ import annotations

from sportsmodel.ingest import odds

# The Odds API / StatsAPI have both called the franchise "Athletics" since the 2025 move;
# fold the older display name onto it so a stale feed still matches.
_ALIASES = {"oakland athletics": "athletics"}


def _norm_name(s: str | None) -> str:
    n = (s or "").strip().lower()
    return _ALIASES.get(n, n)


def match_odds_event(odds_event: dict, games: list[dict]) -> int | None:
    day = odds.resolved_game_date(odds_event.get("commence_time"))
    key = (_norm_name(odds_event.get("home_team")), _norm_name(odds_event.get("away_team")), day)
    hits = [g for g in games
            if (_norm_name(g.get("home_name")), _norm_name(g.get("away_name")),
                str(g.get("game_date"))) == key]
    if len(hits) != 1:
        return None  # no such game, or a doubleheader (ambiguous)
    return int(hits[0]["game_pk"])
