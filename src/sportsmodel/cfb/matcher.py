"""Match a The Odds API CFB event to our ESPN game_pk.

Mirrors sportsmodel.nfl.matcher exactly. `cfb.teams.normalize` was considered
for a more tolerant name match, but it normalizes ESPN *team ids* (passing
through known FBS ids, collapsing everything else to the "FCS" sentinel) --
it does not normalize free-text display-name strings. The Odds API event and
ESPN game dicts here carry only display names (home_team/away_team,
home_name/away_name), no ids, so `normalize` isn't applicable to this
contract; feeding it a name string would always fall through to "FCS" and
silently collapse every unmatched team into one bucket, which is worse than
exact matching. Falling back to the same lowercase/strip `_norm_name` as NFL.
"""
from __future__ import annotations

def _norm_name(s: str) -> str:
    return (s or "").strip().lower()

def _date(iso: str) -> str:
    return (iso or "")[:10]

def match_odds_event(odds_event: dict, espn_games: list[dict]) -> int | None:
    key = (_norm_name(odds_event["home_team"]),
           _norm_name(odds_event["away_team"]),
           _date(odds_event["commence_time"]))
    for g in espn_games:
        if (_norm_name(g["home_name"]), _norm_name(g["away_name"]),
                _date(g["commence_time"])) == key:
            return int(g["game_pk"])
    return None
