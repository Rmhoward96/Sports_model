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
