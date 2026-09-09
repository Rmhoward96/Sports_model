"""The Odds API -> normalized odds rows (game lines + player props).

Snapshots are stored with a captured_at timestamp so the closing line is derivable
as the last snapshot before a game's commence_time. Games join to our game_pk by
team names; props carry the player name (joined to player_id at analysis time).

Requires ODDS_API_KEY. Props are per-event calls and cost credits = markets per
event, so they add up fast — keep an eye on the plan.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from sportsmodel.sports import SportConfig

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
GAME_MARKETS = ["h2h", "totals", "spreads"]

# our market code -> The Odds API market key
PROP_MARKET_MAP = {
    "hits": "batter_hits",
    "total_bases": "batter_total_bases",
    "home_run": "batter_home_runs",
    "hrr": "batter_hits_runs_rbis",
    "pitcher_ks": "pitcher_strikeouts",
    "hits_allowed": "pitcher_hits_allowed",
    "outs_recorded": "pitcher_outs",
}
_ODDS_TO_OURS = {v: k for k, v in PROP_MARKET_MAP.items()}

# credits remaining, updated from the last response header (for logging)
last_requests_remaining: str | None = None


def _key() -> str:
    k = os.getenv("ODDS_API_KEY")
    if not k:
        raise RuntimeError("ODDS_API_KEY is not set (add it to .env / repo secrets).")
    return k


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, params: dict[str, Any]) -> Any:
    global last_requests_remaining
    with httpx.Client(timeout=25) as client:
        resp = client.get(f"{BASE}{path}", params={**params, "apiKey": _key()})
        resp.raise_for_status()
        last_requests_remaining = resp.headers.get("x-requests-remaining")
        return resp.json()


def fetch_events(cfg: "SportConfig | None" = None) -> list[dict]:
    """Upcoming events for the sport (id, home_team, away_team, commence_time)."""
    cfg = cfg or _mlb()
    return _get(f"/sports/{cfg.odds_sport}/events", {"dateFormat": "iso"})


def fetch_game_odds(cfg: "SportConfig | None" = None, regions: str = "us") -> list[dict]:
    cfg = cfg or _mlb()
    return _get(f"/sports/{cfg.odds_sport}/odds", {
        "regions": regions, "markets": ",".join(cfg.game_markets),
        "oddsFormat": "american", "dateFormat": "iso",
    })


def fetch_event_props(event_id: str, markets: list[str], cfg: "SportConfig | None" = None, regions: str = "us") -> dict:
    cfg = cfg or _mlb()
    return _get(f"/sports/{cfg.odds_sport}/events/{event_id}/odds", {
        "regions": regions, "markets": ",".join(markets),
        "oddsFormat": "american", "dateFormat": "iso",
    })


def parse_commence(commence_iso: str | None) -> datetime | None:
    """Tz-aware UTC datetime from an ISO commence string (None if missing/unparseable)."""
    if not commence_iso:
        return None
    try:
        return datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def resolved_game_date(commence_iso: str | None) -> str | None:
    """US game date (YYYY-MM-DD) from a UTC commence time.

    MLB games commence ~17:00 UTC (afternoon) to ~05:00 UTC next day (west-coast
    night). Shifting -10h maps every game in a US day back into that same calendar
    day, so a night game whose UTC date is the next day resolves to its true US date.
    """
    dt = parse_commence(commence_iso)
    if dt is None:
        return None
    return (dt - timedelta(hours=10)).date().isoformat()


def _row(game_pk, market, side, player_name, book, line, price, captured_at, commence):
    return {
        "game_pk": game_pk, "market": market, "side": side,
        "player_name": player_name or "", "book": book, "line": line,
        "price": price, "captured_at": captured_at, "commence_time": commence,
    }


def parse_game_odds(events, game_lookup, captured_at) -> list[dict]:
    """Moneyline/total/spread rows joined to game_pk via (home, away, game_date).

    game_lookup is keyed by (home_team, away_team, US_game_date); the date is derived
    from each event's commence_time so night games map to their correct game day.
    """
    rows: list[dict] = []
    for ev in events:
        gp = game_lookup.get((ev.get("home_team"), ev.get("away_team"),
                              resolved_game_date(ev.get("commence_time"))))
        if gp is None:
            continue
        home, away, commence = ev["home_team"], ev["away_team"], ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk["key"]
            for m in bk.get("markets", []):
                key = m["key"]
                for o in m.get("outcomes", []):
                    if key == "h2h":
                        side = "home" if o["name"] == home else "away" if o["name"] == away else None
                        if side:
                            rows.append(_row(gp, "moneyline", side, None, book, None, o.get("price"), captured_at, commence))
                    elif key == "totals":
                        rows.append(_row(gp, "total", o["name"].lower(), None, book, o.get("point"), o.get("price"), captured_at, commence))
                    elif key == "spreads":
                        side = "home" if o["name"] == home else "away"
                        rows.append(_row(gp, "spread", side, None, book, o.get("point"), o.get("price"), captured_at, commence))
    return rows


def parse_prop_odds(event_props, game_pk, captured_at, cfg: "SportConfig | None" = None) -> list[dict]:
    """Player-prop rows (our market codes) with player name and over/under side.

    The odds-api-key -> our-market-name map defaults to MLB's `_ODDS_TO_OURS`
    (unchanged behavior when `cfg` is omitted). Pass a sport's `SportConfig`
    as `cfg` to map that sport's own odds-api keys instead -- e.g. NFL's
    `player_pass_yds`/`player_anytime_td`/etc, which aren't in the MLB map and
    would otherwise all resolve to None and get silently dropped.
    """
    odds_to_ours = {v: k for k, v in cfg.prop_market_map.items()} if cfg is not None else _ODDS_TO_OURS
    commence = event_props.get("commence_time")
    rows: list[dict] = []
    for bk in event_props.get("bookmakers", []):
        book = bk["key"]
        for m in bk.get("markets", []):
            our = odds_to_ours.get(m["key"])
            if not our:
                continue
            for o in m.get("outcomes", []):
                rows.append(_row(game_pk, our, o["name"].lower(), o.get("description"),
                                 book, o.get("point"), o.get("price"), captured_at, commence))
    return rows


# sports.py imports GAME_MARKETS/PROP_MARKET_MAP from this module, and this module
# needs sports.get() for the MLB default config -- a cycle. Resolving `get("mlb")`
# lazily (on first fetch_* call, cached) rather than at import time sidesteps it
# regardless of which module happens to be imported first: a module-level
# `from sportsmodel.sports import get` here breaks when *sports.py* is imported
# first (e.g. via a test module doing `from sportsmodel.sports import get`), since
# that import re-enters this module before sports.py has finished defining
# SportConfig.
_MLB: "SportConfig | None" = None


def _mlb() -> "SportConfig":
    global _MLB
    if _MLB is None:
        from sportsmodel.sports import get
        _MLB = get("mlb")
    return _MLB
