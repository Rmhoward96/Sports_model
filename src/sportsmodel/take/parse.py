"""Parse Odds API v4 responses into the plain-dict rows the 'take these' pipeline uses."""
from __future__ import annotations

from datetime import datetime

from ..ingest.odds import parse_commence
from .constants import API_MARKET, SOCCER

_OURS = {v: k for k, v in API_MARKET.items()}      # h2h -> ml, spreads -> spread, totals -> total


def parse_events(events: list[dict], league: str) -> list[dict]:
    rows = []
    for ev in events:
        commence = parse_commence(ev.get("commence_time"))
        if not ev.get("id") or commence is None:
            continue
        rows.append({"event_id": ev["id"], "league": league, "home_team": ev.get("home_team"),
                     "away_team": ev.get("away_team"), "commence_time": commence})
    return rows


def _side(name: str | None, market: str, home: str | None, away: str | None, league: str) -> str | None:
    """Our side for an outcome name, or None when it matches nothing (a book's variant team name is dropped)."""
    if market == "total":
        return {"Over": "over", "Under": "under"}.get(name)
    if name == home:
        return "home"
    if name == away:
        return "away"
    if name == "Draw" and market == "ml" and league in SOCCER:
        return "draw"
    return None


def parse_prices(events: list[dict], league: str, pulled_at: datetime) -> list[dict]:
    rows = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        for bk in ev.get("bookmakers") or []:
            for mk in bk.get("markets") or []:
                market = _OURS.get(mk.get("key"))
                if market is None:
                    continue
                last_update = parse_commence(mk.get("last_update"))
                for out in mk.get("outcomes") or []:
                    side = _side(out.get("name"), market, home, away, league)
                    if side is None or out.get("price") is None:
                        continue
                    point = out.get("point")
                    rows.append({
                        "event_id": ev["id"], "league": league, "market": market, "side": side, "book": bk["key"],
                        "line": None if market == "ml" or point is None else float(point),
                        "price": int(out["price"]), "last_update": last_update, "pulled_at": pulled_at,
                    })
    return rows


def parse_scores(scores: list[dict]) -> dict[str, tuple[int, int]]:
    """event_id -> (home_score, away_score) for completed events; scores are matched to home/away by team name."""
    out: dict[str, tuple[int, int]] = {}
    for ev in scores:
        if not ev.get("completed"):
            continue
        by_team = {s.get("name"): s.get("score") for s in ev.get("scores") or []}
        try:
            out[ev["id"]] = (int(by_team[ev["home_team"]]), int(by_team[ev["away_team"]]))
        except (KeyError, TypeError, ValueError):
            continue
    return out
