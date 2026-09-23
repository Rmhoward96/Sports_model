"""+EV Parlays: up to 3 three-leg tickets built from the current +EV picks,
each priced at ONE book (a real parlay is one ticket at one book).

Legs: every current +EV pick -- NFL/CFB game lines and NFL player props -- with
win probability >= MIN_LEG_PROB, one per game within a ticket (independent legs,
so the ticket's true probability is the product). For each US book, take the
legs it prices at the pick's own line that are still +EV at that price, keep
the top N_LEGS by EV-at-book, and price the ticket at that book; the best-EV
book wins. Remove those legs and repeat, up to MAX_TICKETS tickets.

PURE -- no IO. scripts/build_best_parlays.py loads the inputs and persists.
"""
from __future__ import annotations

import math

from .board import MAJOR_BOOKS, decimal_odds
from .parlay import american_from_decimal
from .props_ev import SIM_TO_ODDS_MARKET, normalize_player_name

MIN_LEG_PROB = 0.55
N_LEGS = 3
MAX_TICKETS = 3
MIN_EV = 0.01
EV_CEILING = 1.0
PARLAY_BOOKS = frozenset(b for b in MAJOR_BOOKS if b != "pinnacle")

_PROP_LABEL = {"rush_yds": "Rush Yds", "rec_yds": "Rec Yds", "receptions": "Receptions",
               "pass_yds": "Pass Yds", "pass_tds": "Pass TDs", "rush_att": "Rush Att"}


def _fmt_line(x: float) -> str:
    return f"{x:g}"


def _game_label(market: str, side: str, matchup: str, line) -> str:
    away, _, home = (matchup or " @ ").partition(" @ ")
    if market == "total":
        return f"{'Over' if side == 'over' else 'Under'} {_fmt_line(line)}"
    team = home if side == "home" else away
    if market == "moneyline":
        return f"{team} ML"
    return f"{team} {'+' if line > 0 else ''}{_fmt_line(line)}" if line else f"{team} PK"


def assemble_game_legs(picks: list[dict], odds_rows: list[dict]) -> list[dict]:
    """Game-line +EV picks (ev_current rows with `line` from ev_best_lines) ->
    legs. A book's price counts only at the pick's line (moneyline: any)."""
    legs = []
    for p in picks:
        line = p.get("line")
        if p["market"] != "moneyline" and line is None:
            continue
        prices = {}
        for o in odds_rows:
            if (o["game_pk"] == p["game_pk"] and o["market"] == p["market"] and o["side"] == p["side"]
                    and o["book"] in PARLAY_BOOKS and o.get("price")
                    and (p["market"] == "moneyline" or o.get("line") == line)):
                prices[o["book"]] = o["price"]
        legs.append({
            "key": f"g:{p['game_pk']}:{p['market']}:{p['side']}", "kind": "game",
            "sport": p["sport"], "game_pk": p["game_pk"], "market": p["market"], "side": p["side"],
            "line": line, "prob": float(p["true_prob"]),
            "label": _game_label(p["market"], p["side"], p.get("matchup"), line),
            "matchup": p.get("matchup"), "commence_time": p.get("commence_time"),
            "player_id": None, "player_name": None, "book_prices": prices,
        })
    return legs


def assemble_prop_legs(picks: list[dict], odds_rows: list[dict]) -> list[dict]:
    """NFL prop +EV picks (ev_prop_picks_current) -> legs. A book's price counts
    only for the same player (normalized name), odds market, side and line."""
    legs = []
    for p in picks:
        odds_market = SIM_TO_ODDS_MARKET.get(p["market"])
        if odds_market is None:
            continue
        name = normalize_player_name(p.get("player_name"))
        prices = {}
        for o in odds_rows:
            if (o["game_pk"] == p["game_pk"] and o["market"] == odds_market and o["side"] == p["side"]
                    and o.get("line") == p["line"] and o["book"] in PARLAY_BOOKS and o.get("price")
                    and normalize_player_name(o.get("player_name")) == name):
                prices[o["book"]] = o["price"]
        legs.append({
            "key": f"p:{p['game_pk']}:{p['player_id']}:{p['market']}:{p['side']}", "kind": "prop",
            "sport": "nfl", "game_pk": p["game_pk"], "market": p["market"], "side": p["side"],
            "line": p["line"], "prob": float(p["model_prob"]),
            "label": f"{p.get('player_name')} {_PROP_LABEL.get(p['market'], p['market'])} "
                     f"{'Over' if p['side'] == 'over' else 'Under'} {_fmt_line(p['line'])}",
            "matchup": p.get("matchup"), "commence_time": p.get("commence_time"),
            "player_id": p.get("player_id"), "player_name": p.get("player_name"), "book_prices": prices,
        })
    return legs


def _ticket_at_book(book: str, legs: list[dict]) -> dict | None:
    best_by_game: dict = {}
    for leg in legs:
        price = leg["book_prices"].get(book)
        if price is None:
            continue
        ev = leg["prob"] * decimal_odds(price) - 1
        if ev <= 0:
            continue
        cur = best_by_game.get(leg["game_pk"])
        if cur is None or ev > cur[1]:
            best_by_game[leg["game_pk"]] = (leg, ev, price)
    chosen = sorted(best_by_game.values(), key=lambda t: (-t[1], t[0]["key"]))[:N_LEGS]
    if len(chosen) < N_LEGS:
        return None
    dec = math.prod(decimal_odds(price) for _, _, price in chosen)
    prob = math.prod(leg["prob"] for leg, _, _ in chosen)
    out_legs = [{**{k: v for k, v in leg.items() if k != "book_prices"}, "price": price}
                for leg, _, price in chosen]
    sports = {l["sport"] for l in out_legs}
    times = [l["commence_time"] for l in out_legs if l.get("commence_time") is not None]
    return {
        "parlay_id": book + "|" + "|".join(sorted(l["key"] for l in out_legs)),
        "book": book, "legs": out_legs, "n_legs": len(out_legs),
        "parlay_dec": dec, "parlay_price": american_from_decimal(dec),
        "true_prob": prob, "ev": prob * dec - 1,
        "sport": sports.pop() if len(sports) == 1 else "mixed",
        "first_commence": min(times) if times else None,
        "last_commence": max(times) if times else None,
    }


def build_best_parlays(legs: list[dict], excluded_keys: frozenset = frozenset()) -> list[dict]:
    """Up to MAX_TICKETS three-leg tickets, best-EV first, no shared legs.
    `excluded_keys`: leg keys already riding on a locked (started, ungraded)
    ticket -- never reused."""
    pool = [l for l in legs if l["prob"] >= MIN_LEG_PROB and l["key"] not in excluded_keys]
    tickets: list[dict] = []
    for _ in range(MAX_TICKETS):
        books = sorted({b for l in pool for b in l["book_prices"]})
        best = None
        for book in books:
            t = _ticket_at_book(book, pool)
            if t is not None and MIN_EV < t["ev"] <= EV_CEILING and (best is None or t["ev"] > best["ev"]):
                best = t
        if best is None:
            break
        tickets.append(best)
        used = {l["key"] for l in best["legs"]}
        pool = [l for l in pool if l["key"] not in used]
    return tickets
