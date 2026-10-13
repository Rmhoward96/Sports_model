"""The flagging engine: one poll's prices -> new paper tickets. PURE -- no IO.

For every game and market in the test, Pinnacle's two-way quote in THIS poll gives the fair probability at Pinnacle's
line; every one of the user's nine US books is then valued at its own line and price (adjust.value_side); the highest-EV
book per side becomes a ticket when its EV is at least +1.5% and kickoff is at least 24 hours away. A side that already
has a ticket (any earlier poll) never gets a second one.

`rows` are odds_snapshot-shaped dicts (game_pk, market, side, book, line, price, commence_time, ...) as produced by
`ingest.odds.parse_game_odds`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from ..ingest.odds import parse_commence
from ..serving.board import novig
from . import adjust
from .constants import LINE_EPS, MARKETS, MIN_HOURS_TO_KICK, PINNACLE, SIDES, US_BOOKS
from .landing import LandingTable


@dataclass(frozen=True)
class Reference:
    """Pinnacle's two-way quote for one game and market: the table-space threshold, and per side the no-vig fair
    probability and Pinnacle's own line for that side."""
    threshold: float
    fair: dict[str, float]
    line: dict[str, float]


def _valid_price(price) -> int | None:
    """The price as an int if it is a valid American price (|price| >= 100), else None. Feeds carry junk prices (0,
    -99, strings); decimal_odds would divide by zero on them, so they are dropped here and never reach any maths."""
    try:
        p = int(price)
    except (TypeError, ValueError):
        return None
    return p if abs(p) >= 100 else None


def _offers(rows: list[dict], book: str, side: str) -> list[tuple[float, int]]:
    """(line, price) quotes of `book` for `side`, skipping any with a missing line or an invalid American price."""
    out = []
    for r in rows:
        if r.get("book") != book or r.get("side") != side or r.get("line") is None:
            continue
        price = _valid_price(r.get("price"))
        if price is not None:
            out.append((float(r["line"]), price))
    return out


def pinnacle_reference(rows: list[dict], market: str) -> Reference | None:
    """Pinnacle's two-way quote from `rows` (one game, one market, any books), or None without a complete pair.

    A pair is a spread's home and away quotes at mirrored lines, or a total's Over and Under at the same line. If
    Pinnacle shows several pairs, the one whose fair is closest to 50% (the main line) wins; a remaining tie goes to the
    lower threshold, so the choice is deterministic."""
    first, second = SIDES[market]
    pairs = []
    for l1, p1 in _offers(rows, PINNACLE, first):
        for l2, p2 in _offers(rows, PINNACLE, second):
            mirrored = abs(l1 + l2) <= LINE_EPS if market == "spread" else abs(l1 - l2) <= LINE_EPS
            if mirrored:
                fair_first = novig(p1, p2)
                pairs.append((abs(fair_first - 0.5), adjust.threshold(market, first, l1), fair_first, l1, l2))
    if not pairs:
        return None
    _, c, fair_first, l1, l2 = min(pairs, key=lambda t: (round(t[0], 12), t[1]))
    return Reference(threshold=c, fair={first: fair_first, second: 1.0 - fair_first}, line={first: l1, second: l2})


def best_offer(rows: list[dict], market: str, side: str, ref: Reference, table: LandingTable, *, cap=adjust.MAX_LINE_GAP):
    """The highest-EV (book, line, price, valuation) among the user's US books for `side`, or None. Ties on EV go to the
    alphabetically first book, then the lower line, so results are deterministic."""
    pmf = table.pmf(ref.threshold)
    best = None
    for book in sorted(US_BOOKS):
        for line, price in sorted(_offers(rows, book, side)):
            v = adjust.value_side(market, side, ref.fair[side], ref.line[side], line, price, pmf, cap=cap)
            if v is None:
                continue
            key = (-round(v.ev, 12), book, line)
            if best is None or key < best[0]:
                best = (key, book, line, price, v)
    return None if best is None else best[1:]


def flag_poll(rows: list[dict], *, sport: str, observed_at: datetime, tables: dict, existing: set[tuple],
              poll_run_id: str) -> list[dict]:
    """New tickets for one sport's poll. `tables` is `landing.load_tables()`; `existing` the set of
    (sport, game_pk, market, side) that already have a ticket."""
    by_game_market: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("market") in MARKETS.get(sport, ()) and not r.get("player_name"):
            by_game_market[(r["game_pk"], r["market"])].append(r)
    tickets: list[dict] = []
    for (game_pk, market), group in sorted(by_game_market.items()):
        table = tables.get((sport, market))
        kick = next((k for k in (parse_commence(r.get("commence_time")) for r in group) if k is not None), None)
        if table is None or kick is None:
            continue
        hours = (kick - observed_at).total_seconds() / 3600.0
        if hours < MIN_HOURS_TO_KICK:
            continue
        ref = pinnacle_reference(group, market)
        if ref is None or not adjust.fair_in_range(ref.fair[SIDES[market][0]]):
            continue
        for side in SIDES[market]:
            if (sport, game_pk, market, side) in existing:
                continue
            found = best_offer(group, market, side, ref, table)
            if found is None or not adjust.is_flag(found[3].ev):
                continue
            book, line, price, v = found
            tickets.append({
                "sport": sport, "game_pk": game_pk, "market": market, "side": side, "book": book, "book_line": line,
                "price": price, "pinnacle_line": ref.line[side], "pinnacle_fair": ref.fair[side], "p_win": v.p_win,
                "p_push": v.p_push, "p_loss": v.p_loss, "ev": v.ev, "observed_at": observed_at, "commence_time": kick,
                "hours_to_kick": hours, "poll_run_id": poll_run_id})
    return tickets
