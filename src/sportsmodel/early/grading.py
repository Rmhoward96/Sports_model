"""Grade early tickets against the result and Pinnacle's pre-kick close. PURE -- no IO.

Primary metric `close_ev`: the ticket's book line and price valued at Pinnacle's CLOSING fair, through the same
landing-table adjustment and EV formula the flag used. Unlike the flag, the adjustment is uncapped and clamped here (a
close that moved a long way against the ticket must count as a loss, not drop the ticket from the test).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from ..ledger import reference as ref
from ..ledger.core import as_utc
from ..serving.board import decimal_odds
from . import adjust
from .constants import LINE_EPS, NEXT_POLL_MAX_MINUTES, NEXT_POLL_MIN_MINUTES, PINNACLE
from .flags import _valid_price, pinnacle_reference

MISSING = {"close_kind": "missing", "close_fair": None, "close_line": None, "close_minutes_to_kick": None,
           "close_ev": None}


def closing_view(ticket: dict, close_rows: list[dict], tables: dict) -> dict:
    """close_kind / close_fair / close_line / close_minutes_to_kick / close_ev for one ticket from the close snapshot rows
    of its game. 'missing' when Pinnacle has no complete two-way quote at the close (or its line is off the table's
    half-point grid, or the table has no answer there)."""
    market, side = ticket["market"], ticket["side"]
    rows = [r for r in close_rows if r.get("market") == market and not r.get("player_name")]
    pin = pinnacle_reference(rows, market)
    table = tables.get((ticket["sport"], market))
    if pin is None or table is None:
        return dict(MISSING)
    v = adjust.value_side(market, side, pin.fair[side], pin.line[side], ticket["book_line"], ticket["price"],
                          table.pmf(pin.threshold), cap=None, strict=False)
    if v is None:
        return dict(MISSING)
    kick = as_utc(ticket["commence_time"])
    captured = max(as_utc(r["captured_at"]) for r in rows if r.get("book") == PINNACLE)
    return {"close_kind": "pinnacle_novig", "close_fair": pin.fair[side], "close_line": pin.line[side],
            "close_minutes_to_kick": (kick - captured).total_seconds() / 60.0, "close_ev": v.ev}


def available_next_poll(ticket: dict, later_rows: list[dict]) -> bool | None:
    """Was the same book still offering the same line at the same or a better price one poll later?

    `later_rows`: odds_snapshot rows of the ticket's game and market captured after the ticket. The next poll is the first
    snapshot, at least 30 minutes after the ticket, that carries a Pinnacle quote for the market (the monitor's polls
    always do, so do the regular pulls; a snapshot without Pinnacle is not a poll). It must be within 3 hours of the
    ticket, otherwise the answer is None (not False): the number was not observed again in time."""
    obs = as_utc(ticket["observed_at"])
    by_time = defaultdict(list)
    for r in later_rows:
        if r.get("market") == ticket["market"] and not r.get("player_name"):
            by_time[as_utc(r["captured_at"])].append(r)
    polls = sorted(t for t, rs in by_time.items()
                   if t - obs >= timedelta(minutes=NEXT_POLL_MIN_MINUTES) and any(r.get("book") == PINNACLE for r in rs))
    if not polls or polls[0] - obs > timedelta(minutes=NEXT_POLL_MAX_MINUTES):
        return None
    need = decimal_odds(ticket["price"])
    for r in by_time[polls[0]]:
        price = _valid_price(r.get("price"))                    # junk feed prices (0, -99, strings) are skipped, never maths
        if (r.get("book") == ticket["book"] and r.get("side") == ticket["side"] and r.get("line") is not None
                and price is not None and abs(float(r["line"]) - float(ticket["book_line"])) <= LINE_EPS
                and decimal_odds(price) >= need - 1e-9):
            return True
    return False


def grade_ticket(ticket: dict, final: dict | None, close_rows: list[dict], later_rows: list[dict], tables: dict) -> dict | None:
    """One early_ticket_results row, or None while the game has no result."""
    f = final or {}
    result = ref.settle_game(ticket["market"], ticket["side"], ticket["book_line"], f.get("actual_margin"),
                             f.get("actual_total"))
    if result is None:
        return None
    view = closing_view(ticket, close_rows, tables)
    close_ev = view["close_ev"]
    return {
        "ticket_id": ticket["id"], "sport": ticket["sport"], "game_pk": ticket["game_pk"], "market": ticket["market"],
        "side": ticket["side"], "book": ticket["book"], "book_line": ticket["book_line"], "price": ticket["price"],
        "observed_at": ticket["observed_at"], "commence_time": ticket["commence_time"], "result": result,
        "units": ref.units_at_pick(result, ticket["price"]), **view,
        "beat_close": None if close_ev is None else close_ev > 0,
        "clv_pts": None if view["close_line"] is None else ref.clv_pts(
            ticket["market"], ticket["side"], ticket["book_line"], view["close_line"]),
        "available_next_poll": available_next_poll(ticket, later_rows),
    }


def grade_tickets(tickets: list[dict], finals: dict, close_by_game: dict, later_by_game: dict, tables: dict) -> list[dict]:
    """Grade every ticket whose game is final. `finals` {(sport, game_pk): {actual_margin, actual_total}};
    `close_by_game` / `later_by_game` {game_pk: [snapshot rows]}."""
    out = []
    for t in tickets:
        g = grade_ticket(t, finals.get((t["sport"], t["game_pk"])), close_by_game.get(t["game_pk"], []),
                         later_by_game.get(t["game_pk"], []), tables)
        if g is not None:
            out.append(g)
    return out
