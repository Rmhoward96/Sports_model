"""Auto-parlay builder for the +EV pilot.

Straight picks priced -250 or worse are excluded (see
`ev_pilot.MAX_STRAIGHT_JUICE`): the break-even hit rate is too high to treat as
+EV on its own. This module recombines those juiced favorites into a parlay.

Why a parlay of individually-+EV legs is itself +EV: parlay EV
= Product(true_p_i) * Product(dec_i) - 1 = Product(true_p_i * dec_i) - 1
= Product(1 + ev_i) - 1. If every leg's ev_i > 0, that product exceeds 1, so
the parlay is +EV -- and the combined price is plus-money, which is the whole
point (turn three unpalatable -300 tickets into one +EV plus-money ticket).

A real parlay is ONE ticket at ONE book (you can't line-shop each leg
independently), so a parlay is priced at the best single book that offers every
leg: parlay decimal = Product(that book's leg prices). Legs are one-per-game
(distinct games are independent, so probabilities multiply cleanly).

PURE -- no IO. `build_parlay` takes candidate legs and returns the best parlay
(or None). `build_ev_board` collects the candidates and persists the result.
"""
from __future__ import annotations

from .board import decimal_odds, ev as _ev

MAX_PARLAY_LEGS = 3
MIN_PARLAY_LEGS = 2
# A parlay must clear this EV to surface (same floor as a straight).
PARLAY_MIN_EV = 0.01
# Guard against compounding stale-line artifacts; parlays legitimately reach
# higher EV than a single leg, so this ceiling is looser than the straight 0.25.
PARLAY_EV_CEILING = 1.0


def american_from_decimal(dec: float) -> int:
    """American odds from decimal odds (rounded to the nearest whole number)."""
    if dec >= 2.0:
        return round((dec - 1.0) * 100)
    return round(-100.0 / (dec - 1.0))


def _leg_key(leg: dict) -> str:
    return f"{leg['game_pk']}:{leg['market']}:{leg['side']}"


def parlay_id(legs: list[dict]) -> str:
    """Deterministic id from the (sorted) leg set, so a rebuild with the same
    legs upserts the same parlay row rather than creating a duplicate."""
    return "|".join(sorted(_leg_key(l) for l in legs))


def _dedupe_one_per_game(cands: list[dict]) -> list[dict]:
    """At most one leg per game (same-game legs are correlated, not
    independent) -- keep the highest single-leg ev for each game."""
    best_by_game: dict = {}
    for c in cands:
        cur = best_by_game.get(c["game_pk"])
        if cur is None or c["ev"] > cur["ev"]:
            best_by_game[c["game_pk"]] = c
    return list(best_by_game.values())


def _parlay_at_book(book: str, cands: list[dict], max_legs: int) -> dict | None:
    """Build the parlay for a single book from the candidates that book offers.
    Returns None if fewer than MIN_PARLAY_LEGS are available at this book."""
    legs_here = []
    for c in cands:
        price = dict(c["book_prices"]).get(book)
        if price is not None:
            legs_here.append((c, price))
    legs_here = _dedupe_one_per_game([{**c, "_book_price": p} for c, p in legs_here])
    legs_here.sort(key=lambda c: c["ev"], reverse=True)
    legs_here = legs_here[:max_legs]
    if len(legs_here) < MIN_PARLAY_LEGS:
        return None

    dec = 1.0
    true_prob = 1.0
    for leg in legs_here:
        dec *= decimal_odds(leg["_book_price"])
        true_prob *= leg["true_prob"]
    american = american_from_decimal(dec)
    return {
        "book": book,
        "legs": [
            {"game_pk": leg["game_pk"], "market": leg["market"], "side": leg["side"],
             "matchup": leg["matchup"], "price": leg["_book_price"],
             "true_prob": leg["true_prob"], "commence_time": leg.get("commence_time")}
            for leg in legs_here
        ],
        "parlay_price": american,
        "true_prob": true_prob,
        "ev": _ev(true_prob, american),
        "n_legs": len(legs_here),
        "commence_time": max((leg.get("commence_time") for leg in legs_here), default=None),
    }


def build_parlay(candidates: list[dict], *, max_legs: int = MAX_PARLAY_LEGS,
                 min_ev: float = PARLAY_MIN_EV, ev_ceiling: float = PARLAY_EV_CEILING) -> dict | None:
    """Best single-book parlay from the candidate legs, or None.

    Each candidate: {sport, game_pk, market, side, matchup, commence_time,
    true_prob, ev (single-leg ev at its best price, for ranking),
    book_prices: [(book, american), ...]}.

    For every book any candidate offers, builds that book's parlay (top
    `max_legs` legs by single-leg ev, one per game) and keeps the highest-EV
    parlay across books that clears `min_ev` (and is <= `ev_ceiling`). Returns
    the parlay dict (with a stable `parlay_id`) or None when no book carries
    >= MIN_PARLAY_LEGS or none clear the floor.
    """
    if len(candidates) < MIN_PARLAY_LEGS:
        return None
    books = {bk for c in candidates for bk, _ in c["book_prices"]}
    best = None
    for book in books:
        p = _parlay_at_book(book, candidates, max_legs)
        if p is None:
            continue
        if best is None or p["ev"] > best["ev"]:
            best = p
    if best is None or not (min_ev < best["ev"] <= ev_ceiling):
        return None
    best["parlay_id"] = parlay_id(best["legs"])
    best["sport"] = candidates[0]["sport"]
    return best
