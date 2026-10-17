"""Pick engine: Pinnacle power-devig fair price vs the list books, same line or a better one."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from .constants import EPS, EV_MIN, PIN_MAX_AGE_MIN, PRICE_MAX, PRICE_MIN, REFERENCE, SOCCER, TAKE_BOOKS
from .devig import american_to_decimal, decimal_to_american_safe, novig_power_n

_LINE_TOL = 1e-6
_BOOK_ORDER = {b: i for i, b in enumerate(TAKE_BOOKS)}


def same_line(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= _LINE_TOL


def good_until(fair_prob: float) -> int:
    """The worst American price that still clears the EV threshold (rounded toward the bettor)."""
    return decimal_to_american_safe((1 + EV_MIN) / fair_prob)


def _usable(r: dict, market: str) -> bool:
    """Spread and total rows need a line; moneyline rows have none."""
    return market == "ml" or r["line"] is not None


def _is_better_line(market: str, side: str, book_line: float, pin_line: float) -> bool:
    if market == "total":
        return book_line < pin_line if side == "over" else book_line > pin_line
    return book_line > pin_line      # spread: a larger line is better for either side


def pinnacle_fair(rows_for_market: list[dict], league: str, market: str, pulled_at: datetime) -> dict[str, dict] | None:
    """side -> {"line", "price", "fair"} from Pinnacle's rows, or None when the market is incomplete or stale."""
    sides: dict[str, dict] = {}
    for r in rows_for_market:
        if r["book"] != REFERENCE or not _usable(r, market) or r["side"] in sides:
            continue
        sides[r["side"]] = r
    if market == "ml":
        want = ["home", "draw", "away"] if league in SOCCER else ["home", "away"]
    elif market == "spread":
        want = ["home", "away"]
    else:
        want = ["over", "under"]
    if set(sides) != set(want):
        return None
    if market == "spread" and not same_line(sides["home"]["line"], -sides["away"]["line"]):
        return None
    if market == "total" and not same_line(sides["over"]["line"], sides["under"]["line"]):
        return None
    max_age = timedelta(minutes=PIN_MAX_AGE_MIN)
    for r in sides.values():
        if r["last_update"] is None or pulled_at - r["last_update"] > max_age:
            return None
    fair = novig_power_n([sides[s]["price"] for s in want])
    return {s: {"line": sides[s]["line"], "price": sides[s]["price"], "fair": f} for s, f in zip(want, fair)}


def find_picks(prices: list[dict], events: list[dict], pulled_at: datetime) -> list[dict]:
    live = {e["event_id"]: e for e in events if e["commence_time"] > pulled_at}
    by_market: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in prices:
        if r["event_id"] in live:
            by_market[(r["event_id"], r["market"])].append(r)

    picks: list[dict] = []
    for (event_id, market), rows in by_market.items():
        event = live[event_id]
        pin = pinnacle_fair(rows, event["league"], market, pulled_at)
        if pin is None:
            continue
        cands: dict[str, dict[str, dict]] = defaultdict(dict)      # side -> book -> best qualifying row
        for r in rows:
            side, book = r["side"], r["book"]
            if book not in _BOOK_ORDER or side not in pin or not _usable(r, market):
                continue
            if not PRICE_MIN <= r["price"] <= PRICE_MAX:
                continue
            ref = pin[side]
            better = False
            if market != "ml" and not same_line(r["line"], ref["line"]):
                if not _is_better_line(market, side, r["line"], ref["line"]):
                    continue
                better = True
            ev = ref["fair"] * american_to_decimal(r["price"]) - 1
            if ev < EV_MIN - EPS:
                continue
            if book not in cands[side] or ev > cands[side][book]["ev"]:
                cands[side][book] = {"book": book, "line": r["line"], "price": r["price"], "ev": ev, "better": better}
        for side, by_book in cands.items():
            ranked = sorted(by_book.values(), key=lambda c: (-round(c["ev"], 12), c["better"], _BOOK_ORDER[c["book"]]))
            best, rest = ranked[0], sorted(ranked[1:], key=lambda c: (-round(c["ev"], 12), _BOOK_ORDER[c["book"]]))
            fair = pin[side]["fair"]
            picks.append({
                "event_id": event_id, "league": event["league"], "market": market, "side": side,
                "book": best["book"], "line": best["line"], "price": best["price"], "ev": best["ev"],
                "fair_prob": fair, "good_until": good_until(fair),
                "better_line": best["better"], "ref_line": pin[side]["line"],
                "pin_prices": {s: {"line": v["line"], "price": v["price"]} for s, v in pin.items()},
                "others": [{k: c[k] for k in ("book", "line", "price", "ev")} for c in rest],
                "pulled_at": pulled_at, "commence_time": event["commence_time"],
            })
    return picks
