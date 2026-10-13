"""Shared builders for the early-monitor tests: stub landing tables, odds_snapshot-shaped rows, graded results."""
from datetime import datetime, timezone

from sportsmodel.early.landing import Pmf, on_grid

NOW = datetime(2026, 10, 13, 16, 0, tzinfo=timezone.utc)      # a Tuesday poll


class StubTable:
    """A landing table answering the same Pmf for every on-grid line (None off the grid), like LandingTable."""

    def __init__(self, pmf: Pmf):
        self._pmf = pmf

    def pmf(self, center):
        return self._pmf if on_grid(center) else None


def mass_at(values, m: float) -> Pmf:
    """m of the mass on each of the given integer results, the rest split between two far-away results (never
    between the lines the tests move across)."""
    values = (values,) if isinstance(values, int) else tuple(values)
    rest = (1 - m * len(values)) / 2
    return Pmf(n=1000, window=1.0, probs={**{v: m for v in values}, min(values) - 15: rest, max(values) + 15: rest})


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def row(game_pk, market, side, book, line, price, kick, *, captured_at=NOW, player_name=""):
    return {"game_pk": game_pk, "market": market, "side": side, "player_name": player_name, "book": book,
            "line": line, "price": price, "captured_at": captured_at.isoformat(), "commence_time": iso(kick)}


def two_way(game_pk, market, book, line_a, price_a, line_b, price_b, kick, sides):
    return [row(game_pk, market, sides[0], book, line_a, price_a, kick),
            row(game_pk, market, sides[1], book, line_b, price_b, kick)]


def total_rows(game_pk, kick, pin_line=55.5, pin=(-110, -110), books=None):
    """Pinnacle's Over/Under at pin_line plus {book: (line, over_price, under_price)} for the user's books."""
    out = two_way(game_pk, "total", "pinnacle", pin_line, pin[0], pin_line, pin[1], kick, ("over", "under"))
    for book, (line, over, under) in (books or {}).items():
        out += two_way(game_pk, "total", book, line, over, line, under, kick, ("over", "under"))
    return out


def res(sport="cfb", market="total", game_pk=1, ev=0.05, beat=None, avail=True, kind="pinnacle_novig", result="W", units=0.9):
    return {"sport": sport, "market": market, "game_pk": game_pk, "close_ev": ev, "close_kind": kind,
            "beat_close": (None if ev is None else ev > 0) if beat is None else beat, "available_next_poll": avail, "result": result, "units": units}


def many(n, **kw):
    """n tickets in n different games."""
    base = kw.pop("game_start", 1000)
    return [res(game_pk=base + i, **kw) for i in range(n)]


def totals(n=60, **kw):
    return many(n, sport="cfb", market="total", game_start=1000, **kw)


def spreads(n=45, **kw):
    return many(n, sport="cfb", market="spread", game_start=2000, **kw) + many(n, sport="nfl", market="spread", game_start=3000, **kw)
