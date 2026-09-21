"""Market-microstructure features for the cover/total ensemble (Phase 2). PURE.

Reads `odds_snapshot`-shaped rows (line movement, sharp-vs-soft divergence) and
`nfl_betting_splits`-shaped rows (public %), producing features for
`build_cover_dataset.assemble_rows`. Every function is leakage-safe: it uses
only snapshots captured at/before a decision timestamp. All builders return NaN
(or None) when the underlying data is absent -- which is the norm until capture
accrues -- and the GBM (HistGradientBoosting) consumes the NaNs natively.

A snapshot row is a dict with: game_pk, market ("spread"|"total"|"moneyline"),
side ("home"|"away"|"over"|"under"), book, line, price, captured_at.
"""
from __future__ import annotations

import math
import statistics

from sportsmodel.serving.board import novig

# Reference side per market: the side whose posted line we track over time.
_REF_SIDE = {"spread": "home", "total": "over"}
_OPPOSITE = {"home": "away", "away": "home", "over": "under", "under": "over"}
NAN = float("nan")


def _before(snapshots, game_pk, market, decision_ts):
    return [s for s in (snapshots or [])
            if s.get("game_pk") == game_pk and s.get("market") == market
            and s.get("captured_at") is not None and s.get("captured_at") <= decision_ts]


def _consensus_line(rows):
    """Median posted line across books for one (side) at one capture time."""
    lines = [s["line"] for s in rows if s.get("line") is not None]
    return statistics.median(lines) if lines else None


def line_movement(snapshots, game_pk, market, decision_ts) -> dict:
    """Open->close movement of the reference side's consensus line, using only
    snapshots at/before `decision_ts`. Returns {open_line, close_line, dline,
    abs_dline} (dline = close - open); all NaN if there aren't at least two
    distinct capture times with a line for the reference side."""
    ref = _REF_SIDE.get(market)
    rows = [s for s in _before(snapshots, game_pk, market, decision_ts) if s.get("side") == ref]
    by_ts: dict = {}
    for s in rows:
        by_ts.setdefault(s["captured_at"], []).append(s)
    stamps = sorted(t for t, r in by_ts.items() if _consensus_line(r) is not None)
    if len(stamps) < 2:
        return {"open_line": NAN, "close_line": NAN, "dline": NAN, "abs_dline": NAN}
    open_line = _consensus_line(by_ts[stamps[0]])
    close_line = _consensus_line(by_ts[stamps[-1]])
    dline = close_line - open_line
    return {"open_line": open_line, "close_line": close_line,
            "dline": dline, "abs_dline": abs(dline)}


def sharp_vs_soft(snapshots, game_pk, market, side, decision_ts) -> float | None:
    """Pinnacle's no-vig prob for `side` minus the soft-book consensus no-vig
    prob for `side`, at the latest capture at/before `decision_ts`. A positive
    value = Pinnacle (sharp) rates `side` higher than the crowd's books. None if
    either the sharp or soft two-way price is unavailable."""
    rows = [s for s in _before(snapshots, game_pk, market, decision_ts) if s.get("price") is not None]
    if not rows:
        return None
    latest = max(s["captured_at"] for s in rows)
    rows = [s for s in rows if s["captured_at"] == latest]
    other = _OPPOSITE.get(side)

    def _novig_for(books_rows):
        side_prices = [s["price"] for s in books_rows if s.get("side") == side]
        other_prices = [s["price"] for s in books_rows if s.get("side") == other]
        if not side_prices or not other_prices:
            return None
        return novig(int(statistics.median(side_prices)), int(statistics.median(other_prices)))

    pin = _novig_for([s for s in rows if str(s.get("book", "")).lower() == "pinnacle"])
    soft = _novig_for([s for s in rows if str(s.get("book", "")).lower() != "pinnacle"])
    if pin is None or soft is None:
        return None
    return pin - soft


def split_features(splits_row) -> dict:
    """cash%, ticket%, and their divergence (cash - ticket) for a side. The
    cash>ticket gap is the classic sharp signal (fewer, bigger bets). NaN-safe:
    a None row or missing field yields NaN."""
    if not splits_row:
        return {"cash_pct": NAN, "ticket_pct": NAN, "cash_minus_ticket": NAN}
    cash = splits_row.get("cash_pct")
    tick = splits_row.get("ticket_pct")
    cash = NAN if cash is None else float(cash)
    tick = NAN if tick is None else float(tick)
    diff = NAN if (math.isnan(cash) or math.isnan(tick)) else cash - tick
    return {"cash_pct": cash, "ticket_pct": tick, "cash_minus_ticket": diff}


def reverse_line_movement(home_dline, home_ticket_pct) -> float | None:
    """Signed reverse-line-movement magnitude for the SPREAD, from the home
    line's open->close delta (`home_dline`; negative = home got more favored,
    Odds-API/ESPN sign) and the % of tickets on the home side. RLM = the line
    moving AGAINST the public ticket majority (the sharp-fade signal):
      +abs(dline) when the line moves away from the ticket-majority side,
      -abs(dline) when it moves with the majority, 0.0 when flat or split 50/50.
    None when splits are absent or the movement is unknown."""
    if home_ticket_pct is None or home_dline is None or (isinstance(home_dline, float) and math.isnan(home_dline)):
        return None
    if home_dline == 0 or home_ticket_pct == 50:
        return 0.0
    public_on_home = home_ticket_pct > 50
    line_toward_home = home_dline < 0  # home spread more negative = more favored
    fades_public = (public_on_home and not line_toward_home) or (not public_on_home and line_toward_home)
    return abs(home_dline) if fades_public else -abs(home_dline)
