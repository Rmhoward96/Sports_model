"""Grade a pick against Pinnacle's close (CLV, or line direction when the line moved), then settle it."""
from __future__ import annotations

from datetime import datetime

from .constants import CLOSE_VALID_MAX_MIN, DIRECTION_STEP, SOCCER
from .devig import american_to_decimal, novig_power_n
from .picks import same_line


def close_valid(close: dict | None) -> bool:
    """A close counts only when taken 0-30 minutes before start."""
    return close is not None and 0 <= close["minutes_to_start"] <= CLOSE_VALID_MAX_MIN


def close_fair(close_prices: dict, league: str, market: str, side: str) -> float | None:
    """De-vigged fair probability of `side` at the close, or None when the market is incomplete."""
    if market == "ml":
        want = ["home", "draw", "away"] if league in SOCCER else ["home", "away"]
    elif market == "spread":
        want = ["home", "away"]
    else:
        want = ["over", "under"]
    if set(close_prices) != set(want) or side not in want:
        return None
    if market != "ml" and any(close_prices[s].get("line") is None for s in want):
        return None                      # a spread / total side with no point: the market cannot be read
    if market == "spread" and not same_line(close_prices["home"]["line"], -close_prices["away"]["line"]):
        return None
    if market == "total" and not same_line(close_prices["over"]["line"], close_prices["under"]["line"]):
        return None
    fair = novig_power_n([close_prices[s]["price"] for s in want])
    return fair[want.index(side)]


def _moved_toward(market: str, side: str, ref_line: float, close_line: float) -> float:
    """How far the market line has moved toward the pick since the reference line."""
    if market == "total" and side == "over":
        return close_line - ref_line
    return ref_line - close_line         # spread (either side) and under


def grade_pick(pick: dict, close: dict | None, graded_at: datetime) -> dict:
    """A result row without outcome or units."""
    row = {"pick_id": pick.get("pick_id"), "graded_by": "no_close", "close_fair": None, "clv": None,
           "beat": None, "close_line": None,
           "close_minutes": close["minutes_to_start"] if close is not None else None,
           "graded_at": graded_at}
    if not close_valid(close):
        return row
    market, side = pick["market"], pick["side"]
    fair = close_fair(close["prices"], pick["league"], market, side)
    if fair is None:
        return row
    line = close["prices"][side]["line"] if market != "ml" else None
    if market != "ml" and pick["ref_line"] is None:
        return row                       # no reference line to compare the close with
    if market == "ml" or same_line(line, pick["ref_line"]):
        clv = fair * american_to_decimal(pick["price"]) - 1
        row.update(graded_by="clv", close_fair=fair, clv=clv, beat=clv > 0, close_line=line)
    else:
        move = _moved_toward(market, side, pick["ref_line"], line)
        row.update(graded_by="direction", beat=move >= DIRECTION_STEP - 1e-9, close_line=line)
    return row


def settle(pick: dict, home: int, away: int) -> tuple[str, float]:
    """(outcome, units) at the pick's own line and price."""
    market, side = pick["market"], pick["side"]
    if market == "ml":
        winner = "home" if home > away else "away" if away > home else ("draw" if pick["league"] in SOCCER else None)
        margin = 0 if winner is None else (1 if side == winner else -1)
    elif market == "spread":
        own = home - away if side == "home" else away - home
        margin = own + pick["line"]
    else:
        diff = home + away - pick["line"]
        margin = diff if side == "over" else -diff
    if margin > 0:
        return "win", american_to_decimal(pick["price"]) - 1
    if margin < 0:
        return "loss", -1.0
    return "push", 0.0
