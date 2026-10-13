"""Closing reference prices and CLV maths for ledger picks. PURE -- no IO.

The close reference for a pick's side is, in order:
  1. Pinnacle's no-vig probability at the pick's EXACT line          (close_kind = pinnacle_novig)
  2. the median no-vig probability of >= 3 US books at the exact line (close_kind = us_consensus_novig)
  3. Pinnacle quoting a DIFFERENT line (no comparable probability; only the line move is measured)
  4. nothing                                                           (close_kind = missing)

CLV is computed against a probability only when the close is at the pick's own line; a line that moved is
measured in points instead (`clv_pts`), never by comparing probabilities across different numbers.
"""
from __future__ import annotations

from statistics import median

from ..serving.board import MAJOR_BOOKS, decimal_odds, implied_prob, novig
from ..serving.props_ev import MLB_SIM_TO_ODDS_MARKET, SIM_TO_ODDS_MARKET, normalize_player_name

PINNACLE = "pinnacle"
MIN_CONSENSUS_BOOKS = 3
LINE_EPS = 1e-4                    # odds_snapshot.line is REAL (float32): compare with a tolerance
STANDARD_PRICE = -110              # model spread / total picks carry no market price; graded at flat -110
OTHER_SIDE = {"home": "away", "away": "home", "over": "under", "under": "over"}
PROP_TO_ODDS = {"nfl": SIM_TO_ODDS_MARKET, "mlb": MLB_SIM_TO_ODDS_MARKET}


def _same(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= LINE_EPS


def odds_market(sport: str, market: str, is_prop: bool) -> str | None:
    """The odds_snapshot market code for a ledger market (game markets are identical; props map sim -> odds)."""
    return PROP_TO_ODDS.get(sport, {}).get(market) if is_prop else market


def _mirror(market: str, line):
    """The line the OTHER side of the same two-way market carries at the same number."""
    if line is None:
        return None
    return -float(line) if market == "spread" else float(line)


def _book_entries(rows: list[dict], omarket: str, player_key: str | None) -> dict[str, dict[str, list[tuple]]]:
    """{book: {side: [(line, price), ...]}} for one market (and one player, for props)."""
    out: dict[str, dict[str, list[tuple]]] = {}
    for r in rows:
        if r.get("market") != omarket or r.get("price") is None:
            continue
        if player_key is not None and normalize_player_name(r.get("player_name")) != player_key:
            continue
        if player_key is None and (r.get("player_name") or ""):
            continue
        out.setdefault(r["book"], {}).setdefault(r["side"], []).append((r.get("line"), int(r["price"])))
    return out


def _price_at(entries: list[tuple], line) -> int | None:
    for l, p in entries or []:
        if line is None or _same(l, line):
            return p
    return None


def _book_fair(sides: dict[str, list[tuple]], market: str, side: str, line) -> float | None:
    """One book's no-vig probability for `side` at the exact `line`, or None if either leg is missing there."""
    mine = _price_at(sides.get(side), line)
    theirs = _price_at(sides.get(OTHER_SIDE[side]), _mirror(market, line))
    return None if mine is None or theirs is None else novig(mine, theirs)


def _us_fairs(book_entries: dict, market: str, side: str, line) -> list[float]:
    """No-vig probabilities of the US books that quote BOTH legs at the exact line, deduplicated by display
    name (williamhill_us and caesars are one book; Caesars is kept when both are complete). Pinnacle and
    offshore books are excluded. A one-legged quote never displaces a complete one."""
    by_name: dict[str, float] = {}
    for book in sorted(book_entries, reverse=True):          # 'williamhill_us' first, so 'caesars' overwrites it
        name = MAJOR_BOOKS.get(book)
        if book == PINNACLE or name is None:
            continue
        fair = _book_fair(book_entries[book], market, side, line)
        if fair is not None:
            by_name[name] = fair
    return list(by_name.values())


def close_reference(pick: dict, rows: list[dict]) -> dict:
    """Close reference for one ledger pick from the close snapshot `rows` of its game.

    `pick`: sport, market, side, line, player_name, player_id (props are the picks carrying a player_id).
    Returns {close_kind, fair_prob, close_line, same_line, books}: `fair_prob` is the side's no-vig probability
    AT THE PICK'S LINE (None when only a moved-line Pinnacle quote exists); `close_line` is the side's own line
    at the close (Pinnacle's when it quotes one, else the pick's line)."""
    sport, market, side, line = pick["sport"], pick["market"], pick["side"], pick.get("line")
    is_prop = pick.get("player_id") is not None
    omarket = odds_market(sport, market, is_prop)
    missing = {"close_kind": "missing", "fair_prob": None, "close_line": None, "same_line": None, "books": 0}
    if omarket is None or side not in OTHER_SIDE:
        return missing
    if market != "moneyline" and line is None:          # a lineless spread / total / prop matches nothing
        return missing
    player_key = normalize_player_name(pick.get("player_name")) if is_prop else None
    entries = _book_entries(rows, omarket, player_key)

    pin = entries.get(PINNACLE)
    pin_fair = _book_fair(pin, market, side, line) if pin else None
    if pin_fair is not None:
        return {"close_kind": "pinnacle_novig", "fair_prob": pin_fair, "close_line": line,
                "same_line": True, "books": 1}

    pin_line = None
    if pin and line is not None:                 # Pinnacle quotes this market, but at another number
        cands = [(l, p) for l, p in pin.get(side, []) if l is not None]
        theirs = {round(float(l), 4) for l, _ in pin.get(OTHER_SIDE[side], []) if l is not None}
        cands = [(l, p) for l, p in cands if round(_mirror(market, l), 4) in theirs]
        if cands:
            pin_line = min(cands, key=lambda lp: abs(float(lp[0]) - float(line)))[0]

    fairs = _us_fairs(entries, market, side, line)
    if len(fairs) >= MIN_CONSENSUS_BOOKS:
        close_line = pin_line if pin_line is not None else line
        return {"close_kind": "us_consensus_novig", "fair_prob": median(fairs), "close_line": close_line,
                "same_line": _same(close_line, line) if line is not None else True, "books": len(fairs)}

    if pin_line is not None:
        return {"close_kind": "pinnacle_novig", "fair_prob": None, "close_line": float(pin_line),
                "same_line": False, "books": 1}
    return missing


def clv_prob(fair_prob: float | None, best_price: int | None) -> float | None:
    """Close fair probability minus the implied probability of the price taken (same line only). Positive =
    the price taken was better than the close. The taker's vig is included, so a zero-skill bet scores ~ -2 to -4pp."""
    if fair_prob is None or best_price is None:
        return None
    return float(fair_prob) - implied_prob(int(best_price))


def clv_pts(market: str, side: str, pick_line, close_line) -> float | None:
    """Line points gained versus the close, from the bettor's side: spreads use the side's own line
    (pick - close), Over gains when the close is higher (close - pick), Under the reverse. None for moneylines
    or when either line is missing."""
    if market == "moneyline" or pick_line is None or close_line is None:
        return None
    d = float(pick_line) - float(close_line)
    if market == "spread" or side == "under":
        return d
    return -d if side == "over" else None


def beat_close(prob_clv: float | None, pts_clv: float | None) -> bool | None:
    """True when the pick beat the close: on points if the line moved, else on probability. None when neither
    can be measured."""
    if pts_clv is not None and abs(pts_clv) > LINE_EPS:
        return pts_clv > 0
    if prob_clv is not None:
        return prob_clv > 0
    return None


def settle_game(market: str, side: str, line, margin: float | None, total: float | None) -> str | None:
    """W / L / P for a game-line pick. `margin` = home - away; `line` is the SIDE's own line (spread) or the
    total. None when the inputs needed are missing."""
    if market in ("moneyline", "spread") and side not in ("home", "away"):
        return None
    if market == "total" and side not in ("over", "under"):
        return None
    if market == "moneyline":
        if margin is None:
            return None
        sm = margin if side == "home" else -margin
    elif market == "spread":
        if margin is None or line is None:
            return None
        sm = (margin if side == "home" else -margin) + float(line)
    elif market == "total":
        if total is None or line is None:
            return None
        sm = (total - float(line)) if side == "over" else (float(line) - total)
    else:
        return None
    return "W" if sm > 0 else "L" if sm < 0 else "P"


def settle_prop(side: str, line, actual: float | None) -> str | None:
    if actual is None or line is None or side not in ("over", "under"):
        return None
    d = float(actual) - float(line)
    sm = d if side == "over" else -d
    return "W" if sm > 0 else "L" if sm < 0 else "P"


def units_at_pick(result: str | None, price: int | None) -> float | None:
    """Profit in units for 1 unit staked at `price`."""
    if result not in ("W", "L", "P") or price is None:
        return None
    try:
        p = int(price)
    except (TypeError, ValueError):
        return None
    if abs(p) < 100:
        return None
    return {"W": decimal_odds(p) - 1.0, "L": -1.0, "P": 0.0}[result]
