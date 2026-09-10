"""Pure edge/EV engine for the +EV desk-driven pilot.

Sub-project 4 (+EV desk-driven pilot), task 1 -- see
.superpowers/sdd/2026-09-09-plus-ev-4-desk-driven-pilot/task-1-brief.md.

Design (post sub-project-3 NO-GO, market-anchored): the model does NOT supply the
base true probability. Base true prob = Pinnacle's no-vig implied probability (the
sharp true prob); the desk overlays a clamped probability delta on top of it:

    true_prob = clamp(base_prob + desk_delta, 0, 1)

If the desk has no pick on a market, desk_delta is 0, so true_prob == base_prob.
The board surfaces TWO independent edge sources against this true_prob, and a row
is a pick whenever either one clears the floor:
  - line-shopping: the best available soft-book price is itself +EV vs true_prob
    (ev_best > 0), regardless of whether the desk touched the market at all --
    this is the "best_line_implied" / "soft_vs_sharp_gap" edge;
  - desk overlay: the desk moves true_prob away from base_prob (desk_delta != 0),
    which can also push ev_best (and ev_pinnacle) positive.
`is_pick` is keyed off ev_best alone (see `_finish_row`), so it captures both
sources without needing to special-case which one produced the edge. The model
(via `sportsmodel.model.trueprob_prob`) only supplies sigma + the win/cover/over
probability-conversion functions used to translate the desk's points-shift into a
probability delta -- it is never the base.

PURE -- no IO. Reuses `sportsmodel.serving.board` (implied_prob/novig/ev/best_price/
MAJOR_BOOKS/EV_CEILING) and `sportsmodel.model.trueprob_prob` (win_prob/cover_prob/
over_prob/desk_margin_shift/DESK_MAX_PROB_DELTA) for all probability + EV math --
nothing here re-derives no-vig, EV, or normal-distribution math.
"""
from __future__ import annotations

from ..model.trueprob_prob import (
    DESK_MAX_PROB_DELTA,
    cover_prob,
    desk_margin_shift,
    over_prob,
    win_prob,
)
from .board import EV_CEILING, MAJOR_BOOKS, best_price, ev, implied_prob, novig

# Football margin/total sigmas carried over from the sub-project-3 true-prob model
# (see trueprob_prob.py); the model supplies these + the prob-conversion functions
# only -- the base probability itself is market-anchored (Pinnacle no-vig), not
# model-derived.
SIGMA_MARGIN = 13.2
SIGMA_TOTAL = 10.0

# A market only becomes a pick once the desk-adjusted edge clears this floor.
# No longer consulted by `is_pick` (see module docstring) -- kept defined since
# the test suite still asserts against it as a documented desk-edge floor.
MIN_EDGE = 0.02

# A market only becomes a pick once the best available price clears this much
# EV vs true_prob. Below this, "+EV" is noise (rounding/vig-residue), not a
# real line-shopping or desk edge.
MIN_EV = 0.01


def market_margin(pinnacle_home_spread: float) -> float:
    """Implied home margin from the book's home spread convention (e.g. a
    home_line of -3.0 means the home side is favored by 3 -> margin 3.0)."""
    return -pinnacle_home_spread


def desk_prob_delta(kind: str, base_number: float, line: float, sigma: float,
                     desk_shift: float) -> float:
    """The clamped probability move the desk's points-shift induces, in
    home/over-referenced terms (positive = toward home/over).

    `kind` selects which trueprob_prob conversion is shifted:
      - "ml":     win_prob(base_number+desk_shift, sigma) - win_prob(base_number, sigma)
      - "spread": cover_prob(base_number+desk_shift, sigma, line) - cover_prob(base_number, sigma, line)
      - "total":  over_prob(base_number+desk_shift, sigma, line) - over_prob(base_number, sigma, line)
    Result is clamped to +-DESK_MAX_PROB_DELTA. `desk_shift == 0` -> 0.0 (no desk
    pick on this market -> no delta, regardless of kind).
    """
    if desk_shift == 0:
        return 0.0
    if kind == "ml":
        base = win_prob(base_number, sigma)
        shifted = win_prob(base_number + desk_shift, sigma)
    elif kind == "spread":
        base = cover_prob(base_number, sigma, line)
        shifted = cover_prob(base_number + desk_shift, sigma, line)
    elif kind == "total":
        base = over_prob(base_number, sigma, line)
        shifted = over_prob(base_number + desk_shift, sigma, line)
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    delta = shifted - base
    return max(-DESK_MAX_PROB_DELTA, min(DESK_MAX_PROB_DELTA, delta))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _pick_side(desk_side, prob_a: float, label_a: str, label_b: str) -> str:
    """The side to build a row for: the desk's side if it has one on this market,
    else the market-favored side (higher no-vig probability; ties -> label_a)."""
    if desk_side in (label_a, label_b):
        return desk_side
    return label_a if prob_a >= 0.5 else label_b


def _side_shift(desk: dict | None, desk_side) -> float:
    """Home/away-perspective margin shift (points) for a desk pick's side, via
    trueprob_prob's tested tier-weight logic (home -> +, away -> -)."""
    if not desk or desk_side not in ("home", "away"):
        return 0.0
    return desk_margin_shift({
        "conviction_tier": desk.get("conviction_tier"),
        "confidence": desk.get("confidence"),
        "spread_side": desk_side,
    })


def _total_side_shift(desk: dict | None, desk_side) -> float:
    """Points shift on the total, toward over/under. There's no dedicated
    "total shift" in trueprob_prob, so this reuses desk_margin_shift's tier-weight
    * DESK_MAX_PTS logic (same magnitude as the margin shift) via the pseudo
    mapping over->home(+1), under->away(-1) -- keeps the total shift simple and
    symmetric with the spread/ml shift rather than re-deriving a second constant."""
    if not desk or desk_side not in ("over", "under"):
        return 0.0
    pseudo_side = "home" if desk_side == "over" else "away"
    return desk_margin_shift({
        "conviction_tier": desk.get("conviction_tier"),
        "confidence": desk.get("confidence"),
        "spread_side": pseudo_side,
    })


def _finish_row(game: dict, market: str, side: str, base_prob: float, desk_delta: float,
                 pinnacle_price: int, soft_by_side: dict, conviction_tier) -> dict:
    true_prob = _clamp01(base_prob + desk_delta)
    edge = true_prob - base_prob
    ev_pinnacle = ev(true_prob, pinnacle_price)
    best = best_price(soft_by_side.get(side) or [])
    ev_best = ev(true_prob, best[1]) if best else None
    best_line_implied = implied_prob(best[1]) if best else None
    # Line-shopping gap: the sharp book's own price implied prob minus the best
    # soft price's implied prob -- BOTH vig-inclusive (price vs price), so at
    # equal prices the gap is 0 and a positive gap means the best soft price
    # pays more than the sharp for the same side. (Value-vs-truth is ev_best,
    # which uses the no-vig true_prob; this is the pure price edge to display.)
    soft_vs_sharp_gap = (implied_prob(pinnacle_price) - best_line_implied) if best_line_implied is not None else None
    is_pick = ev_best is not None and MIN_EV < ev_best <= EV_CEILING
    return {
        "sport": game.get("sport"),
        "game_pk": game.get("game_pk"),
        "matchup": game.get("matchup"),
        "commence_time": game.get("commence_time"),
        "market": market,
        "side": side,
        "base_prob": base_prob,
        "true_prob": true_prob,
        "edge": edge,
        "desk_delta": desk_delta,
        "conviction_tier": conviction_tier,
        "pinnacle_price": pinnacle_price,
        "ev_pinnacle": ev_pinnacle,
        "best_book": best[0] if best else None,
        "best_price": best[1] if best else None,
        "ev_best": ev_best,
        "best_line_implied": best_line_implied,
        "soft_vs_sharp_gap": soft_vs_sharp_gap,
        "is_pick": is_pick,
    }


def _ml_row(game: dict) -> dict | None:
    pin = game.get("pinnacle") or {}
    ml, spread = pin.get("moneyline") or {}, pin.get("spread") or {}
    home_price, away_price, home_line = ml.get("home"), ml.get("away"), spread.get("home_line")
    if home_price is None or away_price is None or home_line is None:
        return None

    desk = game.get("desk") or {}
    desk_side = desk.get("ml_pick")
    prob_home = novig(home_price, away_price)
    side = _pick_side(desk_side, prob_home, "home", "away")
    base_prob = prob_home if side == "home" else 1 - prob_home
    pinnacle_price = home_price if side == "home" else away_price

    shift_home = _side_shift(desk, desk_side)
    base_number = market_margin(home_line)
    delta_home = desk_prob_delta("ml", base_number, None, SIGMA_MARGIN, shift_home)
    desk_delta = delta_home if side == "home" else -delta_home

    soft = (game.get("books") or {}).get("moneyline") or {}
    return _finish_row(game, "moneyline", side, base_prob, desk_delta, pinnacle_price,
                        soft, desk.get("conviction_tier"))


def _spread_row(game: dict) -> dict | None:
    pin = game.get("pinnacle") or {}
    spread = pin.get("spread") or {}
    home_price, away_price, home_line = spread.get("home"), spread.get("away"), spread.get("home_line")
    if home_price is None or away_price is None or home_line is None:
        return None

    desk = game.get("desk") or {}
    desk_side = desk.get("spread_side")
    prob_home = novig(home_price, away_price)
    side = _pick_side(desk_side, prob_home, "home", "away")
    base_prob = prob_home if side == "home" else 1 - prob_home
    pinnacle_price = home_price if side == "home" else away_price

    shift_home = _side_shift(desk, desk_side)
    base_number = market_margin(home_line)
    delta_home = desk_prob_delta("spread", base_number, home_line, SIGMA_MARGIN, shift_home)
    desk_delta = delta_home if side == "home" else -delta_home

    soft = (game.get("books") or {}).get("spread") or {}
    return _finish_row(game, "spread", side, base_prob, desk_delta, pinnacle_price,
                        soft, desk.get("conviction_tier"))


def _total_row(game: dict) -> dict | None:
    pin = game.get("pinnacle") or {}
    total = pin.get("total") or {}
    over_price, under_price, line = total.get("over"), total.get("under"), total.get("line")
    if over_price is None or under_price is None or line is None:
        return None

    desk = game.get("desk") or {}
    desk_side = desk.get("total_side")
    prob_over = novig(over_price, under_price)
    side = _pick_side(desk_side, prob_over, "over", "under")
    base_prob = prob_over if side == "over" else 1 - prob_over
    pinnacle_price = over_price if side == "over" else under_price

    shift_over = _total_side_shift(desk, desk_side)
    delta_over = desk_prob_delta("total", line, line, SIGMA_TOTAL, shift_over)
    desk_delta = delta_over if side == "over" else -delta_over

    soft = (game.get("books") or {}).get("total") or {}
    return _finish_row(game, "total", side, base_prob, desk_delta, pinnacle_price,
                        soft, desk.get("conviction_tier"))


def ev_rows_for_game(game: dict) -> list[dict]:
    """One row per market (moneyline/spread/total) that has a two-way Pinnacle
    price, for the side the desk picked (or the market-favored side if the desk
    has no pick there). See module docstring for the true_prob design."""
    rows = [_ml_row(game), _spread_row(game), _total_row(game)]
    return [r for r in rows if r is not None]


def assemble_games(desk_rows: list[dict], odds_rows: list[dict]) -> list[dict]:
    """Join desk_current rows + latest-snapshot odds_snapshot rows (the caller
    pre-filters to the latest snapshot per game/market/side/book) into the `game`
    dict shape `ev_rows_for_game` expects, by game_pk. Only games present in
    `desk_rows` are emitted -- a game the desk hasn't touched at all has no row
    in desk_current and is out of scope for this pilot.

    Pinnacle prices come from book=="pinnacle" rows; soft-book price entries come
    from the remaining MAJOR_BOOKS rows, filtered to the line Pinnacle is posting
    for spread/total (so an alternate-line price doesn't get compared against the
    main-line Pinnacle price / used as a fake best-book edge).
    """
    by_game: dict = {}
    for o in odds_rows:
        by_game.setdefault(o.get("game_pk"), []).append(o)

    games = []
    for d in desk_rows:
        gpk = d.get("game_pk")
        game_odds = by_game.get(gpk, [])

        pinnacle: dict = {"moneyline": {}, "spread": {}, "total": {}}
        for o in game_odds:
            if o.get("book") != "pinnacle":
                continue
            market, side, price, line = o.get("market"), o.get("side"), o.get("price"), o.get("line")
            if market not in pinnacle or price is None:
                continue
            pinnacle[market][side] = int(price)
            if market == "spread" and side == "home" and line is not None:
                pinnacle[market]["home_line"] = float(line)
            if market == "total" and line is not None:
                pinnacle[market]["line"] = float(line)

        books = {"moneyline": {"home": [], "away": []},
                 "spread": {"home": [], "away": []},
                 "total": {"over": [], "under": []}}
        pin_home_line = pinnacle["spread"].get("home_line")
        pin_total_line = pinnacle["total"].get("line")
        for o in game_odds:
            book = o.get("book")
            if book == "pinnacle" or book not in MAJOR_BOOKS:
                continue
            market, side, price, line = o.get("market"), o.get("side"), o.get("price"), o.get("line")
            if market not in books or side not in books[market] or price is None:
                continue
            if market == "spread" and pin_home_line is not None and line is not None:
                want_line = pin_home_line if side == "home" else -pin_home_line
                if abs(float(line) - want_line) > 1e-6:
                    continue
            if market == "total" and pin_total_line is not None and line is not None:
                if abs(float(line) - pin_total_line) > 1e-6:
                    continue
            books[market][side].append((book, int(price)))

        desk = None
        if d.get("ml_pick") or d.get("spread_side") or d.get("total_side"):
            desk = {
                "ml_pick": d.get("ml_pick"),
                "spread_side": d.get("spread_side"),
                "total_side": d.get("total_side"),
                "conviction_tier": d.get("conviction_tier"),
                "confidence": d.get("confidence"),
            }

        games.append({
            "sport": d.get("sport"),
            "game_pk": gpk,
            "matchup": d.get("matchup"),
            "commence_time": d.get("commence_time"),
            "desk": desk,
            "pinnacle": pinnacle,
            "books": books,
        })
    return games
