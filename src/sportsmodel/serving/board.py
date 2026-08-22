"""Pure pick-math for the serving board: best-book selection, EV, no-vig, and
per-market row builders. Shared by scripts/generate_board.py and grade_results.py
so the live board and the graded track record can never drift."""
from __future__ import annotations

from ..model.calibration import calibrate
from ..model.distributions import apply_affine, prob_cover, prob_over_dist


def decimal_odds(american: int) -> float:
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied_prob(american: int) -> float:
    return 1.0 / decimal_odds(american)


def novig(price_side: int, price_other: int) -> float:
    """No-vig implied probability of `price_side` given the two-way market."""
    io, iu = implied_prob(price_side), implied_prob(price_other)
    return io / (io + iu)


def best_price(entries):
    """(book, american) with the highest decimal odds (best for the bettor); None if empty."""
    entries = [(bk, p) for bk, p in (entries or []) if p]
    if not entries:
        return None
    return max(entries, key=lambda e: decimal_odds(e[1]))


def ev(prob: float, american: int) -> float:
    return prob * decimal_odds(american) - 1


def _mkrow(market, side, line, label, model_p, market_p, price, book):
    e = ev(model_p, price)
    return {"market": market, "side": side, "line": line, "pick_label": label,
            "model_prob": model_p, "implied_prob": market_p, "ev": e,
            "odds": price, "book": book, "is_pick": e > 0}


def moneyline_row(home_wp, home_entries, away_entries, home_name, away_name):
    """Favored team (higher model win prob), priced at its best book. ML has no pass;
    is_pick just flags whether that best-book price is +EV."""
    hb, ab = best_price(home_entries), best_price(away_entries)
    if hb is None or ab is None:
        return None
    novig_home = novig(hb[1], ab[1])
    if home_wp >= 1 - home_wp:
        return _mkrow("moneyline", "home", None, f"{home_name} ML", home_wp, novig_home, hb[1], hb[0])
    return _mkrow("moneyline", "away", None, f"{away_name} ML", 1 - home_wp, 1 - novig_home, ab[1], ab[0])


def total_row(total_dist, total_cal, main_line, over_entries, under_entries):
    ob, ub = best_price(over_entries), best_price(under_entries)
    if ob is None or ub is None:
        return None
    p_over = prob_over_dist(apply_affine(total_dist, *total_cal), main_line)
    if p_over != p_over:  # NaN
        return None
    novig_over = novig(ob[1], ub[1])
    if ev(p_over, ob[1]) >= ev(1 - p_over, ub[1]):
        return _mkrow("total", "over", main_line, f"Over {main_line:g}", p_over, novig_over, ob[1], ob[0])
    return _mkrow("total", "under", main_line, f"Under {main_line:g}", 1 - p_over, 1 - novig_over, ub[1], ub[0])


def spread_row(margin_dist, margin_cal, home_line, home_entries, away_entries, home_name, away_name):
    hb, ab = best_price(home_entries), best_price(away_entries)
    if hb is None or ab is None:
        return None
    p_home = prob_cover(apply_affine(margin_dist, *margin_cal), home_line)
    if p_home != p_home:
        return None
    novig_home = novig(hb[1], ab[1])
    if ev(p_home, hb[1]) >= ev(1 - p_home, ab[1]):
        return _mkrow("spread", "home", home_line, f"{home_name} {home_line:+g}", p_home, novig_home, hb[1], hb[0])
    return _mkrow("spread", "away", -home_line, f"{away_name} {-home_line:+g}", 1 - p_home, 1 - novig_home, ab[1], ab[0])


def prop_row(market, dist, cal_target, main_line, over_entries, under_entries):
    """Prop pick by EV from the calibrated P(over) at the book's main line vs best-book
    prices. Over-only markets (e.g. home_run: no under posted) get ev_under=-inf, so the
    over is only a pick when genuinely +EV."""
    ob = best_price(over_entries)
    if ob is None:
        return None
    p_over = calibrate(cal_target, prob_over_dist(dist, main_line))
    if p_over != p_over:  # NaN
        return None
    ub = best_price(under_entries)
    ev_over = ev(p_over, ob[1])
    ev_under = ev(1 - p_over, ub[1]) if ub else float("-inf")
    if ev_over >= ev_under:
        market_p = novig(ob[1], ub[1]) if ub else implied_prob(ob[1])
        return _mkrow(market, "over", main_line, f"Over {main_line:g}", p_over, market_p, ob[1], ob[0])
    return _mkrow(market, "under", main_line, f"Under {main_line:g}", 1 - p_over,
                  1 - novig(ob[1], ub[1]), ub[1], ub[0])
