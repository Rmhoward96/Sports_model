"""Fitting k for the prop shrink. PURE -- no IO.

Spec section 1. For one group of legs the Brier-minimising closed form, clamped:

    k = clamp( sum(y - m)(p - m) / sum(p - m)^2, 0, 1 )      (y = 1 when the OVER hit; pushes excluded; k = 0 if the denominator is 0)

A pooled NFL k is fitted on all eligible legs; each market gets its own k, partially pooled:

    k_market = (n_m * k_m_raw + 300 * k_pool) / (n_m + 300)

If the pooled sample has fewer than 300 legs every k is 0. The sample is the legs of the CURRENT props model version; if
that gives fewer than 300 legs, all versions are pooled.
"""
from __future__ import annotations

import math

from ..serving.props_ev import SIM_TO_ODDS_MARKET, MLB_SIM_TO_ODDS_MARKET
from .formula import market_prob_over

MIN_POOL_LEGS = 300          # pooled sample below this: k = 0 everywhere; version sample below this: pool all versions
PRIOR_LEGS = 300             # weight of the pooled k in each market's partially pooled k
ALL_VERSIONS = "all"         # model_version_used when the version fallback pooled every version
FIT_SPORT = "nfl"


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def legs_from_grades(rows: list[dict], market_map: dict[str, str] | None = None) -> list[dict]:
    """`nfl_prop_grades` rows (plus an optional model_version) -> fit legs {season, week, game_pk, market, p, m, y,
    version}. `market` is the board's ODDS market key (the sim-to-odds mapping is applied; unmapped sim markets such as
    pass_yds and rush_att are dropped). Excluded: a result other than hit / miss (pushes), a lean other than over /
    under, a missing or non-finite p_over, a missing season or week, and any row whose best over or under price is
    missing or not a valid American price (|price| < 100). y is 1 when the OVER hit."""
    market_map = SIM_TO_ODDS_MARKET if market_map is None else market_map
    legs = []
    for r in rows:
        market = market_map.get(r.get("market"))
        result, lean = r.get("result"), r.get("lean")
        if market is None or result not in ("hit", "miss") or lean not in ("over", "under"):
            continue
        p = _num(r.get("p_over"))
        m = market_prob_over(r.get("over_price"), r.get("under_price"))
        if p is None or m is None or r.get("season") is None or r.get("week") is None:
            continue
        legs.append({
            "season": int(r["season"]), "week": int(r["week"]), "game_pk": r.get("game_pk"), "market": market,
            "p": p, "m": m, "y": 1 if (lean == "over") == (result == "hit") else 0,
            "version": r.get("model_version"),
        })
    return legs


def closed_form_k(legs: list[dict]) -> float:
    """The clamped Brier-minimising k for these legs; 0.0 when there are no legs or no variance in p - m."""
    num = sum((l["y"] - l["m"]) * (l["p"] - l["m"]) for l in legs)
    den = sum((l["p"] - l["m"]) ** 2 for l in legs)
    if den <= 0.0:
        return 0.0
    return min(1.0, max(0.0, num / den))


def select_sample(legs: list[dict], current_version: str | None) -> tuple[list[dict], str]:
    """(sample, model_version_used): the legs of the current version if there are at least 300, else every leg."""
    if current_version:
        own = [l for l in legs if l.get("version") == current_version]
        if len(own) >= MIN_POOL_LEGS:
            return own, current_version
    return list(legs), ALL_VERSIONS


def fit_table(legs: list[dict], current_version: str | None) -> dict:
    """k per NFL board market, partially pooled, plus the pooled values. Every board market (SIM_TO_ODDS_MARKET value)
    gets an entry; a market with no legs gets k = 0, n = 0, k_raw = None. MLB gets k = 0 for every market (no unbiased
    fit sample exists in Part A)."""
    sample, used = select_sample(legs, current_version)
    n_pool = len(sample)
    k_pool_raw = closed_form_k(sample)
    enough = n_pool >= MIN_POOL_LEGS
    k_pool = k_pool_raw if enough else 0.0
    nfl = {}
    for market in sorted(set(SIM_TO_ODDS_MARKET.values())):
        own = [l for l in sample if l["market"] == market]
        n = len(own)
        if n == 0:
            nfl[market] = {"k": 0.0, "n": 0, "k_raw": None}
            continue
        raw = closed_form_k(own)
        k = (n * raw + PRIOR_LEGS * k_pool) / (n + PRIOR_LEGS) if enough else 0.0
        nfl[market] = {"k": k, "n": n, "k_raw": raw}
    mlb = {market: {"k": 0.0, "n": 0, "k_raw": None} for market in sorted(set(MLB_SIM_TO_ODDS_MARKET.values()))}
    return {"model_version_used": used, "pooled": {"k": k_pool, "n": n_pool, "k_raw": k_pool_raw},
            "markets": {"nfl": nfl, "mlb": mlb}}


def build_document(legs: list[dict], current_version: str | None, fitted_at: str) -> dict:
    """The contents of assets/shrink_k.json: the fit_table result plus when it was fitted, which version string was
    current, and the data window (first and last (season, week) of the eligible legs and how many there were in all)."""
    doc = fit_table(legs, current_version)
    weeks = sorted({(l["season"], l["week"]) for l in legs})
    return {
        "schema": 1, "fitted_at": fitted_at, "current_version": current_version,
        "model_version_used": doc["model_version_used"],
        "data_window": {"first": list(weeks[0]) if weeks else None, "last": list(weeks[-1]) if weeks else None,
                        "n_legs": len(legs)},
        "pooled": {"sport": FIT_SPORT, **doc["pooled"]}, "markets": doc["markets"],
    }


def has_fewer_legs(existing: dict | None, new: dict) -> bool:
    """True when the committed file was fitted on MORE eligible legs than this fit saw: graded legs only ever accumulate,
    so a smaller count means the read was truncated and the existing file must be kept."""
    try:
        before = existing["data_window"]["n_legs"]
    except (TypeError, KeyError):
        return False
    return isinstance(before, int) and new["data_window"]["n_legs"] < before
