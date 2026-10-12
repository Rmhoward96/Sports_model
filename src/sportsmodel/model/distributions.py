"""Prop outcome distributions from per-PA vectors (docs/methodology.md Part C).

Every hitter prop is an aggregation of a batter's per-PA outcome vectors across
projected plate appearances. Vectors are heterogeneous (starter vs bullpen), so we
use Poisson-binomial / convolution forms rather than a single Binomial.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from math import erf, exp, lgamma, log, prod, sqrt


def nb_pmf(mean: float, var: float, xmax: int = 30) -> list[float]:
    """Negative-binomial PMF from a target mean and variance (Poisson if var<=mean)."""
    if mean <= 0:
        return [1.0] + [0.0] * xmax
    if var <= mean * 1.0000001:  # not overdispersed -> Poisson
        out = [exp(-mean)]
        for x in range(1, xmax + 1):
            out.append(out[-1] * mean / x)
        s = sum(out)
        return [v / s for v in out]
    r = mean * mean / (var - mean)
    p = mean / var
    logp, log1mp = log(p), log(1 - p)
    out = [exp(lgamma(x + r) - lgamma(r) - lgamma(x + 1) + r * logp + x * log1mp)
           for x in range(xmax + 1)]
    s = sum(out)
    return [v / s for v in out]


def normal_sf(line: float, mean: float, sd: float) -> float:
    """P(X > line) for X ~ Normal(mean, sd)."""
    if sd <= 0:
        return 1.0 if mean > line else 0.0
    return 0.5 * (1 - erf((line - mean) / (sd * sqrt(2))))


def poisson_binomial_pmf(probs: Sequence[float]) -> list[float]:
    """PMF of the count of successes among independent Bernoulli(p_j) trials."""
    dp = [1.0]
    for p in probs:
        nxt = [0.0] * (len(dp) + 1)
        for i, val in enumerate(dp):
            nxt[i] += val * (1 - p)
            nxt[i + 1] += val * p
        dp = nxt
    return dp


def prob_over(pmf: Sequence[float], line: float) -> float:
    """P(X > line). Use half-point lines to avoid pushes."""
    return sum(p for k, p in enumerate(pmf) if k > line)


def prob_under(pmf: Sequence[float], line: float) -> float:
    return sum(p for k, p in enumerate(pmf) if k < line)


def prob_at_least_one_hr(hr_probs: Sequence[float]) -> float:
    """Home Run (Y/N): 1 - product(1 - p_hr,j) over projected PA."""
    return 1.0 - prod(1 - p for p in hr_probs)


def prob_over_dist(dist: dict, line: float) -> float:
    """P(X > line) from a serialized distribution.

    `dist` is either {"kind": "pmf", "pmf": [...]} (discrete counts) or
    {"kind": "normal", "mean": m, "sd": s}. Lets us evaluate the model's own
    distribution at *the book's* line, not just a fixed default line.
    """
    if not dist:
        return float("nan")
    if dist.get("kind") == "normal":
        return normal_sf(line, dist["mean"], dist["sd"])
    return sum(p for k, p in enumerate(dist.get("pmf") or []) if k > line)


def prob_cover(margin_dist: dict, home_line: float) -> float:
    """P(home covers a spread `home_line`, e.g. -1.5) from a serialized margin dist.

    Home covers iff margin + home_line > 0, i.e. margin > -home_line. A margin
    exactly equal to -home_line is a PUSH, not a cover, so the comparison is
    strict. `margin_dist` is {"kind": "margin", "offset": o, "pmf": [...]} where
    pmf[i] = P(margin == i - o) (see sim.engine.margin_pmf). Returns NaN if
    `margin_dist` is falsy or the wrong kind.
    """
    if not margin_dist or margin_dist.get("kind") != "margin":
        return float("nan")
    offset = margin_dist["offset"]
    pmf = margin_dist.get("pmf") or []
    threshold = -home_line
    return sum(p for i, p in enumerate(pmf) if (i - offset) > threshold)


# --- pick-side probabilities at the closing line (prediction_accuracy.spread_pick_prob / total_pick_prob) ---------
# These mirror the game page (site/js/pages/game.js gmDists / gmSplit with app.js distParse / distProbs): the same
# stored pmf, the same strict comparisons, pushes excluded and the rest renormalised. A missing / unusable
# distribution gives None -- never a normal approximation.

def _parse_dist(dist) -> dict | None:
    """A stored distribution (dict, or JSON text from a TEXT / unparsed JSONB column) -> dict, else None."""
    if dist is None:
        return None
    if isinstance(dist, (str, bytes)):
        try:
            dist = json.loads(dist)
        except (ValueError, TypeError):
            return None
    return dist if isinstance(dist, dict) else None


def pick_dist(key: str, *sources) -> dict | None:
    """First source row whose `key` ('margin_dist' / 'total_dist') parses to a distribution with a pmf list.

    Mirrors the game page's gmDists: the prediction row (game_predictions / predictions_any) first, then the NFL
    sim row (nfl_sim). A source that lacks a usable pmf falls through to the next."""
    for src in sources:
        d = _parse_dist(src.get(key)) if src else None
        if d is not None and isinstance(d.get("pmf"), list):
            return d
    return None


def dist_split(dist, threshold) -> tuple[float, float] | None:
    """(P(value < threshold), P(value > threshold)) with pushes (value == threshold) excluded and the rest
    renormalised to sum to 1 -- site gmSplit. `dist` is {"kind": "margin", "offset", "pmf"} (value = i - offset) or
    {"kind": "pmf", "pmf"} (value = i). None without a pmf, without a threshold, or when only a push has mass."""
    d = _parse_dist(dist)
    if d is None or threshold is None or not isinstance(d.get("pmf"), list):
        return None
    offset = d.get("offset", 0) if d.get("kind") == "margin" else 0
    under = over = 0.0
    for i, p in enumerate(d["pmf"]):
        v = i - offset
        if v < threshold:
            under += p
        elif v > threshold:
            over += p
    s = under + over
    if not s > 0:
        return None
    return under / s, over / s


def spread_pick_prob(margin_dist, pred_margin, market_spread) -> float | None:
    """P(the model's spread pick covers the closing HOME line `market_spread`), from the margin pmf (HOME - AWAY).

    The pick side matches the site's trackRows: edge = pred_margin + market_spread; edge > 0 -> home, < 0 -> away,
    0 -> no bet (None). Home covers when margin > -market_spread, away when margin < -market_spread."""
    if pred_margin is None or market_spread is None:
        return None
    edge = pred_margin + market_spread
    if edge == 0:
        return None
    split = dist_split(margin_dist, -market_spread)
    if split is None:
        return None
    under, over = split
    return over if edge > 0 else under


def total_pick_prob(total_dist, pred_total, market_total) -> float | None:
    """P(the model's total pick lands) at the closing total: Over when pred_total > market_total, Under when
    below, None when equal (no bet) or without a distribution. Pushes excluded."""
    if pred_total is None or market_total is None or pred_total == market_total:
        return None
    split = dist_split(total_dist, market_total)
    if split is None:
        return None
    under, over = split
    return over if pred_total > market_total else under


def apply_affine(dist: dict, loc: float, scale: float) -> dict:
    """Location+scale remap of a discrete distribution, re-binned to its integer
    support. Calibrates the sim's total (pmf) and margin distributions: each support
    value x is mapped to scale*(x-mean)+mean+loc and its mass re-binned (nearest int,
    clipped to range). loc=0, scale=1 is the identity. Preserves kind/offset."""
    import numpy as np
    if not dist:
        return dist
    kind = dist.get("kind")
    pmf = np.asarray(dist.get("pmf") or [], dtype=float)
    if pmf.size == 0:
        return dist
    offset = dist.get("offset", 0)
    support = np.arange(pmf.size) - offset
    mean = float((support * pmf).sum())
    new_vals = scale * (support - mean) + mean + loc
    new_idx = np.clip(np.rint(new_vals + offset).astype(int), 0, pmf.size - 1)
    out = np.zeros_like(pmf)
    np.add.at(out, new_idx, pmf)
    s = out.sum()
    out = (out / s) if s > 0 else out
    res = {"kind": kind, "pmf": out.tolist()}
    if kind == "margin":
        res["offset"] = offset
    return res


def _convolve(a: Sequence[float], b: Sequence[float]) -> list[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += av * bv
    return out


def _pa_base_pmf(vec) -> list[float]:
    """Per-PA distribution over total bases {0,1,2,3,4} from an outcome vector.

    A walk = 0 total bases (grouped with K/out at index 0).
    """
    zero = vec["p_out"] + vec["p_bb"] + vec["p_k"]
    return [zero, vec["p_1b"], vec["p_2b"], vec["p_3b"], vec["p_hr"]]


def total_bases_pmf(vectors: Sequence[dict]) -> list[float]:
    """PMF of total bases summed across per-PA vectors (exact convolution)."""
    pmf = [1.0]
    for v in vectors:
        pmf = _convolve(pmf, _pa_base_pmf(v))
    return pmf


def hits_pmf(vectors: Sequence[dict]) -> list[float]:
    """PMF of hits across per-PA vectors (Poisson-binomial on p_hit)."""
    hit_probs = [v["p_1b"] + v["p_2b"] + v["p_3b"] + v["p_hr"] for v in vectors]
    return poisson_binomial_pmf(hit_probs)


def normal_to_pmf(mean: float, sd: float, xmax: int) -> list[float]:
    """Discretize Normal(mean, sd) onto integer support 0..xmax via CDF differences
    (P(X==k) ~ CDF(k+0.5)-CDF(k-0.5)), normalized to sum 1. For NFL totals."""
    from math import erf, sqrt
    def cdf(x: float) -> float:
        if sd <= 0:
            return 1.0 if x >= mean else 0.0
        return 0.5 * (1.0 + erf((x - mean) / (sd * sqrt(2))))
    pmf = [max(0.0, cdf(k + 0.5) - cdf(k - 0.5)) for k in range(xmax + 1)]
    s = sum(pmf)
    return [p / s for p in pmf] if s > 0 else pmf


def poisson_pmf(mean: float, xmax: int = 12) -> list[float]:
    """PMF of Poisson(mean) on 0..xmax, normalized. For TD-count props."""
    from math import exp, factorial
    if mean <= 0:
        out = [0.0] * (xmax + 1); out[0] = 1.0; return out
    pmf = [exp(-mean) * mean ** k / factorial(k) for k in range(xmax + 1)]
    s = sum(pmf)
    return [p / s for p in pmf] if s > 0 else pmf


def normal_to_margin_pmf(mean: float, sd: float, offset: int) -> dict:
    """Discretize Normal(mean, sd) onto integer margins -offset..+offset into the
    serving margin-dist format {"kind":"margin","offset":o,"pmf":[...]}, pmf[i]=P(margin==i-o)."""
    from math import erf, sqrt
    def cdf(x: float) -> float:
        if sd <= 0:
            return 1.0 if x >= mean else 0.0
        return 0.5 * (1.0 + erf((x - mean) / (sd * sqrt(2))))
    pmf = []
    for i in range(2 * offset + 1):
        m = i - offset
        pmf.append(max(0.0, cdf(m + 0.5) - cdf(m - 0.5)))
    s = sum(pmf)
    pmf = [p / s for p in pmf] if s > 0 else pmf
    return {"kind": "margin", "offset": offset, "pmf": pmf}
