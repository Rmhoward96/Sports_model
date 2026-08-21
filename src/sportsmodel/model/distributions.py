"""Prop outcome distributions from per-PA vectors (docs/methodology.md Part C).

Every hitter prop is an aggregation of a batter's per-PA outcome vectors across
projected plate appearances. Vectors are heterogeneous (starter vs bullpen), so we
use Poisson-binomial / convolution forms rather than a single Binomial.
"""
from __future__ import annotations

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
