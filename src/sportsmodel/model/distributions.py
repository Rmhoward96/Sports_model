"""Prop outcome distributions from per-PA vectors (docs/methodology.md Part C).

Every hitter prop is an aggregation of a batter's per-PA outcome vectors across
projected plate appearances. Vectors are heterogeneous (starter vs bullpen), so we
use Poisson-binomial / convolution forms rather than a single Binomial.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import prod


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
