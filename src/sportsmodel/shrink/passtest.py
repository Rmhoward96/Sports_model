"""The pre-registered pass test for the prop shrink. PURE (numpy only) -- no IO.

FROZEN as of docs/superpowers/specs/2026-10-13-shrink-props-design.md section 3: nothing here may change once results
exist. Evaluation set: each graded NFL lean in week W (keyed by (season, week)) is scored with the shrunk probability,
using the k fitted by the section 1 rules on legs graded in weeks BEFORE W only; weeks with fewer than 300 earlier legs are
not scored; pushes are excluded. Judged ONCE, at the first weekly report with at least 2,000 scored legs (before that the
status is "collecting"). It passes only if BOTH hold:
  1. Brier(market) - Brier(shrunk) >= 0.001, and the paired game-clustered bootstrap 95% CI of that difference has its
     lower bound > 0 (resamples games with replacement: 10,000 resamples, seed 20261013, 2.5th percentile)
  2. the logistic regression y ~ a + b * logit(shrunk) has slope b in [0.8, 1.2]
A pass means the model adds information beyond the board's market price; it does not prove profit (CLV does).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .fit import MIN_POOL_LEGS, fit_table
from .formula import shrink_prob

JUDGEMENT_LEGS = 2000
MIN_BRIER_GAIN = 0.001
SLOPE_LO, SLOPE_HI = 0.8, 1.2
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261013
LOWER_PERCENTILE = 2.5
CHUNK = 1000                                  # resamples drawn per block: bounds memory, fixed so results are reproducible
EPS = 1e-12                                   # float noise only: a gain of exactly 0.001 or a slope of exactly 0.8 passes
CRITERIA = ("Brier gain over the market >= 0.001", "paired game-clustered 95% lower bound > 0", "calibration slope in [0.8, 1.2]")


def week_key(leg: dict) -> tuple[int, int]:
    return (leg["season"], leg["week"])


def score_walk_forward(legs: list[dict], current_version: str | None) -> list[dict]:
    """Every leg of every week that has at least 300 legs in EARLIER weeks, with the shrunk probability from the k fitted
    on those earlier legs alone (the section 1 rules, including the model-version fallback). A market with no earlier legs
    has k = 0, so its shrunk probability is the market's."""
    by_week: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for leg in legs:
        by_week[week_key(leg)].append(leg)
    scored = []
    for wk in sorted(by_week):
        earlier = [l for l in legs if week_key(l) < wk]
        if len(earlier) < MIN_POOL_LEGS:
            continue
        ks = {m: e["k"] for m, e in fit_table(earlier, current_version)["markets"]["nfl"].items()}
        for l in by_week[wk]:
            k = ks.get(l["market"], 0.0)
            scored.append({"season": l["season"], "week": l["week"], "game_pk": l["game_pk"], "market": l["market"],
                           "p": l["p"], "m": l["m"], "y": l["y"], "k": k, "shrunk": shrink_prob(l["p"], l["m"], k)})
    return scored


def _gains(scored: list[dict]) -> list[float]:
    """Per-leg Brier(market) - Brier(shrunk): positive where the shrunk probability beat the market's."""
    return [(s["m"] - s["y"]) ** 2 - (s["shrunk"] - s["y"]) ** 2 for s in scored]


def brier_gain(scored: list[dict]) -> float | None:
    g = _gains(scored)
    return sum(g) / len(g) if g else None


def cluster_bootstrap_lower(scored: list[dict], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                            percentile: float = LOWER_PERCENTILE) -> float | None:
    """Lower bound of the paired Brier gain: resample GAMES with replacement (all legs of a drawn game come along), take
    each resample's leg-weighted mean gain, return the `percentile`-th percentile. Deterministic for a given seed."""
    per_game: dict = defaultdict(list)
    for s, g in zip(scored, _gains(scored)):
        per_game[s["game_pk"]].append(g)
    if not per_game:
        return None
    games = sorted(per_game, key=str)
    sums = np.array([sum(per_game[g]) for g in games])
    counts = np.array([len(per_game[g]) for g in games], dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, CHUNK):
        idx = rng.integers(0, len(games), size=(min(CHUNK, resamples - start), len(games)))
        means.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    return float(np.percentile(np.concatenate(means), percentile))


def calibration_slope(scored: list[dict]) -> float | None:
    """b in the logistic regression y ~ a + b * logit(shrunk), by Newton-Raphson (IRLS). None when it is undefined: no
    variance in the predictor, a singular fit, or no convergence (for example perfect separation)."""
    if len(scored) < 3:
        return None
    s = np.array([r["shrunk"] for r in scored], dtype=float)
    y = np.array([r["y"] for r in scored], dtype=float)
    x = np.log(s / (1.0 - s))
    if float(np.ptp(x)) < 1e-9:
        return None
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -35.0, 35.0)))
        w = mu * (1.0 - mu)
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not np.all(np.isfinite(beta)) or float(np.max(np.abs(beta))) > 50.0:
            return None
        if float(np.max(np.abs(step))) < 1e-10:
            return float(beta[1])
    return None


def evaluate(scored: list[dict], *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict | None:
    """The running numbers: n, games, both Brier scores, the gain, its clustered lower bound and the slope."""
    if not scored:
        return None
    n = len(scored)
    return {
        "n": n, "games": len({s["game_pk"] for s in scored}),
        "brier_market": sum((s["m"] - s["y"]) ** 2 for s in scored) / n,
        "brier_shrunk": sum((s["shrunk"] - s["y"]) ** 2 for s in scored) / n,
        "gain": brier_gain(scored), "lower": cluster_bootstrap_lower(scored, resamples=resamples),
        "slope": calibration_slope(scored),
    }


def verdict(gain: float | None, lower: float | None, slope: float | None) -> dict:
    """PASS only if the gain is >= 0.001 AND its clustered lower bound is > 0 AND the slope is in [0.8, 1.2]."""
    checks = [
        (gain is not None and gain + EPS >= MIN_BRIER_GAIN,
         "n/a" if gain is None else f"{gain:+.5f} (need >= {MIN_BRIER_GAIN:+.3f})"),
        (lower is not None and lower > 0, "n/a" if lower is None else f"{lower:+.5f} (need > 0)"),
        (slope is not None and SLOPE_LO - EPS <= slope <= SLOPE_HI + EPS,
         "undefined" if slope is None else f"{slope:.3f} (need {SLOPE_LO} to {SLOPE_HI})"),
    ]
    criteria = [{"name": n, "passed": bool(ok), "detail": d} for n, (ok, d) in zip(CRITERIA, checks)]
    return {"verdict": "PASS" if all(c["passed"] for c in criteria) else "FAIL", "criteria": criteria}


def decide(scored: list[dict], previous: dict | None, as_of: str, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The decision block of this week's report. Once a previous report judged the test the verdict is carried forward
    untouched (the test is judged ONCE); otherwise judge now if there are at least 2,000 scored legs, else 'collecting'."""
    if previous and previous.get("status") == "judged":
        return previous
    running = evaluate(scored, resamples=resamples)
    base = {"scored": len(scored), "need_scored": JUDGEMENT_LEGS, "running": running}
    if len(scored) < JUDGEMENT_LEGS:
        return {"status": "collecting", **base}
    v = verdict(running["gain"], running["lower"], running["slope"])
    return {"status": "judged", "judged_on": as_of, **base, **v}
