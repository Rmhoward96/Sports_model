"""The pre-registered CLV verdict for the 'take these' list. PURE (numpy only) -- no IO.

FROZEN once results exist. The test is judged ONCE, on the FIRST `N_REQUIRED` picks to be graded (ordered by
`(graded_at, id)`, because that set is fixed the moment it exists; ordering by first_seen could change as later games
grade). Only picks first seen on or after TEST_START whose grade is a CLV or a line direction count; October picks and
`no_close` picks do not. It passes only if ALL three hold:
  1. beat-close rate >= BEAT_MIN, over the 300 (clv and direction grades alike)
  2. mean CLV > 0, over the picks that have a CLV
  3. the event-clustered 95% lower bound of that mean CLV > 0 (cluster bootstrap: RESAMPLES resamples, seed SEED)
Once a verdict row exists, `decide` returns it unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

from ..ledger.core import as_utc
from .constants import BEAT_MIN, LOWER_PERCENTILE, N_REQUIRED, RESAMPLES, SEED, TEST_START

CHUNK = 1000                     # resamples drawn per block: bounds memory, fixed so results are reproducible
CRITERIA = (f"beat_rate>={BEAT_MIN}", "mean_clv>0", "lower>0")


def cluster_lower(rows: list[dict], *, key: str = "event_id", value: str = "clv", resamples: int = RESAMPLES,
                  seed: int = SEED, percentile: float = LOWER_PERCENTILE) -> float | None:
    """Lower bound of the mean `value`: resample clusters (`key`) with replacement (all rows of a drawn cluster come
    along), take each resample's row-weighted mean, return the `percentile`-th percentile. Rows without a `value` are
    skipped. Deterministic for a given seed. Same algorithm as early.passtest.cluster_bootstrap_lower."""
    per_cluster: dict = defaultdict(list)
    for r in rows:
        if r.get(value) is not None:
            per_cluster[r[key]].append(float(r[value]))
    if not per_cluster:
        return None
    clusters = sorted(per_cluster)
    sums = np.array([sum(per_cluster[c]) for c in clusters])
    counts = np.array([len(per_cluster[c]) for c in clusters], dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, CHUNK):
        idx = rng.integers(0, len(clusters), size=(min(CHUNK, resamples - start), len(clusters)))
        means.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    return float(np.percentile(np.concatenate(means), percentile))


def test_rows(rows: list[dict]) -> list[dict]:
    """The picks that count toward the test, in grading order: first seen on/after TEST_START, graded by clv or
    direction, sorted by (graded_at, id)."""
    keep = [r for r in rows if as_utc(r["first_seen"]) >= TEST_START and r.get("graded_by") in ("clv", "direction")]
    return sorted(keep, key=lambda r: (as_utc(r["graded_at"]), r["id"]))


def stats(rows: list[dict]) -> dict:
    """The test statistics of `rows` (test rows): beat rate over all of them, mean CLV and its clustered lower bound
    over those that have a CLV. None where there is nothing to average."""
    clvs = [float(r["clv"]) for r in rows if r.get("clv") is not None]
    return {
        "n": len(rows),
        "beat_rate": sum(1 for r in rows if r.get("beat") is True) / len(rows) if rows else None,
        "mean_clv": sum(clvs) / len(clvs) if clvs else None,
        "n_clv": len(clvs),
        "lower": cluster_lower(rows),
        "n_direction": sum(1 for r in rows if r.get("graded_by") == "direction"),
        "n_better_line": sum(1 for r in rows if r.get("better_line")),
    }


def criteria(s: dict) -> dict[str, bool]:
    """The three pass criteria for `stats` output; an undefined number fails its criterion."""
    return {
        CRITERIA[0]: s["beat_rate"] is not None and s["beat_rate"] >= BEAT_MIN,
        CRITERIA[1]: s["mean_clv"] is not None and s["mean_clv"] > 0,
        CRITERIA[2]: s["lower"] is not None and s["lower"] > 0,
    }


def decide(all_rows: list[dict], existing: dict | None, as_of: datetime) -> dict:
    """The decision for this report. An existing verdict is carried unchanged (judged ONCE); below N_REQUIRED test
    rows nothing can pass or fail; otherwise judge the first N_REQUIRED graded."""
    if existing:
        return {**existing, "carried": True}
    rows = test_rows(all_rows)
    if len(rows) < N_REQUIRED:
        return {"judged": False, **stats(rows)}
    first = rows[:N_REQUIRED]
    s = stats(first)
    crit = criteria(s)
    return {"judged": True, "passed": all(crit.values()), "pick_ids": [r["id"] for r in first], "criteria": crit, **s}


def verdict_row(decision: dict, judged_at: datetime) -> dict:
    """A judged decision as the take_verdict row (store.VERDICT_COLUMNS without id)."""
    return {"judged_at": judged_at, "n": decision["n"], "beat_rate": decision["beat_rate"],
            "mean_clv": decision["mean_clv"] if decision["mean_clv"] is not None else 0.0,   # NOT NULL; fails its criterion
            "lower": decision["lower"] if decision["lower"] is not None else 0.0,
            "passed": decision["passed"], "criteria": decision["criteria"], "pick_ids": decision["pick_ids"]}
