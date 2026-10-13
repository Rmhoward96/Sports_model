"""The pre-registered pass test for the early-week monitor. PURE (numpy only) -- no IO.

FROZEN as of docs/superpowers/specs/2026-10-13-early-week-monitor-design.md section 4: nothing here may change once
results exist. The test is judged ONCE, at the first weekly report where there are at least 150 graded tickets with a
Pinnacle close across the three markets AND at least 60 of them are CFB totals; before that the status is "collecting"
and running numbers cannot pass or fail anything. Each market is then judged separately (CFB spreads and NFL spreads
need at least 40 graded tickets, else "insufficient") and passes only if ALL four hold:
  1. mean close_ev >= +0.015
  2. the game-clustered 95% lower bound of mean close_ev > 0 (cluster bootstrap over games: 10,000 resamples, seed
     20261013, 2.5th percentile)
  3. beat_close rate >= 55%
  4. mean close_ev over tickets with available_next_poll = true > 0
Realized ROI is reported, never gated.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

MIN_GRADED_TOTAL = 150
MIN_CFB_TOTALS = 60
MIN_OTHER_MARKET = 40
MIN_MEAN_CLOSE_EV = 0.015
MIN_BEAT_RATE = 0.55
EPS = 1e-12                                  # float noise only: a mean of exactly 0.015 passes
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261013
LOWER_PERCENTILE = 2.5
CHUNK = 1000                                  # resamples drawn per block: bounds memory, fixed so results are reproducible
MARKETS = (("cfb", "total"), ("cfb", "spread"), ("nfl", "spread"))
LABELS = {("cfb", "total"): "CFB totals", ("cfb", "spread"): "CFB spreads", ("nfl", "spread"): "NFL spreads"}
CRITERIA = ("mean close_ev >= +1.5%", "clustered 95% lower bound > 0", "beat-close rate >= 55%",
            "mean close_ev where available_next_poll = true > 0")


def usable(r: dict) -> bool:
    """A graded ticket that counts toward the test: it has a Pinnacle close and a close EV."""
    return r.get("close_kind") == "pinnacle_novig" and r.get("close_ev") is not None


def usable_rows(results: list[dict], sport: str | None = None, market: str | None = None) -> list[dict]:
    return [r for r in results if usable(r) and (sport is None or r["sport"] == sport)
            and (market is None or r["market"] == market)]


def cluster_bootstrap_lower(rows: list[dict], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                            percentile: float = LOWER_PERCENTILE) -> float | None:
    """Lower bound of the mean close_ev: resample GAMES with replacement (all tickets of a drawn game come along), take
    each resample's ticket-weighted mean, return the `percentile`-th percentile. Deterministic for a given seed."""
    per_game: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        per_game[(r["sport"], r["game_pk"])].append(float(r["close_ev"]))
    if not per_game:
        return None
    games = sorted(per_game)
    sums = np.array([sum(per_game[g]) for g in games])
    counts = np.array([len(per_game[g]) for g in games], dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, CHUNK):
        idx = rng.integers(0, len(games), size=(min(CHUNK, resamples - start), len(games)))
        means.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    return float(np.percentile(np.concatenate(means), percentile))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def market_stats(rows: list[dict], *, bootstrap: bool = True) -> dict:
    """Running numbers for one market's usable tickets (also the four test statistics)."""
    evs = [float(r["close_ev"]) for r in rows]
    avail = [r for r in rows if r.get("available_next_poll") is True]
    seen = [r for r in rows if r.get("available_next_poll") is not None]
    units = [float(r["units"]) for r in rows if r.get("units") is not None]
    return {
        "n": len(rows), "games": len({(r["sport"], r["game_pk"]) for r in rows}), "mean_close_ev": _mean(evs),
        "lower": cluster_bootstrap_lower(rows) if (bootstrap and rows) else None,
        "beat_rate": _mean([1.0 if r.get("beat_close") else 0.0 for r in rows]),
        "avail_n": len(avail), "avail_observed": len(seen), "avail_share": _mean([1.0 if r["available_next_poll"] else 0.0 for r in seen]),
        "avail_mean_close_ev": _mean([float(r["close_ev"]) for r in avail]),
        "w": sum(1 for r in rows if r.get("result") == "W"), "l": sum(1 for r in rows if r.get("result") == "L"),
        "p": sum(1 for r in rows if r.get("result") == "P"), "units": sum(units), "roi": _mean(units),
    }


def ready(results: list[dict]) -> bool:
    """The judgement point: >= 150 usable graded tickets across the three markets and >= 60 of them CFB totals."""
    rows = [r for r in usable_rows(results) if (r["sport"], r["market"]) in MARKETS]
    return len(rows) >= MIN_GRADED_TOTAL and len(usable_rows(results, "cfb", "total")) >= MIN_CFB_TOTALS


def judge_market(sport: str, market: str, results: list[dict]) -> dict:
    """PASS / FAIL / INSUFFICIENT for one market, with the four criteria and their numbers."""
    rows = usable_rows(results, sport, market)
    stats = market_stats(rows)
    floor = MIN_CFB_TOTALS if (sport, market) == ("cfb", "total") else MIN_OTHER_MARKET
    if stats["n"] < floor:
        return {"sport": sport, "market": market, "verdict": "INSUFFICIENT", "stats": stats, "criteria": [],
                "detail": f"{stats['n']} graded tickets with a Pinnacle close, {floor} needed"}
    checks = [
        (stats["mean_close_ev"] + EPS >= MIN_MEAN_CLOSE_EV, f"{stats['mean_close_ev']:+.4f} (need >= {MIN_MEAN_CLOSE_EV:+.3f})"),
        (stats["lower"] > 0, f"{stats['lower']:+.4f} (need > 0)"),
        (stats["beat_rate"] + EPS >= MIN_BEAT_RATE, f"{stats['beat_rate']:.1%} (need >= {MIN_BEAT_RATE:.0%})"),
        (stats["avail_mean_close_ev"] is not None and stats["avail_mean_close_ev"] > 0,
         "no ticket with available_next_poll = true" if stats["avail_mean_close_ev"] is None
         else f"{stats['avail_mean_close_ev']:+.4f} over {stats['avail_n']} tickets (need > 0)"),
    ]
    criteria = [{"name": n, "passed": bool(ok), "detail": d} for n, (ok, d) in zip(CRITERIA, checks)]
    verdict = "PASS" if all(c["passed"] for c in criteria) else "FAIL"
    return {"sport": sport, "market": market, "verdict": verdict, "stats": stats, "criteria": criteria, "detail": ""}


def decide(results: list[dict], previous: dict | None, as_of: str) -> dict:
    """The decision block of this week's report. Once a previous report judged the test the verdicts are frozen and
    carried forward untouched (the test is judged ONCE); otherwise judge now if the judgement point is reached, else
    'collecting'."""
    if previous and previous.get("status") == "judged":
        return previous
    graded = len([r for r in usable_rows(results) if (r["sport"], r["market"]) in MARKETS])
    progress = {"graded": graded, "need_graded": MIN_GRADED_TOTAL, "cfb_totals": len(usable_rows(results, "cfb", "total")),
                "need_cfb_totals": MIN_CFB_TOTALS}
    if not ready(results):
        return {"status": "collecting", **progress}
    verdicts = {f"{s}_{m}": judge_market(s, m, results) for s, m in MARKETS}
    return {"status": "judged", "judged_on": as_of, **progress, "markets": verdicts}
