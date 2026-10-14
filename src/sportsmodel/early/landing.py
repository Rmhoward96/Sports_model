"""Landing tables: how a game's result lands around its closing line. PURE except `load_tables` (reads JSON).

For one sport and market, and for each line value L on a half-point grid, a table holds the empirical distribution
of the integer RESULT of games whose closing line was within +/-1.0 point of L (widened to +/-2.0 when that gives
fewer than 300 games; a grid value still short at +/-2.0 is left out and answers None). The result is the home margin
for spreads and the total points for totals, so key-number mass (NFL margins 3, 7, 10, 14; common CFB totals) and the
push mass of whole-number lines are kept exactly as history delivered them.

The line value L is the "table-space centre": for spreads it is the expected home margin (positive = home favoured,
the nflverse `spread_line` and the CFB `market_spread` convention, i.e. minus the home team's betting line); for
totals it is the total.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .constants import GRID_EPS, GRID_STEP, LANDING_TABLES, MIN_GAMES, WINDOW_NARROW, WINDOW_WIDE

ASSETS = config.PROJECT_ROOT / "assets" / "early"
RESULT_VARIABLE = {"spread": "home_margin", "total": "total_points"}


@dataclass(frozen=True)
class Pmf:
    """Distribution of the integer result around one grid line. `probs` sums to 1."""
    n: int                       # games in the sample
    window: float                # +/- points the sample was drawn from (1.0 or 2.0)
    probs: dict[int, float]

    def mass(self, value) -> float:
        return self.probs.get(int(round(value)), 0.0)


def table_path(sport: str, market: str, directory: Path | None = None) -> Path:
    return Path(directory or ASSETS) / f"landing_{sport}_{market}.json"


def grid_key(center: float) -> str:
    """JSON key of a grid value. `+ 0.0` turns -0.0 into 0.0 so a pick-em never looks up '-0.0'."""
    return f"{float(center) + 0.0:.1f}"


def on_grid(center) -> bool:
    return center is not None and abs(2.0 * float(center) - round(2.0 * float(center))) < GRID_EPS


def build_table(pairs: list[tuple[float, int]], *, sport: str, market: str, seasons: tuple[int, int],
                source: str) -> dict:
    """The JSON-ready table from (closing line, integer result) pairs. Grid values run from the lowest to the highest
    observed line in half-point steps; each keeps the narrow window if it holds at least MIN_GAMES games, else the wide
    one, else is omitted. Counts (not probabilities) are stored, so the loader normalises exactly."""
    if market not in RESULT_VARIABLE:
        raise ValueError(f"unknown market: {market!r}")
    pairs = sorted((float(line), int(result)) for line, result in pairs)
    if not pairs:
        raise ValueError("no (line, result) pairs")
    lines = [p[0] for p in pairs]
    results = [p[1] for p in pairs]
    first = math.floor(lines[0] / GRID_STEP) * GRID_STEP
    last = math.ceil(lines[-1] / GRID_STEP) * GRID_STEP
    entries: dict[str, dict] = {}
    for i in range(int(round((last - first) / GRID_STEP)) + 1):
        center = first + i * GRID_STEP
        for window in (WINDOW_NARROW, WINDOW_WIDE):
            lo = bisect_left(lines, center - window - 1e-9)
            hi = bisect_right(lines, center + window + 1e-9)
            if hi - lo >= MIN_GAMES:
                counts = Counter(results[lo:hi])
                low, high = min(counts), max(counts)
                entries[grid_key(center)] = {"n": hi - lo, "window": window, "lo": low,
                                             "counts": [counts.get(v, 0) for v in range(low, high + 1)]}
                break
    return {"sport": sport, "market": market, "result": RESULT_VARIABLE[market], "step": GRID_STEP,
            "window_narrow": WINDOW_NARROW, "window_wide": WINDOW_WIDE, "min_games": MIN_GAMES,
            "games": len(pairs), "seasons": list(seasons), "source": source, "entries": entries}


class LandingTable:
    """One loaded table; `pmf(center)` answers None for an off-grid, out-of-range or too-thin line."""

    def __init__(self, raw: dict):
        self.sport, self.market, self.meta = raw["sport"], raw["market"], raw
        self._entries = raw["entries"]
        self._cache: dict[str, Pmf] = {}

    def pmf(self, center) -> Pmf | None:
        if not on_grid(center):
            return None
        key = grid_key(center)
        if key in self._cache:
            return self._cache[key]
        entry = self._entries.get(key)
        if entry is None:
            return None
        total = float(sum(entry["counts"]))
        pmf = Pmf(n=int(entry["n"]), window=float(entry["window"]),
                  probs={entry["lo"] + i: c / total for i, c in enumerate(entry["counts"]) if c})
        self._cache[key] = pmf
        return pmf


def load_tables(directory: Path | None = None) -> dict[tuple[str, str], LandingTable]:
    """{(sport, market): LandingTable} for every market the monitor tests. A missing asset raises
    FileNotFoundError (the JSON is committed; its absence is a broken checkout, not a quiet day)."""
    return {(s, m): LandingTable(json.loads(table_path(s, m, directory).read_text())) for s, m in LANDING_TABLES}
