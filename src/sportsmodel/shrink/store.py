"""The committed k parameters (assets/shrink_k.json) and how the board reads them. The fail-safe lives here.

If the file is missing or unreadable, or the (sport, market) entry is missing or not a number in [0, 1], k is 0: the board
is price-only. It NEVER falls back to the raw model. MLB is always 0 in Part A, whatever the file says.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from .. import config

SHRINK_K_PATH = config.PROJECT_ROOT / "assets" / "shrink_k.json"
MLB_K = 0.0                                  # Part A: no unbiased MLB fit sample, so MLB is price-only


def _valid_k(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0)


class ShrinkTable:
    """k by (sport, odds market). `k()` never raises and never returns anything outside [0, 1]."""

    def __init__(self, ks: Mapping[tuple[str, str], float] | None = None, *, problem: str | None = None):
        self._ks = dict(ks or {})
        self.problem = problem               # why the whole file was unusable, or None
        self.missed: set[tuple[str, str]] = set()

    @classmethod
    def from_dict(cls, doc) -> "ShrinkTable":
        """Build from the parsed JSON document; anything malformed becomes a table that answers 0."""
        if not isinstance(doc, dict) or not isinstance(doc.get("markets"), dict):
            return cls(problem="has no 'markets' object")
        ks, bad = {}, []
        for sport, markets in doc["markets"].items():
            if not isinstance(markets, dict):
                bad.append(str(sport))
                continue
            for market, entry in markets.items():
                k = entry.get("k") if isinstance(entry, dict) else None
                if _valid_k(k):
                    ks[(sport, market)] = float(k)
                else:
                    bad.append(f"{sport}/{market}")
        return cls(ks, problem=("has malformed entries (ignored): " + ", ".join(sorted(bad))) if bad else None)

    def k(self, sport: str, market: str) -> float:
        if sport == "mlb":
            return MLB_K
        key = (sport, market)
        if key in self._ks:
            return self._ks[key]
        self.missed.add(key)
        return 0.0

    def warning(self) -> str | None:
        """One line for the run log if anything fell back to k = 0, else None."""
        parts = []
        if self.problem:
            parts.append(self.problem)
        if self.missed:
            parts.append("no k for " + ", ".join(f"{s}/{m}" for s, m in sorted(self.missed)))
        return ("::warning::shrink: " + "; ".join(parts) + "; using k=0 (price-only) there") if parts else None


def load_table(path: Path | str | None = None) -> ShrinkTable:
    """Read assets/shrink_k.json. A missing, unreadable or malformed file gives a table that answers k = 0 everywhere."""
    path = Path(path) if path is not None else SHRINK_K_PATH
    try:
        text = path.read_text()
    except FileNotFoundError:
        return ShrinkTable(problem=f"{path.name} does not exist yet (first fit has not run)")
    except (OSError, UnicodeDecodeError) as exc:
        return ShrinkTable(problem=f"{path.name} is unreadable ({type(exc).__name__})")
    try:
        doc = json.loads(text)
    except ValueError:
        return ShrinkTable(problem=f"{path.name} is not valid JSON")
    table = ShrinkTable.from_dict(doc)
    if table.problem:
        table.problem = f"{path.name} {table.problem}"
    return table
