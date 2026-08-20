"""Load the advancement transition table into fast lookup structures, expose the
sampling interface the kernel uses, and hold the times-through-order multipliers."""
from __future__ import annotations

from dataclasses import dataclass

# kernel int encoding for the in-play outcomes that use the table
_OUTCOME_NAME = {2: "p_1b", 3: "p_2b", 4: "p_3b", 6: "p_out"}

# League-average times-through-order penalty on a STARTER's per-PA vector, indexed
# [1st time, 2nd time, 3rd+ time] through the order. Values are outcome-rate
# multipliers (renormalized after applying). ~ +0.008 wOBA per time through. [tunable]
TTO_MULT: dict[str, tuple[float, float, float]] = {
    "p_bb": (1.00, 1.04, 1.08),
    "p_k":  (1.00, 0.95, 0.90),
    "p_1b": (1.00, 1.03, 1.06),
    "p_2b": (1.00, 1.04, 1.08),
    "p_3b": (1.00, 1.04, 1.08),
    "p_hr": (1.00, 1.06, 1.12),
    "p_out": (1.00, 0.99, 0.98),
}


@dataclass
class AdvancementTable:
    # keyed (outcome_name, occ) -> (cum_probs, end_occ[], runs[])
    _table: dict

    @classmethod
    def from_rows(cls, rows) -> "AdvancementTable":
        grouped: dict = {}
        for r in rows:
            grouped.setdefault((r["outcome"], int(r["occ"])), []).append(
                (float(r["prob"]), int(r["end_occ"]), int(r["runs"])))
        table = {}
        for key, entries in grouped.items():
            entries.sort(key=lambda e: (-e[0]))  # stable; order within group irrelevant
            cum, ends, runs, acc = [], [], [], 0.0
            for p, e, rn in entries:
                acc += p
                cum.append(acc)
                ends.append(e)
                runs.append(rn)
            cum[-1] = 1.0  # guard fp drift
            table[key] = (cum, ends, runs)
        return cls(table)

    def sample(self, outcome_code: int, occ: int, u: float) -> tuple[int, int]:
        key = (_OUTCOME_NAME[outcome_code], occ)
        entry = self._table.get(key)
        if entry is None:
            # unseen state -> conservative fallback: batter to first if empty-ish, no runs
            return (occ | 1, 0) if outcome_code == 2 else (occ, 0)
        cum, ends, runs = entry
        for i, c in enumerate(cum):
            if u < c:
                return ends[i], runs[i]
        return ends[-1], runs[-1]
