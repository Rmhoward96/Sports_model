"""Game-level ship gate: ML sim vs legacy Elo for NFL moneyline/spread/total (pure, no IO).

Probability conventions (plan rulings):
- home win prob = P(margin > 0) + 0.5 * P(margin == 0)
- cover prob vs closing spread L = P(margin > L), push mass (margin == L, only
  possible for integer L) excluded and renormalized. nflverse ``spread_line`` is
  POSITIVE when the HOME team is favored: home covers iff margin > spread_line.
- over prob vs closing total T = P(total > T), push excluded and renormalized.
- A game without a closing line is excluded from that market's Brier only.

PMF convention: pmfs are over consecutive integers and ``offset`` is the margin
VALUE at index 0, i.e. ``pmf[i] = P(margin == offset + i)``. Note this is the
negation of ``sim.engine.margin_pmf``'s stored ``"offset"`` (which is
``half_range`` with ``pmf[i] = P(margin == i - offset)``), so for engine output
call ``win_prob_pmf(m["pmf"], -m["offset"])``. Total pmfs (``engine.total_pmf``)
start at total 0. Engine pmfs clip their tails onto the end bins; that is a
property of the input, not handled here.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from . import props_eval

METRICS = ("win_brier", "cover_brier", "over_brier", "margin_mae", "total_mae")
_BRIER_METRICS = ("win_brier", "cover_brier", "over_brier")
MAE_TOLERANCE = 1.01

# metric -> per-game error column stem in paired_diffs
_COLUMN = {
    "win_brier": "win_sq",
    "cover_brier": "cover_sq",
    "over_brier": "over_sq",
    "margin_mae": "margin_ae",
    "total_mae": "total_ae",
}
_KEY = ("season", "week", "home")


def _missing(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------- pmf probabilities

def _prob_above_line(pmf, first_value: int, line) -> float | None:
    """P(X > line) with push mass (X == line) excluded and renormalized."""
    if _missing(line):
        return None
    p = np.asarray(pmf, dtype=float)
    values = first_value + np.arange(len(p))
    line = float(line)
    above = float(p[values > line].sum())
    push = float(p[values == line].sum())
    denom = float(p.sum()) - push
    if denom <= 0.0:
        return None
    return above / denom


def win_prob_pmf(margin_pmf, offset: int) -> float:
    """Home win prob from a margin pmf: P(margin > 0) + 0.5 * P(margin == 0)."""
    p = np.asarray(margin_pmf, dtype=float)
    values = offset + np.arange(len(p))
    total = float(p.sum())
    return (float(p[values > 0].sum()) + 0.5 * float(p[values == 0].sum())) / total


def cover_prob_pmf(margin_pmf, offset: int, line) -> float | None:
    """Home cover prob vs closing spread ``line`` (positive = home favored)."""
    return _prob_above_line(margin_pmf, offset, line)


def over_prob_pmf(total_pmf, line) -> float | None:
    """Over prob vs closing total ``line``; index 0 of the pmf is total 0."""
    return _prob_above_line(total_pmf, 0, line)


# ---------------------------------------------------------------- normal probabilities

def _normal_above(mean: float, sigma: float, line) -> float | None:
    if _missing(line):
        return None
    if not sigma > 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    z = (float(line) - float(mean)) / (float(sigma) * math.sqrt(2.0))
    return 0.5 * math.erfc(z)


def cover_prob_normal(mean: float, sigma: float, line) -> float | None:
    """P(margin > line) for margin ~ N(mean, sigma); None for a missing line."""
    return _normal_above(mean, sigma, line)


def over_prob_normal(mean: float, sigma: float, line) -> float | None:
    """P(total > line) for total ~ N(mean, sigma); None for a missing line."""
    return _normal_above(mean, sigma, line)


# ---------------------------------------------------------------- per-game errors

def _sq(prob, outcome) -> float:
    if _missing(prob) or outcome is None:
        return float("nan")
    return (float(prob) - outcome) ** 2


def _line_outcome(actual, line) -> float | None:
    """1 if actual > line, 0 if below, None on push / missing line or actual."""
    if _missing(line) or _missing(actual):
        return None
    a, l = float(actual), float(line)
    if a == l:
        return None
    return 1.0 if a > l else 0.0


def _ae(pred, actual) -> float:
    if _missing(pred) or _missing(actual):
        return float("nan")
    return abs(float(pred) - float(actual))


def _game_errors(r: dict) -> dict:
    """Per-game errors; NaN where a market is not scorable (push, no line)."""
    am = r.get("actual_margin")
    win_outcome = None
    if not _missing(am):
        am = float(am)
        win_outcome = 1.0 if am > 0 else (0.5 if am == 0 else 0.0)
    return {
        "win_sq": _sq(r.get("win_prob"), win_outcome),
        "cover_sq": _sq(r.get("cover_prob"),
                        _line_outcome(r.get("actual_margin"), r.get("spread_line"))),
        "over_sq": _sq(r.get("over_prob"),
                       _line_outcome(r.get("actual_total"), r.get("total_line"))),
        "margin_ae": _ae(r.get("pred_margin"), r.get("actual_margin")),
        "total_ae": _ae(r.get("pred_total"), r.get("actual_total")),
    }


def _nanmean(values) -> tuple[float, int]:
    a = np.asarray(values, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan"), 0
    return float(a.mean()), int(a.size)


def game_metrics(records: Iterable[dict]) -> dict:
    """Win/cover/over Brier and margin/total MAE for one model, with n per metric.

    Win Brier scores ties as outcome 0.5; cover/over Brier drop pushes and games
    without a closing line (or prob); a metric with no scorable games is NaN, n=0.
    """
    errs = [_game_errors(r) for r in records]
    out: dict = {}
    for metric in METRICS:
        col = _COLUMN[metric]
        value, n = _nanmean([e[col] for e in errs])
        out[metric] = value
        out["n_" + metric.split("_")[0]] = n
    return out


# ---------------------------------------------------------------- pairing

def _index(records: Iterable[dict], label: str) -> dict:
    out = {}
    for r in records:
        key = tuple(r[k] for k in _KEY)
        if key in out:
            raise ValueError(f"duplicate {label} record for (season, week, home)={key}")
        out[key] = r
    return out


def paired_diffs(ml_records: Iterable[dict], elo_records: Iterable[dict]) -> pd.DataFrame:
    """Inner-join ML and Elo records on (season, week, home); one row per game.

    Columns: season, week, home, cluster (``f"{season}-{week}"``) and, for each
    error stem in (win_sq, cover_sq, over_sq, margin_ae, total_ae), ``<stem>_ml``
    and ``<stem>_elo`` (NaN where that market is unscorable). Rows sorted by key.
    """
    ml = _index(ml_records, "ML")
    elo = _index(elo_records, "Elo")
    stems = list(_COLUMN.values())
    columns = ["season", "week", "home", "cluster"] + [
        f"{s}_{m}" for s in stems for m in ("ml", "elo")
    ]
    rows = []
    for key in sorted(set(ml) & set(elo)):
        season, week, home = key
        e_ml, e_elo = _game_errors(ml[key]), _game_errors(elo[key])
        row = {"season": season, "week": week, "home": home,
               "cluster": f"{season}-{week}"}
        for s in stems:
            row[f"{s}_ml"] = e_ml[s]
            row[f"{s}_elo"] = e_elo[s]
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------- decision

def _diff_stat(col: str):
    """Mean of (ML - Elo) over games where both models are scorable."""
    def stat(d: pd.DataFrame) -> float:
        diff = (d[f"{col}_ml"] - d[f"{col}_elo"]).to_numpy(dtype=float)
        diff = diff[~np.isnan(diff)]
        return float(diff.mean()) if diff.size else float("nan")
    return stat


def gate_decision(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired ML-vs-Elo gate over a ``paired_diffs`` frame.

    Per metric: ML and Elo means over games where both are scorable, their
    difference ML - Elo, and a 95% bootstrap CI that resamples season-week
    clusters with replacement (``props_eval.cluster_bootstrap``; the same seed per
    metric, so every metric sees the same resampled clusters).

    Pass iff win/cover/over Brier ML <= Elo and margin/total MAE
    ML <= 1.01 * Elo (point estimates). ``reasons`` names each failing metric.
    """
    metrics: dict = {}
    reasons: list[str] = []
    if df.empty:
        return {"metrics": metrics, "pass": False, "reasons": ["no paired games"]}

    for metric in METRICS:
        col = _COLUMN[metric]
        both = df[[f"{col}_ml", f"{col}_elo"]].notna().all(axis=1)
        sub = df[both]
        n = int(len(sub))
        if n == 0:
            metrics[metric] = {"ml": float("nan"), "elo": float("nan"),
                               "diff": float("nan"), "lo": float("nan"),
                               "hi": float("nan"), "n": 0}
            reasons.append(f"{metric}: no paired scorable games")
            continue
        ml_mean = float(sub[f"{col}_ml"].mean())
        elo_mean = float(sub[f"{col}_elo"].mean())
        diff, lo, hi = props_eval.cluster_bootstrap(
            sub, _diff_stat(col), n_boot=n_boot, seed=seed)
        metrics[metric] = {"ml": ml_mean, "elo": elo_mean, "diff": diff,
                           "lo": lo, "hi": hi, "n": n}
        if metric in _BRIER_METRICS:
            if not ml_mean <= elo_mean:
                reasons.append(f"{metric}: ML {ml_mean:.5f} > Elo {elo_mean:.5f}")
        elif not ml_mean <= MAE_TOLERANCE * elo_mean:
            reasons.append(
                f"{metric}: ML {ml_mean:.4f} > {MAE_TOLERANCE} x Elo {elo_mean:.4f}")

    return {"metrics": metrics, "pass": not reasons, "reasons": reasons}
