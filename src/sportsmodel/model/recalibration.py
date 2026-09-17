"""Systematic-bias fitting + evaluation for the game-line model.

Generation is model-only (no market blend), so a systematic model bias flows
straight into predictions. This fits a *shrunk, clamped* bias correction from
walk-forward residuals and evaluates its held-out effect. PURE -- no IO.

Convention: residual = pred - actual. A positive bias means the model runs high,
so `build_gameline` SUBTRACTS the fitted bias (see gameline.build_gameline).

See docs/superpowers/specs/2026-09-17-model-recalibration-design.md.
"""
from __future__ import annotations

DEFAULT_SHRINK = 0.5
DEFAULT_CAP = 3.0


def fit_bias(residuals: list[float], shrink_factor: float = DEFAULT_SHRINK,
             cap: float = DEFAULT_CAP) -> float:
    """Shrunk, clamped mean residual. Empty -> 0.0.

    `shrink_factor` takes only the portion of the raw mean bias expected to
    generalize; `cap` bounds a bad fit. Both guard against overfitting a small
    or noisy sample (e.g. early-season)."""
    if not residuals:
        return 0.0
    raw = sum(residuals) / len(residuals)
    return max(-cap, min(cap, shrink_factor * raw))


def _mae(preds: list[dict], pred_key: str, actual_key: str, bias: float) -> float:
    n = len(preds)
    if n == 0:
        return 0.0
    return sum(abs((p[pred_key] - bias) - p[actual_key]) for p in preds) / n


def bias_eval(preds: list[dict], bias_margin: float, bias_total: float) -> dict:
    """Held-out metrics for a candidate (bias_margin, bias_total) on model-only
    predictions. `preds` items carry pred_margin/pred_total/actual_margin/
    actual_total. Corrected pred = pred - bias.

    Returns margin_mae, total_mae, cover_acc (sign of corrected margin vs actual),
    and ou_acc = P(actual_total > corrected_total) -- ~0.5 means the total is
    unbiased (the metric a total-bias correction should push toward 0.5)."""
    n = len(preds)
    if n == 0:
        return {"margin_mae": 0.0, "total_mae": 0.0, "cover_acc": 0.0, "ou_acc": 0.0, "n": 0}
    cover = sum(int(((p["pred_margin"] - bias_margin) > 0) == (p["actual_margin"] > 0))
                for p in preds) / n
    ou = sum(int(p["actual_total"] > (p["pred_total"] - bias_total)) for p in preds) / n
    return {
        "margin_mae": _mae(preds, "pred_margin", "actual_margin", bias_margin),
        "total_mae": _mae(preds, "pred_total", "actual_total", bias_total),
        "cover_acc": cover,
        "ou_acc": ou,
        "n": n,
    }
