"""Game-line orchestrator: model margin/total -> market-shrunk serving row.

Combines a model margin (e.g. P1 Elo x SoS `ratings.expected_margin`) and a
model total (e.g. `points.expected_total`) with the market line via
`shrink.shrink`, wraps both in Normal-derived distributions via
`model.distributions`, and emits the serving-shaped fields P4's board/grader
consume: `margin_dist`, `total_dist`, `home_win_prob`, and decomposed scores.

Kept decoupled from the rating internals (plain float margin/total in) so it
stays unit-testable with simple numbers.

NOTE: `market["spread_line"]`/`market["total_line"]` must be `float` or
`None` -- never NaN. `shrink()` treats `None` (not NaN) as "no market line"
and falls back to the model value; the caller (Task 6) is responsible for
sanitizing NaN -> None at its boundary before calling `build_gameline`. No
NaN handling is performed here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .shrink import ShrinkParams, shrink
from ..model.distributions import normal_to_pmf, normal_to_margin_pmf, prob_cover


@dataclass(frozen=True)
class GameLineConfig:
    sigma_margin: float = 13.2
    sigma_total: float = 10.0
    offset: int = 75
    total_max: int = 120
    w_margin: ShrinkParams = field(default_factory=ShrinkParams)
    w_total: ShrinkParams = field(default_factory=ShrinkParams)
    # Systematic-bias corrections (points), subtracted from the model estimate
    # BEFORE shrink. Sign: residual = pred - actual, so a positive bias means the
    # model runs high and is subtracted. 0.0 = no correction (backward compatible;
    # loaders default to 0.0 when the field is absent from gameline.json). Fit by
    # scripts/backtest_{nfl,cfb}_gameline.py; see docs/superpowers/specs/2026-09-17.
    bias_margin: float = 0.0
    bias_total: float = 0.0


def build_gameline(model_margin: float, model_total: float, market: dict,
                    week: int, cfg: GameLineConfig) -> dict:
    margin = shrink(model_margin - cfg.bias_margin, market.get("spread_line"), week, cfg.w_margin)
    total = shrink(model_total - cfg.bias_total, market.get("total_line"), week, cfg.w_total)
    margin_dist = normal_to_margin_pmf(margin, cfg.sigma_margin, cfg.offset)
    total_dist = {"kind": "pmf", "pmf": normal_to_pmf(total, cfg.sigma_total, cfg.total_max)}
    win_prob = prob_cover(margin_dist, 0.0)   # P(margin > 0)
    return {
        "margin_dist": margin_dist,
        "total_dist": total_dist,
        "home_win_prob": win_prob,
        "pred_margin": margin,
        "pred_total": total,
        "pred_home_score": (total + margin) / 2.0,
        "pred_away_score": (total - margin) / 2.0,
    }
