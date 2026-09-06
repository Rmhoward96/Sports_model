"""Preseason prior assembly (pure) -> R_pre.

Turns one priors.parquet row (Task 2's build_cfb_priors output) into a
preseason rating on the model's Elo scale. Pure: no network, no DB. The one
allowed file I/O is `load_weights`, which reads a small JSON config of fitted
weights (produced by the backtest, Task 5).

Task 4 adds the decaying blend of R_pre into the in-season rating in this
same module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


@dataclass(frozen=True)
class PriorWeights:
    """Coefficients mapping a priors row + season z-scores to R_pre.

    Defaults reduce `preseason_rating` to an identity SP+ map on the model's
    Elo scale (R_pre = 1500 + sp_rating), i.e. all the qualitative
    adjustments are off until weights are fitted (Task 5).
    """

    sp_scale: float = 1.0
    sp_offset: float = 1500.0
    w_portal: float = 0.0
    w_coach: float = 0.0
    w_qb: float = 0.0
    w_starters: float = 0.0
    w_sos_prior: float = 0.0
    w_sos_shift: float = 0.0


def zscore(values: dict) -> dict:
    """Population z-score each value in `values`; std==0 -> all zeros."""
    keys = list(values.keys())
    vals = [values[k] for k in keys]
    if not vals:
        return {}
    m = mean(vals)
    s = pstdev(vals)
    if s == 0:
        return {k: 0.0 for k in keys}
    return {k: (v - m) / s for k, v in zip(keys, vals)}


def load_weights(path) -> PriorWeights:
    """Load fitted `PriorWeights` from a JSON file; identity-SP+ default if missing.

    Unspecified fields in the JSON fall back to `PriorWeights` defaults.
    """
    p = Path(path)
    if not p.exists():
        return PriorWeights()
    data = json.loads(p.read_text())
    return PriorWeights(**data)


_Z_FEATURES = ("portal_net", "returning_starters", "prior_sos", "forward_sos_shift")


def season_features_z(rows: list[dict]) -> dict:
    """Z-score the season's forward-looking features across all FBS teams.

    Returns {team_espn_id: {"portal_net": z, "returning_starters": z,
    "prior_sos": z, "forward_sos_shift": z}}, using `zscore` independently
    per feature across the season's teams (one shared implementation so the
    backtest and the live producer z-score identically).

    A None value for a feature is treated as missing: it is excluded from
    that feature's mean/std, and the team maps to 0.0 for that feature
    (rather than being dropped or raising).
    """
    result = {row["team_espn_id"]: {} for row in rows}
    for feature in _Z_FEATURES:
        present = {
            row["team_espn_id"]: row.get(feature)
            for row in rows
            if row.get(feature) is not None
        }
        z = zscore(present)
        for row in rows:
            team_id = row["team_espn_id"]
            result[team_id][feature] = z.get(team_id, 0.0)
    return result


def preseason_rating(row: dict, z: dict, weights: PriorWeights) -> float:
    """R_pre for one team-season: SP+ base plus weighted qualitative adjustments.

    `row` is a priors.parquet row (needs sp_rating, coach_first_year,
    qb_returning). `z` carries this season's FBS-wide z-scored features for
    this team (portal_net, returning_starters, prior_sos, forward_sos_shift),
    e.g. from `season_features_z`.
    """
    qb_flag_signed = 1.0 if row["qb_returning"] else -1.0
    coach_flag = 1.0 if row["coach_first_year"] else 0.0
    return (
        weights.sp_offset
        + weights.sp_scale * row["sp_rating"]
        + weights.w_portal * z["portal_net"]
        + weights.w_coach * coach_flag
        + weights.w_qb * qb_flag_signed
        + weights.w_starters * z["returning_starters"]
        + weights.w_sos_prior * z["prior_sos"]
        + weights.w_sos_shift * z["forward_sos_shift"]
    )
