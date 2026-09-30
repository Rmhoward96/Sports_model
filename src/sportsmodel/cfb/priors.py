"""Preseason prior assembly (pure) -> R_pre.

LEAK-FREE definition (fix/cfb-live-ratings): for season S,

    R_pre = sp_offset + sp_scale * prev_sp
            + w_returning * z(returning_pct) + w_recruiting * z(recruiting_points)
            + w_portal * z(portal_net) + w_sos_prior * z(prior_sos)
            + w_qb * qb_flag

  - prev_sp: the team's PREVIOUS season's final SP+ (priors.parquet
    sp_rating at season S-1); missing (NaN / team not rated last year) ->
    that season's league mean (0.0 if no S-1 data at all).
  - z(.): season-S FBS-wide population z-scores of preseason features
    (missing -> 0, e.g. portal_net before 2021).
  - qb_flag: +1 returning starting QB, -1 not, 0 unknown.

NOT allowed (post-season information for season S): priors.parquet's
same-season sp_rating (CFBD /ratings/sp?year=S is END-of-season SP+),
coach_first_year and forward_sos_shift. `season_prior_inputs` only ever reads
ALLOWED_COLUMNS from a season-S row, so they cannot leak in. The pre-fix
definition (R_pre = 1500 + same-season sp_rating) was a leak.

Pure: no network, no DB. The one allowed file I/O is `load_weights` /
`load_decay_config`, which read small JSON configs fitted by
scripts/backtest_cfb_priors.py. Also holds the decaying blend of R_pre into
the in-season rating.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .. import config

# season-S columns a leak-free prior may read (see module docstring)
PRIOR_Z_FEATURES = ("returning_pct", "recruiting_points", "portal_net", "prior_sos")
ALLOWED_COLUMNS = ("season", "team_espn_id", "qb_returning") + PRIOR_Z_FEATURES
_WEIGHT_FOR_FEATURE = {"returning_pct": "w_returning", "recruiting_points": "w_recruiting",
                       "portal_net": "w_portal", "prior_sos": "w_sos_prior"}
# weights of the retired, leaky definition: tolerated in a JSON only as 0.0
LEGACY_LEAKY_WEIGHTS = ("w_coach", "w_starters", "w_sos_shift")


@dataclass(frozen=True)
class PriorWeights:
    """Coefficients mapping previous-season SP+ + season z-scores to R_pre.

    Defaults reduce `preseason_rating` to 1500 + prev_sp (identity map, all
    feature adjustments off) until weights are fitted.
    """

    sp_scale: float = 1.0
    sp_offset: float = 1500.0
    w_returning: float = 0.0
    w_recruiting: float = 0.0
    w_portal: float = 0.0
    w_qb: float = 0.0
    w_sos_prior: float = 0.0


def _is_missing(v) -> bool:
    """True for a missing feature value -- None OR NaN. Parquet nulls come back
    as float NaN (numpy float64, a subclass of float), not None, so a plain
    `is None` check would let them through into the stats and corrupt them."""
    return v is None or (isinstance(v, float) and v != v)


def zscore(values: dict) -> dict:
    """Population z-score each value in `values`; std==0 -> all zeros.

    Uses plain arithmetic (not `statistics.mean`/`pstdev`): values arrive as
    numpy float64 from parquet, and the statistics module routes through
    Fraction and raises on any non-Fraction (including a NaN that reduces the
    mean-of-squares to a bare float). Callers should pre-filter missing values
    (see `_is_missing`); coercing to float here keeps the math built-in-typed."""
    keys = list(values.keys())
    vals = [float(values[k]) for k in keys]
    n = len(vals)
    if n == 0:
        return {}
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    if var == 0:
        return {k: 0.0 for k in keys}
    s = var ** 0.5
    return {k: (v - m) / s for k, v in zip(keys, vals)}


def load_weights(path) -> PriorWeights:
    """Load fitted `PriorWeights` from a JSON file; identity default if missing.

    Unspecified fields fall back to `PriorWeights` defaults. Keys of the
    retired leaky definition (LEGACY_LEAKY_WEIGHTS) are ignored when 0.0 and
    rejected (ValueError) otherwise -- a stale non-zero coach/forward-SoS
    weight must never be silently dropped or applied. Unknown keys raise
    TypeError (typo guard).
    """
    p = Path(path)
    if not p.exists():
        return PriorWeights()
    data = json.loads(p.read_text())
    for k in LEGACY_LEAKY_WEIGHTS:
        if k in data:
            if float(data[k]) != 0.0:
                raise ValueError(f"{p}: {k}={data[k]} belongs to the retired leaky prior "
                                 "definition; refit with scripts/backtest_cfb_priors.py")
            del data[k]
    return PriorWeights(**data)


def season_features_z(rows: list[dict]) -> dict:
    """Z-score the season's preseason features (PRIOR_Z_FEATURES) across all
    FBS teams: {team_espn_id: {feature: z}} (one shared implementation so the
    backtest and the live producer z-score identically).

    A missing value (None, or NaN from a parquet null) is excluded from that
    feature's mean/std and maps to 0.0 for that team.
    """
    result = {row["team_espn_id"]: {} for row in rows}
    for feature in PRIOR_Z_FEATURES:
        present = {
            row["team_espn_id"]: row.get(feature)
            for row in rows
            if not _is_missing(row.get(feature))
        }
        z = zscore(present)
        for row in rows:
            team_id = row["team_espn_id"]
            result[team_id][feature] = z.get(team_id, 0.0)
    return result


def _qb_flag(v) -> float:
    """+1 returning QB, -1 departed, 0 unknown (missing data is neutral)."""
    if _is_missing(v):
        return 0.0
    return 1.0 if bool(v) else -1.0


def season_prior_inputs(rows_by_season: dict, season: int) -> dict:
    """{team_espn_id: {"prev_sp", "qb_flag", "z": {feature: z}}} for `season`.

    `rows_by_season` = {season: [priors.parquet row dicts]}. Reads ONLY
    sp_rating from season-1 rows and ALLOWED_COLUMNS from season rows (the
    rows are projected onto ALLOWED_COLUMNS before anything else touches
    them), so same-season SP+ / coach flag / forward SoS cannot leak in.
    """
    cur = [{k: r.get(k) for k in ALLOWED_COLUMNS} for r in rows_by_season.get(season, [])]
    prev = {r["team_espn_id"]: float(r["sp_rating"])
            for r in rows_by_season.get(season - 1, [])
            if not _is_missing(r.get("sp_rating"))}
    league_mean = sum(prev.values()) / len(prev) if prev else 0.0
    z = season_features_z(cur)
    return {r["team_espn_id"]: {"prev_sp": prev.get(r["team_espn_id"], league_mean),
                                "qb_flag": _qb_flag(r["qb_returning"]),
                                "z": z[r["team_espn_id"]]}
            for r in cur}


def feature_adjustment(inp: dict, weights: PriorWeights) -> float:
    """The weighted feature part of R_pre (everything but offset + SP+ base)."""
    return (sum(getattr(weights, _WEIGHT_FOR_FEATURE[f]) * inp["z"][f]
                for f in PRIOR_Z_FEATURES)
            + weights.w_qb * inp["qb_flag"])


def preseason_rating(inp: dict, weights: PriorWeights) -> float:
    """R_pre for one team-season from its `season_prior_inputs` entry."""
    return weights.sp_offset + weights.sp_scale * inp["prev_sp"] + feature_adjustment(inp, weights)


def season_priors(rows_by_season: dict, season: int, weights: PriorWeights) -> dict:
    """{team_espn_id: R_pre} for `season` (see module docstring)."""
    return {t: preseason_rating(inp, weights)
            for t, inp in season_prior_inputs(rows_by_season, season).items()}


@dataclass(frozen=True)
class DecayConfig:
    """Configuration for the decaying prior blend.

    half_life_games: number of games at which the prior weight decays to 0.5
    prior_floor: minimum prior weight (default 0.0)
    """

    half_life_games: float
    prior_floor: float = 0.0


DECAY_PATH = config.PROJECT_ROOT / "assets" / "cfb" / "priors_decay.json"
DEFAULT_HALF_LIFE_GAMES = 4.0  # documented default half-life; matches the
                               # committed default assets/cfb/priors_decay.json


def load_decay_config(path=DECAY_PATH) -> DecayConfig:
    """Load a fitted `DecayConfig` from its sibling JSON file (see
    backtest_cfb_priors.py's design note on why DecayConfig lives in a
    separate file from PriorWeights); missing file -> the documented default
    half-life, mirroring `load_weights`'s missing-file-safe fallback.

    Kept next to `load_weights` so both the backtest (Task 5) and the live
    producer (Task 6) can load fitted config the same way -- `scripts/` is
    not an importable package, so this can't live in the backtest script.
    """
    p = Path(path)
    if not p.exists():
        return DecayConfig(half_life_games=DEFAULT_HALF_LIFE_GAMES)
    data = json.loads(p.read_text())
    return DecayConfig(**data)


def prior_weight(games_played: float, cfg: DecayConfig) -> float:
    """Decay weight of the preseason prior with exponential half-life.

    Returns a weight in [cfg.prior_floor, 1.0]:
    - At 0 games: weight = 1.0 (prior dominates)
    - At cfg.half_life_games: weight = 0.5 (equal blend)
    - As games increase: weight → cfg.prior_floor

    Formula: max(prior_floor, 0.5 ** (games_played / half_life_games))
    """
    return max(cfg.prior_floor, 0.5 ** (games_played / cfg.half_life_games))


def blend_rating(
    r_pre: float, in_season_rating: float, games_played: float, cfg: DecayConfig
) -> float:
    """Blend preseason prior with in-season rating using decay-based weight.

    Blends:
    - r_pre (preseason rating) with weight w
    - in_season_rating with weight (1 - w)
    where w = prior_weight(games_played, cfg)

    Result: w * r_pre + (1 - w) * in_season_rating
    """
    w = prior_weight(games_played, cfg)
    return w * r_pre + (1 - w) * in_season_rating
