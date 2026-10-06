"""cfb-ratings-v3: the points map, the blend and the serialisable weights. PURE.

    side points   = pm.intercept + sum_f pm.coefs[f] * side_feature[f]     (efficiency -> points, per offense)
    margin_eff    = points(home offense) - points(away offense)
    total_eff     = points(home offense) + points(away offense)
    margin_v3     = c + a1*margin_v2 + a2*margin_eff + a3*prior_margin + h*non_neutral + sum ctx_m * x
    total_v3      = c + b1*total_eff + b2*total_v2 + sum ctx_t * x

`margin_v2` / `total_v2` are the live v2 model's own pre-bias margin / total (Elo + SRS, prior-
blended; walkforward.model_margin_total), `prior_margin` the decaying preseason-prior margin
(Elo scale, /25). Any NaN feature counts as 0.0 so a missing input never poisons a prediction.
Weights live in assets/cfb/v3_weights.json; the market is never an input.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.shrink import ShrinkParams

from .context import MARGIN_CTX, TOTAL_CTX
from .efficiency import POINT_FEATURES

MARGIN_CORE = ("margin_v2", "margin_eff", "prior_margin", "non_neutral")
TOTAL_CORE = ("total_eff", "total_v2")
GAMELINE_OFFSET, GAMELINE_TOTAL_MAX = 110, 150      # same CFB constants as gameline.json
VERSION = "cfb-ratings-v3"


def _f(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass(frozen=True)
class LinearBlend:
    intercept: float
    coefs: dict = field(default_factory=dict)

    def predict(self, feats) -> float:
        return self.intercept + sum(c * _f(feats.get(k)) for k, c in self.coefs.items())

    def to_dict(self) -> dict:
        return {"intercept": float(self.intercept), "coefs": {k: float(v) for k, v in self.coefs.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearBlend":
        return cls(float(d["intercept"]), {k: float(v) for k, v in d["coefs"].items()})


@dataclass(frozen=True)
class V3Weights:
    points_map: LinearBlend
    margin: LinearBlend
    total: LinearBlend
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"version": VERSION, "points_map": self.points_map.to_dict(),
                           "margin": self.margin.to_dict(), "total": self.total.to_dict(),
                           "meta": self.meta}, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "V3Weights":
        d = json.loads(text)
        return cls(LinearBlend.from_dict(d["points_map"]), LinearBlend.from_dict(d["margin"]),
                   LinearBlend.from_dict(d["total"]), d.get("meta", {}))


def load_v3_weights(path) -> V3Weights:
    """V3Weights from assets/cfb/v3_weights.json. STRICT: a missing file raises (v3 must never be
    served on guessed weights); the v2 path does not call this."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run scripts/fit_cfb_v3.py (plan Task 8) before serving v3")
    d = json.loads(p.read_text())
    if d.get("version") != VERSION:
        raise ValueError(f"{p}: version {d.get('version')!r} != {VERSION!r}; refit with scripts/fit_cfb_v3.py")
    for section, allowed in (("points_map", POINT_FEATURES), ("margin", ALL_MARGIN_FEATURES),
                             ("total", ALL_TOTAL_FEATURES)):
        unknown = sorted(set(d[section]["coefs"]) - set(allowed))
        if unknown:                                   # a typo must not silently drop a feature
            raise ValueError(f"{p}: unknown {section} coefficient(s) {unknown}; allowed: {sorted(allowed)}")
    return V3Weights.from_json(p.read_text())


def eff_margin_total(pm: LinearBlend, row) -> tuple[float, float]:
    """(margin_eff, total_eff) from a table row's h_<feature> / a_<feature> side features."""
    home = pm.predict({f: row.get(f"h_{f}") for f in POINT_FEATURES})
    away = pm.predict({f: row.get(f"a_{f}") for f in POINT_FEATURES})
    return home - away, home + away


def predict_v3(row, w: V3Weights) -> tuple[float, float]:
    """The v3 (margin, total) for one feature-table row (a dict / Series). No bias is applied:
    the fitted intercepts already centre the residuals."""
    margin_eff, total_eff = eff_margin_total(w.points_map, row)
    feats = {**row, "margin_eff": margin_eff, "total_eff": total_eff}
    return w.margin.predict(feats), w.total.predict(feats)


def gameline_v3_dict(sigma_margin: float, sigma_total: float) -> dict:
    """assets/cfb/gameline_v3.json: same keys as gameline.json (so generate_cfb.load_gameline reads
    it); sigmas are the v3 train RMSEs, the market-shrink curves are zero (model-only) and the
    bias terms are 0 (the blend intercepts already centre v3)."""
    zero = {"start": 0.0, "floor": 0.0, "decay": 0.0}
    return {"sigma_margin": float(sigma_margin), "sigma_total": float(sigma_total),
            "offset": GAMELINE_OFFSET, "total_max": GAMELINE_TOTAL_MAX,
            "w_margin": dict(zero), "w_total": dict(zero), "bias_margin": 0.0, "bias_total": 0.0}


ALL_MARGIN_FEATURES = MARGIN_CORE + MARGIN_CTX
ALL_TOTAL_FEATURES = TOTAL_CORE + TOTAL_CTX


def gameline_from_dict(j: dict) -> GameLineConfig:
    """GameLineConfig from a gameline.json-shaped dict (see gameline_v3_dict)."""
    return GameLineConfig(sigma_margin=j["sigma_margin"], sigma_total=j["sigma_total"], offset=j["offset"],
                          total_max=j["total_max"], w_margin=ShrinkParams(**j["w_margin"]),
                          w_total=ShrinkParams(**j["w_total"]), bias_margin=j.get("bias_margin", 0.0),
                          bias_total=j.get("bias_total", 0.0))


def load_gameline_config(path) -> GameLineConfig:
    """GameLineConfig from a gameline.json-shaped file (gameline.json for v2, gameline_v3.json for v3)."""
    return gameline_from_dict(json.loads(Path(path).read_text()))
