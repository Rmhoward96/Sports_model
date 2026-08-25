from __future__ import annotations
from dataclasses import dataclass, field
from ..model.distributions import nb_pmf

def _sigma_defaults():
    return {"pass_yds": 65.0, "reception_yds": 30.0, "rush_yds": 28.0, "rush_reception_yds": 35.0}

def _nb_defaults():
    return {"receptions": 1.6}   # var = mean * mult (overdispersion), tuned in backtest

@dataclass(frozen=True)
class PropConfig:
    sigma: dict = field(default_factory=_sigma_defaults)
    nb_var_mult: dict = field(default_factory=_nb_defaults)

def _normal(mean, sd):
    return {"kind": "normal", "mean": mean, "sd": sd}

def build_prop(market: str, volume: dict, eff: dict, cfg: PropConfig) -> dict:
    rec_mean = volume["targets"] * eff["catch_rate"]
    if market == "pass_yds":
        m = volume["pass_att"] * eff["ypa"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["pass_yds"])}
    if market == "reception_yds":
        m = rec_mean * eff["ypr"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["reception_yds"])}
    if market == "rush_yds":
        m = volume["carries"] * eff["ypc"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["rush_yds"])}
    if market == "rush_reception_yds":
        m = volume["carries"] * eff["ypc"] + rec_mean * eff["ypr"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["rush_reception_yds"])}
    if market == "receptions":
        m = rec_mean
        var = max(m * cfg.nb_var_mult["receptions"], m + 1e-6)  # var > mean for NB
        return {"projected_mean": m, "dist": {"kind": "pmf", "pmf": nb_pmf(m, var)}}
    raise ValueError(f"unknown/yardage-phase market: {market}")
