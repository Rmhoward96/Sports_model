from __future__ import annotations
from dataclasses import dataclass, field
from ..model.distributions import nb_pmf, poisson_pmf

def _sigma_defaults():
    return {"pass_yds": 65.0, "reception_yds": 30.0, "rush_yds": 28.0, "rush_reception_yds": 35.0}

def _nb_defaults():
    return {"receptions": 1.6}   # var = mean * mult (overdispersion), tuned in backtest

def _mean_mult_defaults():
    # Population-level mean de-bias multiplier per market, fit in the backtest as
    # mean(actual)/mean(pred_mean). Defaults to 1.0 (no correction) -- the
    # walk-forward that FITS this multiplier must itself run with mean_mult=1.0
    # (i.e. these defaults), or the fit would be circular. See
    # scripts/backtest_nfl_props.py fit_calibration.
    return {"pass_yds": 1.0, "reception_yds": 1.0, "rush_yds": 1.0,
            "rush_reception_yds": 1.0, "receptions": 1.0}

@dataclass(frozen=True)
class PropConfig:
    sigma: dict = field(default_factory=_sigma_defaults)
    nb_var_mult: dict = field(default_factory=_nb_defaults)
    mean_mult: dict = field(default_factory=_mean_mult_defaults)

def _normal(mean, sd):
    return {"kind": "normal", "mean": mean, "sd": sd}

def build_prop(market: str, volume: dict, eff: dict, cfg: PropConfig) -> dict:
    rec_mean = volume["targets"] * eff["catch_rate"]
    if market == "pass_yds":
        m = volume["pass_att"] * eff["ypa"] * cfg.mean_mult["pass_yds"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["pass_yds"])}
    if market == "reception_yds":
        m = rec_mean * eff["ypr"] * cfg.mean_mult["reception_yds"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["reception_yds"])}
    if market == "rush_yds":
        m = volume["carries"] * eff["ypc"] * cfg.mean_mult["rush_yds"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["rush_yds"])}
    if market == "rush_reception_yds":
        m = (volume["carries"] * eff["ypc"] + rec_mean * eff["ypr"]) * cfg.mean_mult["rush_reception_yds"]
        return {"projected_mean": m, "dist": _normal(m, cfg.sigma["rush_reception_yds"])}
    if market == "receptions":
        m = rec_mean * cfg.mean_mult["receptions"]
        var = max(m * cfg.nb_var_mult["receptions"], m + 1e-6)  # var > mean for NB
        return {"projected_mean": m, "dist": {"kind": "pmf", "pmf": nb_pmf(m, var)}}
    if market == "pass_tds":
        m = volume["pass_att"] * eff["pass_td_rate"]
        return {"projected_mean": m, "dist": {"kind": "pmf", "pmf": poisson_pmf(m)}}
    if market == "anytime_td":
        lam = volume["carries"] * eff["rush_td_rate"] + volume["targets"] * eff["rec_td_rate"]
        return {"projected_mean": lam, "dist": {"kind": "pmf", "pmf": poisson_pmf(lam)}}
    raise ValueError(f"unknown market: {market}")
