"""cfb-ratings-v3.1: the v3 pipeline refit every offseason (spec addendum A). PURE (no network).

For a test season S the WHOLE v3 fit -- efficiency hyperparameters (eff_fit, same grid/objective),
points map, margin/total blends with the context lasso and fold-SE one-SE rule (folds = the
training seasons), sigmas -- sees seasons `first` .. S-1 only; it then predicts season S. Nothing
about the method differs from scripts/fit_cfb_eff.py + scripts/fit_cfb_v3.py except the training
window. The schedule frame, efficiency games and prior rows are cut at season S before anything
runs, so a later season cannot reach the fit even by accident; season S's own rows are present
(the table needs them to emit season S's features) but the fit filters them out by season.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from . import eff_fit, v3, v3_fit, v3_table
from .efficiency import EffConfig

FIRST_TRAIN = 2016          # 2015 is warm-up only (no previous-season prior)
TEST_SEASONS = (2023, 2024, 2025)


def train_seasons_for(season: int, first: int = FIRST_TRAIN) -> tuple[int, ...]:
    """The seasons a model predicting `season` may be fitted on: first .. season-1."""
    return tuple(range(first, season))


@dataclass(frozen=True)
class SeasonFit:
    season: int
    eff_cfg: EffConfig
    weights: v3.V3Weights
    gameline: dict
    table: pd.DataFrame        # the feature-table rows of season `season` (built with eff_cfg)


def fit_season(season: int, frame: pd.DataFrame, inp: v3_table.V3Inputs, *, first: int = FIRST_TRAIN,
               grid: dict = eff_fit.GRID, walk_from: int = 2015) -> SeasonFit:
    """Refit the full v3 pipeline on train_seasons_for(season) and build season `season`'s rows.

    `frame` is the merged schedule, `inp` the V3Inputs whose eff_cfg is REPLACED by the refit one."""
    train = train_seasons_for(season, first)
    frame = frame[frame["season"] <= season]
    eff_games = inp.eff_games[inp.eff_games["season"] <= season]
    priors = {s: r for s, r in inp.priors_rows.items() if s <= season}
    cfg, _ = eff_fit.fit_eff_config(eff_games, priors, inp.talent, train_seasons=train, grid=grid)
    inp_s = replace(inp, eff_cfg=cfg, eff_games=eff_games, priors_rows=priors)
    table = v3_table.build_table(frame, inp_s, seasons=range(walk_from, season + 1))
    w = v3_fit.fit_v3(table, train_seasons=train)
    gl = v3.gameline_v3_dict(w.meta["sigma_margin"], w.meta["sigma_total"])
    return SeasonFit(season, cfg, w, gl, table[table["season"] == season].reset_index(drop=True))


def write_season_outputs(fit: SeasonFit, out_dir) -> Path:
    """<out_dir>/<season>/{eff_config,v3_weights,gameline_v3}.json (same formats as the v3 files)."""
    d = Path(out_dir) / str(fit.season)
    d.mkdir(parents=True, exist_ok=True)
    (d / "eff_config.json").write_text(json.dumps({k: float(v) for k, v in asdict(fit.eff_cfg).items()}, indent=2) + "\n")
    (d / "v3_weights.json").write_text(fit.weights.to_json())
    (d / "gameline_v3.json").write_text(json.dumps(fit.gameline, indent=2) + "\n")
    return d
