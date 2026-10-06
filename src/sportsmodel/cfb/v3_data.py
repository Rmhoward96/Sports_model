"""Committed-asset loaders shared by the cfb-ratings-v3 fit, gate and serving code.

All assets live under assets/cfb/. The three the v3 walk cannot run without are
schedules.parquet, advanced_games.parquet and cfbd_games.parquet; everything else
(havoc, drives, weather, talent, venues, priors, prior ratings) is optional and a
missing file simply means that feature is NaN / neutral.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .efficiency import build_eff_games

ASSETS = config.PROJECT_ROOT / "assets" / "cfb"
BACKFILL_HINT = ("run scripts/build_cfb_advanced.py and scripts/build_cfb_game_data.py with "
                 "CFBD_API_KEY (plan Task 3) to create it")


def read_asset(name: str, assets: Path = ASSETS) -> pd.DataFrame | None:
    """assets/cfb/<name> as a DataFrame, or None when the file is absent."""
    p = Path(assets) / name
    return pd.read_parquet(p) if p.exists() else None


def require_asset(name: str, assets: Path = ASSETS) -> pd.DataFrame:
    df = read_asset(name, assets)
    if df is None:
        raise FileNotFoundError(f"{Path(assets) / name} is missing: {BACKFILL_HINT}")
    return df


def load_merged_schedule(assets: Path = ASSETS) -> pd.DataFrame:
    """REG schedules left-joined to closing/opening lines on (season, week, home, away), with the
    CFBD self-match rows (home == away) dropped from both sides first -- identical to
    backtest_cfb_gameline.load_merged_schedule, so the v2 walk-forward numbers are unchanged."""
    sched = require_asset("schedules.parquet", assets)
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    reg = reg[reg["home_team"] != reg["away_team"]].copy()
    lines = require_asset("lines.parquet", assets)
    lines = lines[lines["home_team"] != lines["away_team"]].drop_duplicates(
        subset=["season", "week", "home_team", "away_team"], keep="first")
    return reg.merge(lines, on=["season", "week", "home_team", "away_team"], how="left",
                     validate="one_to_one")


def load_priors_rows(assets: Path = ASSETS) -> dict[int, list[dict]]:
    """priors.parquet -> {season: [row dicts]} (empty dict when the asset is absent)."""
    df = read_asset("priors.parquet", assets)
    if df is None:
        return {}
    return {int(s): sdf.to_dict("records") for s, sdf in df.groupby("season")}


def load_eff_games(assets: Path = ASSETS) -> pd.DataFrame:
    """The efficiency frame (efficiency.build_eff_games) from the committed assets."""
    return build_eff_games(require_asset("advanced_games.parquet", assets),
                           read_asset("havoc_games.parquet", assets),
                           read_asset("drive_games.parquet", assets),
                           require_asset("cfbd_games.parquet", assets),
                           require_asset("schedules.parquet", assets))
