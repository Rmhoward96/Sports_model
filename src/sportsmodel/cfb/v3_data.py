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
