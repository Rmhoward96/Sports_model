"""Resilient nflverse (nfl_data_py) season fetches.

The current NFL season's data files are published incrementally; early in a
season (or before nflverse loads a just-started year) a request for that season
404s. nfl_data_py's own error handling has a bug that surfaces the 404 as a
``NameError: name 'Error' is not defined`` rather than a clean HTTPError, and a
multi-season ``import_*`` call fails atomically -- so one unavailable season
takes the whole fetch down. ``import_by_season`` fetches season-by-season and
skips the seasons that aren't available yet, keeping the rest. The sim's leakage
logic already tolerates a missing current season (``_determine_upto_week`` falls
back to week 1), so degrading to "latest available seasons" is correct.

``load_release`` goes a step further and reads nflverse's release parquets by
URL directly, bypassing nfl_data_py entirely. We need this because the pinned
nfl_data_py 0.3.2 (a) points ``import_weekly_data`` at the RETIRED
``player_stats/player_stats_{year}`` release -- nflverse moved weekly player
stats to ``stats_player/stats_player_week_{year}`` -- so 2025+ weekly 404s, and
(b) 404s on ``import_pbp_data``'s participation sidecar and masks it as the
NameError above. Reading the canonical release URLs ourselves fixes both and
lets the current season's pbp/weekly load. Verified 2026-09-22: pbp,
stats_player_week, and snap_counts all resolve for 2026.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

# Canonical nflverse-data release URLs (per season). nfl_data_py's own paths for
# these are stale (weekly) or buggy (pbp), so we read them directly.
_RELEASE_URL = {
    "pbp": "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet",
    "weekly": "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet",
    "snaps": "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{year}.parquet",
}


def _read_release_season(dataset: str, year: int) -> pd.DataFrame:
    """Read one season's release parquet and normalize it to the column names the
    rest of the code expects. The new weekly release names the team column
    ``team``; downstream (``player_inputs_from_weekly``) expects ``recent_team``."""
    df = pd.read_parquet(_RELEASE_URL[dataset].format(year=year))
    if dataset == "weekly" and "recent_team" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "recent_team"})
    return df


def load_release(dataset: str, seasons: list[int], *, required: bool = True) -> pd.DataFrame:
    """Season-by-season read of an nflverse release (``pbp``/``weekly``/``snaps``),
    skipping seasons whose file isn't published yet, concatenating the rest.

    Same contract as ``import_by_season`` (resilient, ``required`` gating) but
    reads the canonical release URLs directly rather than via nfl_data_py.
    """
    frames: list[pd.DataFrame] = []
    got: list[int] = []
    for yr in seasons:
        try:
            frames.append(_read_release_season(dataset, yr))
            got.append(yr)
        except Exception as exc:  # noqa: BLE001 -- a not-yet-published season 404s; skip it
            print(f"nflverse {dataset}: season {yr} unavailable ({type(exc).__name__}); skipping")
    if not frames:
        if required:
            raise RuntimeError(f"nflverse {dataset}: no seasons available from {seasons}")
        print(f"nflverse {dataset}: no seasons available from {seasons}; continuing empty")
        return pd.DataFrame()
    if got != seasons:
        print(f"nflverse {dataset}: using seasons {got}")
    return pd.concat(frames, ignore_index=True)


def import_by_season(
    import_fn: Callable[[list[int]], pd.DataFrame],
    seasons: list[int],
    label: str,
    *,
    required: bool = True,
) -> pd.DataFrame:
    """Call an nfl_data_py season-list importer once per season, skipping any
    season whose data isn't published yet, and concatenate what's available.

    Args:
        import_fn: e.g. ``nfl_data_py.import_pbp_data`` -- takes a list of years.
        seasons: seasons to try, oldest-to-newest.
        label: short name for log lines (e.g. "pbp").
        required: if True, raise when NO season is available; if False, return
            an empty DataFrame instead (used where "no data" is a valid state,
            e.g. the current-week injury report before it's published).

    Returns:
        Concatenated DataFrame of the available seasons (possibly empty when
        ``required`` is False and none are available).
    """
    frames: list[pd.DataFrame] = []
    got: list[int] = []
    for yr in seasons:
        try:
            frames.append(import_fn([yr]))
            got.append(yr)
        except Exception as exc:  # noqa: BLE001 -- newest season may 404 pre-publish (nfl_data_py masks it as NameError)
            print(f"nflverse {label}: season {yr} unavailable ({type(exc).__name__}); skipping")
    if not frames:
        if required:
            raise RuntimeError(f"nflverse {label}: no seasons available from {seasons}")
        print(f"nflverse {label}: no seasons available from {seasons}; continuing empty")
        return pd.DataFrame()
    if got != seasons:
        print(f"nflverse {label}: using seasons {got}")
    return pd.concat(frames, ignore_index=True)
