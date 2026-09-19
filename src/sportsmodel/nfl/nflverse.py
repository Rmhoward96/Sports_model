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
"""
from __future__ import annotations

from typing import Callable

import pandas as pd


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
