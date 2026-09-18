"""Per-week active-roster usage model for the NFL sim (NFL-only).

This module bridges nflverse's two player-id namespaces (Pro Football
Reference `pfr_id` and nflverse's own `gsis_id`) and fetches the raw
depth-chart / snap-count / id-crosswalk frames the usage model consumes.

`build_pfr_to_gsis` is pure: DataFrame in, dict out. `fetch_usage_sources`
is the only IO in this module and is not unit-tested.
"""
from __future__ import annotations

import pandas as pd


def _is_missing_id(value: object) -> bool:
    """True if `value` is null or a blank/whitespace-only string."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def build_pfr_to_gsis(ids_df: pd.DataFrame) -> dict[str, str]:
    """Map Pro Football Reference `pfr_id` -> nflverse `gsis_id`.

    Expects `ids_df` (nflverse `import_ids()`) with `pfr_id` and `gsis_id`
    columns. Rows with a null/blank id on either side are skipped. If the
    same `pfr_id` appears more than once, the last row wins (matches
    `import_ids()`'s de-facto one-row-per-player shape; duplicates are not
    expected in practice, but last-wins keeps this deterministic).
    """
    mapping: dict[str, str] = {}
    for row in ids_df.itertuples(index=False):
        pfr_id = getattr(row, "pfr_id", None)
        gsis_id = getattr(row, "gsis_id", None)
        if _is_missing_id(pfr_id) or _is_missing_id(gsis_id):
            continue
        mapping[str(pfr_id).strip()] = str(gsis_id).strip()
    return mapping


def fetch_usage_sources(seasons: list[int]) -> dict:
    """Thin IO wrapper around nfl_data_py imports. Not unit-tested.

    Returns {"depth": DataFrame, "snaps": DataFrame, "ids": DataFrame}.
    """
    import nfl_data_py as nfl

    return {
        "depth": nfl.import_depth_charts(seasons),
        "snaps": nfl.import_snap_counts(seasons),
        "ids": nfl.import_ids(),
    }
