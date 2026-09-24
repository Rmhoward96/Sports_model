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

_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
# Canonical nflverse-data release URLs (per season). nfl_data_py's own paths for
# these are stale (weekly) or buggy (pbp), so we read them directly.
_RELEASE_URL = {
    "pbp": f"{_BASE}/pbp/play_by_play_{{year}}.parquet",
    "weekly": f"{_BASE}/stats_player/stats_player_week_{{year}}.parquet",
    "snaps": f"{_BASE}/snap_counts/snap_counts_{{year}}.parquet",
    "depth": f"{_BASE}/depth_charts/depth_charts_{{year}}.parquet",
}
# One file covering every season; read once, filtered to the requested seasons.
_SINGLE_FILE_URL = {
    "schedules": f"{_BASE}/schedules/games.parquet",
    "ngs_receiving": f"{_BASE}/nextgen_stats/ngs_receiving.parquet",
    "ngs_rushing": f"{_BASE}/nextgen_stats/ngs_rushing.parquet",
    "ngs_passing": f"{_BASE}/nextgen_stats/ngs_passing.parquet",
}
# Columns the props-ML features read. A release that drops one fails loudly
# here instead of silently producing NaN features (lesson of the 2025
# depth-chart schema change).
EXPECTED_COLUMNS: dict[str, frozenset[str]] = {
    "schedules": frozenset({"game_id", "season", "week", "game_type", "gameday", "gametime",
                            "home_team", "away_team", "home_rest", "away_rest", "roof", "surface",
                            "temp", "wind", "div_game", "stadium_id", "spread_line", "total_line"}),
    "ngs_receiving": frozenset({"season", "week", "player_gsis_id", "avg_separation", "avg_cushion",
                                "avg_intended_air_yards", "avg_yac_above_expectation"}),
    "ngs_rushing": frozenset({"season", "week", "player_gsis_id", "efficiency",
                              "percent_attempts_gte_eight_defenders", "rush_yards_over_expected_per_att"}),
    "ngs_passing": frozenset({"season", "week", "player_gsis_id", "avg_time_to_throw",
                              "completion_percentage_above_expectation", "aggressiveness"}),
}


def validate_columns(df: pd.DataFrame, dataset: str) -> None:
    """Raise ValueError if `df` lacks any EXPECTED_COLUMNS[dataset] column."""
    missing = sorted(EXPECTED_COLUMNS.get(dataset, frozenset()) - set(df.columns))
    if missing:
        raise ValueError(f"nflverse {dataset}: missing expected columns {missing}")


# nflverse depth charts changed schema in 2025: seasons <= 2024 are the old
# weekly charts, 2025+ timestamped snapshots. Each season's file must match one.
DEPTH_SCHEMAS: dict[str, frozenset[str]] = {
    "old": frozenset({"season", "club_code", "week", "depth_team", "gsis_id", "position"}),
    "snapshot": frozenset({"dt", "team", "gsis_id", "pos_abb", "pos_rank"}),
}


def validate_depth_season(df: pd.DataFrame, season: int) -> str:
    """Schema of one season's depth-chart release ("old" or "snapshot"); raise
    ValueError if it matches neither, or if a snapshot's non-null `pos_rank`
    values are not positive integers."""
    missing = {name: sorted(cols - set(df.columns)) for name, cols in DEPTH_SCHEMAS.items()}
    schema = next((name for name, m in missing.items() if not m), None)
    if schema is None:
        raise ValueError(f"nflverse depth {season}: matches neither depth-chart schema -- old schema "
                         f"missing {missing['old']}; snapshot schema missing {missing['snapshot']}")
    if schema == "snapshot":
        raw = df["pos_rank"]
        num = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & (num.isna() | (num <= 0) | (num % 1 != 0))
        if bad.any():
            raise ValueError(f"nflverse depth {season}: pos_rank must be positive integers; "
                             f"got {sorted(map(str, raw[bad].unique()))[:10]}")
    return schema


def _read_release_season(dataset: str, year: int) -> pd.DataFrame:
    """Read one season's release parquet and normalize it to the column names the
    rest of the code expects. The new weekly release names the team column
    ``team``; downstream (``player_inputs_from_weekly``) expects ``recent_team``."""
    df = pd.read_parquet(_RELEASE_URL[dataset].format(year=year))
    if dataset == "weekly" and "recent_team" not in df.columns and "team" in df.columns:
        df = df.rename(columns={"team": "recent_team"})
    return df


def load_release(dataset: str, seasons: list[int], *, required: bool = True) -> pd.DataFrame:
    """Season-by-season read of an nflverse release (``pbp``/``weekly``/``snaps``/``depth``),
    or single-file datasets (``schedules`` and ``ngs_*``) read once and filtered to ``seasons``,
    skipping seasons whose file isn't published yet, concatenating the rest.

    Same contract as ``import_by_season`` (resilient, ``required`` gating) but
    reads the canonical release URLs directly rather than via nfl_data_py. Every
    dataset in EXPECTED_COLUMNS is schema-validated; ``depth`` is validated per
    season against either of its two schemas (``validate_depth_season``).
    """
    # Handle single-file datasets (read once, filter to seasons)
    if dataset in _SINGLE_FILE_URL:
        df = pd.read_parquet(_SINGLE_FILE_URL[dataset])
        validate_columns(df, dataset)
        df = df[df["season"].isin(seasons)].reset_index(drop=True)
        if df.empty and required:
            raise RuntimeError(f"nflverse {dataset}: no rows for seasons {seasons}")
        return df

    # Handle per-season datasets
    frames: list[pd.DataFrame] = []
    got: list[int] = []
    for yr in seasons:
        try:
            df = _read_release_season(dataset, yr)
        except Exception as exc:  # noqa: BLE001 -- a not-yet-published season 404s; skip it
            print(f"nflverse {dataset}: season {yr} unavailable ({type(exc).__name__}); skipping")
            continue
        if dataset == "depth":
            # per season, outside the try: schema drift must fail loudly, not
            # read as "unavailable" (a concat would also hide it behind the
            # other schema's columns)
            validate_depth_season(df, yr)
        frames.append(df)
        got.append(yr)
    if not frames:
        if required:
            raise RuntimeError(f"nflverse {dataset}: no seasons available from {seasons}")
        print(f"nflverse {dataset}: no seasons available from {seasons}; continuing empty")
        return pd.DataFrame()
    if got != seasons:
        print(f"nflverse {dataset}: using seasons {got}")
    out = pd.concat(frames, ignore_index=True)
    validate_columns(out, dataset)
    return out


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
