"""NFL team-level offensive/defensive EPA from nflverse play-by-play.

`load_team_epa` calls `nfl_data_py.import_pbp_data`, which downloads the full
play-by-play dataset for the requested seasons and is large (multiple
seasons can be hundreds of MB). Callers should pass the minimal season span
they actually need.
"""

import pandas as pd

from .teams import normalize_team


def team_epa_from_pbp(pbp: pd.DataFrame) -> dict[str, dict]:
    """Aggregate per-team offensive/defensive EPA from a play-by-play frame.

    Pure: DataFrame in, dict out, no IO.

    `off_epa` for a team is the mean of `epa` over rows where the team is
    `posteam`; `def_epa` is the mean of `epa` over rows where the team is
    `defteam`. Rows with NaN `epa` are dropped from both aggregations; rows
    with null/NaN `posteam` are dropped only from the offense aggregation,
    and rows with null/NaN `defteam` are dropped only from the defense
    aggregation. Team keys are normalized via `normalize_team`. Returns a
    dict over the union of teams appearing on offense or defense; a team
    appearing on only one side gets `None` for the other.
    """
    valid_epa = pbp[pd.notna(pbp["epa"])]

    off_rows = valid_epa[pd.notna(valid_epa["posteam"])]
    def_rows = valid_epa[pd.notna(valid_epa["defteam"])]

    off = off_rows.groupby("posteam")["epa"].mean()
    def_ = def_rows.groupby("defteam")["epa"].mean()

    off_by_team = {normalize_team(team): value for team, value in off.items()}
    def_by_team = {normalize_team(team): value for team, value in def_.items()}

    teams = set(off_by_team) | set(def_by_team)

    return {
        team: {
            "off_epa": off_by_team.get(team),
            "def_epa": def_by_team.get(team),
        }
        for team in teams
    }


def team_epa_by_season(pbp: pd.DataFrame) -> dict[tuple[int, str], dict]:
    """Aggregate per-(season, team) offensive/defensive EPA from a
    multi-season play-by-play frame.

    Pure: DataFrame in, dict out, no IO. Groups `pbp` by `season` and runs
    `team_epa_from_pbp` independently within each season group, so each
    season's off/def EPA is computed only from that season's plays -- no
    cross-season blending. Returns a dict keyed `(season, team) ->
    {"off_epa", "def_epa"}` over every (season, team) pair that appears in
    the frame.
    """
    out: dict[tuple[int, str], dict] = {}
    for season, group in pbp.groupby("season"):
        season_epa = team_epa_from_pbp(group)
        for team, values in season_epa.items():
            out[(int(season), team)] = values
    return out


def load_team_epa(seasons: list[int]) -> dict[str, dict]:
    """Load team off/def EPA for the given seasons from nflverse pbp data."""
    import nfl_data_py as nfl

    pbp = nfl.import_pbp_data(seasons)
    return team_epa_from_pbp(pbp[["posteam", "defteam", "epa"]])
