"""Leakage-free rate aggregation from nflverse data (NFL-only).

Turns raw nflverse play-by-play and weekly player frames into the
`TeamRates` / `PlayerInput` shapes the drive-based sim engine consumes
(see `sportsmodel.sim.nfl.spec`).

No-leakage contract
--------------------
`team_rates_from_pbp` and `player_inputs_from_weekly` both take an
`upto_season`/`upto_week` cutoff and filter to rows STRICTLY BEFORE it:

    season < upto_season OR (season == upto_season AND week < upto_week)

A row at or after the cutoff (including a later season) is excluded. This
is the only leakage guard — callers must pass the full history frame and
let these functions do the cutting, rather than pre-slicing themselves.

All functions except `fetch_nflverse` are pure: DataFrame in, value out.
They never read files, hit the network, or mutate their inputs.

Expected columns
-----------------
`team_rates_from_pbp(pbp_df, ...)` expects (nflverse `import_pbp_data`):
    season              : int
    week                : int
    posteam             : str  — offense team code for this play
    defteam             : str  — defense team code for this play
    play_type           : str  — 'pass' / 'run' (other values ignored for pass_rate)
    fixed_drive_result  : str  — nflverse's per-drive outcome label, one of
                                  {'Touchdown', 'Field Goal', 'Punt', 'Turnover',
                                  'Turnover on Downs', 'End of Half', 'End of Game',
                                  ...}; mapped case-insensitively to the sim's
                                  6-key drive_outcomes vocabulary (see
                                  `_DRIVE_RESULT_MAP` below). Anything unrecognized
                                  falls into "end" (concern: verify the exact
                                  label set against real nflverse data in Task 9/10).
    drive               : hashable — per-game drive id; used with game_id to
                                  count distinct drives.
    game_id             : hashable — per-game id; used for drives_per_game and
                                  as the "how many games" denominator.
    yardline_100        : float — yards from opponent's end zone (100 = own goal
                                  line); used to gate the red-zone (<=20) subset
                                  for rz_td_rate. Missing/absent column is
                                  tolerated (rz_td_rate becomes 0.0).

`player_inputs_from_weekly(weekly_df, snaps_df, ...)` expects (nflverse
`import_weekly_data`); `snaps_df` (nflverse `import_snap_counts`) is accepted
for interface symmetry with Task 6's brief but is NOT currently required by
any computed field (concern: snap-share weighting can be layered in later if
target/carry shares prove too noisy — verify in Task 9/10):
    player_id           : str
    player_display_name : str
    position             : str
    recent_team          : str
    season, week          : int
    targets, carries, receptions : numeric
    receiving_yards, rushing_yards : numeric
    receiving_tds, rushing_tds     : numeric

`fetch_nflverse(seasons)` is the only IO in this module; it is a thin
wrapper and is not unit-tested.
"""
from __future__ import annotations

import pandas as pd

from sportsmodel.sim.nfl.spec import PlayerInput, TeamRates

_DRIVE_KEYS: tuple[str, ...] = ("td", "fg", "punt", "turnover", "downs", "end")

# Maps nflverse's `fixed_drive_result` values (lowercased) to the sim's
# 6-key drive_outcomes vocabulary. Concern (verify in Task 9/10): confirm
# this covers the full real label set -- unrecognized values fall into "end".
_DRIVE_RESULT_MAP: dict[str, str] = {
    "touchdown": "td",
    "field goal": "fg",
    "punt": "punt",
    "turnover": "turnover",
    "turnover on downs": "downs",
    "downs": "downs",
    "end of half": "end",
    "end of game": "end",
    "end of 4th quarter": "end",
    "quarter end": "end",
}

_RED_ZONE_YARDLINE = 20


def _before_cutoff(df: pd.DataFrame, upto_season: int, upto_week: int) -> pd.DataFrame:
    """Rows strictly before (upto_season, upto_week): the sole leakage guard."""
    mask = (df["season"] < upto_season) | (
        (df["season"] == upto_season) & (df["week"] < upto_week)
    )
    return df[mask]


def team_rates_from_pbp(
    pbp_df: pd.DataFrame, upto_season: int, upto_week: int
) -> dict[str, TeamRates]:
    """Per-offense-team `TeamRates` aggregated from plays strictly before the cutoff.

    See module docstring for expected `pbp_df` columns.
    """
    df = _before_cutoff(pbp_df, upto_season, upto_week)
    result: dict[str, TeamRates] = {}

    for team, team_df in df.groupby("posteam"):
        if pd.isna(team) or team == "":
            continue

        # Drive-outcome distribution: one row per (game_id, drive) using its
        # fixed_drive_result (constant within a drive).
        drives = team_df.drop_duplicates(subset=["game_id", "drive"])
        counts = dict.fromkeys(_DRIVE_KEYS, 0.0)
        for raw_result in drives["fixed_drive_result"]:
            key = _DRIVE_RESULT_MAP.get(str(raw_result).strip().lower(), "end")
            counts[key] += 1.0
        total_drives = sum(counts.values())
        if total_drives > 0:
            drive_outcomes = {k: v / total_drives for k, v in counts.items()}
        else:
            drive_outcomes = dict.fromkeys(_DRIVE_KEYS, 0.0)

        # Pass rate over pass/run plays only.
        n_pass = int((team_df["play_type"] == "pass").sum())
        n_run = int((team_df["play_type"] == "run").sum())
        pass_rate = n_pass / (n_pass + n_run) if (n_pass + n_run) > 0 else 0.0

        # Drives per game.
        n_games = team_df["game_id"].nunique()
        drives_per_game = total_drives / n_games if n_games > 0 else 0.0

        # Red-zone TD rate: of drives that ever reached yardline_100 <= 20,
        # fraction ending in a touchdown.
        if "yardline_100" in team_df.columns:
            rz_drive_ids = team_df.loc[
                team_df["yardline_100"] <= _RED_ZONE_YARDLINE, ["game_id", "drive"]
            ].drop_duplicates()
            if len(rz_drive_ids) > 0:
                rz_drives = drives.merge(rz_drive_ids, on=["game_id", "drive"])
                n_rz = len(rz_drives)
                n_rz_td = int(
                    rz_drives["fixed_drive_result"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("touchdown")
                    .sum()
                )
                rz_td_rate = n_rz_td / n_rz if n_rz > 0 else 0.0
            else:
                rz_td_rate = 0.0
        else:
            rz_td_rate = 0.0

        result[team] = TeamRates(
            drive_outcomes=drive_outcomes,
            pass_rate=pass_rate,
            drives_per_game=drives_per_game,
            rz_td_rate=rz_td_rate,
        )

    return result


def player_inputs_from_weekly(
    weekly_df: pd.DataFrame,
    snaps_df: pd.DataFrame,
    upto_season: int,
    upto_week: int,
) -> dict[str, list[PlayerInput]]:
    """Per-team list of `PlayerInput` aggregated from weekly stats strictly
    before the cutoff.

    `snaps_df` is accepted for interface symmetry (nflverse `import_snap_counts`)
    but is not currently consumed by any computed field. See module docstring
    for expected `weekly_df` columns.
    """
    del snaps_df  # not currently used; kept for interface symmetry
    df = _before_cutoff(weekly_df, upto_season, upto_week)
    result: dict[str, list[PlayerInput]] = {}

    for team, team_df in df.groupby("recent_team"):
        if pd.isna(team) or team == "":
            continue

        team_targets = float(team_df["targets"].sum())
        team_carries = float(team_df["carries"].sum())
        team_tds = float(
            (team_df["receiving_tds"] + team_df["rushing_tds"]).sum()
        )

        players: list[PlayerInput] = []
        for player_id, p_df in team_df.groupby("player_id"):
            targets = float(p_df["targets"].sum())
            carries = float(p_df["carries"].sum())
            receptions = float(p_df["receptions"].sum())
            rec_yards = float(p_df["receiving_yards"].sum())
            rush_yards = float(p_df["rushing_yards"].sum())
            player_tds = float(
                (p_df["receiving_tds"] + p_df["rushing_tds"]).sum()
            )

            target_share = targets / team_targets if team_targets > 0 else 0.0
            carry_share = carries / team_carries if team_carries > 0 else 0.0
            ypt = rec_yards / targets if targets > 0 else 0.0
            ypc = rush_yards / carries if carries > 0 else 0.0
            catch_rate = receptions / targets if targets > 0 else 0.0
            td_share = player_tds / team_tds if team_tds > 0 else 0.0

            name = str(p_df["player_display_name"].iloc[0])
            pos = str(p_df["position"].iloc[0])

            players.append(
                PlayerInput(
                    player_id=str(player_id),
                    name=name,
                    pos=pos,
                    target_share=target_share,
                    carry_share=carry_share,
                    ypt=ypt,
                    ypc=ypc,
                    catch_rate=catch_rate,
                    td_share=td_share,
                )
            )

        result[team] = players

    return result


def fetch_nflverse(seasons: list[int]) -> dict:
    """Thin IO wrapper around nfl_data_py imports. Not unit-tested.

    Returns {"pbp": DataFrame, "weekly": DataFrame, "snaps": DataFrame}.
    """
    import nfl_data_py as nfl

    return {
        "pbp": nfl.import_pbp_data(seasons),
        "weekly": nfl.import_weekly_data(seasons),
        "snaps": nfl.import_snap_counts(seasons),
    }
