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
                                  {'Touchdown', 'Field Goal', 'Missed Field Goal',
                                  'Punt', 'Turnover', 'Turnover on Downs', 'Safety',
                                  'Opp touchdown', 'End of Half', ...}; mapped
                                  case-insensitively to the sim's 6-key
                                  drive_outcomes vocabulary (see
                                  `_DRIVE_RESULT_MAP` below, verified against a
                                  live 2024 nflverse pull). Anything unforeseen
                                  falls into "end" as a safe fallback.

                                  v1 simplification: "Safety" and "Opp touchdown"
                                  (defensive/return TD) are opponent-scoring
                                  drives from the offense's point of view. Both
                                  are mapped to "turnover" — i.e. treated as a
                                  0-point offensive turnover. This deliberately
                                  does NOT credit the opponent's +2 (safety) or
                                  +7 (opp touchdown) anywhere in TeamRates; that
                                  scoring is out of scope for this rate-aggregation
                                  layer and would need to be handled by whatever
                                  consumes drive_outcomes if/when it matters.
    drive               : hashable — per-game drive id; used with game_id to
                                  count distinct drives.
    game_id             : hashable — per-game id; used for drives_per_game and
                                  as the "how many games" denominator.
    yardline_100        : float — yards from opponent's end zone (100 = own goal
                                  line); used to gate the red-zone (<=20) subset
                                  for rz_td_rate. Missing/absent column is
                                  tolerated (rz_td_rate becomes 0.0).
    sack                : int/float 0/1 — 1 if the pass play was a sack (B.3).
                                  Read unconditionally (import_pbp_data always
                                  supplies it); sacks are pass PLAYS that are NOT
                                  attempts, so pass_att_pg = pass plays minus
                                  sacks. NaN counts as non-sack.
    complete_pass       : int/float 0/1 — 1 if the pass attempt was completed
                                  (B.3); used for completion_pct. Read
                                  unconditionally.

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
# 6-key drive_outcomes vocabulary. Verified against a live 2024 nflverse pull.
# All non-td/fg buckets are 0-point outcomes for the offense; this mapping
# does not change scoring, only the drive-outcome distribution's accuracy.
#
# v1 simplification: "safety" and "opp touchdown" are opponent-scoring drives
# (the defense/return unit scores against this offense). Both are bucketed as
# "turnover" -- a 0-point offensive turnover -- and this module does NOT
# credit the opponent's +2 (safety) or +7 (opp touchdown) points anywhere.
_DRIVE_RESULT_MAP: dict[str, str] = {
    "touchdown": "td",
    "field goal": "fg",  # made field goals only
    "missed field goal": "downs",  # failed attempt: opponent takes over, 0 pts
    "punt": "punt",
    "turnover": "turnover",
    "turnover on downs": "downs",
    "downs": "downs",
    "safety": "turnover",  # 0 offensive points; opponent's +2 not credited (v1)
    "opp touchdown": "turnover",  # defensive/return TD; opponent's +7 not credited (v1)
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


def season_weights(seasons, upto_season: int, decay: float):
    """Per-row recency weight ``decay ** (upto_season - season)``.

    The current season (``upto_season``) gets weight 1.0 and each earlier season
    is multiplied by an extra factor of ``decay`` (0 < decay <= 1). ``decay == 1``
    is uniform weighting (every season equal); ``decay == 0.5`` makes each
    current-season game count double a one-year-old game and quadruple a
    two-year-old one. The exponent is clipped at 0 so any stray post-cutoff row
    can never exceed weight 1.0.
    """
    exponent = (upto_season - pd.to_numeric(seasons)).clip(lower=0)
    return pd.Series(float(decay), index=seasons.index).pow(exponent)


def team_rates_from_pbp(
    pbp_df: pd.DataFrame, upto_season: int, upto_week: int, season_decay: float = 1.0
) -> dict[str, TeamRates]:
    """Per-offense-team `TeamRates` aggregated from plays strictly before the cutoff.

    Every count/rate below is a season-weighted sum: each play (or drive, or
    game) is weighted by ``season_weights(...)`` so recent seasons count more
    heavily. ``season_decay=1.0`` (default) recovers the original equal-weight
    aggregation. See module docstring for expected `pbp_df` columns.
    """
    df = _before_cutoff(pbp_df, upto_season, upto_week).copy()
    df["_w"] = season_weights(df["season"], upto_season, season_decay)
    result: dict[str, TeamRates] = {}

    for team, team_df in df.groupby("posteam"):
        if pd.isna(team) or team == "":
            continue

        w = team_df["_w"]

        # Drive-outcome distribution: one weight-carrying row per (game_id,
        # drive) using its fixed_drive_result (constant within a drive).
        drives = team_df.drop_duplicates(subset=["game_id", "drive"]).copy()
        drives["_k"] = (
            drives["fixed_drive_result"].astype(str).str.strip().str.lower()
            .map(lambda r: _DRIVE_RESULT_MAP.get(r, "end"))
        )
        by_key = drives.groupby("_k")["_w"].sum()
        counts = {k: float(by_key.get(k, 0.0)) for k in _DRIVE_KEYS}
        total_drives = float(drives["_w"].sum())
        if total_drives > 0:
            drive_outcomes = {k: v / total_drives for k, v in counts.items()}
        else:
            drive_outcomes = dict.fromkeys(_DRIVE_KEYS, 0.0)

        # Pass rate over pass/run plays only (weighted).
        is_pass = team_df["play_type"] == "pass"
        is_run = team_df["play_type"] == "run"
        w_pass = float(w[is_pass].sum())
        w_run = float(w[is_run].sum())
        pass_rate = w_pass / (w_pass + w_run) if (w_pass + w_run) > 0 else 0.0

        # Weighted game count (a game's plays share one season -> one weight).
        w_games = float(team_df.drop_duplicates("game_id")["_w"].sum())
        drives_per_game = total_drives / w_games if w_games > 0 else 0.0

        # Red-zone TD rate: of drives that ever reached yardline_100 <= 20, the
        # weighted fraction ending in a touchdown.
        if "yardline_100" in team_df.columns:
            rz_drive_ids = team_df.loc[
                team_df["yardline_100"] <= _RED_ZONE_YARDLINE, ["game_id", "drive"]
            ].drop_duplicates()
            if len(rz_drive_ids) > 0:
                rz_drives = drives.merge(rz_drive_ids, on=["game_id", "drive"])
                n_rz = float(rz_drives["_w"].sum())
                n_rz_td = float(rz_drives.loc[rz_drives["_k"] == "td", "_w"].sum())
                rz_td_rate = n_rz_td / n_rz if n_rz > 0 else 0.0
            else:
                rz_td_rate = 0.0
        else:
            rz_td_rate = 0.0

        # Per-game volume + efficiency rates (B.3 Task 1), weighted. Reuses
        # is_pass/is_run/w_pass/w_run from above; `is_pass` counts all pass
        # PLAYS including sacks, so attempts subtract sacks. nflverse
        # passing_yards is gross, so sacks affect target counts only.
        is_sack = team_df["sack"] == 1
        is_attempt = is_pass & ~is_sack

        w_attempts = float(w[is_attempt].sum())
        w_sacks = float(w[is_sack].sum())
        w_completions = float(w[is_attempt & (team_df["complete_pass"] == 1)].sum())

        pass_att_pg = w_attempts / w_games if w_games > 0 else 0.0
        rush_att_pg = w_run / w_games if w_games > 0 else 0.0
        sack_rate = w_sacks / w_pass if w_pass > 0 else 0.0
        completion_pct = w_completions / w_attempts if w_attempts > 0 else 0.0

        # Passing-TD share of offensive TDs (for the sim's TD split: passing TDs
        # credit the QB + the receiver; rushing TDs credit the rusher). nflverse
        # pbp flags pass_touchdown/rush_touchdown per play. League fallback ~0.58
        # when a team has no offensive TDs yet in the window.
        w_pass_td = float(w[team_df.get("pass_touchdown", 0) == 1].sum()) if "pass_touchdown" in team_df.columns else 0.0
        w_rush_td = float(w[team_df.get("rush_touchdown", 0) == 1].sum()) if "rush_touchdown" in team_df.columns else 0.0
        pass_td_share = w_pass_td / (w_pass_td + w_rush_td) if (w_pass_td + w_rush_td) > 0 else 0.58

        result[team] = TeamRates(
            drive_outcomes=drive_outcomes,
            pass_rate=pass_rate,
            drives_per_game=drives_per_game,
            rz_td_rate=rz_td_rate,
            pass_att_pg=pass_att_pg,
            rush_att_pg=rush_att_pg,
            sack_rate=sack_rate,
            completion_pct=completion_pct,
            pass_td_share=pass_td_share,
        )

    return result


def player_inputs_from_weekly(
    weekly_df: pd.DataFrame,
    snaps_df: pd.DataFrame,
    upto_season: int,
    upto_week: int,
    season_decay: float = 1.0,
) -> dict[str, list[PlayerInput]]:
    """Per-team list of `PlayerInput` aggregated from weekly stats strictly
    before the cutoff.

    Each weekly (per-game) row is season-weighted via ``season_weights(...)`` so
    a player's recent-season usage/efficiency counts more heavily; shares stay
    consistent because numerator and denominator carry the same weights.
    ``season_decay=1.0`` (default) recovers equal weighting. `snaps_df` is
    accepted for interface symmetry (nflverse `import_snap_counts`) but is not
    currently consumed by any computed field. See module docstring for expected
    `weekly_df` columns.
    """
    del snaps_df  # not currently used; kept for interface symmetry
    df = _before_cutoff(weekly_df, upto_season, upto_week).copy()
    df["_w"] = season_weights(df["season"], upto_season, season_decay)
    result: dict[str, list[PlayerInput]] = {}

    def _wsum(frame: pd.DataFrame, col) -> float:
        """Weighted sum of `col` (a column name or a precomputed Series)."""
        series = frame[col] if isinstance(col, str) else col
        return float((series * frame["_w"]).sum())

    for team, team_df in df.groupby("recent_team"):
        if pd.isna(team) or team == "":
            continue

        team_targets = _wsum(team_df, "targets")
        team_carries = _wsum(team_df, "carries")
        team_tds = _wsum(team_df, team_df["receiving_tds"] + team_df["rushing_tds"])

        players: list[PlayerInput] = []
        for player_id, p_df in team_df.groupby("player_id"):
            targets = _wsum(p_df, "targets")
            carries = _wsum(p_df, "carries")
            receptions = _wsum(p_df, "receptions")
            rec_yards = _wsum(p_df, "receiving_yards")
            rush_yards = _wsum(p_df, "rushing_yards")
            player_tds = _wsum(p_df, p_df["receiving_tds"] + p_df["rushing_tds"])

            target_share = targets / team_targets if team_targets > 0 else 0.0
            carry_share = carries / team_carries if team_carries > 0 else 0.0
            ypt = rec_yards / targets if targets > 0 else 0.0
            ypc = rush_yards / carries if carries > 0 else 0.0
            ypr = rec_yards / receptions if receptions > 0 else 0.0
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
                    ypr=ypr,
                    catch_rate=catch_rate,
                    td_share=td_share,
                )
            )

        result[team] = players

    return result


def fetch_nflverse(seasons: list[int]) -> dict:
    """Thin IO wrapper reading nflverse release parquets directly. Not unit-tested.

    Returns {"pbp": DataFrame, "weekly": DataFrame, "snaps": DataFrame}.

    Uses `nflverse.load_release` (canonical release URLs) rather than
    nfl_data_py, whose pinned 0.3.2 can't fetch the current season's weekly
    (retired URL) or pbp (participation-sidecar bug). snaps is optional -- it's
    accepted for interface symmetry but not required by any computed field -- so
    a missing snaps season degrades to empty instead of aborting the build.
    """
    from sportsmodel.nfl.nflverse import load_release

    return {
        "pbp": load_release("pbp", seasons),
        "weekly": load_release("weekly", seasons),
        "snaps": load_release("snaps", seasons, required=False),
    }
