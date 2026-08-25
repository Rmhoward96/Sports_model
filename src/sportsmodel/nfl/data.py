import nfl_data_py as nfl
import pandas as pd
from .teams import normalize_team

_SCHED_COLS = ["game_id", "season", "week", "game_type", "gameday", "gametime",
               "home_team", "away_team", "home_score", "away_score", "espn"]


def normalize_team_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].map(normalize_team)
    return df


def normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in _SCHED_COLS if c in df.columns]].copy()
    for col in ("home_team", "away_team"):
        df = normalize_team_col(df, col)
    return df


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    return normalize_schedule(nfl.import_schedules(seasons))


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    return normalize_team_col(nfl.import_weekly_data(seasons), "recent_team")


# nflverse's 2015 seasonal-roster snapshot uses PFR-style 3/4-char codes for a
# handful of teams instead of the standard nflverse abbreviation used everywhere
# else (verified live: only season==2015 rows carry these; all other seasons use
# the standard codes already covered by teams._ALIASES). Map them to the standard
# code before the usual normalize_team_col pass so load_rosters doesn't 404/raise
# on real, expected data.
_ROSTER_EXTRA_ALIASES = {"ARZ": "ARI", "BLT": "BAL", "HST": "HOU", "SL": "LA", "CLV": "CLE"}


def load_rosters(seasons: list[int]) -> pd.DataFrame:
    df = nfl.import_seasonal_rosters(seasons).copy()
    df["team"] = df["team"].replace(_ROSTER_EXTRA_ALIASES)
    # jersey_number and draft_number come back as object columns mixing float
    # and str values for the same whole numbers (verified live: all non-null
    # values are numeric, no fractional or non-numeric strings) — unify each to
    # a nullable int dtype so pyarrow can serialize the columns at all.
    for col in ("jersey_number", "draft_number"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return normalize_team_col(df, "team")


def load_injuries(seasons: list[int]) -> pd.DataFrame:
    df = nfl.import_injuries(seasons)
    return normalize_team_col(df, "team")
