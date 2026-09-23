"""Pure resolvers mapping the sim's projected player markets to their realized
nflverse `import_weekly_data` values, and a game's NFL week via the schedules
`espn` join. Shared by scripts/capture_nfl_player_actuals.py (and available to
grade_ev_props.py). No IO.
"""
from __future__ import annotations

import pandas as pd

# Projected market -> the nflverse weekly column(s) carrying its realized value.
# anytime_td has no single column: it's "did the player score", = rush + rec TDs.
WEEKLY_STAT_COLS: dict[str, tuple[str, ...]] = {
    "pass_yds": ("passing_yards",),
    "pass_tds": ("passing_tds",),
    "rush_yds": ("rushing_yards",),
    "rec_yds": ("receiving_yards",),
    "receptions": ("receptions",),
    "anytime_td": ("rushing_tds", "receiving_tds"),
    "rush_att": ("carries",),
}


def week_for_game(sched_df: pd.DataFrame, game_pk) -> int | None:
    """NFL week for `game_pk` (an ESPN event id) via an exact match against
    nflverse `import_schedules`' `espn` column. None if unresolvable."""
    if sched_df is None or sched_df.empty or "espn" not in sched_df.columns:
        return None
    matches = sched_df[sched_df["espn"].astype(str) == str(game_pk)]
    if matches.empty:
        return None
    week = matches.iloc[0]["week"]
    return int(week) if pd.notna(week) else None


def player_market_actual(weekly_row: pd.Series, market: str) -> float | None:
    """Realized value for `market` from one weekly-data row. Sums the columns
    for `anytime_td`. None if every needed column is absent or NaN."""
    cols = WEEKLY_STAT_COLS.get(market)
    if not cols:
        return None
    total = 0.0
    seen = False
    for col in cols:
        if col not in weekly_row:
            continue
        val = weekly_row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        total += float(val)
        seen = True
    return total if seen else None
