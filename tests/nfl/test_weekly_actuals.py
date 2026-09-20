import pandas as pd

from sportsmodel.nfl.weekly_actuals import (
    WEEKLY_STAT_COLS,
    player_market_actual,
    week_for_game,
)


def _sched(rows):
    return pd.DataFrame(rows, columns=["espn", "week"])


def test_week_for_game_matches_espn_column():
    sched = _sched([["401872934", 3], ["401872947", 3]])
    assert week_for_game(sched, 401872934) == 3          # int game_pk vs str espn
    assert week_for_game(sched, "401872947") == 3


def test_week_for_game_no_match_or_missing_col_returns_none():
    assert week_for_game(_sched([["1", 1]]), 999) is None
    assert week_for_game(pd.DataFrame({"week": [1]}), 1) is None
    assert week_for_game(pd.DataFrame(), 1) is None


def test_all_six_markets_mapped():
    assert set(WEEKLY_STAT_COLS) == {
        "pass_yds", "pass_tds", "rush_yds", "rec_yds", "receptions", "anytime_td"}


def test_player_market_actual_reads_and_sums():
    row = pd.Series({"passing_yards": 251.0, "rushing_tds": 1, "receiving_tds": 2,
                     "receptions": 5})
    assert player_market_actual(row, "pass_yds") == 251.0
    assert player_market_actual(row, "anytime_td") == 3.0     # rush+rec TDs
    assert player_market_actual(row, "receptions") == 5.0


def test_player_market_actual_nan_or_missing_is_none():
    row = pd.Series({"passing_yards": float("nan")})
    assert player_market_actual(row, "pass_yds") is None
    assert player_market_actual(row, "rush_yds") is None      # column absent
    assert player_market_actual(pd.Series({"rushing_tds": float("nan"),
                                           "receiving_tds": 1}), "anytime_td") == 1.0
