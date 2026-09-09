import pandas as pd
import pytest

from sportsmodel.features.schedule import (
    last10_win_pct,
    sos,
    sov,
    rest_days_nfl,
    rest_weeks_cfb,
)

# Team "A" across 4 prior games, mixed home/away, mixed wins/losses.
# Opponent pre-game Elo, gameday, and week are all hand-picked so every
# stat can be verified by hand.
#
#   G1: wk1, A home vs B, A wins 24-17. Opp(B) pre-game elo = elo_away = 1450.
#   G2: wk2, A away vs C, A loses 20-28. Opp(C) pre-game elo = elo_home = 1550.
#   G3: wk4 (bye in wk3), A home vs D, A wins 30-10. Opp(D) pre-game elo = elo_away = 1600.
#   G4: wk5, A away vs E, A wins 21-14. Opp(E) pre-game elo = elo_home = 1480.
G1 = {
    "season": 2025, "week": 1, "home_team": "A", "away_team": "B",
    "home_score": 24, "away_score": 17,
    "elo_home": 1500, "elo_away": 1450, "gameday": "2025-09-01",
}
G2 = {
    "season": 2025, "week": 2, "home_team": "C", "away_team": "A",
    "home_score": 28, "away_score": 20,
    "elo_home": 1550, "elo_away": 1510, "gameday": "2025-09-08",
}
G3 = {
    "season": 2025, "week": 4, "home_team": "A", "away_team": "D",
    "home_score": 30, "away_score": 10,
    "elo_home": 1520, "elo_away": 1600, "gameday": "2025-09-22",
}
G4 = {
    "season": 2025, "week": 5, "home_team": "E", "away_team": "A",
    "home_score": 14, "away_score": 21,
    "elo_home": 1480, "elo_away": 1540, "gameday": "2025-09-29",
}

ALL_GAMES = pd.DataFrame([G1, G2, G3, G4])


def _frame(*games):
    return pd.DataFrame(list(games))


# ---------------------------------------------------------------------------
# last10_win_pct
# ---------------------------------------------------------------------------

def test_last10_win_pct_over_four_prior_games():
    # A: win, loss, win, win -> 3/4
    assert last10_win_pct(ALL_GAMES, "A") == pytest.approx(0.75)


def test_last10_win_pct_single_game():
    assert last10_win_pct(_frame(G1), "A") == pytest.approx(1.0)


def test_last10_win_pct_no_prior_games_is_none():
    assert last10_win_pct(_frame(), "A") is None


def test_last10_win_pct_ignores_incomplete_games():
    incomplete = dict(G4, home_score=None, away_score=None)
    frame = _frame(G1, G2, G3, incomplete)
    # incomplete game must not count toward the 3/3 -> still 2/3 (G1 win, G2 loss, G3 win)
    assert last10_win_pct(frame, "A") == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# sos
# ---------------------------------------------------------------------------

def test_sos_mean_opponent_pregame_elo():
    # (1450 + 1550 + 1600 + 1480) / 4 = 1520.0
    assert sos(ALL_GAMES, "A") == pytest.approx(1520.0)


def test_sos_no_prior_games_is_none():
    assert sos(_frame(), "A") is None


# ---------------------------------------------------------------------------
# sov
# ---------------------------------------------------------------------------

def test_sov_mean_opponent_pregame_elo_wins_only():
    # A won G1 (opp elo 1450), G3 (opp elo 1600), G4 (opp elo 1480)
    # (1450 + 1600 + 1480) / 3 = 1510.0
    assert sov(ALL_GAMES, "A") == pytest.approx(1510.0)


def test_sov_no_wins_is_none():
    assert sov(_frame(G2), "A") is None


def test_sov_no_prior_games_is_none():
    assert sov(_frame(), "A") is None


# ---------------------------------------------------------------------------
# rest_days_nfl
# ---------------------------------------------------------------------------

def test_rest_days_nfl_exact_day_diff_to_most_recent_game():
    # most recent prior game is G4 on 2025-09-29; next game 2025-10-06 -> 7 days
    assert rest_days_nfl(ALL_GAMES, "A", "2025-10-06") == 7


def test_rest_days_nfl_no_prior_games_is_none():
    assert rest_days_nfl(_frame(), "A", "2025-09-01") is None


def test_rest_days_nfl_never_uses_same_or_later_game():
    # even if a same-day/future row leaks into prior_games, it must be excluded
    leaking = dict(G4, gameday="2025-10-06")
    frame = _frame(G1, G2, G3, leaking)
    # falls back to G3 (2025-09-22) -> 2025-10-06 is 14 days later
    assert rest_days_nfl(frame, "A", "2025-10-06") == 14


# ---------------------------------------------------------------------------
# rest_weeks_cfb
# ---------------------------------------------------------------------------

def test_rest_weeks_cfb_off_bye_true_across_skipped_week():
    # prior games end at G2 (week 2); target game is week 4 (week 3 was a bye)
    result = rest_weeks_cfb(_frame(G1, G2), "A", 4)
    assert result == {"weeks_since_prev": 2, "off_bye": True}


def test_rest_weeks_cfb_normal_week_no_bye():
    # prior games end at G4 (week 5); target game is week 6
    result = rest_weeks_cfb(ALL_GAMES, "A", 6)
    assert result == {"weeks_since_prev": 1, "off_bye": False}


def test_rest_weeks_cfb_no_prior_games():
    result = rest_weeks_cfb(_frame(), "A", 1)
    assert result == {"weeks_since_prev": None, "off_bye": False}


def test_rest_weeks_cfb_never_uses_same_or_later_game():
    # a leaking row at/after this_week must be excluded from "most recent prior"
    leaking = dict(G3, week=4)
    frame = _frame(G1, G2, leaking)
    result = rest_weeks_cfb(frame, "A", 4)
    # must fall back to G2 (week 2), not the leaking week-4 row
    assert result == {"weeks_since_prev": 2, "off_bye": True}
