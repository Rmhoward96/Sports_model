import math

import pandas as pd
import pytest

from sportsmodel.cfb import panels


def _game(season, home, away, hq, aq):
    return {"season": season, "season_type": "regular", "home_team": home, "away_team": away,
            **{f"home_q{i + 1}": float(v) for i, v in enumerate(hq)},
            **{f"away_q{i + 1}": float(v) for i, v in enumerate(aq)}}


def test_quarter_shares_sum_to_one_and_follow_the_pattern():
    games = pd.DataFrame([_game(2025, "A", "B", [14, 0, 0, 0], [0, 7, 0, 0]),
                          _game(2025, "A", "C", [7, 0, 0, 0], [0, 0, 7, 0]),
                          _game(2026, "B", "C", [0, 0, 0, 7], [0, 0, 0, 7])])
    sh = panels.quarter_shares(games, 2026).set_index("team")
    assert list(sh.index) == ["A", "B", "C"] and list(sh["games_used"]) == [2, 2, 2]
    for t in sh.index:
        assert sum(sh.loc[t, f"scored_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
        assert sum(sh.loc[t, f"allowed_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
    lq = [21 / 49, 7 / 49, 7 / 49, 14 / 49]          # league scored points per quarter over all six team-games: 21 / 7 / 7 / 14 of 49
    w = 2 / 14                                                           # n / (n + 12)
    assert sh.loc["A", "scored_q1"] == pytest.approx(w * 1.0 + (1 - w) * lq[0])
    assert sh.loc["A", "scored_q2"] == pytest.approx((1 - w) * lq[1])
    assert sh.loc["A", "allowed_q2"] == pytest.approx(w * 0.5 + (1 - w) * lq[1])


def test_quarter_shares_small_n_is_close_to_the_league_and_large_n_to_the_team():
    big = [_game(2026, "A", "B", [10, 0, 0, 0], [0, 0, 0, 10]) for _ in range(120)]
    one = [_game(2026, "C", "D", [0, 10, 0, 0], [0, 0, 10, 0])]
    sh = panels.quarter_shares(pd.DataFrame(big + one), 2026).set_index("team")
    lg = 10 / 2420                                   # league share of q2: only C's single game scored there (10 of 2,420 points)
    assert sh.loc["A", "scored_q1"] == pytest.approx(120 / 132 * 1.0 + 12 / 132 * (1200 / 2420))      # 120 games: w = 120 / 132
    assert sh.loc["C", "scored_q2"] == pytest.approx(1 / 13 + 12 / 13 * lg)                          # one game: w = 1 / 13, so mostly league
    assert sh.loc["C", "games_used"] == 1


def test_quarter_shares_window_missing_quarters_fbs_filter_and_empty():
    games = pd.DataFrame([_game(2023, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]),          # too old (2026 - 2 = 2024)
                          _game(2025, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]),
                          {**_game(2025, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]), "home_q3": float("nan")}])
    sh = panels.quarter_shares(games, 2026)
    assert list(sh["games_used"]) == [1, 1]
    assert list(panels.quarter_shares(games, 2026, fbs={"A"})["team"]) == ["A"]
    assert panels.quarter_shares(games.iloc[0:1], 2026).empty
    assert list(panels.quarter_shares(games.iloc[0:0], 2026).columns) == panels.SHARE_COLUMNS


def test_quarter_shares_a_team_that_never_scored_falls_back_to_the_league():
    games = pd.DataFrame([_game(2026, "A", "B", [0, 0, 0, 0], [7, 7, 7, 7]), _game(2026, "A", "C", [0, 0, 0, 0], [3, 3, 3, 3])])
    sh = panels.quarter_shares(games, 2026).set_index("team")
    assert sum(sh.loc["A", f"scored_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
    assert sh.loc["A", "scored_q1"] == pytest.approx(0.25)


def test_to_rows_python_scalars_and_none():
    rows = panels.to_rows(panels.quarter_shares(pd.DataFrame([_game(2026, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7])]), 2026))
    assert type(rows[0]["games_used"]) is int and type(rows[0]["scored_q1"]) is float and rows[0]["team"] == "A"
    assert panels.to_rows(pd.DataFrame({"x": [float("nan"), 1.5], "n_ranked": [3.0, None]})) == [{"x": None, "n_ranked": 3}, {"x": 1.5, "n_ranked": None}]
