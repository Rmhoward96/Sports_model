"""build_features.assemble is PURE: an Elo-augmented per-game schedule frame
(the shape `run_elo(...).games` produces) + a (season, team)->EPA/PPA dict
in, one feature row per game out. No IO here -- main()'s parquet/CFBD/EPA
loading is the thin live wrapper, exercised only by scripts/generate_*.py-style
live runs, not unit tested.

Fixture: 3 NFL teams (A, B, C), 3 games across weeks 1-3 of one season.
    G1: wk1, A(home) beats B(away) 24-17.  elo_home=1500 elo_away=1450
    G2: wk2, B(home) beats C(away) 20-14.  elo_home=1430 elo_away=1500
    G3: wk3, A(home) beats C(away) 30-10.  elo_home=1520 elo_away=1480  <- target row

For G3, A's ONLY prior (same-season, strictly-earlier-week) game is G1, and
C's only prior game is G2 -- hand-verified below. G1 itself is A's and B's
season debut, so G1's own row must show None/NaN trailing features (no
leakage from games that haven't happened yet).
"""
import importlib.util
import math
import pathlib

import pandas as pd
import pytest

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_features.py"
_spec = importlib.util.spec_from_file_location("build_features", _p)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

G1 = {
    "game_id": "2025_01_A_B", "season": 2025, "week": 1,
    "home_team": "A", "away_team": "B",
    "home_score": 24, "away_score": 17,
    "elo_home": 1500, "elo_away": 1450, "gameday": "2025-09-01",
}
G2 = {
    "game_id": "2025_02_B_C", "season": 2025, "week": 2,
    "home_team": "B", "away_team": "C",
    "home_score": 20, "away_score": 14,
    "elo_home": 1430, "elo_away": 1500, "gameday": "2025-09-08",
}
G3 = {
    "game_id": "2025_03_A_C", "season": 2025, "week": 3,
    "home_team": "A", "away_team": "C",
    "home_score": 30, "away_score": 10,
    "elo_home": 1520, "elo_away": 1480, "gameday": "2025-09-15",
}

ELO_GAMES = pd.DataFrame([G1, G2, G3])

# EPA_BY_TEAM is keyed (season, team) -- this is what a caller who has
# already shifted prior-season EPA/PPA onto the current season's key looks
# like: G1/G2/G3 are all season 2025, so their EPA/PPA must be looked up
# under (2025, team), NOT under (2024, team) or any other season.
#
# A deliberately different (2024, "A") entry is included so that a bug which
# looked EPA up by team alone (ignoring season) or which fell through to the
# wrong season would be caught by test_assemble_g3_epa_join_is_season_scoped
# below -- if the lookup used (2024, "A")'s value instead of (2025, "A")'s,
# the off_epa assertion there would fail.
#
# C is deliberately absent from season 2025 entirely to exercise the
# "missing team" graceful-NaN path.
EPA_BY_TEAM = {
    (2025, "A"): {"off_epa": 0.10, "def_epa": -0.05},
    (2025, "B"): {"off_epa": 0.20, "def_epa": 0.00},
    (2024, "A"): {"off_epa": 99.0, "def_epa": 99.0},
}


def _row(df: pd.DataFrame, game_id: str) -> pd.Series:
    matches = df[df["game_id"] == game_id]
    assert len(matches) == 1, f"expected exactly one row for {game_id}"
    return matches.iloc[0]


def test_assemble_nfl_row_count_and_columns_present():
    out = bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM, upcoming=False)
    assert len(out) == 3  # all 3 games completed
    for col in [
        "home_elo", "away_elo", "elo_diff",
        "home_last10", "away_last10", "last10_diff",
        "home_sos", "away_sos", "sos_diff",
        "home_sov", "away_sov", "sov_diff",
        "home_rest", "away_rest", "rest_diff",
        "home_off_epa", "home_def_epa", "away_off_epa", "away_def_epa",
        "off_epa_diff", "def_epa_diff",
        "margin", "total", "season", "week", "home_team", "away_team", "game_id",
    ]:
        assert col in out.columns, col


def test_assemble_g3_elo_and_diff():
    row = _row(bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM), "2025_03_A_C")
    assert row["home_elo"] == 1520
    assert row["away_elo"] == 1480
    assert row["elo_diff"] == 40


def test_assemble_g3_schedule_features_scoped_to_prior_same_season_games():
    row = _row(bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM), "2025_03_A_C")
    # A's only prior game is G1 (week1<week3): A won, opp(B) pregame elo=1450
    assert row["home_last10"] == pytest.approx(1.0)
    assert row["home_sos"] == pytest.approx(1450.0)
    assert row["home_sov"] == pytest.approx(1450.0)
    # C's only prior game is G2 (week2<week3): C lost, opp(B) pregame elo=1430
    assert row["away_last10"] == pytest.approx(0.0)
    assert row["away_sos"] == pytest.approx(1430.0)
    assert math.isnan(row["away_sov"])  # no wins among C's prior games -> None -> NaN
    assert row["last10_diff"] == pytest.approx(1.0)
    assert row["sos_diff"] == pytest.approx(20.0)
    assert math.isnan(row["sov_diff"])  # away side is None -> diff must be NaN, not crash


def test_assemble_g3_rest_days():
    row = _row(bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM), "2025_03_A_C")
    # A last played 2025-09-01 -> 14 days rest into 2025-09-15
    assert row["home_rest"] == 14
    # C last played 2025-09-08 -> 7 days rest into 2025-09-15
    assert row["away_rest"] == 7
    assert row["rest_diff"] == 7


def test_assemble_g3_epa_join_with_missing_team_is_nan_not_crash():
    row = _row(bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM), "2025_03_A_C")
    assert row["home_off_epa"] == pytest.approx(0.10)
    assert row["home_def_epa"] == pytest.approx(-0.05)
    # C is missing from EPA_BY_TEAM entirely (any season)
    assert math.isnan(row["away_off_epa"])
    assert math.isnan(row["away_def_epa"])
    assert math.isnan(row["off_epa_diff"])
    assert math.isnan(row["def_epa_diff"])


def test_assemble_g3_epa_join_is_season_scoped_not_leaked():
    """G3 is a season-2025 game. EPA_BY_TEAM carries BOTH (2025, "A") =
    {0.10, -0.05} and a deliberately different (2024, "A") = {99.0, 99.0}.
    The join must use the (2025, "A") value -- if `assemble` looked EPA up
    by team alone (ignoring season, the original leakage bug's shape) or
    picked up the wrong season's entry, home_off_epa would come back 99.0
    instead of 0.10."""
    row = _row(bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM), "2025_03_A_C")
    assert row["home_off_epa"] == pytest.approx(0.10)
    assert row["home_def_epa"] == pytest.approx(-0.05)
    assert row["home_off_epa"] != pytest.approx(99.0)


def test_assemble_epa_nan_for_season_with_no_stored_entry():
    """A game in a season that has no (season, team) key at all in the dict
    (e.g. the earliest season in the schedule, which has no prior season to
    draw leakage-free EPA from) must come back NaN for every team, not raise
    and not silently fall back to some other season's value."""
    game_2023 = {
        "game_id": "2023_01_A_B", "season": 2023, "week": 1,
        "home_team": "A", "away_team": "B",
        "home_score": 21, "away_score": 14,
        "elo_home": 1500, "elo_away": 1500, "gameday": "2023-09-01",
    }
    out = bf.assemble("nfl", pd.DataFrame([game_2023]), EPA_BY_TEAM, upcoming=False)
    row = _row(out, "2023_01_A_B")
    # EPA_BY_TEAM has no (2023, "A") or (2023, "B") key at all.
    assert math.isnan(row["home_off_epa"])
    assert math.isnan(row["home_def_epa"])
    assert math.isnan(row["away_off_epa"])
    assert math.isnan(row["away_def_epa"])
    assert math.isnan(row["off_epa_diff"])
    assert math.isnan(row["def_epa_diff"])


def test_assemble_g3_targets():
    row = _row(bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM), "2025_03_A_C")
    assert row["margin"] == 20
    assert row["total"] == 40


def test_assemble_no_leakage_debut_game_has_none_trailing_features():
    """G1 is A's and B's first game of the season -- no prior games exist, so
    every trailing feature must be NaN, proving the prior-games slice cannot
    reach into games that haven't been played yet (or the target game
    itself)."""
    row = _row(bf.assemble("nfl", ELO_GAMES, EPA_BY_TEAM), "2025_01_A_B")
    assert math.isnan(row["home_last10"])
    assert math.isnan(row["away_last10"])
    assert math.isnan(row["home_sos"])
    assert math.isnan(row["away_sos"])
    assert math.isnan(row["home_sov"])
    assert math.isnan(row["away_sov"])
    assert math.isnan(row["home_rest"])
    assert math.isnan(row["away_rest"])


def test_assemble_upcoming_true_includes_unplayed_games_no_targets():
    upcoming_game = {
        "game_id": "2025_04_A_B", "season": 2025, "week": 4,
        "home_team": "A", "away_team": "B",
        "home_score": None, "away_score": None,
        "elo_home": 1560, "elo_away": 1400, "gameday": "2025-09-22",
    }
    frame = pd.concat([ELO_GAMES, pd.DataFrame([upcoming_game])], ignore_index=True)
    out = bf.assemble("nfl", frame, EPA_BY_TEAM, upcoming=True)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["game_id"] == "2025_04_A_B"
    assert "margin" not in out.columns
    assert "total" not in out.columns
    assert row["home_elo"] == 1560


def test_assemble_default_upcoming_false_excludes_unplayed_games():
    upcoming_game = {
        "game_id": "2025_04_A_B", "season": 2025, "week": 4,
        "home_team": "A", "away_team": "B",
        "home_score": None, "away_score": None,
        "elo_home": 1560, "elo_away": 1400, "gameday": "2025-09-22",
    }
    frame = pd.concat([ELO_GAMES, pd.DataFrame([upcoming_game])], ignore_index=True)
    out = bf.assemble("nfl", frame, EPA_BY_TEAM, upcoming=False)
    assert len(out) == 3
    assert "2025_04_A_B" not in set(out["game_id"])


def test_assemble_cfb_rest_weeks_and_bye_columns():
    cfb_games = pd.DataFrame([
        {
            "game_id": "c1", "season": 2025, "week": 1,
            "home_team": "A", "away_team": "B",
            "home_score": 24, "away_score": 17,
            "elo_home": 1500, "elo_away": 1450,
        },
        {
            "game_id": "c2", "season": 2025, "week": 4,  # bye in weeks 2-3
            "home_team": "A", "away_team": "C",
            "home_score": 30, "away_score": 10,
            "elo_home": 1520, "elo_away": 1480,
        },
    ])
    ppa_by_season_team = {(2025, "A"): {"off_ppa": 0.3, "def_ppa": -0.1}}
    out = bf.assemble("cfb", cfb_games, ppa_by_season_team, upcoming=False)
    row = _row(out, "c2")
    assert row["home_rest_weeks"] == 3
    assert row["home_off_bye"] == True  # noqa: E712
    assert math.isnan(row["away_rest_weeks"])  # C has no prior game at all
    assert row["away_off_bye"] == False  # noqa: E712
    assert row["home_off_ppa"] == pytest.approx(0.3)
    assert math.isnan(row["away_off_ppa"])  # C missing from ppa dict
    assert math.isnan(row["off_ppa_diff"])
    assert "home_rest" not in out.columns  # NFL-only column name must not leak into CFB output
