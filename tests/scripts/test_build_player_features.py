"""Tests for the pure seams in build_player_features.py (`active_stubs`,
`dropped_snap_mappings`, `nan_share_by_group`). main()'s nflverse IO and
parquet writes are not unit tested. Synthetic frames only -- no network."""
import importlib.util
import math
import pathlib

import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_player_features.py"
_spec = importlib.util.spec_from_file_location("build_player_features", _p)
bpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpf)


def _depth(rows):
    return pd.DataFrame(rows, columns=["season", "week", "club_code", "depth_team", "position", "gsis_id"])


def _sched():
    return pd.DataFrame({
        "season": [2024, 2024], "week": [1, 1], "game_type": ["REG", "REG"],
        "home_team": ["KC", "LAR"], "away_team": ["BAL", "DET"],
    })


def _injuries(rows):
    return pd.DataFrame(rows, columns=["season", "week", "team", "gsis_id", "report_status"])


def _pg(rows):
    return pd.DataFrame(rows, columns=["player_id", "season", "week"])


def test_active_stubs_excludes_played_and_out_players():
    depth = _depth([
        (2024, 1, "KC", 1, "QB", "QB1"),
        (2024, 1, "KC", 2, "QB", "QB2"),
        (2024, 1, "KC", 1, "WR", "WR1"),
        (2024, 1, "KC", 2, "WR", "WR2"),
    ])
    inj = _injuries([(2024, 1, "KC", "WR1", "Out")])
    pg = _pg([("QB1", 2024, 1)])
    out = bpf.active_stubs(depth, inj, pg, _sched())
    assert list(out.columns) == ["player_id", "season", "week", "team", "opponent", "position"]
    got = out.sort_values("player_id").reset_index(drop=True)
    assert got["player_id"].tolist() == ["QB2", "WR2"]
    assert got["team"].tolist() == ["KC", "KC"]
    assert got["opponent"].tolist() == ["BAL", "BAL"]
    assert got["position"].tolist() == ["QB", "WR"]
    assert got["season"].tolist() == [2024, 2024] and got["week"].tolist() == [1, 1]


def test_active_stubs_doubtful_excluded_questionable_kept_and_non_skill_dropped():
    depth = _depth([
        (2024, 1, "KC", 1, "RB", "RB1"),
        (2024, 1, "KC", 1, "TE", "TE1"),
        (2024, 1, "KC", 1, "LT", "OL1"),        # not a skill position
        (2024, 1, "KC", 1, "WR", None),         # no gsis id
    ])
    inj = _injuries([(2024, 1, "KC", "RB1", "Doubtful"), (2024, 1, "KC", "TE1", "Questionable")])
    out = bpf.active_stubs(depth, inj, _pg([]), _sched())
    assert out["player_id"].tolist() == ["TE1"]


def test_active_stubs_normalizes_codes_and_uses_opponent_from_schedule():
    # depth lists LAR (alias of LA); schedule's home team is LAR too
    depth = _depth([(2024, 1, "LAR", 1, "WR", "W"), (2024, 1, "DET", 1, "TE", "T")])
    out = bpf.active_stubs(depth, _injuries([]), _pg([]), _sched()).set_index("player_id")
    assert out.loc["W", "team"] == "LA" and out.loc["W", "opponent"] == "DET"
    assert out.loc["T", "team"] == "DET" and out.loc["T", "opponent"] == "LA"


def test_active_stubs_injury_only_matches_its_own_week_and_dedupes():
    depth = _depth([
        (2024, 1, "KC", 1, "WR", "WR1"),
        (2024, 1, "KC", 1, "WR", "WR1"),        # listed twice (two formations)
        (2024, 2, "KC", 1, "WR", "WR1"),        # week 2 has no REG game in schedule
    ])
    inj = _injuries([(2024, 2, "KC", "WR1", "Out")])   # a different week
    out = bpf.active_stubs(depth, inj, _pg([("WR1", 2023, 1)]), _sched())
    assert len(out) == 1 and out.iloc[0]["week"] == 1 and out.iloc[0]["player_id"] == "WR1"


def test_active_stubs_ignores_non_reg_games_and_handles_empty_injuries():
    sched = _sched().assign(game_type=["REG", "POST"])
    depth = _depth([(2024, 1, "LA", 1, "QB", "Q"), (2024, 1, "KC", 1, "QB", "K")])
    out = bpf.active_stubs(depth, pd.DataFrame(), _pg([]), sched)
    assert out["player_id"].tolist() == ["K"]


def test_dropped_snap_mappings_counts_unmapped_skill_rows_per_season():
    snaps = pd.DataFrame({
        "season": [2023, 2023, 2023, 2024, 2024],
        "game_type": ["REG"] * 5,
        "position": ["WR", "QB", "LT", "RB", "TE"],
        "offense_snaps": [10, 20, 30, 5, 0],
        "pfr_player_id": ["a", "zz", "yy", "b", "xx"],
    })
    got = bpf.dropped_snap_mappings(snaps, {"a": "G1", "b": "G2"})
    # 2023: QB "zz" unmapped (LT is not skill); 2024: TE with 0 snaps doesn't count
    assert got == {2023: 1, 2024: 0}


def test_nan_share_by_group():
    df = pd.DataFrame({"p_a": [1.0, math.nan], "p_b": [math.nan, math.nan], "cx_x": [1.0, 2.0],
                       "y_targets": [math.nan, 1.0]})
    got = bpf.nan_share_by_group(df)
    assert got["p_"] == 0.75 and got["cx_"] == 0.0
    assert "y_" not in got and math.isnan(got["mk_"])
