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


def test_active_stubs_fall_back_to_latest_earlier_chart_like_active_usage():
    """A team-week with no exact chart uses the team's latest earlier chart
    (usage.active_usage's fallback), so an active non-player still gets a row."""
    sched = pd.DataFrame({"season": [2024, 2024, 2024], "week": [1, 2, 3], "game_type": ["REG"] * 3,
                          "home_team": ["KC", "KC", "BAL"], "away_team": ["BAL", "DET", "KC"]})
    depth = _depth([
        (2024, 1, "KC", 1, "WR", "WR1"),
        (2024, 1, "KC", 2, "WR", "WR2"),
        (2024, 3, "KC", 1, "WR", "WR1"),        # week 3: exact chart, WR2 no longer listed
    ])
    inj = _injuries([(2024, 2, "KC", "WR1", "Out")])
    out = bpf.active_stubs(depth, inj, _pg([]), sched)
    got = sorted(zip(out["player_id"], out["week"], out["opponent"]))
    # wk1 exact chart; wk2 has no chart -> wk1's (WR1 Out that week); wk3 exact chart
    assert got == [("WR1", 1, "BAL"), ("WR1", 3, "BAL"), ("WR2", 1, "BAL"), ("WR2", 2, "DET")]
    assert set(out.loc[out["team"] == "BAL", "player_id"]) == set()   # BAL never had a chart


def test_chart_coverage_counts_exact_fallback_and_none():
    sched = pd.DataFrame({"season": [2024, 2024], "week": [1, 2], "game_type": ["REG"] * 2,
                          "home_team": ["KC", "KC"], "away_team": ["BAL", "BAL"]})
    depth = _depth([(2024, 1, "KC", 1, "WR", "W")])
    # KC wk1 exact, KC wk2 fallback to wk1, BAL both weeks none
    assert bpf.chart_coverage(depth, sched) == {"exact": 1, "fallback": 1, "none": 2}


def test_depth_rank_distribution_shares_by_season():
    feats = pd.DataFrame({"season": [2024] * 4 + [2025] * 2, "position": ["WR"] * 5 + ["RB"],
                          "p_depth_rank": [1.0, 2.0, 6.0, math.nan, 1.0, 1.0]})
    got = bpf.depth_rank_distribution(feats, "WR")
    assert got.loc[2024, "rows"] == 4 and got.loc[2025, "rows"] == 1
    assert got.loc[2024, ["1", "2", "5+", "nan"]].tolist() == [0.25, 0.25, 0.25, 0.25]
    assert got.loc[2024, "mean"] == 3.0 and got.loc[2025, "1"] == 1.0


def test_build_tables_applies_ctx_fill_before_both_builders(monkeypatch):
    """build_tables (shared by the parquet build and live serving) passes
    ctx_fill's output -- called as ctx_fill(ctx, stadiums, sched) -- to both
    table builders; without ctx_fill the raw context goes through."""
    import sportsmodel.nfl.context as ctx_mod
    import sportsmodel.nfl.efficiency as eff_mod
    import sportsmodel.nfl.player_features as pf_mod

    raw_ctx, filled = pd.DataFrame({"c": [1]}), pd.DataFrame({"c": [2]})
    seen: dict = {}
    monkeypatch.setattr(ctx_mod, "load_stadiums", lambda: {"S": {}})
    monkeypatch.setattr(ctx_mod, "team_game_context", lambda sched, st: raw_ctx)
    monkeypatch.setattr(eff_mod, "team_game_epa", lambda pbp: {})
    for name in ("player_games", "team_games", "player_redzone"):
        monkeypatch.setattr(pf_mod, name, lambda *a: pd.DataFrame())
    monkeypatch.setattr(bpf, "active_stubs", lambda *a: pd.DataFrame())
    monkeypatch.setattr(pf_mod, "build_feature_table",
                        lambda pg, tg, rz, ctx, *a, **k: seen.setdefault("feats_ctx", ctx))
    monkeypatch.setattr(pf_mod, "build_team_table", lambda tg, ctx, epa: seen.setdefault("team_ctx", ctx))
    src = {"sched": pd.DataFrame({"s": [1]}), "pbp": None, "injuries": None, "depth": None,
           "weekly": None, "snaps": None, "pfr2gsis": {}, "ngs": {}}

    def fill(ctx, stadiums, sched):
        seen["fill_args"] = (ctx, stadiums, sched)
        return filled

    built = bpf.build_tables(src, ctx_fill=fill)
    assert seen["fill_args"][0] is raw_ctx and seen["fill_args"][1] == {"S": {}}
    assert seen["fill_args"][2] is src["sched"]
    assert seen["feats_ctx"] is filled and seen["team_ctx"] is filled
    assert built["feats"] is filled and built["team"] is filled

    seen.clear()
    bpf.build_tables(src)
    assert seen["feats_ctx"] is raw_ctx and seen["team_ctx"] is raw_ctx


def test_fetch_sources_seasons_default_and_override():
    import inspect

    sig = inspect.signature(bpf.fetch_sources)
    assert "seasons" in sig.parameters and sig.parameters["seasons"].default is None
