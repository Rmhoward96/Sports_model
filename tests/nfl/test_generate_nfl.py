import importlib.util, pathlib
import pytest
_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_nfl.py"
_s = importlib.util.spec_from_file_location("generate_nfl", _p)
gn = importlib.util.module_from_spec(_s); _s.loader.exec_module(gn)
from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.props import PropConfig

def test_build_game_row_is_serving_shaped():
    game = {"game_pk": 401, "game_date": "2026-09-10", "commence_time": "2026-09-11T00:20Z",
            "home_team": "KC", "away_team": "BAL", "home_name": "Kansas City Chiefs",
            "away_name": "Baltimore Ravens"}
    ctx = {"model_margin": 3.0, "model_total": 45.0, "market": {"spread_line": None, "total_line": None},
           "week": 1}
    row = gn.build_game_row(game, ctx, GameLineConfig())
    assert row["sport"] == "nfl" and row["model_version"] == "nfl-elo-v1"
    assert row["game_pk"] == 401 and row["margin_dist"]["kind"] == "margin"
    assert row["total_dist"]["kind"] == "pmf" and 0 <= row["home_win_prob"] <= 1

def test_build_prop_rows_excludes_inactive_and_tags_sport():
    game = {"game_pk": 401, "game_date": "2026-09-10", "home_team": "KC", "away_team": "BAL"}
    universe = [{"player_id": "wr1", "player_name": "WR One", "team": "KC", "position": "WR"}]
    shares = {"wr1": {"target_share": 0.25, "carry_share": 0.0, "pass_att_share": 0.0,
                      "position": "WR", "team": "KC", "player_name": "WR One"}}
    eff = {"wr1": {"ypa":0,"pass_td_rate":0,"catch_rate":0.65,"ypr":11.0,"rec_td_rate":0.06,"ypc":0,"rush_td_rate":0}}
    tv = {"KC": {"pass_att": 34.0, "rush_att": 24.0, "plays": 58.0}}
    rows = gn.build_prop_rows(game, universe, shares, eff, tv, PropConfig())
    assert rows and all(r["sport"] == "nfl" for r in rows)
    assert any(r["market"] == "reception_yds" for r in rows)
    assert all(r["player_id"] == "wr1" for r in rows)   # only the active WR


def test_redistribute_out_shares_bumps_backup():
    # rb1 (depth 1) is OUT this week; universe (already OUT-filtered) only
    # has the backup rb2 (depth 2) left at RB/KC -- rb1's share should land
    # entirely on rb2, not evaporate.
    universe = [
        {"player_id": "rb2", "player_name": "RB Two", "team": "KC", "position": "RB",
         "depth_chart_position": 2},
        {"player_id": "wr1", "player_name": "WR One", "team": "KC", "position": "WR",
         "depth_chart_position": 1},
    ]
    shares = {
        "rb1": {"target_share": 0.05, "carry_share": 0.6, "pass_att_share": 0.0,
                "position": "RB", "team": "KC", "player_name": "RB One"},
        "rb2": {"target_share": 0.02, "carry_share": 0.1, "pass_att_share": 0.0,
                "position": "RB", "team": "KC", "player_name": "RB Two"},
        "wr1": {"target_share": 0.25, "carry_share": 0.0, "pass_att_share": 0.0,
                "position": "WR", "team": "KC", "player_name": "WR One"},
    }
    adj = gn.redistribute_out_shares(shares, {"rb1"}, universe)
    assert adj["rb2"]["carry_share"] == pytest.approx(0.7)
    assert adj["rb2"]["target_share"] == pytest.approx(0.07)
    assert adj["wr1"]["target_share"] == pytest.approx(0.25)     # untouched
    assert shares["rb2"]["carry_share"] == pytest.approx(0.1)    # input not mutated


def test_redistribute_out_shares_no_backup_leaves_shares_untouched():
    universe = [{"player_id": "wr1", "player_name": "WR One", "team": "KC", "position": "WR"}]
    shares = {"rb1": {"target_share": 0.05, "carry_share": 0.6, "pass_att_share": 0.0,
                      "position": "RB", "team": "KC", "player_name": "RB One"}}
    adj = gn.redistribute_out_shares(shares, {"rb1"}, universe)
    assert adj["rb1"]["carry_share"] == pytest.approx(0.6)   # no RB backup to route it to
