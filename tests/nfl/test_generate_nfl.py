import importlib.util, pathlib
import pytest
import sportsmodel.db as db_module
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
    gsis_to_espn = {"wr1": 123456}
    rows = gn.build_prop_rows(game, universe, shares, eff, tv, PropConfig(), gsis_to_espn)
    assert rows and all(r["sport"] == "nfl" for r in rows)
    assert any(r["market"] == "reception_yds" for r in rows)
    # player_id served is the int ESPN athlete id (BIGINT-compatible), not the gsis string
    assert all(r["player_id"] == 123456 for r in rows)
    assert all(isinstance(r["player_id"], int) for r in rows)
    assert all(r["game_date"] == "2026-09-10" for r in rows)  # game_date must round-trip onto every prop row


def test_build_prop_rows_skips_player_missing_from_espn_crosswalk():
    # A player with no (or non-numeric) espn_id in the rosters crosswalk can't
    # be written to the BIGINT player_id column -- must be skipped, not crash.
    game = {"game_pk": 401, "game_date": "2026-09-10", "home_team": "KC", "away_team": "BAL"}
    universe = [{"player_id": "wr1", "player_name": "WR One", "team": "KC", "position": "WR"}]
    shares = {"wr1": {"target_share": 0.25, "carry_share": 0.0, "pass_att_share": 0.0,
                      "position": "WR", "team": "KC", "player_name": "WR One"}}
    eff = {"wr1": {"ypa":0,"pass_td_rate":0,"catch_rate":0.65,"ypr":11.0,"rec_td_rate":0.06,"ypc":0,"rush_td_rate":0}}
    tv = {"KC": {"pass_att": 34.0, "rush_att": 24.0, "plays": 58.0}}
    rows = gn.build_prop_rows(game, universe, shares, eff, tv, PropConfig(), {})  # wr1 not in crosswalk
    assert rows == []


def test_gsis_to_espn_crosswalk_dedups_to_latest_season_and_casts_int():
    import pandas as pd
    rosters = pd.DataFrame([
        {"season": 2023, "player_id": "wr1", "espn_id": "111"},
        {"season": 2024, "player_id": "wr1", "espn_id": "222"},   # latest season wins
        {"season": 2024, "player_id": "wr2", "espn_id": None},    # no espn_id -> dropped
        {"season": 2024, "player_id": "wr3", "espn_id": "not-a-number"},  # non-numeric -> dropped
    ])
    cw = gn._gsis_to_espn_crosswalk(rosters)
    assert cw == {"wr1": 222}
    assert isinstance(cw["wr1"], int)


def test_redistribute_out_shares_splits_proportionally_to_existing_share():
    # wr1 (the starter) is OUT; wr2 and wr3 survive with EXISTING target
    # shares 0.2 and 0.1 -- wr1's 0.3 target_share should split 2:1 between
    # them, IN PROPORTION TO THEIR OWN EXISTING SHARE -- not by depth-chart
    # rank. The committed assets/nfl/rosters.parquet's depth_chart_position
    # column is just the position CODE ("WR") for every same-position
    # player on a team, so a rank-based pick would tie and be arbitrary;
    # depth_chart_position is included below (as the real data shapes it)
    # to prove the function no longer relies on it.
    universe = [
        {"player_id": "wr2", "player_name": "WR Two", "team": "KC", "position": "WR",
         "depth_chart_position": "WR"},
        {"player_id": "wr3", "player_name": "WR Three", "team": "KC", "position": "WR",
         "depth_chart_position": "WR"},
    ]
    shares = {
        "wr1": {"target_share": 0.3, "carry_share": 0.0, "pass_att_share": 0.0,
                "position": "WR", "team": "KC", "player_name": "WR One"},
        "wr2": {"target_share": 0.2, "carry_share": 0.0, "pass_att_share": 0.0,
                "position": "WR", "team": "KC", "player_name": "WR Two"},
        "wr3": {"target_share": 0.1, "carry_share": 0.0, "pass_att_share": 0.0,
                "position": "WR", "team": "KC", "player_name": "WR Three"},
    }
    adj = gn.redistribute_out_shares(shares, {"wr1"}, universe)
    assert adj["wr2"]["target_share"] == pytest.approx(0.2 + 0.3 * (0.2 / 0.3))  # +0.2
    assert adj["wr3"]["target_share"] == pytest.approx(0.1 + 0.3 * (0.1 / 0.3))  # +0.1
    assert shares["wr2"]["target_share"] == pytest.approx(0.2)   # input not mutated


def test_redistribute_out_shares_drops_dimension_with_no_survivor_share():
    # Sole RB survivor has ZERO existing carry_share (e.g. a pure receiving
    # back) -- the OUT starter's carry_share has no legitimate weight to
    # split by and must be DROPPED, not handed whole to a 0-carry player.
    # Their target_share, however, has a real (sole) survivor to land on.
    universe = [{"player_id": "rb2", "player_name": "RB Two", "team": "KC", "position": "RB",
                "depth_chart_position": "RB"}]
    shares = {
        "rb1": {"target_share": 0.05, "carry_share": 0.6, "pass_att_share": 0.0,
                "position": "RB", "team": "KC", "player_name": "RB One"},
        "rb2": {"target_share": 0.3, "carry_share": 0.0, "pass_att_share": 0.0,
                "position": "RB", "team": "KC", "player_name": "RB Two"},
    }
    adj = gn.redistribute_out_shares(shares, {"rb1"}, universe)
    assert adj["rb2"]["carry_share"] == pytest.approx(0.0)         # dropped, not guessed
    assert adj["rb2"]["target_share"] == pytest.approx(0.3 + 0.05)  # sole survivor gets all of it


def test_redistribute_out_shares_no_same_position_survivor_leaves_share_unclaimed():
    universe = [{"player_id": "wr1", "player_name": "WR One", "team": "KC", "position": "WR",
                "depth_chart_position": "WR"}]
    shares = {"rb1": {"target_share": 0.05, "carry_share": 0.6, "pass_att_share": 0.0,
                      "position": "RB", "team": "KC", "player_name": "RB One"}}
    adj = gn.redistribute_out_shares(shares, {"rb1"}, universe)
    assert adj["rb1"]["carry_share"] == pytest.approx(0.6)   # no RB survivor at all -> unclaimed


def test_latest_market_line_flips_home_spread_sign_to_nflverse_convention(monkeypatch):
    # odds_snapshot stores the raw sportsbook convention: side='home' `line`
    # is NEGATIVE when home is favored (e.g. -3.5). build_gameline/shrink use
    # nflverse's convention instead -- POSITIVE spread_line = home favored,
    # matching model_margin's sign. A -3.5 home line must come out as +3.5.
    rows = [("spread", "home", -3.5), ("total", "over", 45.5)]

    class FakeCursor:
        def execute(self, *a, **k):
            pass
        def fetchall(self):
            return rows
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class FakeConn:
        def cursor(self):
            return FakeCursor()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn())
    market = gn._latest_market_line(401)
    assert market["spread_line"] == pytest.approx(3.5)    # sign flipped: home favored by 3.5
    assert market["total_line"] == pytest.approx(45.5)    # totals pass through unchanged
