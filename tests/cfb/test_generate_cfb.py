import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_cfb.py"
_s = importlib.util.spec_from_file_location("generate_cfb", _p)
gc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gc)

from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.ratings import BlendConfig

ELO_CFG = EloConfig()
BLEND_CFG = BlendConfig(w_sos=0.0, srs_min_games=3)  # w_sos=0 -> pure elo margin, no SRS needed
GL_CFG = GameLineConfig()


def _ratings(elo_final=None, srs_now=None, points_ratings=None, lg_avg=55.0,
            games_played=None) -> dict:
    return {
        "elo_final": elo_final or {},
        "srs_now": srs_now or {},
        "points_ratings": points_ratings or {},
        "lg_avg": lg_avg,
        "games_played": games_played or {},
        "elo_cfg": ELO_CFG,
        "blend_cfg": BLEND_CFG,
    }


def test_build_game_row_is_serving_shaped():
    game = {"game_pk": 401628354, "game_date": "2026-09-05",
            "home_team": "96", "away_team": "61",
            "home_name": "Kentucky Wildcats", "away_name": "Georgia Bulldogs"}
    ctx = {"model_margin": -7.0, "model_total": 58.0, "week": 3}
    row = gc.build_game_row(game, ctx, GL_CFG)
    assert row["sport"] == "cfb"
    assert row["model_version"] == "cfb-ratings-v1"
    assert row["game_pk"] == 401628354
    assert row["game_date"] == "2026-09-05"
    assert row["home_team_name"] == "Kentucky Wildcats"
    assert row["away_team_name"] == "Georgia Bulldogs"
    assert 0 < row["home_win_prob"] < 1
    assert isinstance(row["pred_home_score"], float)
    assert isinstance(row["pred_away_score"], float)
    assert row["margin_dist"]["kind"] == "margin"
    assert row["total_dist"]["kind"] == "pmf"
    # model-only: no market was ever passed in, so pred_margin/pred_total
    # must equal the raw model inputs untouched by any shrink
    assert row["pred_margin"] == ctx["model_margin"]
    assert row["pred_total"] == ctx["model_total"]


def test_build_game_rows_skips_fcs_games():
    games = [
        {"game_pk": 1, "home_team": "96", "away_team": "61",
         "home_name": "Kentucky Wildcats", "away_name": "Georgia Bulldogs",
         "game_date": "2026-09-05"},
        {"game_pk": 2, "home_team": "158", "away_team": "FCS",
         "home_name": "Nebraska Cornhuskers", "away_name": "Northern Iowa Panthers",
         "game_date": "2026-09-05"},
        {"game_pk": 3, "home_team": "FCS", "away_team": "12",
         "home_name": "Some FCS School", "away_name": "Arizona Wildcats",
         "game_date": "2026-09-05"},
    ]
    ratings = _ratings(elo_final={"96": 1550.0, "61": 1620.0})
    rows = gc.build_game_rows(games, ratings, week=3, gl_cfg=GL_CFG)
    assert len(rows) == 1
    assert rows[0]["game_pk"] == 1
    assert all(r["sport"] == "cfb" for r in rows)


def test_build_game_rows_uses_ratings_to_derive_margin_and_total():
    games = [
        {"game_pk": 10, "home_team": "H", "away_team": "A",
         "home_name": "Home Team", "away_name": "Away Team", "game_date": "2026-09-05"},
    ]
    ratings = _ratings(
        elo_final={"H": 1700.0, "A": 1500.0},
        points_ratings={"H": {"off": 5.0, "def": -2.0}, "A": {"off": -3.0, "def": 1.0}},
        lg_avg=55.0,
        games_played={"H": 5, "A": 5},
    )
    rows = gc.build_game_rows(games, ratings, week=3, gl_cfg=GL_CFG)
    assert len(rows) == 1
    row = rows[0]
    # home is much stronger (elo 1700 vs 1500 + HFA) -> favored, positive margin
    assert row["pred_margin"] > 0
    assert row["home_win_prob"] > 0.5
    assert row["pred_home_score"] > 0 and row["pred_away_score"] > 0


def test_build_game_rows_defaults_missing_ratings_to_base_elo():
    # Teams absent from elo_final/srs_now/points_ratings (e.g. first game of
    # the historical window) must fall back gracefully rather than KeyError.
    games = [
        {"game_pk": 20, "home_team": "H", "away_team": "A",
         "home_name": "Home Team", "away_name": "Away Team", "game_date": "2026-09-05"},
    ]
    ratings = _ratings()  # everything empty
    rows = gc.build_game_rows(games, ratings, week=1, gl_cfg=GL_CFG)
    assert len(rows) == 1
    assert 0 < rows[0]["home_win_prob"] < 1
