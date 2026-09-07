import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_cfb.py"
_s = importlib.util.spec_from_file_location("generate_cfb", _p)
gc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gc)

from sportsmodel.cfb.priors import DecayConfig, PriorWeights, preseason_rating, season_features_z
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.ratings import BlendConfig

ELO_CFG = EloConfig()
BLEND_CFG = BlendConfig(w_sos=0.0, srs_min_games=3)  # w_sos=0 -> pure elo margin, no SRS needed
GL_CFG = GameLineConfig()


def _ratings(elo_final=None, srs_now=None, points_ratings=None, lg_avg=55.0,
            games_played=None, r_pre=None, decay_cfg=None) -> dict:
    return {
        "elo_final": elo_final or {},
        "srs_now": srs_now or {},
        "points_ratings": points_ratings or {},
        "lg_avg": lg_avg,
        "games_played": games_played or {},
        "elo_cfg": ELO_CFG,
        "blend_cfg": BLEND_CFG,
        "r_pre": r_pre or {},
        "decay_cfg": decay_cfg or DecayConfig(half_life_games=4.0),
    }


def test_build_game_row_is_serving_shaped():
    game = {"game_pk": 401628354, "game_date": "2026-09-05",
            "commence_time": "2026-09-05T23:30Z",
            "market_spread": -3.5, "market_total": 55.5,
            "home_team": "96", "away_team": "61",
            "home_name": "Kentucky Wildcats", "away_name": "Georgia Bulldogs"}
    ctx = {"model_margin": -7.0, "model_total": 58.0, "week": 3}
    row = gc.build_game_row(game, ctx, GL_CFG)
    assert row["market_spread"] == -3.5 and row["market_total"] == 55.5  # Vegas line for the lean
    assert row["sport"] == "cfb"
    assert row["model_version"] == "cfb-ratings-v1"
    assert row["game_pk"] == 401628354
    assert row["game_date"] == "2026-09-05"
    assert row["commence_time"] == "2026-09-05T23:30Z"  # kickoff carried through for time sort
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


# --------------------------------------------------------------------------
# Forward-looking priors (Task 6): build_game_rows blends each team's R_pre
# (from priors.parquet + PriorWeights, via preseason_rating/season_features_z)
# into the pre-game Elo rating with decay-by-games_played, mirroring
# backtest_cfb_priors.py's prior_seeded_margin design (design decision #1:
# blend at the Elo level, feed the blended value into expected_margin).
# --------------------------------------------------------------------------

_PRIOR_WEIGHTS = PriorWeights()  # identity default: R_pre = 1500 + sp_rating
_PRIORS_ROW_H = {"team_espn_id": "H", "sp_rating": 250.0, "coach_first_year": False,
                 "qb_returning": True, "portal_net": 0.0, "returning_starters": 0.0,
                 "prior_sos": 0.0, "forward_sos_shift": 0.0}
_Z = season_features_z([_PRIORS_ROW_H])
_R_PRE_H = preseason_rating(_PRIORS_ROW_H, _Z["H"], _PRIOR_WEIGHTS)  # 1500 + 250 = 1750


def _games_h_vs_a():
    return [{"game_pk": 30, "home_team": "H", "away_team": "A",
             "home_name": "Home Team", "away_name": "Away Team", "game_date": "2026-09-05"}]


def test_build_game_rows_margin_driven_by_prior_at_zero_games_played():
    # Team H has a strong R_pre (1750) but no in-season Elo history yet
    # (games_played=0 for both sides, elo_final empty -> both default to
    # elo_cfg.base=1500). At games_played=0 the decay weight is 1.0, so the
    # blended rating for H should equal R_pre exactly, producing a
    # noticeably larger home margin than the no-prior baseline (a 250 elo
    # gap is a 10-point margin swing at this engine's 25-elo-per-point scale).
    baseline = gc.build_game_rows(_games_h_vs_a(), _ratings(games_played={"H": 0, "A": 0}),
                                  week=1, gl_cfg=GL_CFG)
    with_prior = gc.build_game_rows(
        _games_h_vs_a(),
        _ratings(games_played={"H": 0, "A": 0}, r_pre={"H": _R_PRE_H}),
        week=1, gl_cfg=GL_CFG)
    assert with_prior[0]["pred_margin"] - baseline[0]["pred_margin"] > 9.9


def test_build_game_rows_prior_influence_negligible_at_high_games_played():
    # Once a team has played many games, the decay weight collapses toward
    # the DecayConfig floor, so the prior-seeded margin should be within a
    # hair of the no-prior baseline (not identical, since the weight never
    # hits exactly 0, but negligible).
    baseline = gc.build_game_rows(_games_h_vs_a(), _ratings(games_played={"H": 100, "A": 100}),
                                  week=1, gl_cfg=GL_CFG)
    with_prior = gc.build_game_rows(
        _games_h_vs_a(),
        _ratings(games_played={"H": 100, "A": 100}, r_pre={"H": _R_PRE_H}),
        week=1, gl_cfg=GL_CFG)
    assert abs(with_prior[0]["pred_margin"] - baseline[0]["pred_margin"]) < 0.01


def test_build_game_rows_missing_priors_falls_back_to_todays_behavior():
    # No "r_pre"/"decay_cfg" injected at all (mirrors main() when
    # assets/cfb/priors.parquet doesn't exist -- `_ratings()`'s own defaults
    # of r_pre={} apply) -- must be byte-for-byte the pre-Task-6 behavior.
    games = _games_h_vs_a()
    ratings = _ratings(elo_final={"H": 1700.0, "A": 1500.0}, games_played={"H": 5, "A": 5})
    rows = gc.build_game_rows(games, ratings, week=3, gl_cfg=GL_CFG)
    assert len(rows) == 1
    assert rows[0]["pred_margin"] > 0
    assert rows[0]["home_win_prob"] > 0.5
