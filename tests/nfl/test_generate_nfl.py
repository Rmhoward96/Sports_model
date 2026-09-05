"""generate_nfl is a MODEL-ONLY game-line producer (no props, no market line),
structurally identical to generate_cfb. build_game_row is pure and unit-tested
here; main()'s ESPN/DB I/O is the thin live wrapper."""
import importlib.util, pathlib

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_nfl.py"
_s = importlib.util.spec_from_file_location("generate_nfl", _p)
gn = importlib.util.module_from_spec(_s); _s.loader.exec_module(gn)
from sportsmodel.nfl.gameline import GameLineConfig


def test_build_game_row_is_serving_shaped():
    game = {"game_pk": 401, "game_date": "2026-09-10", "commence_time": "2026-09-11T00:20Z",
            "home_team": "KC", "away_team": "BAL", "home_name": "Kansas City Chiefs",
            "away_name": "Baltimore Ravens"}
    ctx = {"model_margin": 3.0, "model_total": 45.0, "week": 1}
    row = gn.build_game_row(game, ctx, GameLineConfig())
    assert row["sport"] == "nfl" and row["model_version"] == "nfl-elo-v1"
    assert row["game_pk"] == 401 and row["margin_dist"]["kind"] == "margin"
    assert row["total_dist"]["kind"] == "pmf" and 0 <= row["home_win_prob"] <= 1
    assert row["home_team_name"] == "Kansas City Chiefs"
    assert row["away_team_name"] == "Baltimore Ravens"
    assert row["commence_time"] == "2026-09-11T00:20Z"  # kickoff carried through for time sort


def test_build_game_row_is_model_only():
    """A large model margin must flow straight through -- no market shrink. Two
    very different model margins must yield two different home_win_probs (a
    market-anchored build would pull both toward the same line)."""
    game = {"game_pk": 1, "game_date": "2026-09-10", "home_name": "H", "away_name": "A",
            "home_team": "KC", "away_team": "BAL"}
    small = gn.build_game_row(game, {"model_margin": 1.0, "model_total": 40.0, "week": 1}, GameLineConfig())
    big = gn.build_game_row(game, {"model_margin": 14.0, "model_total": 40.0, "week": 1}, GameLineConfig())
    assert big["home_win_prob"] > small["home_win_prob"]
    assert big["pred_margin"] > small["pred_margin"]


def test_game_date_from_commence_shifts_back_to_us_day():
    # A 00:20 UTC Monday SNF/MNF kickoff belongs to the Sunday US game day.
    assert gn._game_date_from_commence("2026-09-14T00:20Z") == "2026-09-13"


def test_producer_has_no_prop_or_market_surface():
    # The model-only rewrite must not reintroduce props/odds coupling.
    for gone in ("build_prop_rows", "_latest_market_line", "redistribute_out_shares",
                 "PROP_MODEL_VERSION", "_gsis_to_espn_crosswalk"):
        assert not hasattr(gn, gone), f"{gone} should be gone from the model-only producer"
