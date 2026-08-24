import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_board.py"
_s = importlib.util.spec_from_file_location("generate_board", _p)
gb = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gb)


def _game():
    g = {"game_pk": 1, "game_date": "2026-08-22", "commence_time": None,
         "home_name": "Orioles", "away_name": "Yankees", "matchup": "Yankees @ Orioles",
         "pred_total": 8.2, "total_dist": {"kind": "pmf", "pmf": [0.0] * 8 + [0.4, 0.6] + [0.0] * 11},
         "pred_margin": 0.3, "margin_dist": {"kind": "margin", "offset": 10, "pmf": [0.0] * 21},
         "home_win_prob": 0.55}
    g["margin_dist"]["pmf"][2 + 10] = 1.0  # all mass at margin +2
    return g


def test_build_rows_game_lines_and_prop():
    game = _game()
    odds = {
        ("moneyline", "home", ""): {None: [("draftkings", -120), ("fanduel", -115)]},
        ("moneyline", "away", ""): {None: [("betmgm", 105)]},
        ("total", "over", ""): {8.5: [("draftkings", -110)]},
        ("total", "under", ""): {8.5: [("fanduel", -105)]},
        ("spread", "home", ""): {-1.5: [("draftkings", 130)]},
        ("spread", "away", ""): {1.5: [("fanduel", -150)]},
        ("hits", "over", "aaron judge"): {1.5: [("fanduel", -110)]},
        ("hits", "under", "aaron judge"): {1.5: [("draftkings", -120)]},
    }
    props = [{"player_id": 99, "player_name": "Aaron Judge", "team": "NYY",
              "market": "hits", "dist": {"kind": "pmf", "pmf": [0.2, 0.2, 0.6]}}]  # P(over 1.5)=0.6

    rows = gb.build_rows(game, props, odds, ((0.0, 1.0), (0.0, 1.0)))
    mk = {r["market"]: r for r in rows}
    assert mk["moneyline"]["pick_label"] == "Orioles ML"
    assert mk["moneyline"]["book"] == "FanDuel"  # display name; -115 (dec 1.87) beats -120
    assert "total" in mk and "spread" in mk
    assert mk["hits"]["player_name"] == "Aaron Judge" and mk["hits"]["is_pick"] is True
    for r in rows:
        assert r["sport"] == "mlb" and r["game_pk"] == 1 and r["market_label"]


def test_home_run_excluded_from_prop_markets():
    assert "home_run" not in gb.PROP_MARKETS


def test_main_line_prefers_most_booked_then_lowest():
    by_line = {8.5: [("A", -110)], 9.0: [("B", -110), ("C", -108)]}
    assert gb._main_line(by_line) == 9.0   # 9.0 has 2 books
    tie = {8.5: [("A", -110)], 9.0: [("B", -110)]}
    assert gb._main_line(tie) == 8.5       # tie -> lowest


def test_build_rows_tags_sport():
    game = _game()
    rows = gb.build_rows(game, [], {
        ("moneyline", "home", ""): {None: [("draftkings", -120)]},
        ("moneyline", "away", ""): {None: [("fanduel", 110)]},
    }, ((0.0, 1.0), (0.0, 1.0)), sport="nfl")
    assert rows and all(r["sport"] == "nfl" for r in rows)
