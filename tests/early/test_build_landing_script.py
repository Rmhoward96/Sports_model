"""scripts/build_landing_tables.py on tiny synthetic frames: joins, sign convention, filters, output files."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("build_landing_tables", ROOT / "scripts" / "build_landing_tables.py")
bl = importlib.util.module_from_spec(_spec)
sys.modules["build_landing_tables"] = bl
_spec.loader.exec_module(bl)


def lines(rows):
    return pd.DataFrame(rows, columns=["season", "week", "home_team", "away_team", "market_spread", "market_total"])


def schedules(rows):
    return pd.DataFrame(rows, columns=["season", "week", "home_team", "away_team", "home_score", "away_score"])


def test_cfb_pairs_join_on_season_week_and_both_team_ids():
    ln = lines([(2024, 3, "12", "99", 7.0, 55.5), (2024, 3, "13", "98", -3.0, 48.5)])
    sc = schedules([(2024, 3, "12", "99", 31, 20), (2024, 3, "13", "98", 10, 17), (2024, 4, "12", "99", 0, 0)])
    out = bl.cfb_pairs(ln, sc)
    assert out["spread"] == [(7.0, 11), (-3.0, -7)]                 # (expected home margin, home margin)
    assert out["total"] == [(55.5, 51), (48.5, 27)]


def test_cfb_spread_sign_is_positive_for_a_home_favourite_as_in_the_source():
    ln = lines([(2024, 1, "1", "2", 14.0, 50.0)])
    sc = schedules([(2024, 1, "1", "2", 35, 14)])
    assert bl.cfb_pairs(ln, sc)["spread"] == [(14.0, 21)]


def test_cfb_rows_without_a_line_or_a_score_or_outside_the_seasons_are_dropped():
    ln = lines([(2024, 1, "1", "2", 3.0, None), (2024, 1, "3", "4", 3.0, 50.0), (2014, 1, "5", "6", 3.0, 50.0),
                (2026, 1, "7", "8", 3.0, 50.0), (2024, 1, "9", "10", 3.0, 50.0)])
    sc = schedules([(2024, 1, "1", "2", 20, 10), (2024, 1, "3", "4", 20, None), (2014, 1, "5", "6", 20, 10),
                    (2026, 1, "7", "8", 20, 10), (2024, 1, "9", "10", 20, 10)])
    out = bl.cfb_pairs(ln, sc)
    assert out["spread"] == [(3.0, 10), (3.0, 10)] and out["total"] == [(50.0, 30)]


def test_nfl_pairs_use_spread_line_and_scores_for_2010_to_2025():
    sc = pd.DataFrame({"season": [2009, 2010, 2025, 2026, 2024], "spread_line": [3.0, -4.0, 7.0, 1.0, None],
                       "home_score": [20, 13, 27, 10, 20.0], "away_score": [10, 16, 20, 9, 17.0]})
    assert bl.nfl_pairs(sc) == [(-4.0, -3), (7.0, 7)]


def test_main_writes_the_three_tables_into_the_output_folder(tmp_path, monkeypatch):
    fake = {("cfb", "spread"): {"sport": "cfb", "market": "spread", "entries": {}},
            ("cfb", "total"): {"sport": "cfb", "market": "total", "entries": {}, "games": 0},
            ("nfl", "spread"): {"sport": "nfl", "market": "spread", "entries": {}, "games": 0}}
    for v in fake.values():
        v.setdefault("games", 0)
    monkeypatch.setattr(bl, "build_all", lambda assets: fake)
    assert bl.main(["--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["landing_cfb_spread.json", "landing_cfb_total.json",
                                                         "landing_nfl_spread.json"]
    assert json.loads((tmp_path / "landing_nfl_spread.json").read_text())["sport"] == "nfl"
