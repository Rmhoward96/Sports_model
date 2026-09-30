"""Pure helpers of scripts/compare_cfb_live_fix.py (part C)."""
import importlib.util
import pathlib

import pandas as pd
import pytest

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "compare_cfb_live_fix.py"
_s = importlib.util.spec_from_file_location("compare_cfb_live_fix", _p)
cmp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(cmp)

from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig


def _sched():
    rows = []
    for season in (2021, 2022):
        for week in (1, 2, 3):
            rows += [{"season": season, "week": week, "game_type": "REG", "home_team": "A",
                      "away_team": "B", "home_score": 30, "away_score": 20},
                     {"season": season, "week": week, "game_type": "REG", "home_team": "C",
                      "away_team": "D", "home_score": 14, "away_score": 21}]
    return pd.DataFrame(rows)


def test_legacy_state_reproduces_todays_pooling():
    st = cmp.legacy_season_to_date_ratings(_sched(), 2022, 2, EloConfig(), BlendConfig())
    # 3 games in 2021 + 1 in 2022 week 1: pooled across seasons (the bug)
    assert st["games_played"] == {"A": 4, "B": 4, "C": 4, "D": 4}
    assert st["srs_now"] and st["points_ratings"]


def test_legacy_prior_is_same_season_sp():
    pri = pd.DataFrame([{"season": 2022, "team_espn_id": "A", "sp_rating": 12.0},
                        {"season": 2021, "team_espn_id": "A", "sp_rating": -3.0}])
    assert cmp.legacy_priors_for_season(pri, 2022) == {"A": 1512.0}


def test_score_and_paired_diff():
    rows = [
        # home favored by 7 (home-margin convention); model on home; home wins by 10 -> win
        {"week": 1, "m_margin": 9.0, "m_total": 50.0, "actual_margin": 10.0,
         "actual_total": 52.0, "market_spread": 7.0},
        # model on away (2 < 7); home wins by 3 -> away covers -> win
        {"week": 6, "m_margin": 2.0, "m_total": 40.0, "actual_margin": 3.0,
         "actual_total": 44.0, "market_spread": 7.0},
        # push
        {"week": 6, "m_margin": 2.0, "m_total": 40.0, "actual_margin": 7.0,
         "actual_total": 44.0, "market_spread": 7.0},
    ]
    s = cmp.score(rows, "m")
    assert (s["n"], s["n_early"], s["ats_w"], s["ats_l"], s["ats_push"]) == (3, 1, 2, 0, 1)
    assert s["ats_pct"] == 1.0
    assert s["margin_mae"] == pytest.approx((1 + 1 + 5) / 3)
    assert s["margin_mae_wk1_5"] == pytest.approx(1.0)
    assert s["n_late"] == 2 and s["margin_mae_wk6p"] == pytest.approx(3.0)
    assert s["total_mae"] == pytest.approx((2 + 4 + 4) / 3)
    pr = [{**r, "a_margin": r["m_margin"], "b_margin": r["m_margin"] + 1} for r in rows]
    d, se = cmp.paired_diff(pr, "a", "b")
    # a errs 1,1,5 ; b = a + 1 errs 0,0,4 -> every diff is +1
    assert d == pytest.approx(1.0) and se == pytest.approx(0.0)
