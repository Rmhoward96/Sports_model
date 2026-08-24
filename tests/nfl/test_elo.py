import math
import pandas as pd
from sportsmodel.nfl.elo import (
    EloConfig, expected_home, mov_multiplier, elo_expected_margin, run_elo,
)

def test_expected_home_neutral_with_hfa():
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75)
    assert expected_home(1500, 1500, cfg) == 0.5
    cfg2 = EloConfig(k=20, hfa_elo=65, carryover=0.75)
    assert expected_home(1500, 1500, cfg2) > 0.5   # HFA tilts home

def test_mov_multiplier_monotone_and_tie():
    assert mov_multiplier(3, 0) < mov_multiplier(21, 0)   # bigger win => bigger mult
    assert mov_multiplier(0, 0) == mov_multiplier(1, 0)   # tie treated as mov_input 1

def test_expected_margin_scale_and_sign():
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75)
    assert elo_expected_margin(1525, 1500, cfg) == (25 / 25)   # +1 point
    assert elo_expected_margin(1500, 1525, cfg) < 0

def test_run_elo_single_game_hand_computed():
    # base 1500, K=20, HFA 0, home wins by 7 -> E=0.5, mult=ln(8), delta=20*ln(8)*0.5
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75, base=1500)
    sched = pd.DataFrame([{
        "season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
        "home_score": 27, "away_score": 20,
    }])
    res = run_elo(sched, cfg)
    delta = 20 * math.log(8) * 0.5
    assert math.isclose(res.final["KC"], 1500 + delta, rel_tol=1e-9)
    assert math.isclose(res.final["BAL"], 1500 - delta, rel_tol=1e-9)
    row = res.games.iloc[0]
    assert row["elo_home"] == 1500 and row["e_home"] == 0.5   # pre-game values

def test_carryover_regresses_toward_1500():
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75, base=1500)
    # season 2023 gives KC a lead; 2024 opener should start from a regressed rating
    s2023 = pd.DataFrame([{"season": 2023, "week": 1, "home_team": "KC",
                           "away_team": "BAL", "home_score": 30, "away_score": 0}])
    s2024 = pd.DataFrame([{"season": 2024, "week": 1, "home_team": "KC",
                           "away_team": "BAL", "home_score": 20, "away_score": 17}])
    res = run_elo(pd.concat([s2023, s2024], ignore_index=True), cfg)
    kc_end_2023 = 1500 + 20 * math.log(31) * 0.5
    kc_start_2024 = 1500 + 0.75 * (kc_end_2023 - 1500)
    e = expected_home(kc_start_2024, 1500 + 0.75 * ((1500 - 20 * math.log(31) * 0.5) - 1500), cfg)
    assert res.games.iloc[1]["elo_home"] == kc_start_2024
