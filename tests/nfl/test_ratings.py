import math
from sportsmodel.nfl.elo import EloConfig, elo_expected_margin
from sportsmodel.nfl.ratings import BlendConfig, expected_margin

ELO = EloConfig(k=20, hfa_elo=0, carryover=0.75)

def test_pure_elo_below_min_games():
    b = BlendConfig(w_sos=0.5, srs_min_games=4)
    got = expected_margin(1525, 1500, 3.0, -3.0, games_home=2, games_away=9,
                          elo_cfg=ELO, blend_cfg=b)
    assert got == elo_expected_margin(1525, 1500, ELO)   # cold-start -> pure elo

def test_pure_elo_when_srs_none():
    b = BlendConfig(w_sos=0.5, srs_min_games=1)
    got = expected_margin(1525, 1500, None, None, games_home=9, games_away=9,
                          elo_cfg=ELO, blend_cfg=b)
    assert got == elo_expected_margin(1525, 1500, ELO)

def test_blend_when_available():
    b = BlendConfig(w_sos=0.5, srs_min_games=1)
    elo_m = elo_expected_margin(1525, 1500, ELO)   # +1.0
    srs_m = 6.0 - (-2.0) + ELO.hfa_elo / 25         # 8.0
    got = expected_margin(1525, 1500, 6.0, -2.0, games_home=9, games_away=9,
                          elo_cfg=ELO, blend_cfg=b)
    assert math.isclose(got, 0.5 * elo_m + 0.5 * srs_m)

def test_w_zero_reproduces_elo():
    b = BlendConfig(w_sos=0.0, srs_min_games=1)
    got = expected_margin(1525, 1500, 6.0, -2.0, 9, 9, ELO, b)
    assert got == elo_expected_margin(1525, 1500, ELO)
