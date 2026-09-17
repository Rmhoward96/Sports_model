import numpy as np
from sportsmodel.sim.nfl.spec import TeamRates, PlayerInput, NflGameSpec, NflGameSims
from sportsmodel.sim.engine import home_win_prob, margin_pmf


def test_nflgamesims_duck_types_into_engine():
    s = NflGameSims(home_score=np.array([24, 17]), away_score=np.array([20, 21]),
                    player_stats={})
    assert home_win_prob(s) == 0.5
    assert margin_pmf(s)["kind"] == "margin"


def test_team_rates_and_spec_construct():
    tr = TeamRates(drive_outcomes={"td": .25, "fg": .15, "punt": .4, "turnover": .1, "downs": .05, "end": .05},
                   pass_rate=0.58, drives_per_game=11.0, rz_td_rate=0.6)
    assert abs(sum(tr.drive_outcomes.values()) - 1.0) < 1e-9
