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


def test_player_input_fields_round_trip():
    p = PlayerInput(
        player_id="player_123",
        name="John Doe",
        pos="WR",
        target_share=0.25,
        carry_share=0.05,
        ypt=8.5,
        ypc=4.2,
        catch_rate=0.72,
        td_share=0.15
    )
    assert p.player_id == "player_123"
    assert p.name == "John Doe"
    assert p.pos == "WR"
    assert p.target_share == 0.25
    assert p.carry_share == 0.05
    assert p.ypt == 8.5
    assert p.ypc == 4.2
    assert p.catch_rate == 0.72
    assert p.td_share == 0.15


def test_nfl_game_spec_fields():
    home_tr = TeamRates(
        drive_outcomes={"td": .25, "fg": .15, "punt": .4, "turnover": .1, "downs": .05, "end": .05},
        pass_rate=0.58, drives_per_game=11.0, rz_td_rate=0.6
    )
    away_tr = TeamRates(
        drive_outcomes={"td": .22, "fg": .16, "punt": .42, "turnover": .1, "downs": .05, "end": .05},
        pass_rate=0.55, drives_per_game=10.8, rz_td_rate=0.58
    )

    home_player = PlayerInput(
        player_id="home_wr_1",
        name="Home WR",
        pos="WR",
        target_share=0.3,
        carry_share=0.0,
        ypt=9.0,
        ypc=0.0,
        catch_rate=0.75,
        td_share=0.2
    )
    away_player = PlayerInput(
        player_id="away_rb_1",
        name="Away RB",
        pos="RB",
        target_share=0.1,
        carry_share=0.4,
        ypt=6.0,
        ypc=5.0,
        catch_rate=0.6,
        td_share=0.25
    )

    spec = NflGameSpec(
        home_team="Team A",
        away_team="Team B",
        home=home_tr,
        away=away_tr,
        home_players=[home_player],
        away_players=[away_player]
    )

    assert spec.home_team == "Team A"
    assert spec.away_team == "Team B"
    assert spec.home is home_tr
    assert spec.away is away_tr
    assert len(spec.home_players) == 1
    assert spec.home_players[0] is home_player
    assert len(spec.away_players) == 1
    assert spec.away_players[0] is away_player
