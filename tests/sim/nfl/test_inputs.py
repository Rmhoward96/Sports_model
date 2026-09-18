"""Tests for NflGameSpec assembly with injury zeroing/renormalization."""
import pytest

from sportsmodel.sim.nfl.spec import TeamRates, PlayerInput, NflGameSpec
from sportsmodel.sim.nfl.inputs import build_spec


def _team_rates(td=0.25):
    return TeamRates(
        drive_outcomes={"td": td, "fg": .15, "punt": .4, "turnover": .1, "downs": .05, "end": .1 - (td - .15)},
        pass_rate=0.58,
        drives_per_game=11.0,
        rz_td_rate=0.6,
    )


def _wr(player_id, name, target_share, td_share, carry_share=0.0):
    return PlayerInput(
        player_id=player_id,
        name=name,
        pos="WR",
        target_share=target_share,
        carry_share=carry_share,
        ypt=8.0,
        ypc=0.0,
        catch_rate=0.65,
        td_share=td_share,
    )


def test_out_wr_removed_and_remaining_shares_renormalize():
    home_players = [
        _wr("wr1", "Alpha One", 0.5, 0.5),
        _wr("wr2", "Beta Two", 0.3, 0.3),
        _wr("wr3", "Gamma Three", 0.2, 0.2),
    ]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"HOME": [{"player": "Alpha One", "position": "WR", "status": "Out", "note": ""}]}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    names = {p.name for p in spec.home_players}
    assert "Alpha One" not in names
    assert names == {"Beta Two", "Gamma Three"}

    target_sum = sum(p.target_share for p in spec.home_players)
    assert target_sum == pytest.approx(1.0)
    td_sum = sum(p.td_share for p in spec.home_players)
    assert td_sum == pytest.approx(1.0)

    # 0.3 / (0.3 + 0.2) = 0.6, 0.2 / 0.5 = 0.4
    by_name = {p.name: p for p in spec.home_players}
    assert by_name["Beta Two"].target_share == pytest.approx(0.6)
    assert by_name["Gamma Three"].target_share == pytest.approx(0.4)


def test_doubtful_player_removed_by_default():
    home_players = [
        _wr("wr1", "Alpha One", 0.6, 0.6),
        _wr("wr2", "Beta Two", 0.4, 0.4),
    ]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"HOME": [{"player": "Alpha One", "position": "WR", "status": "Doubtful", "note": ""}]}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    names = {p.name for p in spec.home_players}
    assert names == {"Beta Two"}
    assert spec.home_players[0].target_share == pytest.approx(1.0)


def test_questionable_player_kept():
    home_players = [
        _wr("wr1", "Alpha One", 0.6, 0.6),
        _wr("wr2", "Beta Two", 0.4, 0.4),
    ]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"HOME": [{"player": "Alpha One", "position": "WR", "status": "Questionable", "note": ""}]}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    names = {p.name for p in spec.home_players}
    assert names == {"Alpha One", "Beta Two"}
    # unchanged since nobody was dropped
    by_name = {p.name: p for p in spec.home_players}
    assert by_name["Alpha One"].target_share == pytest.approx(0.6)
    assert by_name["Beta Two"].target_share == pytest.approx(0.4)


def test_name_matching_is_case_insensitive():
    home_players = [
        _wr("wr1", "Alpha One", 0.6, 0.6),
        _wr("wr2", "Beta Two", 0.4, 0.4),
    ]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"HOME": [{"player": "alpha ONE", "position": "WR", "status": "out", "note": ""}]}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    names = {p.name for p in spec.home_players}
    assert names == {"Beta Two"}


def test_spec_has_both_teams_rates_and_players():
    home_players = [_wr("wr1", "Alpha One", 1.0, 1.0)]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    home_rates = _team_rates(td=0.25)
    away_rates = _team_rates(td=0.22)
    rates = {"HOME": home_rates, "AWAY": away_rates}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    assert isinstance(spec, NflGameSpec)
    assert spec.home_team == "HOME"
    assert spec.away_team == "AWAY"
    assert spec.home is home_rates
    assert spec.away is away_rates
    assert len(spec.home_players) == 1
    assert len(spec.away_players) == 1


def test_no_injuries_for_team_keeps_all_players_unchanged():
    home_players = [
        _wr("wr1", "Alpha One", 0.6, 0.6),
        _wr("wr2", "Beta Two", 0.4, 0.4),
    ]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"AWAY": []}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    assert len(spec.home_players) == 2


def test_missing_team_in_rates_raises():
    players = {"HOME": [_wr("wr1", "Alpha One", 1.0, 1.0)], "AWAY": [_wr("awr1", "Away One", 1.0, 1.0)]}
    rates = {"AWAY": _team_rates()}
    with pytest.raises(KeyError):
        build_spec("HOME", "AWAY", rates, players, {})


def test_missing_team_in_players_raises():
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"AWAY": [_wr("awr1", "Away One", 1.0, 1.0)]}
    with pytest.raises(KeyError):
        build_spec("HOME", "AWAY", rates, players, {})


def test_all_players_dropped_leaves_shares_unchanged_uniform_guard():
    home_players = [_wr("wr1", "Alpha One", 1.0, 1.0)]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"HOME": [{"player": "Alpha One", "position": "WR", "status": "Out", "note": ""}]}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    assert spec.home_players == []
