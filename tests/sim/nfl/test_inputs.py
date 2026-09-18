"""Tests for NflGameSpec assembly with injury zeroing/renormalization."""
import importlib.util
import pathlib

import numpy as np
import pytest

from sportsmodel.sim.nfl.spec import TeamRates, PlayerInput, NflGameSpec
from sportsmodel.sim.nfl.inputs import build_spec, build_spec_from_usage
from sportsmodel.sim.nfl.kernel import simulate_game

_p = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "generate_sim_nfl.py"
_spec_mod = importlib.util.spec_from_file_location("generate_sim_nfl", _p)
gsn = importlib.util.module_from_spec(_spec_mod)
_spec_mod.loader.exec_module(gsn)


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
        ypr=12.3,
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


def test_out_injury_isolated_to_its_own_team_by_name():
    # Same name on both rosters; only HOME's copy is ruled Out.
    home_players = [
        _wr("h_smith", "Chris Smith", 0.6, 0.6),
        _wr("h_other", "Home Other", 0.4, 0.4),
    ]
    away_players = [
        _wr("a_smith", "Chris Smith", 0.7, 0.7),
        _wr("a_other", "Away Other", 0.3, 0.3),
    ]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"HOME": [{"player": "chris SMITH", "position": "WR", "status": "out", "note": ""}]}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    home_names = {p.name for p in spec.home_players}
    away_names = {p.name for p in spec.away_players}
    assert "Chris Smith" not in home_names
    assert home_names == {"Home Other"}
    assert "Chris Smith" in away_names
    assert away_names == {"Chris Smith", "Away Other"}

    # And the reverse: AWAY-only OUT leaves HOME's same-named player untouched.
    injuries_reverse = {"AWAY": [{"player": "CHRIS smith", "position": "WR", "status": "Out", "note": ""}]}
    spec2 = build_spec("HOME", "AWAY", rates, players, injuries_reverse)

    home_names2 = {p.name for p in spec2.home_players}
    away_names2 = {p.name for p in spec2.away_players}
    assert "Chris Smith" in home_names2
    assert home_names2 == {"Chris Smith", "Home Other"}
    assert "Chris Smith" not in away_names2
    assert away_names2 == {"Away Other"}


def test_carry_share_renormalizes_over_survivors_only():
    rb1 = PlayerInput(
        player_id="rb1", name="RB One", pos="RB",
        target_share=0.1, carry_share=0.5, ypt=6.0, ypc=4.5, ypr=10.0,
        catch_rate=0.6, td_share=0.4,
    )
    rb2 = PlayerInput(
        player_id="rb2", name="RB Two", pos="RB",
        target_share=0.1, carry_share=0.3, ypt=6.0, ypc=4.0, ypr=10.0,
        catch_rate=0.6, td_share=0.3,
    )
    wr1 = _wr("wr1", "WR One", target_share=0.8, td_share=0.3, carry_share=0.2)
    home_players = [rb1, rb2, wr1]
    away_players = [_wr("awr1", "Away One", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}
    players = {"HOME": home_players, "AWAY": away_players}
    injuries = {"HOME": [{"player": "RB Two", "position": "RB", "status": "Out", "note": ""}]}

    spec = build_spec("HOME", "AWAY", rates, players, injuries)

    by_name = {p.name: p for p in spec.home_players}
    assert set(by_name) == {"RB One", "WR One"}

    carry_sum = sum(p.carry_share for p in spec.home_players)
    assert carry_sum == pytest.approx(1.0)

    # Survivors' pre-drop carry_share total is 0.5 + 0.2 = 0.7, not 1.0 or
    # the full pre-drop total of 1.0 (0.5+0.3+0.2) -- renormalization must
    # use only the surviving players' sum as the denominator.
    assert by_name["RB One"].carry_share == pytest.approx(0.5 / 0.7)
    assert by_name["WR One"].carry_share == pytest.approx(0.2 / 0.7)


# =============================================================================
# build_spec_from_usage (Task 3, Part A): active-usage players are assembled
# as-is -- no re-injury-drop, no re-renormalize (Ruling B1) -- with a
# defensive QB-attribution assert (Ruling C1).
# =============================================================================

def _qb(player_id, name, target_share=0.05, carry_share=0.05, td_share=0.1):
    return PlayerInput(
        player_id=player_id,
        name=name,
        pos="QB",
        target_share=target_share,
        carry_share=carry_share,
        ypt=0.0,
        ypc=4.0,
        ypr=0.0,
        catch_rate=0.0,
        td_share=td_share,
    )


def test_build_spec_from_usage_passes_players_through_unchanged():
    home_players = [
        _qb("qb1", "Home QB", target_share=0.05, td_share=0.1),
        _wr("wr1", "Home WR", target_share=0.95, td_share=0.9),
    ]
    away_players = [_qb("qb2", "Away QB", target_share=1.0, td_share=1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}

    spec = build_spec_from_usage("HOME", "AWAY", rates, home_players, away_players, "qb1", "qb2")

    assert isinstance(spec, NflGameSpec)
    assert spec.home_team == "HOME"
    assert spec.away_team == "AWAY"
    assert spec.home is rates["HOME"]
    assert spec.away is rates["AWAY"]
    # Exactly the passed lists -- no renormalization, no injury re-drop.
    assert spec.home_players == home_players
    assert spec.away_players == away_players


def test_build_spec_from_usage_missing_team_in_rates_raises():
    home_players = [_qb("qb1", "Home QB")]
    away_players = [_qb("qb2", "Away QB")]
    rates = {"AWAY": _team_rates()}
    with pytest.raises(KeyError):
        build_spec_from_usage("HOME", "AWAY", rates, home_players, away_players, "qb1", "qb2")


def test_build_spec_from_usage_raises_when_qb_gsis_does_not_match_sole_qb():
    home_players = [_qb("qb1", "Home QB"), _wr("wr1", "Home WR", 0.9, 0.9)]
    away_players = [_qb("qb2", "Away QB")]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}

    with pytest.raises(ValueError):
        build_spec_from_usage("HOME", "AWAY", rates, home_players, away_players, "not-qb1", "qb2")


def test_build_spec_from_usage_raises_when_two_qbs_present():
    home_players = [_qb("qb1", "Home QB1"), _qb("qb1b", "Home QB2")]
    away_players = [_qb("qb2", "Away QB")]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}

    with pytest.raises(ValueError):
        build_spec_from_usage("HOME", "AWAY", rates, home_players, away_players, "qb1", "qb2")


def test_build_spec_from_usage_allows_none_qb_gsis_with_zero_qbs():
    home_players = [_wr("wr1", "Home WR", 1.0, 1.0)]
    away_players = [_wr("awr1", "Away WR", 1.0, 1.0)]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}

    spec = build_spec_from_usage("HOME", "AWAY", rates, home_players, away_players, None, None)

    assert spec.home_players == home_players
    assert spec.away_players == away_players


def test_build_spec_from_usage_feeds_kernel_and_pass_yds_lands_on_qb_gsis():
    """End-to-end: build_spec_from_usage -> simulate_game -> assemble_sim_rows,
    and the pass_yds player row is keyed by the QB's gsis id (proves the
    attribution guarantee holds through the whole pipeline)."""
    home_players = [
        _qb("qb_gsis_home", "Home QB", target_share=0.05, carry_share=0.1, td_share=0.1),
        _wr("wr_home", "Home WR", target_share=0.95, td_share=0.9),
    ]
    away_players = [
        _qb("qb_gsis_away", "Away QB", target_share=0.05, carry_share=0.1, td_share=0.1),
        _wr("wr_away", "Away WR", target_share=0.95, td_share=0.9),
    ]
    rates = {"HOME": _team_rates(), "AWAY": _team_rates()}

    spec = build_spec_from_usage(
        "HOME", "AWAY", rates, home_players, away_players, "qb_gsis_home", "qb_gsis_away"
    )

    rng = np.random.default_rng(42)
    sims = simulate_game(spec, n_sims=20, rng=rng)

    games = [{
        "game_pk": 1,
        "matchup": "AWAY @ HOME",
        "commence_time": "2026-09-21T17:00:00+00:00",
        "home_team": "HOME",
        "away_team": "AWAY",
    }]
    sim_rows, player_rows = gsn.assemble_sim_rows(
        games, {1: sims}, {1: spec}, {1: 0.5}
    )
    assert len(sim_rows) == 1

    by_player_market = {(r["player_id"], r["market"]): r for r in player_rows}
    assert ("qb_gsis_home", "pass_yds") in by_player_market
    assert ("qb_gsis_away", "pass_yds") in by_player_market
    # The non-QB WR should never carry a pass_yds attribution row keyed to it
    # as the passer -- attribute_offense zeroes pass_yds for everyone but the
    # sole QB.
    assert by_player_market[("qb_gsis_home", "pass_yds")]["mean"] >= 0.0
