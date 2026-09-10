"""_sport_config selects the right adapter, paths, and crosswalk per sport."""
import importlib.util
import pathlib

_p = pathlib.Path(__file__).parents[2] / "scripts" / "desk_inputs.py"
_spec = importlib.util.spec_from_file_location("desk_inputs", _p)
desk_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(desk_inputs)


def test_cfb_config():
    cfg = desk_inputs._sport_config("cfb")
    assert cfg["adapter"].__name__ == "sportsmodel.cfb.sportsdata"
    assert cfg["adapter"].INJURED_PLAYERS_PATH == "/scores/json/InjuredPlayers"
    assert cfg["schedules_path"].as_posix().endswith("assets/cfb/schedules.parquet")
    assert cfg["crosswalk_path"].as_posix().endswith("assets/cfb/fbs_teams.json")
    assert cfg["out_default"].as_posix().endswith("cfb/desk_bundle.json")


def test_nfl_config():
    cfg = desk_inputs._sport_config("nfl")
    assert cfg["adapter"].__name__ == "sportsmodel.nfl.sportsdata"
    assert cfg["adapter"].INJURED_PLAYERS_PATH == "/projections/json/InjuredPlayers"
    assert cfg["schedules_path"].as_posix().endswith("assets/nfl/schedules.parquet")
    assert cfg["crosswalk_path"].as_posix().endswith("assets/nfl/nfl_teams.json")
    assert cfg["out_default"].as_posix().endswith("nfl/desk_bundle.json")


def test_unknown_sport_raises():
    import pytest
    with pytest.raises((KeyError, ValueError, SystemExit)):
        desk_inputs._sport_config("mlb")
