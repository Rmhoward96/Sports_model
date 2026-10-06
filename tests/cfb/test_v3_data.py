import pandas as pd
import pytest

from sportsmodel.cfb import v3_data


def test_missing_required_asset_names_the_backfill(tmp_path):
    with pytest.raises(FileNotFoundError, match="CFBD_API_KEY"):
        v3_data.require_asset("advanced_games.parquet", tmp_path)
    assert v3_data.read_asset("talent.parquet", tmp_path) is None


def test_read_and_require_asset_roundtrip(tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "talent.parquet")
    assert list(v3_data.read_asset("talent.parquet", tmp_path)["a"]) == [1, 2]
    assert len(v3_data.require_asset("talent.parquet", tmp_path)) == 2


def write(tmp, name, df):
    df.to_parquet(tmp / name)


def test_merged_schedule_drops_self_matches_and_joins_lines(tmp_path):
    write(tmp_path, "schedules.parquet", pd.DataFrame({
        "season": [2023] * 3, "week": [1] * 3, "home_team": ["a", "b", "c"], "away_team": ["b", "a", "c"],
        "home_score": [1, 2, 3], "away_score": [0, 1, 2], "game_type": ["REG", "REG", "REG"],
        "game_pk": [1, 2, 3]}))
    write(tmp_path, "lines.parquet", pd.DataFrame({
        "season": [2023], "week": [1], "home_team": ["a"], "away_team": ["b"], "market_spread": [-3.0],
        "market_total": [50.0]}))
    m = v3_data.load_merged_schedule(tmp_path)
    assert list(m["game_pk"]) == [1, 2]                              # c-vs-c self match dropped
    assert m.loc[m.game_pk == 1, "market_spread"].iloc[0] == -3.0
    assert pd.isna(m.loc[m.game_pk == 2, "market_spread"].iloc[0])


def test_missing_optional_assets_and_required_loader(tmp_path):
    with pytest.raises(FileNotFoundError, match="CFBD_API_KEY"):
        v3_data.load_eff_games(tmp_path)
    assert v3_data.read_asset("talent.parquet", tmp_path) is None
    assert v3_data.load_priors_rows(tmp_path) == {}


def test_load_eff_games_works_without_optional_assets(tmp_path):
    write(tmp_path, "advanced_games.parquet", pd.DataFrame({
        "season": [2023, 2023], "week": [1, 1], "season_type": ["regular"] * 2, "game_id": [1, 1],
        "team": ["a", "b"], "opponent": ["b", "a"], "off_plays": [70.0, 60.0], "off_ppa": [0.2, 0.1],
        "off_success": [0.5, 0.4], "off_explosiveness": [1.2, 1.1], "off_pass_ppa": [0.2, 0.1],
        "off_rush_ppa": [0.1, 0.0], "off_pass_plays": [40.0, 30.0], "off_rush_plays": [30.0, 30.0]}))
    write(tmp_path, "cfbd_games.parquet", pd.DataFrame({"game_id": [1], "home_team": ["a"],
                                                         "neutral_site": [False]}))
    write(tmp_path, "schedules.parquet", pd.DataFrame({"game_pk": [1], "week": [1], "game_type": ["REG"]}))
    g = v3_data.load_eff_games(tmp_path)
    assert len(g) == 2 and g["y_havoc"].isna().all() and g["y_ppo"].isna().all()
    assert list(g["home"]) == [1.0, -1.0]


def test_load_v3_inputs_reads_configs_and_tolerates_missing_optional_assets(tmp_path):
    import json
    write(tmp_path, "advanced_games.parquet", pd.DataFrame({
        "season": [2023, 2023], "week": [1, 1], "season_type": ["regular"] * 2, "game_id": [1, 1],
        "team": ["a", "b"], "opponent": ["b", "a"], "off_plays": [70.0, 60.0], "off_ppa": [0.2, 0.1],
        "off_success": [0.5, 0.4], "off_explosiveness": [1.2, 1.1], "off_pass_ppa": [0.2, 0.1],
        "off_rush_ppa": [0.1, 0.0], "off_pass_plays": [40.0, 30.0], "off_rush_plays": [30.0, 30.0]}))
    write(tmp_path, "cfbd_games.parquet", pd.DataFrame({
        "game_id": [1], "season": [2023], "venue_id": [7.0], "home_team": ["a"], "neutral_site": [False]}))
    write(tmp_path, "schedules.parquet", pd.DataFrame({"game_pk": [1], "week": [1], "game_type": ["REG"]}))
    (tmp_path / "rating.json").write_text(json.dumps({"k": 40, "hfa_elo": 70, "carryover": 0.9,
                                                      "w_sos": 0.45, "srs_min_games": 3}))
    inp = v3_data.load_v3_inputs(tmp_path)                    # no weights/decay/eff json: documented defaults
    assert inp.elo_cfg.k == 40 and inp.blend_cfg.srs_min_games == 3
    assert inp.eff_cfg.ridge == 4.0 and inp.decay.half_life_games > 0
    assert inp.talent is None and inp.priors_rows == {} and len(inp.eff_games) == 2
    assert inp.ctx.game_venue == {1: 7}


def test_load_v3_inputs_is_strict_about_prior_weights_when_priors_exist(tmp_path):
    import json
    write(tmp_path, "priors.parquet", pd.DataFrame({"season": [2023], "team_espn_id": ["a"]}))
    (tmp_path / "rating.json").write_text(json.dumps({"k": 40, "hfa_elo": 70, "carryover": 0.9,
                                                      "w_sos": 0.45, "srs_min_games": 3}))
    with pytest.raises(FileNotFoundError, match="priors_weights.json"):
        v3_data.load_v3_inputs(tmp_path)


def test_load_v3_inputs_weather_override_replaces_the_weather_asset(tmp_path):
    import json
    write(tmp_path, "advanced_games.parquet", pd.DataFrame({
        "season": [2023, 2023], "week": [1, 1], "season_type": ["regular"] * 2, "game_id": [1, 1],
        "team": ["a", "b"], "opponent": ["b", "a"], "off_plays": [70.0, 60.0], "off_ppa": [0.2, 0.1],
        "off_success": [0.5, 0.4], "off_explosiveness": [1.2, 1.1], "off_pass_ppa": [0.2, 0.1],
        "off_rush_ppa": [0.1, 0.0], "off_pass_plays": [40.0, 30.0], "off_rush_plays": [30.0, 30.0]}))
    write(tmp_path, "cfbd_games.parquet", pd.DataFrame({
        "game_id": [1], "season": [2023], "venue_id": [7.0], "home_team": ["a"], "neutral_site": [False]}))
    write(tmp_path, "schedules.parquet", pd.DataFrame({"game_pk": [1], "week": [1], "game_type": ["REG"]}))
    write(tmp_path, "weather_games.parquet", pd.DataFrame({"game_id": [1], "wind_speed": [3.0]}))
    (tmp_path / "rating.json").write_text(json.dumps({"k": 40, "hfa_elo": 70, "carryover": 0.9,
                                                      "w_sos": 0.45, "srs_min_games": 3}))
    assert v3_data.load_v3_inputs(tmp_path).ctx.weather[1]["wind_speed"] == 3.0
    live = pd.DataFrame({"game_id": [1], "wind_speed": [21.0]})
    assert v3_data.load_v3_inputs(tmp_path, weather=live).ctx.weather[1]["wind_speed"] == 21.0
