import numpy as np
import pandas as pd
import pytest

from sportsmodel.nfl.context import haversine_km, load_stadiums, team_game_context

STAD = {"KAN00": {"lat": 39.049, "lon": -94.484, "tz": "America/Chicago", "roof": "outdoors"},
        "LAX01": {"lat": 33.953, "lon": -118.339, "tz": "America/Los_Angeles", "roof": "dome"},
        "LON02": {"lat": 51.604, "lon": -0.066, "tz": "Europe/London", "roof": "retractable"}}

STAD_WITH_IND = {
    **STAD,
    "IND00": {"lat": 39.760, "lon": -86.164, "tz": "America/Indiana/Indianapolis", "roof": "retractable"}
}


def _games():
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"], "season": [2024] * 3, "week": [1, 2, 3], "game_type": ["REG"] * 3,
        "gameday": ["2024-09-08", "2024-09-15", "2024-09-22"], "gametime": ["16:25", "13:00", "09:30"],
        "home_team": ["KC", "LAC", "KC"], "away_team": ["LAC", "KC", "LAC"],
        "home_rest": [7, 7, 7], "away_rest": [7, 7, 14], "roof": ["outdoors", "dome", ""],
        "surface": ["grass", "matrixturf", "grass"], "temp": [80.0, None, None], "wind": [9.0, None, None],
        "div_game": [1, 1, 1], "stadium_id": ["KAN00", "LAX01", "LON02"],
        "spread_line": [3.0, -2.5, 6.0], "total_line": [48.0, 45.0, 47.5],
    })


def _games_with_nans():
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4", "g5"],
        "season": [2024] * 5,
        "week": [1, 2, 3, 4, 5],
        "game_type": ["REG"] * 5,
        "gameday": ["2024-09-08", "2024-09-15", "2024-09-22", "2024-09-29", "2024-10-06"],
        "gametime": ["16:25", "13:00", "09:30", np.nan, "13:00"],  # NaN gametime for g4
        "home_team": ["KC", "LAC", "KC", "LAC", "KC"],
        "away_team": ["LAC", "KC", "LAC", "IND", "LAC"],
        "home_rest": [7, 7, 7, 7, 7],
        "away_rest": [7, 7, 14, 7, 7],
        "roof": ["outdoors", "dome", np.nan, "retractable", "outdoors"],  # NaN roof for g3
        "surface": ["grass", "matrixturf", np.nan, "", "grass"],  # NaN and blank surface for g3, g4
        "temp": [80.0, None, None, None, None],
        "wind": [9.0, None, None, None, None],
        "div_game": [1, 1, 1, 0, 1],
        "stadium_id": ["KAN00", "LAX01", "LON02", "IND00", np.nan],  # NaN stadium_id for g5
        "spread_line": [3.0, -2.5, 6.0, -4.0, 2.0],
        "total_line": [48.0, 45.0, 47.5, 44.0, 46.0],
    })


def test_haversine_known_distance():
    assert haversine_km(39.049, -94.484, 33.953, -118.339) == pytest.approx(2200, rel=0.03)


def test_all_stadium_ids_have_coords():
    st = load_stadiums()
    assert len(st) == 47 and all({"lat", "lon", "tz", "roof"} <= set(v) for v in st.values())


def test_context_rows_and_features():
    cx = team_game_context(_games(), STAD).set_index(["week", "team"])
    assert len(cx) == 6
    # home game: no travel; LAC at KC travels ~2200 km, crosses 2 time zones
    assert cx.loc[(1, "KC"), "cx_travel_km"] == 0.0
    assert cx.loc[(1, "LAC"), "cx_travel_km"] == pytest.approx(2200, rel=0.03)
    assert cx.loc[(1, "LAC"), "cx_tz_shift"] == 2.0
    # dome: indoor fill (70F, 0 mph); retractable w/ blank roof: unknown -> NaN weather
    assert cx.loc[(2, "KC"), "cx_indoor"] == 1 and cx.loc[(2, "KC"), "cx_temp"] == 70.0 and cx.loc[(2, "KC"), "cx_wind"] == 0.0
    assert pd.isna(cx.loc[(3, "KC"), "cx_indoor"]) and pd.isna(cx.loc[(3, "KC"), "cx_wind"])
    # rest + bye
    assert cx.loc[(3, "LAC"), "cx_off_bye"] == 1 and cx.loc[(3, "LAC"), "cx_rest_diff"] == 7
    # market: home favored by 3, total 48 -> KC implied 25.5, LAC 22.5
    assert cx.loc[(1, "KC"), "mk_spread"] == 3.0 and cx.loc[(1, "LAC"), "mk_spread"] == -3.0
    assert cx.loc[(1, "KC"), "mk_implied"] == 25.5 and cx.loc[(1, "LAC"), "mk_implied"] == 22.5
    # turf flag
    assert cx.loc[(2, "LAC"), "cx_turf"] == 1 and cx.loc[(1, "KC"), "cx_turf"] == 0


def test_context_with_nans_and_eastern_tz():
    """Test handling of NaN values (roof, surface, gametime, stadium_id) and Indianapolis Eastern timezone."""
    cx = team_game_context(_games_with_nans(), STAD_WITH_IND).set_index(["week", "team"])

    # NaN roof in week 3: retractable stadium so indoor=NaN, weather=NaN
    assert pd.isna(cx.loc[(3, "KC"), "cx_indoor"])
    assert pd.isna(cx.loc[(3, "KC"), "cx_temp"])

    # NaN surface in week 3: cx_turf should be NaN
    assert pd.isna(cx.loc[(3, "KC"), "cx_turf"])
    assert pd.isna(cx.loc[(3, "LAC"), "cx_turf"])

    # Blank surface in week 4: cx_turf should be NaN
    assert pd.isna(cx.loc[(4, "LAC"), "cx_turf"])
    assert pd.isna(cx.loc[(4, "IND"), "cx_turf"])

    # NaN gametime defaults to 13:00 (kick_h=13): LAC (Pacific home) at IND00 (Eastern) -> cx_west_early=1
    assert cx.loc[(4, "LAC"), "cx_west_early"] == 1

    # NaN stadium_id in week 5: travel and tz_shift become NaN (no exception raised)
    assert pd.isna(cx.loc[(5, "KC"), "cx_travel_km"])
    assert pd.isna(cx.loc[(5, "KC"), "cx_tz_shift"])
