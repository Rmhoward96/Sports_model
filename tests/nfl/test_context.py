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


def test_short_week_and_off_bye_are_nan_when_rest_is_unknown():
    g = _games().assign(home_rest=[7, 4, np.nan], away_rest=[np.nan, 13, 7])
    cx = team_game_context(g, STAD).set_index(["week", "team"])
    for key in ((1, "LAC"), (3, "KC")):                      # rest NaN -> flags NaN, not 0.0
        assert pd.isna(cx.loc[key, "cx_rest"])
        assert pd.isna(cx.loc[key, "cx_short_week"]) and pd.isna(cx.loc[key, "cx_off_bye"])
    assert cx.loc[(2, "LAC"), "cx_short_week"] == 1 and cx.loc[(2, "LAC"), "cx_off_bye"] == 0
    assert cx.loc[(2, "KC"), "cx_short_week"] == 0 and cx.loc[(2, "KC"), "cx_off_bye"] == 1
    assert cx.loc[(1, "KC"), "cx_short_week"] == 0 and cx.loc[(1, "KC"), "cx_off_bye"] == 0


# --- live kickoff forecast (Open-Meteo hourly, requested in degF / mph, UTC) ---

def _hourly(times, temps, winds):
    return {"hourly": {"time": times, "temperature_2m": temps, "wind_speed_10m": winds}}


def test_parse_kickoff_forecast_picks_nearest_hour_and_converts_nothing():
    from sportsmodel.nfl.context import parse_kickoff_forecast
    p = _hourly(["2026-09-27T16:00", "2026-09-27T17:00", "2026-09-27T18:00"],
                [60.1, 62.5, 64.0], [5.0, 7.5, 9.0])
    assert parse_kickoff_forecast(p, pd.Timestamp("2026-09-27T17:25Z")) == (62.5, 7.5)
    assert parse_kickoff_forecast(p, pd.Timestamp("2026-09-27T17:40Z")) == (64.0, 9.0)
    assert parse_kickoff_forecast(p, pd.Timestamp("2026-09-27T15:50Z")) == (60.1, 5.0)


def test_parse_kickoff_forecast_missing_is_nan():
    from sportsmodel.nfl.context import parse_kickoff_forecast
    ko = pd.Timestamp("2026-09-27T17:00Z")
    for payload in ({}, {"hourly": {}}, _hourly([], [], []),
                    _hourly(["2026-09-27T17:00"], [None], [None]),
                    _hourly(["2026-09-29T17:00"], [70.0], [4.0])):   # nowhere near kickoff
        temp, wind = parse_kickoff_forecast(payload, ko)
        assert pd.isna(temp) and pd.isna(wind)
    temp, wind = parse_kickoff_forecast(_hourly(["2026-09-27T17:00"], [66.0], [None]), ko)
    assert temp == 66.0 and pd.isna(wind)


def _live_games():
    # now = 2026-09-24 12:00Z. Gametimes are US-Eastern (13:00 EDT = 17:00Z).
    return pd.DataFrame({
        "game_id": ["past", "near", "dome", "far"], "season": [2026] * 4, "week": [2, 3, 3, 4],
        "game_type": ["REG"] * 4,
        "gameday": ["2026-09-20", "2026-09-27", "2026-09-27", "2026-10-04"],
        "gametime": ["13:00", "13:00", "16:25", "13:00"],
        "home_team": ["KC", "KC", "LAC", "KC"], "away_team": ["DEN", "BUF", "LV", "LV"],
        "home_rest": [7] * 4, "away_rest": [7] * 4, "roof": ["outdoors", "", "", ""],
        "surface": ["grass"] * 4, "temp": [75.0, None, None, None], "wind": [8.0, None, None, None],
        "div_game": [1, 0, 1, 1], "stadium_id": ["KAN00", "KAN00", "LAX01", "KAN00"],
        "spread_line": [3.0] * 4, "total_line": [45.0] * 4,
    })


def test_fill_forecast_weather_fills_only_near_outdoor_nan_rows():
    from sportsmodel.nfl.context import fill_forecast_weather
    calls = []

    def fake_fetch(lat, lon):
        calls.append((lat, lon))
        return _hourly(["2026-09-27T16:00", "2026-09-27T17:00", "2026-09-27T18:00"],
                       [55.0, 58.0, 61.0], [12.0, 14.0, 16.0])

    games = _live_games()
    ctx = team_game_context(games, STAD)
    before = ctx.copy()
    out = fill_forecast_weather(ctx, STAD, games, fake_fetch, now=pd.Timestamp("2026-09-24T12:00Z"))
    pd.testing.assert_frame_equal(ctx, before)                     # input not mutated
    assert calls == [(STAD["KAN00"]["lat"], STAD["KAN00"]["lon"])]  # only the near outdoor game
    o = out.set_index(["week", "team"])
    for team in ("KC", "BUF"):                                     # kickoff 17:00Z
        assert o.loc[(3, team), "cx_temp"] == 58.0 and o.loc[(3, team), "cx_wind"] == 14.0
    assert o.loc[(2, "KC"), "cx_temp"] == 75.0 and o.loc[(2, "KC"), "cx_wind"] == 8.0   # past: recorded
    assert o.loc[(3, "LAC"), "cx_temp"] == 70.0 and o.loc[(3, "LAC"), "cx_wind"] == 0.0  # dome untouched
    assert pd.isna(o.loc[(4, "KC"), "cx_temp"]) and pd.isna(o.loc[(4, "LV"), "cx_wind"])  # > 7 days out
    other = [c for c in out.columns if c not in ("cx_temp", "cx_wind")]
    pd.testing.assert_frame_equal(out[other], ctx[other])


def test_fill_forecast_weather_missing_payload_leaves_nan():
    from sportsmodel.nfl.context import fill_forecast_weather
    games = _live_games()
    out = fill_forecast_weather(team_game_context(games, STAD), STAD, games, lambda lat, lon: None,
                                now=pd.Timestamp("2026-09-24T12:00Z")).set_index(["week", "team"])
    assert pd.isna(out.loc[(3, "KC"), "cx_temp"]) and pd.isna(out.loc[(3, "BUF"), "cx_wind"])
