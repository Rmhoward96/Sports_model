import math

import pandas as pd
import pytest

from sportsmodel.cfb import context as cx

NYC, LA = (40.7128, -74.0060), (34.0522, -118.2437)


def test_haversine_known_distance():
    assert cx.haversine_miles(*NYC, *LA) == pytest.approx(2445, abs=10)
    assert cx.haversine_miles(*NYC, *NYC) == 0.0


def test_utc_offset_is_dst_aware_and_unknown_is_nan():
    sept, jan = "2024-09-14T16:00Z", "2024-01-08T16:00Z"
    assert cx.utc_offset_hours("America/New_York", sept) == -4.0
    assert cx.utc_offset_hours("America/New_York", jan) == -5.0
    assert cx.utc_offset_hours("America/Los_Angeles", sept) == -7.0
    assert math.isnan(cx.utc_offset_hours("Not/AZone", sept)) and math.isnan(cx.utc_offset_hours("", sept))


# ----------------------------------------------------------------- weather --

def test_weather_thresholds_and_zero_fill():
    f = cx.weather_features({"wind_speed": 22.0, "temperature": 31.0, "precipitation": 0.2,
                             "game_indoors": False}, None)
    assert f == {"wind_excess": 7.0, "cold_excess": 9.0, "precip": 0.2, "weather_missing": 0.0}
    calm = cx.weather_features({"wind_speed": 5.0, "temperature": 75.0, "precipitation": 0.0}, None)
    assert calm["wind_excess"] == 0.0 and calm["cold_excess"] == 0.0 and calm["weather_missing"] == 0.0


def test_dome_means_no_weather_effect_even_with_bad_readings():
    row = {"wind_speed": 30.0, "temperature": 10.0, "precipitation": 1.0, "game_indoors": True}
    assert cx.weather_features(row, None) == {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0,
                                              "weather_missing": 0.0}
    assert cx.weather_features({"wind_speed": 30.0}, {"dome": True})["wind_excess"] == 0.0


def test_missing_weather_is_flagged_not_fabricated():
    for row in (None, {}, {"wind_speed": float("nan"), "temperature": None, "precipitation": float("nan")}):
        f = cx.weather_features(row, None)
        assert f == {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0, "weather_missing": 1.0}
    part = cx.weather_features({"wind_speed": 20.0, "temperature": float("nan")}, None)
    assert part["wind_excess"] == 5.0 and part["cold_excess"] == 0.0 and part["weather_missing"] == 0.0


# ------------------------------------------------------------ travel / tz --

def assets_for_travel():
    venues = pd.DataFrame({"venue_id": [1, 2, 3], "name": ["ny", "la", "ny2"],
                           "timezone": ["America/New_York", "America/Los_Angeles", "America/New_York"],
                           "latitude": [NYC[0], LA[0], NYC[0] + 0.1], "longitude": [NYC[1], LA[1], NYC[1]],
                           "elevation": [0.0] * 3, "dome": [0.0, 0.0, float("nan")]})
    meta = pd.DataFrame({"game_id": [10, 11], "season": [2023, 2023], "venue_id": [1, 2],
                         "home_team": ["EAST", "WEST"], "neutral_site": [False, False]})
    return cx.build_context_assets(venues, meta, None, None)


def test_home_venue_comes_from_non_neutral_home_games():
    a = assets_for_travel()
    assert cx.home_venue_of(a, "EAST", 2023) == 1 and cx.home_venue_of(a, "EAST", 2025) == 1
    assert cx.home_venue_of(a, "EAST", 2022) is None and cx.home_venue_of(a, "NOBODY", 2023) is None


def test_travel_diff_sign_away_team_burden_is_positive_for_home():
    a = assets_for_travel()
    f = cx.travel_features(a, 2023, "EAST", "WEST", 1, "2023-09-16T16:00Z")   # west team flies to the east
    assert f["travel_far_diff"] == 1.0 and f["travel_mid_diff"] == 0.0
    assert f["tz_east_diff"] == 1.0 and f["tz_west_diff"] == 0.0              # traveled 3h east
    g = cx.travel_features(a, 2023, "WEST", "EAST", 2, "2023-09-16T16:00Z")   # east team flies west
    assert g["tz_west_diff"] == 1.0 and g["tz_east_diff"] == 0.0


def test_neutral_site_burdens_both_teams_and_unknowns_are_zero():
    a = assets_for_travel()
    f = cx.travel_features(a, 2023, "EAST", "WEST", 3, "2023-09-16T16:00Z")   # neutral venue 3 (near NYC)
    assert f["travel_far_diff"] == 1.0                                        # WEST far, EAST ~7 miles
    assert f["travel_mid_diff"] == 0.0
    assert cx.travel_features(a, 2023, "EAST", "WEST", None, "")["travel_far_diff"] == 0.0
    assert cx.travel_features(a, 2023, "EAST", "GHOST", 1, "2023-09-16T16:00Z")["travel_far_diff"] == 0.0


# ------------------------------------------------------------ rest / talent --

def test_rest_features_signs():
    assert cx.rest_features(10, 5) == {"short_week_diff": 1.0, "bye_diff": 0.0}    # away on a short week helps home
    assert cx.rest_features(5, 14) == {"short_week_diff": -1.0, "bye_diff": 1.0}
    assert cx.rest_features(float("nan"), 7) == {"short_week_diff": 0.0, "bye_diff": 0.0}


def test_rest_table_days_since_previous_game_and_opener_nan():
    frame = pd.DataFrame({"season": [2023] * 3, "game_pk": [1, 2, 3],
                          "start_date": ["2023-09-02T17:00Z", "2023-09-09T17:00Z", "2023-09-21T23:30Z"],
                          "home_team": ["a", "b", "a"], "away_team": ["c", "a", "b"]})
    r = cx.rest_table(frame)
    assert math.isnan(r[(1, "a")]) and r[(2, "a")] == 7.0 and r[(3, "a")] == 12.0
    assert math.isnan(r[(2, "b")]) and r[(3, "b")] == 12.0


def test_talent_gap_fades_with_games_and_z_scores_per_season():
    talent = pd.DataFrame({"season": [2023, 2023, 2023], "team": ["a", "b", "c"], "talent": [900.0, 700.0, 500.0]})
    a = cx.build_context_assets(None, None, None, talent)
    z = a.talent_z
    assert z[(2023, "a")] == pytest.approx(-z[(2023, "c")]) and z[(2023, "b")] == pytest.approx(0.0)
    g0 = cx.talent_gap(a, 2023, "a", "c", 0, 0)
    assert g0 == pytest.approx(z[(2023, "a")] - z[(2023, "c")])
    assert cx.talent_gap(a, 2023, "a", "c", 3, 3) == pytest.approx(0.5 * g0)
    assert cx.talent_gap(a, 2023, "a", "ghost", 0, 0) == 0.0


# --------------------------------------------------------------- assembled --

def test_context_features_assembles_every_column_and_never_nan():
    a = assets_for_travel()
    game = {"season": 2023, "game_pk": 77, "home_team": "EAST", "away_team": "WEST",
            "start_date": "2023-09-16T16:00Z", "neutral_site": False}
    f = cx.context_features(game, a, {(77, "EAST"): 6.0, (77, "WEST"): 14.0}, 2, 2)
    assert set(f) == set(cx.CONTEXT_COLS)
    assert f["travel_far_diff"] == 1.0 and f["short_week_diff"] == -1.0 and f["bye_diff"] == 1.0
    assert f["weather_missing"] == 1.0 and not any(math.isnan(v) for v in f.values())


# ---------------------------------------- ruling T1: missing venue metadata --

SEPT = "2023-09-16T16:00Z"


def _frames(venue_rows):
    venues = pd.DataFrame(venue_rows, columns=["venue_id", "name", "timezone", "latitude", "longitude",
                                               "elevation", "dome"])
    meta = pd.DataFrame({"game_id": [10, 11], "season": [2023, 2023], "venue_id": [1, 2],
                         "home_team": ["EAST", "WEST"], "neutral_site": [False, False]})
    return cx.build_context_assets(venues, meta, None, None)


def test_travel_uses_longitude_zone_for_venue_with_missing_timezone():
    nan = float("nan")
    a = _frames([(1, "ny", "America/New_York", NYC[0], NYC[1], nan, 0.0),
                 (2, "la", None, LA[0], LA[1], nan, 0.0)])                    # LA timezone is null
    f = cx.travel_features(a, 2023, "EAST", "WEST", 1, SEPT)                  # west team flies east: -8 -> -4
    assert f["travel_far_diff"] == 1.0 and f["tz_east_diff"] == 1.0 and f["tz_west_diff"] == 0.0
    g = cx.travel_features(a, 2023, "WEST", "EAST", 2, SEPT)                  # east team flies west
    assert g["tz_west_diff"] == 1.0 and g["tz_east_diff"] == 0.0


def test_missing_coordinates_zero_travel_and_time_zone_even_with_known_timezone():
    nan = float("nan")
    a = _frames([(1, "ny", "America/New_York", NYC[0], NYC[1], nan, 0.0),
                 (2, "la", "America/Los_Angeles", nan, nan, nan, 0.0)])        # tz known, coords null
    f = cx.travel_features(a, 2023, "EAST", "WEST", 1, SEPT)
    assert f == {"travel_mid_diff": 0.0, "travel_far_diff": 0.0, "tz_east_diff": 0.0, "tz_west_diff": 0.0}
    only_lat = _frames([(1, "ny", "America/New_York", NYC[0], NYC[1], nan, 0.0),
                        (2, "la", "America/Los_Angeles", LA[0], nan, nan, 0.0)])
    assert cx.travel_features(only_lat, 2023, "EAST", "WEST", 1, SEPT)["travel_far_diff"] == 0.0


def test_missing_timezone_and_longitude_zero_time_zone_but_keep_distance():
    nan = float("nan")
    a = _frames([(1, "ny", "America/New_York", NYC[0], NYC[1], nan, 0.0),
                 (2, "la", "", LA[0], nan, nan, 0.0)])                          # lat known, lon + tz null
    f = cx.travel_features(a, 2023, "EAST", "WEST", 1, SEPT)
    assert f["tz_east_diff"] == 0.0 and f["tz_west_diff"] == 0.0


def test_unknown_dome_and_elevation_never_fabricate_a_weather_effect():
    nan = float("nan")
    a = _frames([(1, "ny", "America/New_York", NYC[0], NYC[1], nan, nan),     # dome NaN, elevation NaN
                 (2, "la", "", nan, nan, nan, 1.0)])                           # dome known
    assert a.venues[1]["dome"] is False and a.venues[2]["dome"] is True
    assert cx.weather_features({"wind_speed": 25.0}, a.venues[1])["wind_excess"] == 10.0
    assert cx.weather_features({"wind_speed": 25.0}, a.venues[2])["wind_excess"] == 0.0


# ------------------------- ruling T2: never compare DST-aware with standard-time --

OCT = "2023-10-14T16:00Z"
CHI = (41.88, -87.63)          # Central-longitude zone: round(-87.63 / 15) = -6


def test_utc_shift_uses_longitude_zone_for_both_when_either_timezone_missing():
    east = {"tz": "America/New_York", "lon": NYC[1]}
    central_fallback = {"tz": "", "lon": CHI[1]}
    # lon zones: -6 -> -5 = 1h. Mixing DST-aware EDT (-4) with standard -6 would wrongly read 2h.
    assert cx.utc_shift_hours(central_fallback, east, OCT) == 1.0     # Chicago -> NY: 1h east
    assert cx.utc_shift_hours(east, central_fallback, OCT) == -1.0


def test_utc_shift_stays_dst_aware_when_both_timezones_known():
    ny = {"tz": "America/New_York", "lon": NYC[1]}
    chi = {"tz": "America/Chicago", "lon": CHI[1]}
    assert cx.utc_shift_hours(chi, ny, OCT) == 1.0                    # EDT -4 vs CDT -5
    la = {"tz": "America/Los_Angeles", "lon": LA[1]}
    assert cx.utc_shift_hours(la, ny, OCT) == 3.0
    # longitude is irrelevant (and may be missing) when both zones are known
    assert cx.utc_shift_hours({"tz": "America/Chicago", "lon": float("nan")}, ny, OCT) == 1.0


def test_utc_shift_is_nan_when_a_needed_longitude_is_missing():
    ny = {"tz": "America/New_York", "lon": NYC[1]}
    assert math.isnan(cx.utc_shift_hours(ny, {"tz": "", "lon": float("nan")}, OCT))
    assert math.isnan(cx.utc_shift_hours({"tz": "", "lon": None}, ny, OCT))


def test_travel_october_known_eastern_vs_central_fallback_is_one_hour_not_two():
    nan = float("nan")
    a = _frames([(1, "ny", "America/New_York", NYC[0], NYC[1], nan, 0.0),
                 (2, "chi", "", CHI[0], CHI[1], nan, 0.0)])
    f = cx.travel_features(a, 2023, "EAST", "WEST", 1, OCT)           # WEST flies Chicago -> NY: 1h east
    assert f["tz_east_diff"] == 0.0 and f["tz_west_diff"] == 0.0 and f["travel_far_diff"] == 0.0
    assert f["travel_mid_diff"] == 1.0                                # ~710 mi: distance unaffected


def test_travel_missing_longitude_with_missing_timezone_gives_zero_features():
    nan = float("nan")
    a = _frames([(1, "ny", "America/New_York", NYC[0], NYC[1], nan, 0.0),
                 (2, "la", "", LA[0], nan, nan, 0.0)])
    assert cx.travel_features(a, 2023, "EAST", "WEST", 1, OCT) == {
        "travel_mid_diff": 0.0, "travel_far_diff": 0.0, "tz_east_diff": 0.0, "tz_west_diff": 0.0}


# ------------------------------- rest days are US Eastern calendar days, not UTC --

def _rest(*kickoffs_utc):
    """rest_table for team 'a' playing at each UTC kickoff (opponents are throwaway)."""
    frame = pd.DataFrame({"season": [2023] * len(kickoffs_utc), "game_pk": range(1, len(kickoffs_utc) + 1),
                          "start_date": list(kickoffs_utc),
                          "home_team": ["a"] * len(kickoffs_utc),
                          "away_team": [f"opp{i}" for i in range(len(kickoffs_utc))]})
    r = cx.rest_table(frame)
    return [r[(i, "a")] for i in range(1, len(kickoffs_utc) + 1)]


def test_late_saturday_kickoff_to_next_saturday_noon_is_seven_days_not_a_short_week():
    # Sat 2023-09-09 10:30pm EDT = Sun 02:30Z; next Sat 2023-09-16 12:00 EDT = 16:00Z (UTC dates: 6 apart)
    r = _rest("2023-09-10T02:30Z", "2023-09-16T16:00Z")[1]
    assert r == 7.0
    assert cx.rest_features(r, 7.0) == {"short_week_diff": 0.0, "bye_diff": 0.0}


def test_real_thursday_to_saturday_gap_stays_short():
    # Thu 2023-09-14 8pm EDT (= Fri 00:00Z) -> Sat 2023-09-16 noon EDT: 2 days
    r = _rest("2023-09-15T00:00Z", "2023-09-16T16:00Z")[1]
    assert r == 2.0 and cx.rest_features(r, 7.0)["short_week_diff"] == -1.0


def test_bye_threshold_uses_eastern_dates_across_a_late_kickoff():
    # 13 ET days (bye) that UTC dates would call 12: Sat 9/9 10:30pm EDT -> Fri 9/22 7pm EDT (= 23:00Z)
    r13 = _rest("2023-09-10T02:30Z", "2023-09-22T23:00Z")[1]
    assert r13 == 13.0 and cx.rest_features(r13, 7.0)["bye_diff"] == -1.0
    # 12 ET days (no bye) that UTC dates would call 13: Sat 9/9 noon EDT -> Thu 9/21 9pm EDT (= Fri 01:00Z)
    r12 = _rest("2023-09-09T16:00Z", "2023-09-22T01:00Z")[1]
    assert r12 == 12.0 and cx.rest_features(r12, 7.0)["bye_diff"] == 0.0
    # 14 ET days stays a bye
    assert _rest("2023-09-09T16:00Z", "2023-09-23T16:00Z")[1] == 14.0


def test_rest_days_are_exact_across_the_november_dst_change():
    # Sat 2023-11-04 noon EDT -> Sat 2023-11-11 noon EST (clocks fell back on 11-05): still 7 days
    assert _rest("2023-11-04T16:00Z", "2023-11-11T17:00Z")[1] == 7.0
