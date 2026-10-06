import numpy as np
import pandas as pd

from sportsmodel.cfb import data_checks as dc


def weather(temp=62.0, wind=7.0, precip=0.0, indoors=False, season=2023):
    return pd.DataFrame({"season": [season] * 4, "game_id": range(4), "game_indoors": [indoors] * 4,
                         "temperature": [temp] * 4, "wind_speed": [wind] * 4, "precipitation": [precip] * 4})


def test_weather_units_flagged_when_not_fahrenheit_mph_inches():
    assert dc.check_weather(weather()) == []
    assert any("degrees F" in p for p in dc.check_weather(weather(temp=17.0)))          # Celsius
    assert any("mph" in p for p in dc.check_weather(weather(wind=2.5 * 0.44)))          # m/s-ish
    assert any("precipitation" in p for p in dc.check_weather(weather(precip=80.0)))
    assert dc.check_weather(weather(temp=17.0, indoors=True)) == []                    # domes are ignored
    assert dc.check_weather(weather().iloc[0:0]) == ["weather: no rows"]


def havoc(pair_gap=0.0):
    rows = []
    for gid in (1, 2, 3):
        rows.append({"season": 2023, "game_id": gid, "team": "a", "opponent": "b", "off_havoc": 0.12,
                     "def_havoc": 0.14})
        rows.append({"season": 2023, "game_id": gid, "team": "b", "opponent": "a", "off_havoc": 0.14 + pair_gap,
                     "def_havoc": 0.12})
    return pd.DataFrame(rows)


def test_havoc_sides_must_pair_up_across_the_two_rows_of_a_game():
    assert dc.check_havoc(havoc()) == []
    bad = dc.check_havoc(havoc(pair_gap=0.08))
    assert any("other way round" in p for p in bad)
    out_of_range = havoc().assign(off_havoc=0.9, def_havoc=0.9)
    assert any("outside" in p for p in dc.check_havoc(out_of_range))


def drives(points=(10.0, 7.0)):
    base = {"season": 2023, "game_id": 1, "off_start_yd": 30.0}
    a = {**base, "team": "a", "opponent": "b", "off_ppd": 2.0, "off_points": points[0], "def_points": points[1]}
    b = {**base, "team": "b", "opponent": "a", "off_ppd": 1.5, "off_points": points[1], "def_points": points[0]}
    return pd.DataFrame([a, b])


def test_drive_checks_ranges_and_defense_equals_opponent_offense():
    assert dc.check_drives(drives()) == []
    assert any("def_points" in p for p in dc.check_drives(drives().assign(def_points=[1.0, 1.0])))
    assert any("points/drive" in p for p in dc.check_drives(drives().assign(off_ppd=9.0)))


def test_coverage_flags_thin_seasons_but_not_pre_2018():
    sched = pd.DataFrame({"season": [2016] * 2 + [2023] * 2, "game_pk": [1, 2, 3, 4], "game_type": ["REG"] * 4,
                          "home_team": ["a"] * 4, "away_team": ["b"] * 4})
    thin = pd.DataFrame({"season": [2016, 2023], "game_id": [1, 3], "team": ["a", "a"], "season_type": ["regular"] * 2})
    full = pd.DataFrame({"season": [2023] * 4, "game_id": [3, 3, 4, 4], "team": ["a", "b"] * 2,
                         "season_type": ["regular"] * 4})
    wx = pd.DataFrame({"season": [2016, 2016, 2023, 2023], "game_id": [1, 2, 3, 4], "season_type": ["regular"] * 4})
    p = dc.check_coverage({"advanced": full, "havoc": full, "drives": full, "weather": wx}, sched)
    assert p == []
    p = dc.check_coverage({"advanced": full, "havoc": thin, "drives": full, "weather": wx}, sched)
    assert len(p) == 1 and p[0].startswith("havoc") and "2023" in p[0]
    assert dc.check_coverage({"advanced": full}, sched)[0].startswith("havoc: asset missing")
