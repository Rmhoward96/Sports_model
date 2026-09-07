"""Pure-parser tests for the SportsDataIO adapter (CFB injuries/news/weather).
No network; all fixtures are committed sample JSON under tests/fixtures/cfb/,
shaped to match SportsDataIO's real CFB schema field names (`Team`/`Name`/
`Position`/`Status`/`BodyPart`/`Practice` for injuries, `Title`/`Content`/
`Updated`/`Team` for news, `GameID`/`ForecastTempLow`/`ForecastTempHigh`/
`ForecastWindSpeed`/`ForecastDescription`/`Stadium.Type` for weather)."""
import json
import pathlib

from sportsmodel.cfb import sportsdata

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "cfb"


def _load(name: str):
    return json.loads((FIX / name).read_text())


# ------------------------------------------------------------ parse_injuries --

def test_parse_injuries_groups_by_team_with_status():
    payload = _load("sportsdata_injuries.json")
    out = sportsdata.parse_injuries(payload)
    assert "Alabama" in out
    qb = next(p for p in out["Alabama"] if p["position"] == "QB")
    assert qb["status"] in {"Out", "Questionable", "Doubtful", "Probable"}
    assert qb["player"] == "John Smith"
    assert qb["note"] == "Shoulder - Limited"


def test_parse_injuries_skips_rows_with_null_team():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    assert None not in out
    assert set(out) == {"Alabama", "Georgia"}


def test_parse_injuries_null_bodypart_and_practice_does_not_crash():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    rb = next(p for p in out["Alabama"] if p["position"] == "RB")
    assert rb["note"] is None


# ---------------------------------------------------------------- parse_news --

def test_parse_news_extracts_headline_teams_published_summary():
    payload = _load("sportsdata_news.json")
    out = sportsdata.parse_news(payload)
    assert len(out) == 3

    ala = next(n for n in out if n["headline"] == "Alabama QB day-to-day with shoulder")
    assert ala["teams"] == ["ALA"]
    assert ala["published"] == "2024-09-01T12:00:00"
    assert "day-to-day" in ala["summary"]


def test_parse_news_null_team_yields_empty_teams_list():
    out = sportsdata.parse_news(_load("sportsdata_news.json"))
    general = next(n for n in out if n["headline"] == "General CFB Week 1 roundup")
    assert general["teams"] == []


# ------------------------------------------------------------ parse_weather --

def test_parse_weather_keys_by_game_id_with_temp_wind_precip_dome():
    payload = _load("sportsdata_weather.json")
    out = sportsdata.parse_weather(payload)

    outdoor = out[12345]
    assert outdoor["temp"] == 65.0
    assert outdoor["wind"] == 10
    assert outdoor["precip"] is False
    assert outdoor["dome"] is False

    rainy = out[12346]
    assert rainy["temp"] == 47.5
    assert rainy["precip"] is True
    assert rainy["dome"] is False


def test_parse_weather_dome_game_with_null_forecast_does_not_crash():
    out = sportsdata.parse_weather(_load("sportsdata_weather.json"))
    dome_game = out[12347]
    assert dome_game["temp"] is None
    assert dome_game["wind"] is None
    assert dome_game["precip"] is None
    assert dome_game["dome"] is True
