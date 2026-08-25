import pandas as pd
from sportsmodel.nfl.data import normalize_schedule, normalize_team_col


def test_normalize_schedule_normalizes_and_selects_columns():
    raw = pd.DataFrame([{
        "game_id": "2016_01_LA_SF", "season": 2016, "week": 1, "game_type": "REG",
        "gameday": "2016-09-12", "gametime": "20:20", "home_team": "SF",
        "away_team": "LAR", "home_score": 28, "away_score": 0, "espn": 400874518,
        "extra_col": "dropped",
    }])
    out = normalize_schedule(raw)
    assert out.loc[0, "away_team"] == "LA"      # LAR -> LA
    assert out.loc[0, "home_team"] == "SF"
    assert "extra_col" not in out.columns
    assert out.loc[0, "espn"] == 400874518      # game_pk source


def test_normalize_team_col_handles_relocation():
    raw = pd.DataFrame([{"recent_team": "OAK"}, {"recent_team": "WSH"}])
    out = normalize_team_col(raw, "recent_team")
    assert list(out["recent_team"]) == ["LV", "WAS"]


def test_normalize_schedule_keeps_market_columns():
    raw = pd.DataFrame([{
        "game_id": "2023_01_x", "season": 2023, "week": 1, "game_type": "REG",
        "gameday": "2023-09-10", "gametime": "13:00", "home_team": "KC",
        "away_team": "LAR", "home_score": 21, "away_score": 20, "espn": 401547353,
        "result": 1, "total": 41, "spread_line": 3.5, "total_line": 44.5,
        "away_moneyline": 150, "home_moneyline": -170, "unwanted": "drop",
    }])
    out = normalize_schedule(raw)
    for col in ("result", "total", "spread_line", "total_line",
                "away_moneyline", "home_moneyline"):
        assert col in out.columns
    assert "unwanted" not in out.columns
    assert out.loc[0, "away_team"] == "LA"   # normalization still applied
