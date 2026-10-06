"""build_game_info.py: the daily game-info job on stubbed ESPN / CFBD clients. No network, no DB."""
from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest

from sportsmodel import db
from sportsmodel.cfb import espn as real_cfb_espn

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_game_info.py"
_s = importlib.util.spec_from_file_location("build_game_info", _p)
bgi = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bgi)

NOW = pd.Timestamp("2026-10-07T14:00:00Z")


def _ev(pk, home_id, away_id, kick, status="STATUS_SCHEDULED", ls=None):
    comp = [{"homeAway": "home", "team": {"id": home_id, "displayName": "H"}, "score": "0"},
            {"homeAway": "away", "team": {"id": away_id, "displayName": "A"}, "score": "0"}]
    if ls:
        comp[0]["linescores"] = [{"value": v} for v in ls[0]]
        comp[1]["linescores"] = [{"value": v} for v in ls[1]]
        comp[0]["score"], comp[1]["score"] = str(sum(ls[0])), str(sum(ls[1]))   # parse_line_scores requires the periods to sum to the score
    return {"id": str(pk), "date": kick, "status": {"type": {"name": status}}, "week": {"number": 6},
            "season": {"year": 2026}, "competitions": [{"competitors": comp}]}


class FakeCfbEspn:
    """The real parsers over canned scoreboards; only the two network calls are faked."""
    parse_schedule, parse_line_scores = staticmethod(real_cfb_espn.parse_schedule), staticmethod(real_cfb_espn.parse_line_scores)

    def __init__(self, cur, fail_weeks=()):
        self.cur, self.fail, self.asked = cur, set(fail_weeks), []

    def fetch_current_week(self):
        return self.cur

    def fetch_scoreboard(self, season, week, st):
        self.asked.append((season, week, st))
        if week in self.fail:
            raise RuntimeError("espn down")
        if week != 6:
            return {"events": []}
        return {"events": [_ev(1, "333", "61", "2026-10-09T22:00Z"),
                           _ev(2, "333", "61", "2026-10-06T00:00Z", "STATUS_FINAL", ([7, 7, 7, 7], [0, 3, 3, 0]))]}


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, dict(params)))
        return [{"id": 1, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z",
                 "gameIndoors": False, "homeTeam": "Alabama", "awayTeam": "Georgia", "venueId": 3657, "venue": "Bryant-Denny Stadium",
                 "temperature": 41, "windSpeed": 14, "precipitation": 0}]


VENUES = pd.DataFrame({"venue_id": [3657], "name": ["Bryant-Denny Stadium"], "city": ["Tuscaloosa"], "state": ["AL"], "dome": [0.0]})


def test_window_weeks_only_counts_games_inside_the_window():
    games = [{"season": 2026, "week": 6, "commence_time": "2026-10-09T22:00Z"},
             {"season": 2026, "week": 7, "commence_time": "2026-10-16T22:00Z"},
             {"season": 2026, "week": 12, "commence_time": "2026-11-20T22:00Z"},
             {"season": None, "week": 6, "commence_time": "2026-10-09T22:00Z"}]
    assert bgi.window_weeks(games, NOW) == [(2026, 6), (2026, 7)]


def test_build_cfb_one_weather_call_per_window_week_and_rows(tmp_path):
    vp = tmp_path / "venues.parquet"
    VENUES.to_parquet(vp)
    espn, client = FakeCfbEspn({"season": 2026, "week": 6, "season_type": 2}), FakeClient()
    rows = bgi.build_cfb(NOW, client, espn, vp)
    assert espn.asked == [(2026, 5, 2), (2026, 6, 2), (2026, 7, 2)]
    assert client.calls == [("/games/weather", {"year": 2026, "week": 6, "seasonType": "regular"})]
    by = {r["game_pk"]: r for r in rows}
    assert by[1]["venue_name"] == "Bryant-Denny Stadium" and by[1]["city"] == "Tuscaloosa" and by[1]["weather_kind"] == "forecast"
    assert by[2]["line_score"] == {"home": [7, 7, 7, 7], "away": [0, 3, 3, 0]} and by[2]["venue_name"] is None


def test_build_cfb_offseason_and_a_down_espn_week(tmp_path):
    assert bgi.build_cfb(NOW, FakeClient(), FakeCfbEspn({"season": 2026, "week": 1, "season_type": 4}), tmp_path / "x") == []
    espn = FakeCfbEspn({"season": 2026, "week": 6, "season_type": 3}, fail_weeks={5})
    client = FakeClient()
    rows = bgi.build_cfb(NOW, client, espn, tmp_path / "missing.parquet")           # no venues asset: names only, still runs
    assert client.calls[0][1]["seasonType"] == "postseason" and len(rows) == 2


def test_build_nfl_walks_three_weeks_and_fetches_each_game_once():
    asked, fetched = [], []

    def fetch_schedule(season, week, season_type=2):
        asked.append((season, week, season_type))
        return [{"game_pk": week * 10, "commence_time": "2026-10-11T17:00:00Z"}] if week == 5 else []

    espn = SimpleNamespace(fetch_current_week=lambda: {"season": 2026, "week": 5, "season_type": 2},
                           target_week=lambda cur: cur, fetch_schedule=fetch_schedule,
                           fetch_game_info=lambda pk: fetched.append(pk) or {"venue_name": "V", "city": "C", "state": "S", "indoor": False,
                                                                           "temp_f": 50, "wind_mph": None, "precip_chance": None, "conditions": None})
    rows = bgi.build_nfl(NOW, espn)
    assert asked == [(2026, 4, 2), (2026, 5, 2), (2026, 6, 2)] and fetched == [50]
    assert rows[0]["temp_f"] == 50.0 and rows[0]["weather_kind"] == "forecast"


def test_main_dry_run_writes_nothing_and_a_failed_sport_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("no DB in a dry run")))
    monkeypatch.setattr(bgi, "build_nfl", lambda now, espn=None: [{"sport": "nfl", "game_pk": 1, "weather_kind": "forecast", "indoor": False, "line_score": None, "temp_f": 40.0}])
    bgi.main(["--sport", "nfl", "--dry-run", "--now", "2026-10-07T14:00:00Z"])
    assert "rows=1" in capsys.readouterr().out

    def boom(now, espn=None):
        raise RuntimeError("espn down")

    monkeypatch.setattr(bgi, "build_nfl", boom)
    with pytest.raises(SystemExit) as e:
        bgi.main(["--sport", "nfl", "--dry-run"])
    assert "nfl" in str(e.value)


def test_main_upserts_rows_and_requires_database_url(monkeypatch):
    sent = []
    monkeypatch.setattr(bgi.config, "DATABASE_URL", "postgres://x")
    monkeypatch.setattr(bgi.db, "upsert_game_info", lambda rows: sent.append(rows) or len(rows))
    monkeypatch.setattr(bgi, "build_nfl", lambda now, espn=None: [{"sport": "nfl", "game_pk": 1, "weather_kind": None, "indoor": None, "line_score": None}])
    bgi.main(["--sport", "nfl"])
    assert sent == [[{"sport": "nfl", "game_pk": 1, "weather_kind": None, "indoor": None, "line_score": None}]]
    monkeypatch.setattr(bgi.config, "DATABASE_URL", None)
    with pytest.raises(SystemExit):
        bgi.main(["--sport", "nfl"])
