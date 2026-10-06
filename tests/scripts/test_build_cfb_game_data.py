"""build_cfb_game_data: pure merge + describe helpers and a stubbed-client run (no network)."""
import importlib.util
import pathlib

import pandas as pd
import pytest

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_game_data.py"
_s = importlib.util.spec_from_file_location("build_cfb_game_data", _p)
bgd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bgd)


def test_merge_asset_replaces_pulled_seasons_and_keeps_empty_ones():
    existing = pd.DataFrame({"season": [2022, 2023], "team": ["a", "a"], "talent": [1.0, 2.0]})
    new = pd.DataFrame({"season": [2023, 2023], "team": ["a", "b"], "talent": [9.0, 8.0]})
    out = bgd.merge_asset(existing, new, ["season", "team"])
    assert list(out["talent"]) == [1.0, 9.0, 8.0]                   # 2022 kept, 2023 replaced
    empty = new.iloc[0:0]
    assert bgd.merge_asset(existing, empty, ["season", "team"]).equals(existing)   # empty pull wipes nothing
    assert len(bgd.merge_asset(None, new, ["season", "team"])) == 2


def test_merge_asset_venues_replaced_whole():
    old = pd.DataFrame({"venue_id": [1, 2], "name": ["a", "b"]})
    new = pd.DataFrame({"venue_id": [2, 3], "name": ["B", "C"]})
    assert list(bgd.merge_asset(old, new, ["venue_id"])["name"]) == ["B", "C"]


def test_describe_shape_hides_values_and_follows_first_record():
    shape = bgd.describe_shape([{"id": 7, "team": "Alabama", "offense": {"ppa": 0.3, "tags": [1, 2]}}, {"x": 1}])
    assert shape == [{"id": "int", "team": "str", "offense": {"ppa": "float", "tags": ["int"]}}]
    assert bgd.describe_shape([]) == []


class StubClient:
    def __init__(self, routes):
        self.routes, self.calls, self.seen = routes, 0, []

    def get(self, path, params=None):
        self.calls += 1
        self.seen.append((path, dict(params or {})))
        return self.routes[path]

    def summary(self):
        return f"CFBD calls this run: {self.calls}"


GAMES = [{"id": 401, "season": 2023, "week": 3, "seasonType": "regular", "homeTeam": "Alabama",
          "awayTeam": "Georgia", "venueId": 7, "homePoints": 20, "awayPoints": 10}]
DRIVES = [{"gameId": 401, "offense": "Alabama", "defense": "Georgia", "startYardsToGoal": 70,
           "endYardsToGoal": 0, "startOffenseScore": 0, "endOffenseScore": 7, "driveResult": "TD"}]


def test_run_pulls_games_then_drives_with_meta_and_merges(tmp_path):
    client = StubClient({"/games": GAMES, "/drives": DRIVES})
    # same payload for both season types: the (season, game_id, team) merge de-duplicates it
    wrote = bgd.run(client, ["drives", "games"], [2023], tmp_path)
    assert wrote["games"] == 1 and wrote["drives"] >= 1
    drives = pd.read_parquet(tmp_path / "drive_games.parquet")
    assert set(drives["week"]) == {3} and drives["off_points"].iloc[0] == 7.0
    assert [p for p, _ in client.seen[:2]] == ["/games", "/games"]         # games ran first (meta for drives)
    assert {q["seasonType"] for p, q in client.seen if p == "/drives"} == {"regular", "postseason"}


def test_run_drives_without_games_asset_stops(tmp_path):
    with pytest.raises(SystemExit):
        bgd.run(StubClient({"/drives": DRIVES}), ["drives"], [2023], tmp_path)


def test_run_empty_pull_keeps_committed_season_and_warns(tmp_path, capsys):
    pd.DataFrame({"season": [2023], "team": ["333"], "talent": [900.0]}).to_parquet(tmp_path / "talent.parquet")
    bgd.run(StubClient({"/talent": []}), ["talent"], [2023], tmp_path)
    got = pd.read_parquet(tmp_path / "talent.parquet")
    assert list(got["talent"]) == [900.0]
    assert "::warning::" in capsys.readouterr().out
