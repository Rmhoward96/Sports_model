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


# ------------------------------------------------ season_type-aware merge (fix round 1) --

def _blocks(season_type, ids, season=2023, val=1.0):
    return pd.DataFrame({"season": season, "season_type": season_type, "game_id": ids, "team": "a", "v": val})


def test_merge_asset_regular_only_pull_keeps_committed_postseason():
    existing = pd.concat([_blocks("regular", [1, 2], val=1.0), _blocks("postseason", [9], val=1.0)])
    out = bgd.merge_asset(existing, _blocks("regular", [1, 3], val=5.0), ["season", "game_id", "team"])
    got = {(r.season_type, r.game_id): r.v for r in out.itertuples()}
    assert got == {("regular", 1): 5.0, ("regular", 3): 5.0, ("postseason", 9): 1.0}   # reg 2 replaced, bowl kept


def test_merge_asset_postseason_only_pull_keeps_committed_regular():
    existing = pd.concat([_blocks("regular", [1, 2]), _blocks("postseason", [9])])
    out = bgd.merge_asset(existing, _blocks("postseason", [10], val=7.0), ["season", "game_id", "team"])
    got = {(r.season_type, r.game_id): r.v for r in out.itertuples()}
    assert got == {("regular", 1): 1.0, ("regular", 2): 1.0, ("postseason", 10): 7.0}   # old bowl 9 replaced


def test_merge_asset_nonempty_blocks_replace_and_other_seasons_kept():
    existing = pd.concat([_blocks("regular", [1], season=2022), _blocks("regular", [2]), _blocks("postseason", [9])])
    new = pd.concat([_blocks("regular", [3], val=2.0), _blocks("postseason", [8], val=2.0)])
    out = bgd.merge_asset(existing, new, ["season", "game_id", "team"])
    assert sorted(zip(out["season"], out["season_type"], out["game_id"])) == [
        (2022, "regular", 1), (2023, "postseason", 8), (2023, "regular", 3)]


def test_merge_asset_column_order_follows_new_frame():
    existing = _blocks("regular", [1])[["v", "team", "game_id", "season_type", "season"]]
    new = _blocks("regular", [2])
    assert list(bgd.merge_asset(existing, new, ["season", "game_id", "team"]).columns) == list(new.columns)


def test_run_regular_only_refresh_keeps_committed_postseason_rows(tmp_path):
    bowl = {**GAMES[0], "id": 500, "seasonType": "postseason", "week": 1}
    bgd.run(StubClient({"/games": [GAMES[0], bowl]}), ["games"], [2023], tmp_path)
    reg_only = StubClient({"/games": [GAMES[0]]})
    bgd.run(reg_only, ["games"], [2023], tmp_path)
    got = pd.read_parquet(tmp_path / "cfbd_games.parquet")
    assert set(got["game_id"]) == {401, 500}


class ProbeStub(StubClient):
    SENTINEL = "SENTINEL-VALUE-12345"

    def __init__(self):
        super().__init__({})

    def get(self, path, params=None):
        self.calls += 1
        if path == "/ratings/srs":          # an error-style dict body, not a list
            return {"error": self.SENTINEL, "status": 429}
        return [{"id": 1, "name": self.SENTINEL, "nested": {"x": 1.5}}]


def test_probe_makes_nine_calls_prints_shapes_not_values_and_survives_dict_bodies(capsys):
    client = ProbeStub()
    bgd.probe(client)
    out = capsys.readouterr().out
    assert client.calls == 9
    assert ProbeStub.SENTINEL not in out and "error" in out and '"str"' in out


def test_run_prints_call_counter_even_on_early_exit(tmp_path, capsys):
    with pytest.raises(SystemExit):
        bgd.run(StubClient({"/drives": DRIVES}), ["drives"], [2023], tmp_path)
    assert "CFBD calls this run" in capsys.readouterr().out
