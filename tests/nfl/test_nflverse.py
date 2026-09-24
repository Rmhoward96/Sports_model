"""Resilient per-season nflverse fetch (skips seasons not yet published)."""
import pandas as pd
import pytest

from sportsmodel.nfl import nflverse
from sportsmodel.nfl.nflverse import import_by_season, load_release


def _fake_import(available: set[int]):
    """An nfl_data_py-style importer that returns a 1-row frame per available
    season and raises (like nfl_data_py's 404-masking NameError) otherwise."""
    def _imp(seasons: list[int]) -> pd.DataFrame:
        (yr,) = seasons  # import_by_season calls one season at a time
        if yr not in available:
            raise NameError("name 'Error' is not defined")  # nfl_data_py's real failure shape
        return pd.DataFrame({"season": [yr], "x": [yr * 10]})
    return _imp


def test_skips_unavailable_newest_season_keeps_rest():
    df = import_by_season(_fake_import({2024, 2025}), [2024, 2025, 2026], "pbp")
    assert sorted(df["season"].tolist()) == [2024, 2025]  # 2026 skipped, not fatal


def test_all_available_concatenates_all():
    df = import_by_season(_fake_import({2024, 2025, 2026}), [2024, 2025, 2026], "pbp")
    assert sorted(df["season"].tolist()) == [2024, 2025, 2026]


def test_required_raises_when_none_available():
    with pytest.raises(RuntimeError):
        import_by_season(_fake_import(set()), [2026], "pbp")


def test_not_required_returns_empty_when_none_available():
    df = import_by_season(_fake_import(set()), [2026], "injuries", required=False)
    assert df.empty


# --- load_release (direct nflverse release URL reads) ---

def _patch_release(monkeypatch, available: set[int], *, weekly_team_col=False):
    """Stub pd.read_parquet so load_release can be exercised offline. Available
    seasons return a 1-row frame; others raise a 404-shaped error. When
    weekly_team_col, the frame uses the NEW release's `team` column (no
    `recent_team`) to exercise the rename."""
    def _fake_read(path, *a, **k):
        yr = int(path.rsplit("_", 1)[1].split(".")[0])
        if yr not in available:
            raise FileNotFoundError(f"404: {path}")
        row = {"season": [yr], "week": [1]}
        row["team" if weekly_team_col else "recent_team"] = ["KC"]
        return pd.DataFrame(row)
    monkeypatch.setattr(nflverse.pd, "read_parquet", _fake_read)


def test_load_release_skips_unavailable_season(monkeypatch):
    _patch_release(monkeypatch, {2024, 2025})
    df = load_release("pbp", [2024, 2025, 2026])
    assert sorted(df["season"].tolist()) == [2024, 2025]  # 2026 not published -> skipped


def test_load_release_renames_weekly_team_to_recent_team(monkeypatch):
    _patch_release(monkeypatch, {2026}, weekly_team_col=True)
    df = load_release("weekly", [2026])
    assert "recent_team" in df.columns and "team" not in df.columns
    assert df["recent_team"].iloc[0] == "KC"


def test_load_release_not_required_returns_empty(monkeypatch):
    _patch_release(monkeypatch, set())
    assert load_release("snaps", [2026], required=False).empty


def test_load_release_required_raises_when_none(monkeypatch):
    _patch_release(monkeypatch, set())
    with pytest.raises(RuntimeError):
        load_release("pbp", [2026])


def test_single_file_dataset_is_read_once_and_filtered(monkeypatch):
    calls = []
    frame = pd.DataFrame({"season": [2020, 2021, 2022], "week": [1, 1, 1],
                          "player_gsis_id": ["a", "b", "c"], "avg_separation": [3.0, 3.1, 3.2],
                          "avg_cushion": [6.0] * 3, "avg_intended_air_yards": [9.0] * 3,
                          "avg_yac_above_expectation": [0.1] * 3, "season_type": ["REG"] * 3})
    monkeypatch.setattr(nflverse.pd, "read_parquet", lambda url: calls.append(url) or frame)
    out = nflverse.load_release("ngs_receiving", [2021, 2022])
    assert len(calls) == 1 and calls[0].endswith("nextgen_stats/ngs_receiving.parquet")
    assert sorted(out["season"].tolist()) == [2021, 2022]


def test_validate_columns_names_missing(monkeypatch):
    with pytest.raises(ValueError, match="avg_separation"):
        nflverse.validate_columns(pd.DataFrame({"season": [2021]}), "ngs_receiving")


def test_depth_release_url_is_per_season(monkeypatch):
    urls = []
    old = pd.DataFrame({"season": [2021], "club_code": ["KC"], "week": [1], "depth_team": ["1"],
                        "gsis_id": ["x"], "position": ["QB"], "football_name": ["P"], "last_name": ["M"],
                        "first_name": ["P"]})
    monkeypatch.setattr(nflverse.pd, "read_parquet", lambda url: urls.append(url) or old)
    nflverse.load_release("depth", [2021])
    assert urls == ["https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2021.parquet"]


def test_schedules_validated(monkeypatch):
    bad = pd.DataFrame({"season": [2021], "week": [1]})
    monkeypatch.setattr(nflverse.pd, "read_parquet", lambda url: bad)
    with pytest.raises(ValueError, match="home_rest"):
        nflverse.load_release("schedules", [2021])
