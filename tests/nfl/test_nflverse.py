"""Resilient per-season nflverse fetch (skips seasons not yet published)."""
import pandas as pd
import pytest

from sportsmodel.nfl.nflverse import import_by_season


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
