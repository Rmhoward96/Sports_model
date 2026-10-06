import pandas as pd
import pytest

from sportsmodel.cfb import v3_data


def test_missing_required_asset_names_the_backfill(tmp_path):
    with pytest.raises(FileNotFoundError, match="CFBD_API_KEY"):
        v3_data.require_asset("advanced_games.parquet", tmp_path)
    assert v3_data.read_asset("talent.parquet", tmp_path) is None


def test_read_and_require_asset_roundtrip(tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "talent.parquet")
    assert list(v3_data.read_asset("talent.parquet", tmp_path)["a"]) == [1, 2]
    assert len(v3_data.require_asset("talent.parquet", tmp_path)) == 2
