"""Tests for the pfr<->gsis id bridge in sportsmodel.sim.nfl.usage."""
import pandas as pd

from sportsmodel.sim.nfl.usage import build_pfr_to_gsis


def _ids_rows():
    """Small synthetic frame mirroring nfl_data_py's `import_ids` output.

    Columns: gsis_id, pfr_id. Includes rows with missing/blank ids on
    either side that must be skipped, and a duplicate pfr_id where the
    later row should win.
    """
    rows = [
        dict(gsis_id="00-0033873", pfr_id="MahoPa00"),
        dict(gsis_id="00-0034796", pfr_id="JackLa00"),
        # Missing pfr_id: skip.
        dict(gsis_id="00-0031234", pfr_id=None),
        # Blank/whitespace pfr_id: skip.
        dict(gsis_id="00-0031235", pfr_id="   "),
        # Missing gsis_id: skip.
        dict(gsis_id=None, pfr_id="SomeP00"),
        # Blank/whitespace gsis_id: skip.
        dict(gsis_id="  ", pfr_id="OtheP00"),
        # Duplicate pfr_id "MahoPa00": last one wins.
        dict(gsis_id="00-9999999", pfr_id="MahoPa00"),
    ]
    return pd.DataFrame(rows)


def test_build_pfr_to_gsis_maps_and_skips_missing_ids():
    ids_df = _ids_rows()
    mapping = build_pfr_to_gsis(ids_df)

    assert mapping == {
        "MahoPa00": "00-9999999",  # last-wins over the first "MahoPa00" row
        "JackLa00": "00-0034796",
    }
    assert "SomeP00" not in mapping
    assert "OtheP00" not in mapping
    assert len(mapping) == 2


def test_fetch_usage_sources_can_skip_the_old_depth_import(monkeypatch):
    import sys
    import types

    import pandas as pd

    from sportsmodel.nfl import nflverse
    from sportsmodel.sim.nfl import usage

    calls = []

    def boom(seasons):
        raise AssertionError("old depth import ran")

    fake = types.SimpleNamespace(import_depth_charts=boom,
                                 import_snap_counts=lambda seasons: calls.append("snaps") or pd.DataFrame(),
                                 import_ids=lambda: calls.append("ids") or pd.DataFrame())
    monkeypatch.setitem(sys.modules, "nfl_data_py", fake)
    monkeypatch.setattr(nflverse, "import_by_season", lambda fn, seasons, name, **kw: fn(seasons))
    out = usage.fetch_usage_sources([2026], include_depth=False)
    assert set(out) == {"snaps", "ids"} and calls == ["snaps", "ids"]
