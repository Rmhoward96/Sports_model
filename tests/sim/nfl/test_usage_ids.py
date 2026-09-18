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
