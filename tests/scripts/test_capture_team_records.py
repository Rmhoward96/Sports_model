"""Test scripts/capture_team_records.py pure functions using importlib load."""
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".." / "scripts" / "capture_team_records.py"
_spec = importlib.util.spec_from_file_location("capture_team_records", _SCRIPT_PATH)
capture_team_records = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture_team_records)


def test_resolve_names_matched_name_replaced():
    """Matched AN names are replaced with our names; unmatched stay as AN names."""
    rows = [
        {"an_team_name": "Buffalo Bills", "records": {}},
        {"an_team_name": "Alabama Crimson Tide", "records": {}},
        {"an_team_name": "Nowhere Goblins", "records": {}},
    ]
    ours = ["Buffalo Bills", "Alabama Crimson Tide"]
    result_rows, unmatched = capture_team_records.resolve_names(rows, ours)

    assert len(result_rows) == 3
    assert result_rows[0]["team_name"] == "Buffalo Bills"
    assert result_rows[1]["team_name"] == "Alabama Crimson Tide"
    assert result_rows[2]["team_name"] == "Nowhere Goblins"  # unmatched, stays as AN name
    assert unmatched == ["Nowhere Goblins"]


def test_resolve_names_only_unmatched_reported():
    """Only AN names not in our list are reported as unmatched."""
    rows = [
        {"an_team_name": "Stanford Cardinal", "records": {}},
        {"an_team_name": "UCLA Bruins", "records": {}},
    ]
    ours = ["UCLA Bruins"]
    result_rows, unmatched = capture_team_records.resolve_names(rows, ours)

    assert unmatched == ["Stanford Cardinal"]
    assert result_rows[1]["team_name"] == "UCLA Bruins"
