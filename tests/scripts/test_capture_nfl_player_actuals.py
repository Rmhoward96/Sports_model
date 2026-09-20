"""Tests for the pure assembly seam in capture_nfl_player_actuals.py. main()'s
DB/nflverse IO is thin and not unit-tested (see the module docstring)."""
import importlib.util
import pathlib

import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "capture_nfl_player_actuals.py"
_spec = importlib.util.spec_from_file_location("capture_nfl_player_actuals", _p)
cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap)


def _weekly(rows):
    return pd.DataFrame(rows)


def test_assemble_one_row_per_resolved_player_market():
    projected = [("00-A", "Star WR"), ("00-B", "Backup")]
    weekly = _weekly([
        {"player_id": "00-A", "week": 3, "receiving_yards": 88.0, "receptions": 6,
         "rushing_yards": 0.0, "receiving_tds": 1, "rushing_tds": 0,
         "passing_yards": 0.0, "passing_tds": 0},
    ])  # 00-B absent from weekly -> skipped entirely
    rows = cap.assemble_actual_rows(projected, weekly, 401, 2026, 3)
    by = {(r["player_id"], r["market"]): r["actual"] for r in rows}
    assert by[("00-A", "rec_yds")] == 88.0
    assert by[("00-A", "anytime_td")] == 1.0
    assert by[("00-A", "receptions")] == 6.0
    assert not any(r["player_id"] == "00-B" for r in rows)   # unresolved player skipped
    assert all(r["game_pk"] == 401 and r["season"] == 2026 and r["week"] == 3
               and r["player_name"] == "Star WR" for r in rows if r["player_id"] == "00-A")


def test_assemble_skips_all_nan_market_but_keeps_others():
    projected = [("00-A", "QB")]
    weekly = _weekly([{"player_id": "00-A", "week": 3, "passing_yards": 260.0,
                       "passing_tds": 2, "rushing_yards": float("nan"),
                       "receiving_yards": float("nan"), "receptions": float("nan"),
                       "receiving_tds": float("nan"), "rushing_tds": float("nan")}])
    markets = {r["market"] for r in cap.assemble_actual_rows(projected, weekly, 401, 2026, 3)}
    assert "pass_yds" in markets and "pass_tds" in markets
    assert "rush_yds" not in markets            # all-NaN -> None -> skipped


def test_assemble_empty_when_week_absent_from_frame():
    projected = [("00-A", "WR")]
    weekly = _weekly([{"player_id": "00-A", "week": 5, "receiving_yards": 50.0}])
    assert cap.assemble_actual_rows(projected, weekly, 401, 2026, 3) == []   # no week-3 rows
    assert cap.assemble_actual_rows(projected, None, 401, 2026, 3) == []
