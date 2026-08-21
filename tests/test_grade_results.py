"""Unit tests for scripts/grade_results.py grading logic.

Loads the script module directly (it isn't a package) via importlib, the same
pattern tests/test_backtest_sim.py uses. Focus: the spread (run line) branch in
_grade_game, whose sign logic is easy to get subtly wrong and must have teeth —
these tests are written so that flipping a sign in the implementation makes them
fail (verified manually; see clv-grading-fix-report.md for the flip-and-run check).
"""
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "grade_results.py"
_spec = importlib.util.spec_from_file_location("grade_results", _SCRIPT_PATH)
grade_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grade_results)

_grade_game = grade_results._grade_game


def test_grade_results_script_imports_cleanly():
    assert hasattr(grade_results, "main")
    assert hasattr(grade_results, "_grade_game")
    assert hasattr(grade_results, "_grade_prop")


def _close(sl, home_price=-110, away_price=-110, books=5):
    """Minimal closing-lines dict with just a spread market at line `sl` (home)."""
    return {
        ("spread", "home", ""): [(sl, home_price, books)],
        ("spread", "away", ""): [(-sl, away_price, books)],
    }


def _res(hr, ar):
    return {"home_runs": hr, "away_runs": ar}


def test_spread_home_favored_covers_model_leans_home_win():
    """Home -1.5, home wins by 3 (margin +3): 3 + -1.5 = 1.5 > 0 -> home covers.
    Model pred_margin=+2: 2 + -1.5 = 0.5 > 0 -> model leans home. Home covers and
    model leans home -> win."""
    sl = -1.5
    res = _res(5, 2)  # margin = +3
    close = _close(sl)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=2.0)
    spread_rows = [r for r in out if r["market"] == "spread"]
    assert len(spread_rows) == 1
    row = spread_rows[0]
    assert row["lean"] == "home"
    assert row["result"] == "win"
    assert row["profit"] > 0
    assert row["actual"] == 3


def test_spread_home_favored_does_not_cover_away_backer_wins():
    """Home -1.5, home wins by only 1 (margin +1): 1 + -1.5 = -0.5 < 0 -> home does
    NOT cover; away +1.5 covers. Model still leans home here (pred_margin=+2 -> lean
    home), so the model's home bet should LOSE."""
    sl = -1.5
    res = _res(3, 2)  # margin = +1
    close = _close(sl)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=2.0)
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["lean"] == "home"
    assert row["result"] == "loss"
    assert row["profit"] == -1.0
    assert row["actual"] == 1


def test_spread_away_favored_home_dog_covers():
    """Away favored: home line sl=+1.5. Home loses by 1 (margin -1):
    -1 + 1.5 = 0.5 > 0 -> home covers. Model pred_margin=-2 (favors away):
    -2 + 1.5 = -0.5 < 0 -> model leans away, and away does NOT cover
    (home covers instead) -> the away bet LOSES."""
    sl = 1.5
    res = _res(2, 3)  # margin = -1
    close = _close(sl)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=-2.0)
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["lean"] == "away"
    assert row["result"] == "loss"
    assert row["actual"] == -1


def test_spread_away_favored_home_dog_covers_model_leans_home_win():
    """Same game as above (home covers +1.5) but model leans HOME
    (pred_margin=+0.5 -> 0.5+1.5=2.0>0 -> lean home). Home covers and model
    leans home -> win. Confirms the home-covers computation independent of lean."""
    sl = 1.5
    res = _res(2, 3)  # margin = -1, home covers (+0.5 > 0)
    close = _close(sl)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=0.5)
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["lean"] == "home"
    assert row["result"] == "win"
    assert row["profit"] > 0


def test_spread_push_on_integer_line():
    """Integer alt-line sl=-1 (home -1), home wins by exactly 1 (margin +1):
    1 + -1 = 0 -> push. Standard +/-1.5 lines can never push; this only happens
    on integer alt-lines."""
    sl = -1.0
    res = _res(3, 2)  # margin = +1
    close = _close(sl)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=2.0)
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["result"] == "push"
    assert row["profit"] == 0.0


def test_total_branch_skipped_when_pred_total_none():
    """A NULL pred_total must not raise (guards a TypeError that would otherwise
    abort the whole grading run on one bad row)."""
    close = {
        ("total", "over", ""): [(8.5, -110, 5)],
        ("total", "under", ""): [(8.5, -110, 5)],
    }
    res = _res(4, 3)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=None)
    assert not any(r["market"] == "total" for r in out)


def test_spread_branch_skipped_when_pred_margin_none():
    """A NULL pred_margin must not raise and must not produce a spread row."""
    close = _close(-1.5)
    res = _res(5, 2)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=None)
    assert not any(r["market"] == "spread" for r in out)
