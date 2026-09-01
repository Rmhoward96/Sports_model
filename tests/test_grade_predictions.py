"""Unit tests for scripts/grade_predictions.py's pure accuracy computation.

Loads the script module directly (it isn't a package) via importlib, the same
pattern tests/test_grade_results.py uses. No network, no DB -- _accuracy_row is
a pure function of (prediction dict, final dict).
"""
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "grade_predictions.py"
_spec = importlib.util.spec_from_file_location("grade_predictions", _SCRIPT_PATH)
grade_predictions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grade_predictions)

_accuracy_row = grade_predictions._accuracy_row


def _prediction(**overrides) -> dict:
    base = {
        "sport": "nfl",
        "game_pk": 401671789,
        "game_date": "2026-09-01",
        "home_team_name": "Chiefs",
        "away_team_name": "Ravens",
        "home_win_prob": 0.62,
        "pred_home_score": 27.0,
        "pred_away_score": 20.0,
    }
    base.update(overrides)
    return base


def test_correct_winner_home_favored_and_home_wins():
    prediction = _prediction(home_win_prob=0.62, pred_home_score=27.0, pred_away_score=20.0)
    final = {"home_score": 24, "away_score": 17, "final": True}

    row = _accuracy_row(prediction, final)

    assert row["predicted_winner"] == "Chiefs"
    assert row["actual_winner"] == "Chiefs"
    assert row["winner_correct"] is True
    # pred_margin = 27-20 = 7; actual_margin = 24-17 = 7 -> margin_error 0
    assert row["pred_margin"] == 7.0
    assert row["actual_margin"] == 7
    assert row["margin_error"] == 0.0
    # pred_total = 47; actual_total = 41 -> total_error 6
    assert row["pred_total"] == 47.0
    assert row["actual_total"] == 41
    assert row["total_error"] == 6.0
    assert row["sport"] == "nfl"
    assert row["game_pk"] == 401671789
    assert row["win_prob"] == 0.62


def test_wrong_winner_home_favored_but_away_wins():
    prediction = _prediction(home_win_prob=0.62, pred_home_score=27.0, pred_away_score=20.0)
    final = {"home_score": 14, "away_score": 21, "final": True}

    row = _accuracy_row(prediction, final)

    assert row["predicted_winner"] == "Chiefs"   # model favored home (wp >= 0.5)
    assert row["actual_winner"] == "Ravens"      # away actually won
    assert row["winner_correct"] is False
    assert row["actual_margin"] == -7
    # pred_margin=7, actual_margin=-7 -> margin_error = 14
    assert row["margin_error"] == 14.0
    assert row["actual_total"] == 35


def test_away_favored_and_away_wins_is_correct():
    prediction = _prediction(home_win_prob=0.35, pred_home_score=17.0, pred_away_score=24.0)
    final = {"home_score": 10, "away_score": 27, "final": True}

    row = _accuracy_row(prediction, final)

    assert row["predicted_winner"] == "Ravens"
    assert row["actual_winner"] == "Ravens"
    assert row["winner_correct"] is True


def test_tie_has_no_actual_winner_and_is_not_correct():
    prediction = _prediction(home_win_prob=0.55, pred_home_score=20.0, pred_away_score=17.0)
    final = {"home_score": 20, "away_score": 20, "final": True}

    row = _accuracy_row(prediction, final)

    assert row["predicted_winner"] == "Chiefs"
    assert row["actual_winner"] is None
    assert row["winner_correct"] is False
    assert row["actual_margin"] == 0
    assert row["actual_total"] == 40


def test_missing_predicted_scores_leave_margin_and_total_fields_none():
    prediction = _prediction(pred_home_score=None, pred_away_score=None)
    final = {"home_score": 24, "away_score": 17, "final": True}

    row = _accuracy_row(prediction, final)

    assert row["winner_correct"] is True  # winner call only needs win_prob, not scores
    assert row["pred_margin"] is None
    assert row["margin_error"] is None
    assert row["pred_total"] is None
    assert row["total_error"] is None


def test_grade_predictions_script_imports_cleanly():
    assert hasattr(grade_predictions, "main")
    assert hasattr(grade_predictions, "_accuracy_row")
    assert hasattr(grade_predictions, "FINAL_PROVIDERS")
    assert set(grade_predictions.FINAL_PROVIDERS) == {"nfl", "cfb"}
