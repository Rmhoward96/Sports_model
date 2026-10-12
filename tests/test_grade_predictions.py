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
    # home favored by 7, won by exactly 7 -> favorite covered the model spread
    assert row["spread_covered"] is True
    # actual total 41 < model total 47 -> did NOT go over
    assert row["total_over"] is False
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
    # home favored by 7 but LOST by 7 -> favorite did not cover
    assert row["spread_covered"] is False
    # total 35 < model total 47 -> not over
    assert row["total_over"] is False


def test_spread_cover_and_total_over_away_favored():
    # away favored by 7 (pred 24-17), away wins by 10 (27-17) and total 44 > 41
    prediction = _prediction(home_win_prob=0.35, pred_home_score=17.0, pred_away_score=24.0)
    final = {"home_score": 17, "away_score": 27, "final": True}
    row = _accuracy_row(prediction, final)
    # away favored: pred_margin=-7, actual_margin=-10 <= -7 -> favorite covered
    assert row["spread_covered"] is True
    assert row["total_over"] is True   # actual 44 > model 41


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
    assert row["spread_covered"] is None
    assert row["total_over"] is None
    assert row["spread_pick_correct"] is None
    assert row["total_pick_correct"] is None


def test_no_market_line_leaves_pick_grades_none():
    # final without market_spread/market_total -> vs-market grades are None,
    # even though the diagnostic (vs model's own number) grades still compute.
    prediction = _prediction(pred_home_score=27.0, pred_away_score=20.0)
    final = {"home_score": 24, "away_score": 17, "final": True}
    row = _accuracy_row(prediction, final)
    assert row["market_spread"] is None and row["market_total"] is None
    assert row["spread_pick_correct"] is None
    assert row["total_pick_correct"] is None
    assert row["spread_covered"] is True  # diagnostic still computed


def test_market_spread_pick_home_covers():
    # model projects home -7 (27-20); market home line -3. model_cover = -7? no:
    # pred_margin=+7 (home by 7), market_spread=-3 -> model_cover=+4>0 -> likes home.
    # home wins by 10 (31-21): actual_cover = 10 + (-3) = +7 > 0 -> home covered. correct.
    prediction = _prediction(pred_home_score=27.0, pred_away_score=20.0)
    final = {"home_score": 31, "away_score": 21, "final": True,
             "market_spread": -3.0, "market_total": 48.0}
    row = _accuracy_row(prediction, final)
    assert row["market_spread"] == -3.0
    assert row["spread_pick_correct"] is True


def test_market_spread_pick_home_fails_to_cover():
    # model likes home (pred +7 vs -3 line). home wins by only 1 (24-23):
    # actual_cover = 1 + (-3) = -2 < 0 -> away covered -> model wrong.
    prediction = _prediction(pred_home_score=27.0, pred_away_score=20.0)
    final = {"home_score": 24, "away_score": 23, "final": True,
             "market_spread": -3.0, "market_total": 48.0}
    row = _accuracy_row(prediction, final)
    assert row["spread_pick_correct"] is False


def test_market_total_over_pick_hits_and_misses():
    # model total 47 (27+20). Over the 45 line -> leans Over.
    prediction = _prediction(pred_home_score=27.0, pred_away_score=20.0)
    over = _accuracy_row(prediction, {"home_score": 30, "away_score": 24, "final": True,
                                      "market_spread": -3.0, "market_total": 45.0})
    assert over["total_pick_correct"] is True   # actual 54 > 45, model said over
    under = _accuracy_row(prediction, {"home_score": 14, "away_score": 13, "final": True,
                                       "market_spread": -3.0, "market_total": 45.0})
    assert under["total_pick_correct"] is False  # actual 27 < 45, model said over -> wrong


def test_market_push_grades_none():
    # model likes home (pred +7 vs -3). home wins by exactly 3 (24-21):
    # actual_cover = 3 + (-3) = 0 -> spread push -> None. total lands on 47 line -> push.
    prediction = _prediction(pred_home_score=27.0, pred_away_score=20.0)
    final = {"home_score": 24, "away_score": 21, "final": True,
             "market_spread": -3.0, "market_total": 45.0}
    row = _accuracy_row(prediction, final)
    assert row["spread_pick_correct"] is None   # actual pushed on the spread
    assert row["total_pick_correct"] is None     # actual total 45 == line -> push


def test_grade_predictions_script_imports_cleanly():
    assert hasattr(grade_predictions, "main")
    assert hasattr(grade_predictions, "_accuracy_row")
    assert hasattr(grade_predictions, "FINAL_PROVIDERS")
    assert set(grade_predictions.FINAL_PROVIDERS) == {"nfl", "cfb", "mlb"}


def test_mlb_final_provider_is_statsapi_and_gets_closing_lines_from_odds():
    from sportsmodel.ingest import mlb_statsapi
    assert grade_predictions.FINAL_PROVIDERS["mlb"] is mlb_statsapi
    assert "mlb" in grade_predictions.CLOSING_LINES_FROM_ODDS
    assert "nfl" not in grade_predictions.CLOSING_LINES_FROM_ODDS  # ESPN already carries its line


def test_closing_lines_from_rows_takes_newest_per_market():
    # rows are (market, line) newest-first; spread is the HOME line.
    rows = [("total", 8.5), ("spread", -1.5), ("total", 9.0), ("spread", -1.5)]
    assert grade_predictions.closing_lines_from_rows(rows) == {"market_spread": -1.5, "market_total": 8.5}


def test_closing_lines_from_rows_missing_market_or_null_line_is_none():
    assert grade_predictions.closing_lines_from_rows([]) == {"market_spread": None, "market_total": None}
    assert grade_predictions.closing_lines_from_rows([("total", None), ("total", 7.5)]) == {
        "market_spread": None, "market_total": 7.5}
    assert grade_predictions.closing_lines_from_rows([("moneyline", None)]) == {
        "market_spread": None, "market_total": None}


def test_mlb_accuracy_row_grades_total_pick_against_odds_derived_line():
    # An MLB final merged with the Pinnacle close grades exactly like a football final.
    prediction = _prediction(sport="mlb", game_pk=849825, home_team_name="Milwaukee Brewers",
                             away_team_name="San Diego Padres", home_win_prob=0.55,
                             pred_home_score=4.6, pred_away_score=3.9)
    final = {"home_score": 4, "away_score": 3, "final": True, "market_spread": -1.5, "market_total": 6.5}
    row = _accuracy_row(prediction, final)
    assert row["sport"] == "mlb"
    assert row["winner_correct"] is True
    assert row["total_pick_correct"] is True    # model 8.5 leans over 6.5; the game went 7
    assert row["spread_pick_correct"] is True   # model margin +0.7 -> away +1.5 side; home won by 1


# --- spread_pick_prob / total_pick_prob (model's probability that the pick wins at the closing line) ------------
import json as _json
import math as _math

_MARGIN = {"kind": "margin", "offset": 2, "pmf": [0.1, 0.2, 0.3, 0.25, 0.15]}   # margins -2..2
_TOTAL = {"kind": "pmf", "pmf": [0, 0.1, 0.2, 0.4, 0.2, 0.1]}                    # totals 0..5


def _final(**o):
    base = {"home_score": 24, "away_score": 17, "final": True, "market_spread": -1.5, "market_total": 2.5}
    base.update(o)
    return base


def test_accuracy_row_pick_probs_from_stored_distributions():
    # pred margin 7 + (-1.5) > 0 -> home pick, covers when margin > 1.5 -> .15 ; total 47 > 2.5 -> over -> .7
    pred = _prediction(margin_dist=_MARGIN, total_dist=_TOTAL)
    row = _accuracy_row(pred, _final())
    assert _math.isclose(row["spread_pick_prob"], 0.15)
    assert _math.isclose(row["total_pick_prob"], 0.7)


def test_accuracy_row_pick_probs_away_and_under():
    # model margin 1 (27-26) + (-1.5) < 0 -> away pick: covers when margin < 1.5 -> .85
    pred = _prediction(pred_home_score=27.0, pred_away_score=26.0, margin_dist=_MARGIN,
                       total_dist={"kind": "pmf", "pmf": [0.25] * 4})
    row = _accuracy_row(pred, _final(market_total=1.5))
    assert _math.isclose(row["spread_pick_prob"], 0.85)
    # pred_total 53 > market 1.5 -> over; uniform pmf on totals 0..3: P(total > 1.5) = .5
    assert _math.isclose(row["total_pick_prob"], 0.5)
    under = _accuracy_row(_prediction(pred_home_score=1.0, pred_away_score=0.0, total_dist=_TOTAL), _final(market_total=2.5))
    assert _math.isclose(under["total_pick_prob"], 0.3)   # pred_total 1 < 2.5 -> under: P(total < 2.5) = .3


def test_accuracy_row_pick_probs_none_without_distribution_or_line():
    row = _accuracy_row(_prediction(), _final())          # no dists stored: NULL, never a normal approximation
    assert row["spread_pick_prob"] is None and row["total_pick_prob"] is None
    assert "spread_pick_prob" in row and "total_pick_prob" in row
    pred = _prediction(margin_dist=_MARGIN, total_dist=_TOTAL)
    no_lines = _accuracy_row(pred, _final(market_spread=None, market_total=None))
    assert no_lines["spread_pick_prob"] is None and no_lines["total_pick_prob"] is None
    # model exactly on the line: not a bet
    on_line = _accuracy_row(_prediction(pred_home_score=10.0, pred_away_score=8.5, margin_dist=_MARGIN), _final(market_spread=-1.5))
    assert on_line["spread_pick_prob"] is None


def test_accuracy_row_pick_probs_accept_json_text_and_sim_fallback():
    # a TEXT / unparsed JSONB column arrives as a string; the NFL sim row backs up a prediction without a pmf
    pred = _prediction(margin_dist=_json.dumps(_MARGIN), total_dist=None)
    row = _accuracy_row(pred, _final(), sim={"total_dist": _TOTAL})
    assert _math.isclose(row["spread_pick_prob"], 0.15)
    assert _math.isclose(row["total_pick_prob"], 0.7)


def test_pending_predictions_selects_the_distributions():
    class Cur:
        def __init__(self):
            self.sql = None

        def execute(self, sql, params=None):
            self.sql = sql

        def fetchall(self):
            return [(1, "2026-10-01", "H", "A", 0.6, 24.0, 20.0, _MARGIN, _TOTAL)]

    cur = Cur()
    rows = grade_predictions._pending_predictions(cur, "cfb", "2026-09-01")
    assert "gp.margin_dist" in cur.sql and "gp.total_dist" in cur.sql
    assert rows[0]["margin_dist"] == _MARGIN and rows[0]["total_dist"] == _TOTAL and rows[0]["sport"] == "cfb"


def test_sim_dists_by_game_nfl_only_latest_per_game():
    class Cur:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))

        def fetchall(self):
            return [(7, _MARGIN, _TOTAL)]

    cur = Cur()
    assert grade_predictions._sim_dists_by_game(cur, "cfb", [7]) == {}
    assert cur.calls == []                                      # only NFL has a sim table
    assert grade_predictions._sim_dists_by_game(cur, "nfl", []) == {}
    out = grade_predictions._sim_dists_by_game(cur, "nfl", [7])
    assert out == {7: {"margin_dist": _MARGIN, "total_dist": _TOTAL}}
    sql = cur.calls[0][0]
    assert "FROM nfl_sim" in sql and "created_at DESC" in sql


# --- grading must survive anything the DB hands back ---------------------------------------------------------------
def test_accuracy_row_survives_malformed_distributions_and_keeps_the_rest_of_the_row():
    bad = {"kind": "margin", "offset": None, "pmf": [0.5, None]}
    row = _accuracy_row(_prediction(margin_dist=bad, total_dist={"kind": "pmf", "pmf": ["a"]}), _final())
    assert row["spread_pick_prob"] is None and row["total_pick_prob"] is None
    assert row["winner_correct"] is True and row["spread_pick_correct"] is True   # the real grade is untouched


def test_accuracy_row_survives_an_exception_from_the_probability_maths(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(grade_predictions, "spread_pick_prob", boom)
    monkeypatch.setattr(grade_predictions, "total_pick_prob", boom)
    row = _accuracy_row(_prediction(margin_dist=_MARGIN, total_dist=_TOTAL), _final())
    assert row["spread_pick_prob"] is None and row["total_pick_prob"] is None
    assert row["winner_correct"] is True


def test_accuracy_row_survives_a_distribution_of_the_wrong_type():
    row = _accuracy_row(_prediction(margin_dist=12345, total_dist=["x"]), _final(), sim={"margin_dist": object()})
    assert row["spread_pick_prob"] is None and row["total_pick_prob"] is None


# --- nfl_sim fallback: served version preferred; a failing lookup must not abort grading -------------------------------
def test_sim_dists_prefer_the_served_version_then_the_newest_row():
    class Cur:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params=None):
            self.calls.append((sql, params))

        def fetchall(self):
            return [(7, _MARGIN, _TOTAL)]

    cur = Cur()
    grade_predictions._sim_dists_by_game(cur, "nfl", [7], served_version="nfl-sim-ml-v2")
    sql, params = cur.calls[0]
    assert "(model_version = %s) DESC" in sql and sql.index("(model_version = %s) DESC") < sql.index("created_at DESC")
    assert params == ("nfl-sim-ml-v2", [7])
    cur2 = Cur()
    grade_predictions._sim_dists_by_game(cur2, "nfl", [7], served_version=None)    # no served version: newest row of any version
    assert "model_version" not in cur2.calls[0][0] and cur2.calls[0][1] == ([7],)


class _RollbackConn:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


def test_safe_sim_dists_rolls_back_warns_and_returns_empty_on_failure(capsys):
    class BadCur:
        def execute(self, *a, **k):
            raise RuntimeError("relation nfl_sim does not exist")

    conn = _RollbackConn()
    assert grade_predictions._safe_sim_dists(conn, BadCur(), "nfl", [7], None) == {}
    assert conn.rollbacks == 1                                   # the aborted transaction is cleared so later reads work
    out = capsys.readouterr().out
    assert "WARN" in out and "nfl_sim" in out


def test_safe_sim_dists_passes_through_on_success_without_rollback():
    class Cur:
        def execute(self, *a, **k):
            pass

        def fetchall(self):
            return [(7, _MARGIN, _TOTAL)]

    conn = _RollbackConn()
    assert grade_predictions._safe_sim_dists(conn, Cur(), "nfl", [7], "v1") == {7: {"margin_dist": _MARGIN, "total_dist": _TOTAL}}
    assert conn.rollbacks == 0
