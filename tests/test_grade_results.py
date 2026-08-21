"""Unit tests for scripts/grade_results.py grading logic.

Loads the script module directly (it isn't a package) via importlib, the same
pattern tests/test_backtest_sim.py uses. Focus: the spread (run line) branch in
_grade_game, whose sign logic is easy to get subtly wrong and must have teeth —
these tests are written so that flipping a sign in the implementation makes them
fail (verified manually; see clv-grading-fix-report.md for the flip-and-run check).
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "grade_results.py"
_spec = importlib.util.spec_from_file_location("grade_results", _SCRIPT_PATH)
grade_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grade_results)

_grade_game = grade_results._grade_game


@pytest.fixture(autouse=True)
def _identity_dist_cal(monkeypatch):
    """Grading-logic tests assert against UNcalibrated total/margin probs, so pin the
    dist calibration to identity — they shouldn't depend on the fitted calibration.json.
    The calibration application itself is covered by test_grader_calibrated_total_*."""
    monkeypatch.setattr(grade_results, "_TOTAL_CAL", (0.0, 1.0), raising=False)
    monkeypatch.setattr(grade_results, "_MARGIN_CAL", (0.0, 1.0), raising=False)


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


# --- Change 2: model_prob/market_prob/ev for totals + spread, and pick-team naming ---

def _total_close(line=8.5, over_price=-110, under_price=-110, books=5):
    return {
        ("total", "over", ""): [(line, over_price, books)],
        ("total", "under", ""): [(line, under_price, books)],
    }


def test_grade_game_total_over_lean_uses_dist_model_prob_and_names_pick():
    # pmf over totals 0..N: mass 0.4 at total=8, 0.6 at total=9 -> P(>8.5) = 0.6.
    pmf = [0.0] * 13
    pmf[8], pmf[9] = 0.4, 0.6
    total_dist = {"kind": "pmf", "pmf": pmf}
    close = _total_close()
    res = _res(5, 4)  # actual total = 9
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=9.0, home_wp=0.5, pred_margin=None,
                       total_dist=total_dist)
    row = [r for r in out if r["market"] == "total"][0]
    assert row["lean"] == "over"
    assert row["player_name"] == "Over 8.5"
    assert row["model_prob"] == pytest.approx(grade_results.prob_over_dist(total_dist, 8.5))
    assert row["model_prob"] == pytest.approx(0.6)
    assert row["market_prob"] == pytest.approx(0.5)  # symmetric -110/-110 -> no-vig 0.5
    # model favors the over (0.6) more than the no-vig market (0.5) -> positive EV.
    assert row["ev"] > 0


def test_grade_game_total_under_lean_uses_complement_and_names_pick():
    pmf = [0.0] * 13
    pmf[8], pmf[9] = 0.4, 0.6  # same dist; P(>8.5) = 0.6, P(under) side = 1 - 0.6 = 0.4
    total_dist = {"kind": "pmf", "pmf": pmf}
    close = _total_close()
    res = _res(4, 4)  # actual total = 8
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=8.0, home_wp=0.5, pred_margin=None,
                       total_dist=total_dist)
    row = [r for r in out if r["market"] == "total"][0]
    assert row["lean"] == "under"
    assert row["player_name"] == "Under 8.5"
    p_over = grade_results.prob_over_dist(total_dist, 8.5)
    assert row["model_prob"] == pytest.approx(1 - p_over)
    assert row["market_prob"] == pytest.approx(0.5)
    assert row["ev"] < 0  # model (0.4) is worse than no-vig market (0.5) on the under


def test_grade_game_total_legacy_row_without_dist_leaves_prob_ev_none():
    close = _total_close()
    res = _res(5, 4)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=9.0, home_wp=0.5, pred_margin=None,
                       total_dist=None)
    row = [r for r in out if r["market"] == "total"][0]
    assert row["model_prob"] is None
    assert row["market_prob"] is None
    assert row["ev"] is None


def test_grade_game_spread_home_lean_uses_cover_prob_and_names_covered_team():
    # margin_dist all mass at +3: home -1.5 -> P(cover) = P(margin > 1.5) = 1.0.
    offset = 10
    pmf = [0.0] * (2 * offset + 1)
    pmf[3 + offset] = 1.0
    margin_dist = {"kind": "margin", "offset": offset, "pmf": pmf}
    sl = -1.5
    close = _close(sl)
    res = _res(5, 2)  # actual margin = +3, home covers
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=3.0,
                       margin_dist=margin_dist, home_name="HomeTeam", away_name="AwayTeam")
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["lean"] == "home"
    assert row["player_name"] == "HomeTeam -1.5"
    assert row["model_prob"] == pytest.approx(grade_results.prob_cover(margin_dist, sl))
    assert row["model_prob"] == pytest.approx(1.0)
    assert row["market_prob"] == pytest.approx(0.5)  # symmetric -110/-110 -> no-vig 0.5


def test_grade_game_spread_away_lean_uses_complement_and_names_covered_team():
    # margin_dist all mass at -3: home -1.5 covers iff margin > 1.5 -> P(cover)=0.0,
    # so away (getting +1.5) covers with probability 1.0.
    offset = 10
    pmf = [0.0] * (2 * offset + 1)
    pmf[-3 + offset] = 1.0
    margin_dist = {"kind": "margin", "offset": offset, "pmf": pmf}
    sl = -1.5
    close = _close(sl)
    res = _res(2, 5)  # actual margin = -3, away covers
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=-3.0,
                       margin_dist=margin_dist, home_name="HomeTeam", away_name="AwayTeam")
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["lean"] == "away"
    assert row["player_name"] == "AwayTeam +1.5"  # away line is -sl
    p_home_cover = grade_results.prob_cover(margin_dist, sl)
    assert row["model_prob"] == pytest.approx(1 - p_home_cover)
    assert row["model_prob"] == pytest.approx(1.0)
    assert row["market_prob"] == pytest.approx(0.5)


def test_grade_game_spread_legacy_row_without_dist_leaves_prob_ev_none():
    sl = -1.5
    close = _close(sl)
    res = _res(5, 2)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=3.0,
                       margin_dist=None, home_name="HomeTeam", away_name="AwayTeam")
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["model_prob"] is None
    assert row["market_prob"] is None
    assert row["ev"] is None


def test_grader_calibrated_total_uses_loaded_params(monkeypatch):
    # A +2 location calibration must move total mass right, raising P(over) at a fixed line.
    monkeypatch.setattr(grade_results, "_TOTAL_CAL", (2.0, 1.0), raising=False)
    d = {"kind": "pmf", "pmf": [0.0] * 8 + [1.0] + [0.0] * 11}  # mass at 8, headroom
    out = grade_results._calibrated_total(d)
    assert grade_results.prob_over_dist(out, 9) > 0.99   # mass now at 10 (8 + 2)
    assert grade_results.prob_over_dist(d, 9) < 0.01     # raw: mass at 8, nothing > 9


def test_grade_game_spread_picks_favorite_when_juiced_dog_is_negative_ev():
    # The +1.5 dog is MORE likely to cover (58%) but is juiced to -175 (breakeven
    # 63.6%), so it is -EV. The -1.5 favorite at +150 needs only 40% and the model
    # gives 42%, so it is +EV. EV-based selection must take the favorite -1.5 even
    # though a mean-margin rule (pred_margin=0.84 < 1.5) would take the +1.5 dog.
    offset = 10
    pmf = [0.0] * (2 * offset + 1)
    pmf[2 + offset] = 0.42   # P(margin == 2)  -> home -1.5 covers
    pmf[0 + offset] = 0.58   # P(margin == 0)  -> home -1.5 does NOT cover
    margin_dist = {"kind": "margin", "offset": offset, "pmf": pmf}
    sl = -1.5
    close = _close(sl, home_price=150, away_price=-175)
    res = _res(5, 2)  # actual margin +3, home -1.5 covers -> the +EV pick wins
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.5, pred_margin=0.84,
                       margin_dist=margin_dist, home_name="HomeTeam", away_name="AwayTeam")
    row = [r for r in out if r["market"] == "spread"][0]
    assert row["lean"] == "home"                     # favorite, not the likelier dog
    assert row["player_name"] == "HomeTeam -1.5"
    assert row["model_prob"] == pytest.approx(0.42)  # home cover prob, not the 0.58 dog
    assert row["ev"] > 0
    assert row["result"] == "win"


def test_grade_game_moneyline_names_picked_team():
    close = {
        ("moneyline", "home", ""): [(-150, -150, 5)],
        ("moneyline", "away", ""): [(130, 130, 5)],
    }
    res = _res(5, 2)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.7, pred_margin=None,
                       home_name="HomeTeam", away_name="AwayTeam")
    row = [r for r in out if r["market"] == "moneyline"][0]
    assert row["lean"] == "home"  # home_wp=0.7 > no-vig home implied prob
    assert row["player_name"] == "HomeTeam"


def test_grade_game_moneyline_names_away_pick_when_leaning_away():
    close = {
        ("moneyline", "home", ""): [(-150, -150, 5)],
        ("moneyline", "away", ""): [(130, 130, 5)],
    }
    res = _res(2, 5)
    out = _grade_game(1, "2026-08-20", "mlb-hybrid-v1", res, close,
                       pred_total=None, home_wp=0.3, pred_margin=None,
                       home_name="HomeTeam", away_name="AwayTeam")
    row = [r for r in out if r["market"] == "moneyline"][0]
    assert row["lean"] == "away"
    assert row["player_name"] == "AwayTeam"


def test_window_start_floors_at_fresh_start():
    """The rolling window never starts before FRESH_START (hard CLV floor)."""
    from datetime import date
    assert grade_results.FRESH_START == "2026-08-21"
    # today-5 = 8/15, floored up to the fresh-start date
    assert grade_results._window_start(5, date(2026, 8, 20)) == "2026-08-21"
    # on 8/25, today-5 = 8/20 is still before the floor -> floored
    assert grade_results._window_start(5, date(2026, 8, 25)) == "2026-08-21"
    # well past the floor: the rolling window governs (9/1 - 5 = 8/27)
    assert grade_results._window_start(5, date(2026, 9, 1)) == "2026-08-27"
