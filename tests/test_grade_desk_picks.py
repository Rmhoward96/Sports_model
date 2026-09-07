"""Unit tests for scripts/grade_desk_picks.py's pure forward-CLV grading.

Loads the script module directly (it isn't a package) via importlib, same
pattern as tests/test_grade_predictions.py. No network, no DB -- grade_pick
is a pure function of (desk_pick dict, final dict).

Sign conventions under test (see grade_pick's docstring for the full
derivation):
  - Both `spread_line` (pick-time) and `market_spread` (closing, from
    fetch_final) are ESPN pickcenter SPORTSBOOK convention: the HOME team's
    line, negative when home is favored. No conversion between them.
  - clv_spread: home side -> pick_line - closing_line; away side -> the
    negation of that (closing_line - pick_line). Positive means the desk's
    number was better than the close for the side it took.
  - clv_total: over -> closing_total - pick_total; under -> the negation
    (pick_total - closing_total). Positive means the desk's number was
    better than the close for the side it took.
"""
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "grade_desk_picks.py"
_spec = importlib.util.spec_from_file_location("grade_desk_picks", _SCRIPT_PATH)
grade_desk_picks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grade_desk_picks)

grade_pick = grade_desk_picks.grade_pick


def _pick(**overrides) -> dict:
    base = {
        "sport": "cfb",
        "game_pk": 401671789,
        "ml_pick": "home",
        "spread_side": "home",
        "spread_line": -3.0,
        "total_side": "over",
        "total_line": 47.0,
    }
    base.update(overrides)
    return base


def _final(**overrides) -> dict:
    base = {
        "home_score": 27,
        "away_score": 20,
        "final": True,
        "market_spread": -5.0,
        "market_total": 45.0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# ml_correct
# --------------------------------------------------------------------------

def test_ml_correct_home_pick_wins():
    row = grade_pick(_pick(ml_pick="home"), _final(home_score=27, away_score=20))
    assert row["ml_correct"] is True


def test_ml_correct_home_pick_loses():
    row = grade_pick(_pick(ml_pick="home"), _final(home_score=17, away_score=20))
    assert row["ml_correct"] is False


def test_ml_correct_none_on_tie():
    row = grade_pick(_pick(ml_pick="home"), _final(home_score=20, away_score=20))
    assert row["ml_correct"] is None


# --------------------------------------------------------------------------
# spread_cover -- graded vs the CLOSING line only
# --------------------------------------------------------------------------

def test_spread_cover_home_pick_covers_closing():
    # closing home line -5 (home favored by 5). Home wins by 7 (27-20):
    # cover_margin = 7 + (-5) = 2 > 0 -> home covers -> home pick correct.
    row = grade_pick(_pick(spread_side="home"), _final(home_score=27, away_score=20, market_spread=-5.0))
    assert row["spread_cover"] is True


def test_spread_cover_home_pick_fails_to_cover():
    # home wins by only 3 (23-20): cover_margin = 3 + (-5) = -2 < 0 -> away covers.
    row = grade_pick(_pick(spread_side="home"), _final(home_score=23, away_score=20, market_spread=-5.0))
    assert row["spread_cover"] is False


def test_spread_cover_away_pick_covers_closing():
    row = grade_pick(_pick(spread_side="away"), _final(home_score=23, away_score=20, market_spread=-5.0))
    assert row["spread_cover"] is True


def test_spread_cover_push_is_none():
    # home wins by exactly 5: cover_margin = 5 + (-5) = 0 -> push.
    row = grade_pick(_pick(spread_side="home"), _final(home_score=25, away_score=20, market_spread=-5.0))
    assert row["spread_cover"] is None


def test_spread_cover_none_when_closing_line_missing():
    row = grade_pick(_pick(spread_side="home"), _final(market_spread=None))
    assert row["spread_cover"] is None


def test_spread_cover_none_when_no_pick_made():
    row = grade_pick(_pick(spread_side=None, spread_line=None), _final())
    assert row["spread_cover"] is None


# --------------------------------------------------------------------------
# total_result -- graded vs the CLOSING total only
# --------------------------------------------------------------------------

def test_total_result_over_pick_beats_closing():
    # actual total 47 > closing 45 -> over hits.
    row = grade_pick(_pick(total_side="over"), _final(home_score=27, away_score=20, market_total=45.0))
    assert row["total_result"] is True


def test_total_result_over_pick_misses_closing():
    row = grade_pick(_pick(total_side="over"), _final(home_score=17, away_score=20, market_total=45.0))
    assert row["total_result"] is False


def test_total_result_under_pick_hits_closing():
    row = grade_pick(_pick(total_side="under"), _final(home_score=17, away_score=20, market_total=45.0))
    assert row["total_result"] is True


def test_total_result_push_is_none():
    row = grade_pick(_pick(total_side="over"), _final(home_score=25, away_score=20, market_total=45.0))
    assert row["total_result"] is None


def test_total_result_none_when_closing_total_missing():
    row = grade_pick(_pick(total_side="over"), _final(market_total=None))
    assert row["total_result"] is None


def test_total_result_none_when_no_pick_made():
    row = grade_pick(_pick(total_side=None, total_line=None), _final())
    assert row["total_result"] is None


# --------------------------------------------------------------------------
# clv_spread -- sign locked with the brief's worked example
# --------------------------------------------------------------------------

def test_clv_spread_home_pick_improves_positive():
    # took home -3 at pick time, closed -5 -> home's number improved (laid
    # fewer points than the market ultimately demanded) -> positive clv_spread.
    row = grade_pick(_pick(spread_side="home", spread_line=-3.0), _final(market_spread=-5.0))
    assert row["clv_spread"] == 2.0


def test_clv_spread_home_pick_worsens_negative():
    # took home -5, closed -3 -> home's number got worse -> negative.
    row = grade_pick(_pick(spread_side="home", spread_line=-5.0), _final(market_spread=-3.0))
    assert row["clv_spread"] == -2.0


def test_clv_spread_away_pick_is_inverted():
    # took away when home line was -3 (i.e. away +3), closed -5 (away +5):
    # the market moved further away's way after the pick -> the away side's
    # own number got WORSE relative to the closing number for a home-line
    # comparison; per the away-side inversion, clv_spread = -(pick - closing)
    # = -((-3) - (-5)) = -2.
    row = grade_pick(_pick(spread_side="away", spread_line=-3.0), _final(market_spread=-5.0))
    assert row["clv_spread"] == -2.0


def test_clv_spread_away_pick_positive_case():
    # took away when home line was -5 (away +5), closed -3 (away +3): the
    # away number shrank by close, so the earlier (larger) away number was
    # better -> positive clv_spread = -((-5) - (-3)) = 2.
    row = grade_pick(_pick(spread_side="away", spread_line=-5.0), _final(market_spread=-3.0))
    assert row["clv_spread"] == 2.0


def test_clv_spread_none_when_pick_line_missing():
    row = grade_pick(_pick(spread_side=None, spread_line=None), _final())
    assert row["clv_spread"] is None


def test_clv_spread_none_when_closing_line_missing():
    row = grade_pick(_pick(spread_side="home", spread_line=-3.0), _final(market_spread=None))
    assert row["clv_spread"] is None


# --------------------------------------------------------------------------
# clv_total -- sign locked with the brief's worked example
# --------------------------------------------------------------------------

def test_clv_total_over_pick_improves_positive():
    # took Over at 47, closed 50 -> the total rose after the pick, so the
    # desk's earlier (lower) number was better for an Over bettor -> positive.
    row = grade_pick(_pick(total_side="over", total_line=47.0), _final(market_total=50.0))
    assert row["clv_total"] == 3.0


def test_clv_total_over_pick_worsens_negative():
    # took Over at 50, closed 47 -> worse for Over -> negative.
    row = grade_pick(_pick(total_side="over", total_line=50.0), _final(market_total=47.0))
    assert row["clv_total"] == -3.0


def test_clv_total_under_pick_is_inverted():
    # took Under at 47, closed 50 -> total rose, meaning an Under bettor's
    # earlier (lower) number was WORSE (Under wants a HIGH number) -> negative.
    row = grade_pick(_pick(total_side="under", total_line=47.0), _final(market_total=50.0))
    assert row["clv_total"] == -3.0


def test_clv_total_under_pick_positive_case():
    # took Under at 50, closed 47 -> total fell, so the desk's earlier
    # (higher) number was better for Under -> positive.
    row = grade_pick(_pick(total_side="under", total_line=50.0), _final(market_total=47.0))
    assert row["clv_total"] == 3.0


def test_clv_total_none_when_pick_line_missing():
    row = grade_pick(_pick(total_side=None, total_line=None), _final())
    assert row["clv_total"] is None


def test_clv_total_none_when_closing_total_missing():
    row = grade_pick(_pick(total_side="over", total_line=47.0), _final(market_total=None))
    assert row["clv_total"] is None


# --------------------------------------------------------------------------
# output shape / passthrough
# --------------------------------------------------------------------------

def test_grade_pick_carries_sport_and_game_pk():
    row = grade_pick(_pick(sport="cfb", game_pk=401671789), _final())
    assert row["sport"] == "cfb"
    assert row["game_pk"] == 401671789


def test_grade_desk_picks_script_imports_cleanly():
    assert hasattr(grade_desk_picks, "main")
    assert hasattr(grade_desk_picks, "grade_pick")
    assert hasattr(grade_desk_picks, "FINAL_PROVIDERS")
