"""Unit tests for scripts/grade_ev.py's pure forward-CLV grading.

Loads the script module directly (it isn't a package) via importlib, same
pattern as tests/test_grade_desk_picks.py. No network, no DB --
grade_ev_pick is a pure function of (ev_picks-shaped pick dict, final dict,
pinnacle_close dict).

Sign conventions under test (see grade_ev_pick's docstring for the full
derivation):
  - `won` for market="spread"/"total" is graded against the PICK-TIME line
    (`pick["line"]`) -- NOT the closing line -- using the exact same cover/
    over math as grade_desk_picks.grade_pick (home-referenced spread,
    negative = home favored; push -> None).
  - `clv` is primarily price-based: the signed improvement in single-sided
    implied probability between the pick-time Pinnacle price
    (`pick["pinnacle_price"]`) and the closing Pinnacle price for the same
    side (`pinnacle_close["price"]`). Positive means the pick-time price
    was better (cheaper) than what the market closed at for that side.
  - If the closing price for that exact side is unavailable, `clv` falls
    back to grade_desk_picks's number-based clv_spread/clv_total formula,
    using `pick["line"]` (pick-time) vs `final["market_spread"/"market_total"]`
    (closing number, from the sport's espn.fetch_final).
"""
import importlib.util
import math
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "grade_ev.py"
_spec = importlib.util.spec_from_file_location("grade_ev", _SCRIPT_PATH)
grade_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grade_ev)

grade_ev_pick = grade_ev.grade_ev_pick


def _pick(**overrides) -> dict:
    base = {
        "sport": "cfb",
        "game_pk": 401671789,
        "market": "moneyline",
        "side": "home",
        "line": None,
        "pinnacle_price": -110,
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


def _close(**overrides) -> dict:
    base = {"price": -110}
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# won -- moneyline
# --------------------------------------------------------------------------

def test_moneyline_home_pick_wins():
    row = grade_ev_pick(_pick(market="moneyline", side="home"),
                         _final(home_score=27, away_score=20), _close())
    assert row["won"] is True


def test_moneyline_home_pick_loses():
    row = grade_ev_pick(_pick(market="moneyline", side="home"),
                         _final(home_score=17, away_score=20), _close())
    assert row["won"] is False


def test_moneyline_away_pick_wins():
    row = grade_ev_pick(_pick(market="moneyline", side="away"),
                         _final(home_score=17, away_score=20), _close())
    assert row["won"] is True


def test_moneyline_none_on_tie():
    row = grade_ev_pick(_pick(market="moneyline", side="home"),
                         _final(home_score=20, away_score=20), _close())
    assert row["won"] is None


# --------------------------------------------------------------------------
# won -- spread, graded vs the PICK-TIME line (cross-check against
# grade_desk_picks's cover math, just with the pick-time number substituted
# for the closing number it uses).
# --------------------------------------------------------------------------

def test_spread_home_pick_covers_pick_time_line():
    # took home -3 at pick time; home wins by 7 (27-20):
    # cover_margin = 7 + (-3) = 4 > 0 -> home covers.
    row = grade_ev_pick(_pick(market="spread", side="home", line=-3.0),
                         _final(home_score=27, away_score=20), _close())
    assert row["won"] is True


def test_spread_home_pick_fails_to_cover_pick_time_line():
    # home wins by only 2 (22-20): cover_margin = 2 + (-3) = -1 < 0 -> away covers.
    row = grade_ev_pick(_pick(market="spread", side="home", line=-3.0),
                         _final(home_score=22, away_score=20), _close())
    assert row["won"] is False


def test_spread_away_pick_covers_pick_time_line():
    row = grade_ev_pick(_pick(market="spread", side="away", line=-3.0),
                         _final(home_score=22, away_score=20), _close())
    assert row["won"] is True


def test_spread_push_is_none():
    # home wins by exactly 3: cover_margin = 3 + (-3) = 0 -> push.
    row = grade_ev_pick(_pick(market="spread", side="home", line=-3.0),
                         _final(home_score=23, away_score=20), _close())
    assert row["won"] is None


def test_spread_none_when_pick_time_line_missing():
    row = grade_ev_pick(_pick(market="spread", side="home", line=None),
                         _final(), _close())
    assert row["won"] is None


# --------------------------------------------------------------------------
# won -- total, graded vs the PICK-TIME line
# --------------------------------------------------------------------------

def test_total_over_pick_hits_pick_time_line():
    # actual total 47 > pick-time 45 -> over hits.
    row = grade_ev_pick(_pick(market="total", side="over", line=45.0),
                         _final(home_score=27, away_score=20), _close())
    assert row["won"] is True


def test_total_over_pick_misses_pick_time_line():
    row = grade_ev_pick(_pick(market="total", side="over", line=50.0),
                         _final(home_score=27, away_score=20), _close())
    assert row["won"] is False


def test_total_under_pick_hits_pick_time_line():
    row = grade_ev_pick(_pick(market="total", side="under", line=50.0),
                         _final(home_score=27, away_score=20), _close())
    assert row["won"] is True


def test_total_push_is_none():
    # actual total 47 == pick-time 47 -> push.
    row = grade_ev_pick(_pick(market="total", side="over", line=47.0),
                         _final(home_score=27, away_score=20), _close())
    assert row["won"] is None


def test_total_none_when_pick_time_line_missing():
    row = grade_ev_pick(_pick(market="total", side="over", line=None),
                         _final(), _close())
    assert row["won"] is None


# --------------------------------------------------------------------------
# clv -- price-based (primary path)
# --------------------------------------------------------------------------

def test_clv_price_positive_when_pick_time_beats_close():
    # pick-time -110 (implied .5238) vs closing -150 (implied .6): the market
    # moved toward this side after the pick -> the pick-time price was
    # cheaper/better -> positive clv.
    row = grade_ev_pick(_pick(pinnacle_price=-110), _final(), _close(price=-150))
    assert row["clv"] > 0
    assert math.isclose(row["clv"], 0.6 - (110 / 210), abs_tol=1e-9)


def test_clv_price_negative_when_pick_time_worse_than_close():
    # pick-time -150 (implied .6) vs closing -110 (implied .5238): the market
    # drifted away from this side -> pick-time price was worse -> negative.
    row = grade_ev_pick(_pick(pinnacle_price=-150), _final(), _close(price=-110))
    assert row["clv"] < 0


def test_clv_price_zero_when_price_unchanged():
    row = grade_ev_pick(_pick(pinnacle_price=-110), _final(), _close(price=-110))
    assert math.isclose(row["clv"], 0.0, abs_tol=1e-9)


# --------------------------------------------------------------------------
# clv -- fallback to the number-based (spread/total) CLV when the closing
# price for that exact side isn't available, using grade_desk_picks's
# convention (pick-time line vs final's closing market_spread/market_total).
# --------------------------------------------------------------------------

def test_clv_fallback_spread_home_pick_matches_grade_desk_picks_convention():
    # took home -3, closed -5 -> clv_spread = -3 - (-5) = +2 (same worked
    # example as test_grade_desk_picks.test_clv_spread_home_pick_improves_positive).
    row = grade_ev_pick(_pick(market="spread", side="home", line=-3.0, pinnacle_price=None),
                         _final(market_spread=-5.0), _close(price=None))
    assert row["clv"] == 2.0


def test_clv_fallback_spread_away_pick_is_inverted():
    row = grade_ev_pick(_pick(market="spread", side="away", line=-3.0, pinnacle_price=None),
                         _final(market_spread=-5.0), _close(price=None))
    assert row["clv"] == -2.0


def test_clv_fallback_total_over_pick_matches_grade_desk_picks_convention():
    # took Over 47, closed 50 -> clv_total = 50 - 47 = +3.
    row = grade_ev_pick(_pick(market="total", side="over", line=47.0, pinnacle_price=None),
                         _final(market_total=50.0), _close(price=None))
    assert row["clv"] == 3.0


def test_clv_fallback_total_under_pick_is_inverted():
    row = grade_ev_pick(_pick(market="total", side="under", line=47.0, pinnacle_price=None),
                         _final(market_total=50.0), _close(price=None))
    assert row["clv"] == -3.0


def test_clv_none_moneyline_when_close_price_missing():
    # no number-based fallback exists for moneyline (no line to compare).
    row = grade_ev_pick(_pick(market="moneyline", side="home", pinnacle_price=None),
                         _final(), _close(price=None))
    assert row["clv"] is None


def test_clv_none_when_fallback_data_also_missing():
    row = grade_ev_pick(_pick(market="spread", side="home", line=None, pinnacle_price=None),
                         _final(market_spread=None), _close(price=None))
    assert row["clv"] is None


# --------------------------------------------------------------------------
# output shape / passthrough
# --------------------------------------------------------------------------

def test_grade_ev_pick_carries_identity_fields():
    row = grade_ev_pick(_pick(sport="nfl", game_pk=401671999, market="total", side="over"),
                         _final(), _close())
    assert row["sport"] == "nfl"
    assert row["game_pk"] == 401671999
    assert row["market"] == "total"
    assert row["side"] == "over"


def test_grade_ev_script_imports_cleanly():
    assert hasattr(grade_ev, "main")
    assert hasattr(grade_ev, "grade_ev_pick")
    assert hasattr(grade_ev, "FINAL_PROVIDERS")
