"""The shrink inside serving.props_ev.assemble_prop_rows: formula, k per sport and market, MLB order, the pick decision,
the audit fields and the k = 0 board. Pure."""
import pytest

from sportsmodel.model import calibration
from sportsmodel.serving.board import EV_CEILING, decimal_odds, ev as _ev, novig as _novig
from sportsmodel.serving.props_ev import (
    MLB_SIM_TO_ODDS_MARKET,
    always_propable,
    assemble_prop_rows,
)
from sportsmodel.shrink.formula import shrink_prob
from sportsmodel.shrink.store import ShrinkTable

MV = "props-sim-v1-test"


def _pmf(mapping, length):
    pmf = [0.0] * length
    for k, p in mapping.items():
        pmf[k] = p
    return {"kind": "pmf", "pmf": pmf}


def _sim(market="rec_yds", p_over=0.69, name="A.J. Brown"):
    """A sim row whose P(over) at line 68.5 is exactly p_over (mass at 75 vs 50)."""
    return {"game_pk": 1001, "player_id": "p1", "name": name, "market": market, "mean": 70.0,
            "dist": _pmf({50: 1 - p_over, 75: p_over}, 76),
            "commence_time": "2026-09-21T17:00:00Z", "matchup": "PHI @ DAL"}


def _o(side, book, price, market="reception_yds", line=68.5, name="AJ Brown"):
    return {"game_pk": 1001, "market": market, "side": side, "player_name": name, "book": book, "line": line, "price": price}


ODDS = [_o("over", "draftkings", -125), _o("under", "fanduel", 105), _o("over", "pinnacle", -130)]
M = _novig(-125, 105)                         # the market's no-vig over probability for ODDS


class KStub:
    """Records which (sport, market) the board asks for and answers a fixed k."""

    def __init__(self, k):
        self.k_value, self.asked = k, []

    def k(self, sport, market):
        self.asked.append((sport, market))
        return self.k_value


def nfl(sim=None, odds=None, **kw):
    return assemble_prop_rows([sim or _sim()], odds or ODDS, MV, **kw)


def mlb(sim, odds, **kw):
    return assemble_prop_rows([sim], odds, MV, sport="mlb", market_map=MLB_SIM_TO_ODDS_MARKET, gate=always_propable, **kw)


# ---- no table: the legacy path used by the join tests ---------------------------------------------------------------

def test_without_a_table_the_model_probability_is_untouched_and_shrink_k_is_none():
    [r] = nfl()
    assert r["model_prob"] == pytest.approx(0.69) and r["raw_model_prob"] == pytest.approx(0.69) and r["shrink_k"] is None


# ---- the formula inside the board --------------------------------------------------------------------------------

def test_k_one_third_moves_the_model_probability_a_third_of_the_way_to_the_market():
    [r] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 1 / 3}))
    expected = shrink_prob(0.69, M, 1 / 3)
    assert r["side"] == "over" and r["model_prob"] == pytest.approx(expected) and expected == pytest.approx(M + (0.69 - M) / 3)
    assert r["raw_model_prob"] == pytest.approx(0.69) and r["shrink_k"] == pytest.approx(1 / 3)
    assert r["market_prob"] == pytest.approx(M) and r["edge"] == pytest.approx(expected - M)
    assert r["ev_best"] == pytest.approx(_ev(expected, -125)) and r["best_price"] == -125 and r["best_book"] == "draftkings"


def test_k_one_is_the_raw_model_and_k_zero_is_the_market():
    [full] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert full["model_prob"] == pytest.approx(0.69) and full["edge"] == pytest.approx(0.69 - M)
    [flat] = nfl(shrink=ShrinkTable())
    assert flat["model_prob"] == pytest.approx(M) and flat["edge"] == pytest.approx(0.0, abs=1e-12) and flat["shrink_k"] == 0.0
    assert flat["raw_model_prob"] == pytest.approx(0.69)


def test_the_shrunk_probability_is_clipped_to_the_specs_bounds():
    [r] = nfl(sim=_sim(p_over=1.0), shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert r["model_prob"] == 0.99 and r["raw_model_prob"] == 1.0
    [u] = nfl(sim=_sim(p_over=0.0), odds=[_o("over", "draftkings", 400), _o("under", "fanduel", -150)],
              shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert u["side"] == "under" and u["model_prob"] == pytest.approx(0.99) and u["raw_model_prob"] == 1.0


def test_the_under_side_uses_one_minus_the_shrunk_probability_and_records_the_raw_under_probability():
    odds = [_o("over", "draftkings", 140), _o("under", "fanduel", -165)]
    [r] = nfl(sim=_sim(p_over=0.20), odds=odds, shrink=ShrinkTable({("nfl", "reception_yds"): 0.5}))
    m = _novig(140, -165)
    shrunk_over = shrink_prob(0.20, m, 0.5)
    assert r["side"] == "under" and r["model_prob"] == pytest.approx(1 - shrunk_over)
    assert r["raw_model_prob"] == pytest.approx(0.80) and r["market_prob"] == pytest.approx(1 - m)
    assert r["edge"] == pytest.approx((1 - shrunk_over) - (1 - m)) and r["ev_best"] == pytest.approx(_ev(1 - shrunk_over, -165))


# ---- which k, for which sport and market ----------------------------------------------------------------------------

def test_the_board_asks_for_k_by_sport_and_the_odds_market_key_not_the_sim_market():
    stub = KStub(0.5)
    nfl(shrink=stub)
    assert stub.asked == [("nfl", "reception_yds")]                    # rec_yds (sim) -> reception_yds (odds)
    stub = KStub(0.5)
    sim = _sim(market="total_bases", p_over=0.55, name="Shohei Ohtani") | {"game_pk": 849819}
    odds = [{"game_pk": 849819, "market": "total_bases", "side": "over", "player_name": "Shohei Ohtani", "book": "draftkings", "line": 68.5, "price": 110},
            {"game_pk": 849819, "market": "total_bases", "side": "under", "player_name": "Shohei Ohtani", "book": "fanduel", "line": 68.5, "price": -130}]
    mlb(sim, odds, shrink=stub)
    assert stub.asked == [("mlb", "total_bases")]


def test_each_market_gets_its_own_k():
    table = ShrinkTable({("nfl", "reception_yds"): 0.4, ("nfl", "rush_yds"): 0.1})
    [rec] = nfl(shrink=table)
    odds_rush = [_o("over", "draftkings", -125, market="rush_yds"), _o("under", "fanduel", 105, market="rush_yds")]
    [rush] = nfl(sim=_sim(market="rush_yds"), odds=odds_rush, shrink=table)
    assert rec["shrink_k"] == 0.4 and rush["shrink_k"] == 0.1
    assert rec["model_prob"] == pytest.approx(shrink_prob(0.69, M, 0.4)) and rush["model_prob"] == pytest.approx(shrink_prob(0.69, M, 0.1))


def test_a_market_missing_from_the_table_is_price_only_never_the_raw_model():
    odds = [_o("over", "draftkings", -125, market="pass_tds"), _o("under", "fanduel", 105, market="pass_tds")]
    sim = _sim(market="pass_tds") | {"mean": 1.5}
    table = ShrinkTable({("nfl", "rush_yds"): 0.3})
    [r] = nfl(sim=sim, odds=odds, shrink=table)
    assert r["shrink_k"] == 0.0 and r["model_prob"] == pytest.approx(M) and r["raw_model_prob"] == pytest.approx(0.69)
    assert ("nfl", "pass_tds") in table.missed


def test_mlb_is_price_only_in_part_a_even_if_a_table_lists_mlb_markets():
    sim = _sim(market="total_bases", p_over=0.55, name="Shohei Ohtani") | {"game_pk": 849819}
    odds = [{"game_pk": 849819, "market": "total_bases", "side": "over", "player_name": "Shohei Ohtani", "book": "draftkings", "line": 68.5, "price": 110},
            {"game_pk": 849819, "market": "total_bases", "side": "under", "player_name": "Shohei Ohtani", "book": "fanduel", "line": 68.5, "price": -130}]
    [r] = mlb(sim, odds, shrink=ShrinkTable({("mlb", "total_bases"): 0.9}))
    assert r["shrink_k"] == 0.0 and r["model_prob"] == pytest.approx(_novig(110, -130)) and r["raw_model_prob"] == pytest.approx(0.55)


def test_mlb_calibration_runs_before_the_shrink():
    sim = _sim(market="total_bases", p_over=0.55, name="Shohei Ohtani") | {"game_pk": 849819}
    odds = [{"game_pk": 849819, "market": "total_bases", "side": "over", "player_name": "Shohei Ohtani", "book": "draftkings", "line": 68.5, "price": 110},
            {"game_pk": 849819, "market": "total_bases", "side": "under", "player_name": "Shohei Ohtani", "book": "fanduel", "line": 68.5, "price": -130}]
    [r] = mlb(sim, odds, calibrate_fn=lambda market, p: 0.40, shrink=KStub(0.5))
    m = _novig(110, -130)
    over = shrink_prob(0.40, m, 0.5)                                   # calibrated 0.40 (not the raw 0.55), then shrunk
    assert r["side"] == "under" and r["model_prob"] == pytest.approx(1 - over) and r["raw_model_prob"] == pytest.approx(0.60)


# ---- the pick decision uses the shrunk EV -----------------------------------------------------------------------------

def test_a_raw_pick_that_the_shrink_removes_is_not_a_pick():
    [raw] = nfl()
    assert raw["is_pick"] is True and 0 < raw["ev_best"] <= EV_CEILING
    [shrunk] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 0.05}))
    assert shrunk["ev_best"] == pytest.approx(_ev(shrink_prob(0.69, M, 0.05), -125)) and shrunk["ev_best"] < 0
    assert shrunk["is_pick"] is False


def test_a_shrunk_probability_can_pick_the_other_side():
    odds = [_o("over", "draftkings", -150), _o("under", "fanduel", 130)]
    [raw] = nfl(sim=_sim(p_over=0.62), odds=odds)
    [shrunk] = nfl(sim=_sim(p_over=0.62), odds=odds, shrink=ShrinkTable({("nfl", "reception_yds"): 0.0}))
    assert raw["side"] == "over"
    m = _novig(-150, 130)
    ev_over, ev_under = _ev(m, -150), _ev(1 - m, 130)
    assert shrunk["side"] == ("over" if ev_over >= ev_under else "under")          # decided by the shrunk probability


def test_the_ev_ceiling_still_applies_to_the_shrunk_ev():
    odds = [_o("over", "draftkings", 300), _o("under", "fanduel", -400)]
    [r] = nfl(sim=_sim(p_over=0.60), odds=odds, shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert r["ev_best"] > EV_CEILING and r["is_pick"] is False


def test_at_k_zero_only_cross_book_pricing_gaps_are_picks_and_both_sides_tie():
    arb = [_o("over", "draftkings", 105), _o("under", "fanduel", 105)]             # best prices from two books imply under 100%
    [r] = nfl(odds=arb, shrink=ShrinkTable())
    assert r["is_pick"] is True and r["ev_best"] == pytest.approx(1 / (2 / 2.05) - 1)       # the tie: either side pays the same
    normal = [_o("over", "draftkings", -110), _o("under", "fanduel", -110)]
    [n] = nfl(odds=normal, shrink=ShrinkTable())
    assert n["is_pick"] is False and n["ev_best"] < 0


def test_rows_always_carry_the_audit_keys():
    [r] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 0.2}))
    assert {"raw_model_prob", "shrink_k"} <= set(r)
    assert r["model_prob"] != r["raw_model_prob"]
