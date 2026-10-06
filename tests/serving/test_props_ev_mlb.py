"""MLB path of serving.props_ev.assemble_prop_rows + the accent-folding name key (pure)."""
import pytest

from sportsmodel.model import calibration
from sportsmodel.serving.board import EV_CEILING, ev as _ev, novig as _novig
from sportsmodel.serving.props_ev import (
    MLB_SIM_TO_ODDS_MARKET,
    always_propable,
    assemble_prop_rows,
    normalize_player_name,
)

MV = "props-mlb-v1"


def _pmf(mapping, length):
    pmf = [0.0] * length
    for k, p in mapping.items():
        pmf[k] = p
    return {"kind": "pmf", "pmf": pmf}


def _sim(market="total_bases", mean=1.6, name="Shohei Ohtani", dist=None, pid="660271"):
    return {"game_pk": 849819, "player_id": pid, "name": name, "market": market, "mean": mean,
            "dist": dist or _pmf({0: 0.25, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.15}, 11),   # P(>1.5)=0.55
            "commence_time": "2026-10-06T22:00:00Z", "matchup": "Los Angeles Dodgers @ Atlanta Braves"}


def _o(market, side, line, book, price, name="Shohei Ohtani"):
    return {"game_pk": 849819, "market": market, "side": side, "player_name": name,
            "book": book, "line": line, "price": price}


def _mlb(sim_rows, odds_rows, **kw):
    return assemble_prop_rows(sim_rows, odds_rows, MV, sport="mlb", market_map=MLB_SIM_TO_ODDS_MARKET,
                              gate=always_propable, **kw)


def test_mlb_market_map_is_the_four_live_markets_identity():
    assert MLB_SIM_TO_ODDS_MARKET == {m: m for m in ("total_bases", "pitcher_ks", "hits_allowed", "outs_recorded")}
    for dropped in ("hits", "hrr", "home_run"):
        assert dropped not in MLB_SIM_TO_ODDS_MARKET


def test_mlb_row_is_tagged_mlb_and_picks_the_ev_side_with_no_usage_gate():
    # mean 1.6 would fail NFL's gate lookup (unknown market -> never propable); MLB has none.
    rows = _mlb([_sim()], [_o("total_bases", "over", 1.5, "draftkings", 110),
                           _o("total_bases", "under", 1.5, "fanduel", -130),
                           _o("total_bases", "over", 1.5, "pinnacle", 105)])
    assert len(rows) == 1
    r = rows[0]
    assert r["sport"] == "mlb" and r["model_version"] == MV and r["market"] == "total_bases"
    assert r["side"] == "over" and r["line"] == 1.5
    assert r["model_prob"] == pytest.approx(0.55)
    assert r["best_book"] == "draftkings" and r["best_price"] == 110
    assert r["ev_best"] == pytest.approx(_ev(0.55, 110))
    assert r["market_prob"] == pytest.approx(_novig(110, -130))
    assert r["pinnacle_price"] == 105
    assert r["is_pick"] is True and 0 < r["ev_best"] <= EV_CEILING
    assert r["matchup"].startswith("Los Angeles Dodgers")


def test_nfl_defaults_are_unchanged_by_the_new_keywords():
    # No kwargs -> NFL behavior: an MLB market code is simply unknown to the NFL map.
    assert assemble_prop_rows([_sim()], [_o("total_bases", "over", 1.5, "draftkings", 110),
                                         _o("total_bases", "under", 1.5, "fanduel", -130)], MV) == []


def test_calibrate_fn_is_applied_to_the_model_probability_at_the_book_line():
    odds = [_o("total_bases", "over", 1.5, "draftkings", 110), _o("total_bases", "under", 1.5, "fanduel", -130)]
    [raw] = _mlb([_sim()], odds)
    [cal] = _mlb([_sim()], odds, calibrate_fn=lambda market, p: 0.40)
    assert raw["model_prob"] == pytest.approx(0.55) and raw["side"] == "over"
    assert cal["model_prob"] == pytest.approx(0.60)     # over EV is now negative -> the under side (1 - 0.40) wins
    assert cal["side"] == "under"


def test_real_calibration_is_per_market_and_pure():
    p = calibration.calibrate("total_bases", 0.6)
    assert 0.0 < p < 1.0 and p == calibration.calibrate("total_bases", 0.6)


def test_dropped_markets_never_reach_the_board_even_with_odds():
    sim = _sim(market="hits")
    odds = [_o("hits", "over", 0.5, "draftkings", -150), _o("hits", "under", 0.5, "fanduel", 120)]
    assert _mlb([sim], odds) == []


def test_name_key_folds_accents_for_mlb_names():
    assert normalize_player_name("José Ramírez") == normalize_player_name("Jose Ramirez") == "jose ramirez"
    assert normalize_player_name("Ronald Acuña Jr.") == "ronald acuna"
    # NFL behavior unchanged
    assert normalize_player_name("A.J. Brown Jr.") == "aj brown"
    rows = _mlb([_sim(name="Ronald Acuña Jr.")],
                [_o("total_bases", "over", 1.5, "draftkings", 110, name="Ronald Acuna Jr"),
                 _o("total_bases", "under", 1.5, "fanduel", -130, name="Ronald Acuna Jr")])
    assert len(rows) == 1


def test_pitcher_markets_match_on_identity_codes():
    sim = _sim(market="pitcher_ks", name="Yoshinobu Yamamoto", mean=6.0,
               dist=_pmf({4: 0.1, 5: 0.15, 6: 0.25, 7: 0.3, 8: 0.2}, 16))   # P(>5.5) = 0.75
    odds = [_o("pitcher_ks", "over", 5.5, "draftkings", 120, name="Yoshinobu Yamamoto"),
            _o("pitcher_ks", "under", 5.5, "fanduel", -150, name="Yoshinobu Yamamoto")]
    [r] = _mlb([sim], odds)
    assert r["market"] == "pitcher_ks" and r["side"] == "over" and r["model_prob"] == pytest.approx(0.75)
