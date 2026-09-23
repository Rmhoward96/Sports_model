"""Tests for prop market/name reconciliation (sportsmodel.serving.props_ev).

Covers the sim-market -> odds-market mapping, player-name normalization for
joining sim player names to Odds API `player_name`, and the projected-usage
gate relocated here from scripts/backtest_sim_nfl.py (see Task 2 brief,
Ruling C1)."""
import math

import pytest

from sportsmodel.serving.props_ev import (
    LINE_MARKETS,
    PROJECTED_USAGE_GATE,
    SIM_TO_ODDS_MARKET,
    assemble_prop_line_rows,
    assemble_prop_rows,
    is_propable_projected,
    normalize_player_name,
    odds_market_for,
)


# =============================================================================
# SIM_TO_ODDS_MARKET / odds_market_for
# =============================================================================

def test_odds_market_for_rec_yds_maps_to_reception_yds():
    assert odds_market_for("rec_yds") == "reception_yds"


def test_odds_market_for_rush_yds_is_identity():
    assert odds_market_for("rush_yds") == "rush_yds"


def test_odds_market_for_receptions_is_identity():
    assert odds_market_for("receptions") == "receptions"


def test_odds_market_for_pass_yds_is_excluded_returns_none():
    # pass_yds is intentionally excluded from C v1 (see module docstring).
    assert odds_market_for("pass_yds") is None


def test_odds_market_for_unknown_market_returns_none():
    assert odds_market_for("not_a_market") is None


def test_sim_to_odds_market_entries():
    assert SIM_TO_ODDS_MARKET == {
        "rush_yds": "rush_yds",
        "rec_yds": "reception_yds",
        "receptions": "receptions",
        "pass_tds": "pass_tds",
    }
    assert "anytime_td" not in SIM_TO_ODDS_MARKET  # projection-only (single-sided)
    assert "pass_yds" not in SIM_TO_ODDS_MARKET    # excluded from the board


# =============================================================================
# normalize_player_name
# =============================================================================

def test_normalize_player_name_periods_and_no_periods_match():
    assert normalize_player_name("A.J. Brown") == normalize_player_name("AJ Brown")


def test_normalize_player_name_strips_generational_suffix():
    assert normalize_player_name("A.J. Brown Jr.") == normalize_player_name("A.J. Brown")


def test_normalize_player_name_all_three_aj_brown_variants_equal():
    variants = ["A.J. Brown", "AJ Brown", "A.J. Brown Jr."]
    keys = {normalize_player_name(v) for v in variants}
    assert len(keys) == 1
    assert keys == {"aj brown"}


def test_normalize_player_name_strips_ii_suffix():
    assert normalize_player_name("Patrick Mahomes II") == normalize_player_name("Patrick Mahomes")
    assert normalize_player_name("Patrick Mahomes II") == "patrick mahomes"


def test_normalize_player_name_collapses_internal_whitespace():
    assert normalize_player_name("Ja'Marr  Chase") == normalize_player_name("Ja'Marr Chase")


def test_normalize_player_name_strips_apostrophes():
    assert normalize_player_name("Ja'Marr Chase") == "jamarr chase"


def test_normalize_player_name_strips_surrounding_whitespace():
    assert normalize_player_name("  Justin Jefferson  ") == "justin jefferson"


def test_normalize_player_name_distinct_players_stay_distinct():
    assert normalize_player_name("Mike Evans") != normalize_player_name("Mike Williams")


# =============================================================================
# is_propable_projected / PROJECTED_USAGE_GATE (relocated from
# scripts/backtest_sim_nfl.py -- Ruling C1)
# =============================================================================

def test_is_propable_projected_pass_yds_at_threshold_is_propable():
    assert is_propable_projected("pass_yds", 150.0) is True


def test_is_propable_projected_pass_yds_below_threshold_is_not_propable():
    assert is_propable_projected("pass_yds", 149.9) is False


def test_is_propable_projected_rush_yds_at_threshold_is_propable():
    assert is_propable_projected("rush_yds", 25.0) is True


def test_is_propable_projected_rush_yds_below_threshold_is_not_propable():
    assert is_propable_projected("rush_yds", 24.9) is False


def test_is_propable_projected_rec_yds_at_threshold_is_propable():
    assert is_propable_projected("rec_yds", 25.0) is True


def test_is_propable_projected_rec_yds_below_threshold_is_not_propable():
    assert is_propable_projected("rec_yds", 24.9) is False


def test_is_propable_projected_receptions_at_threshold_is_propable():
    assert is_propable_projected("receptions", 2.5) is True


def test_is_propable_projected_receptions_below_threshold_is_not_propable():
    assert is_propable_projected("receptions", 2.4) is False


def test_is_propable_projected_unknown_market_is_never_propable():
    assert is_propable_projected("not_a_market", 9999.0) is False


def test_projected_usage_gate_has_expected_thresholds():
    assert PROJECTED_USAGE_GATE == {
        "pass_yds": 150.0,
        "rush_yds": 25.0,
        "rec_yds": 25.0,
        "receptions": 2.5,
        "pass_tds": 1.0,
    }


# =============================================================================
# assemble_prop_rows (Task 4)
# =============================================================================

from sportsmodel.serving.board import EV_CEILING  # noqa: E402
from sportsmodel.serving.board import decimal_odds as _decimal_odds  # noqa: E402
from sportsmodel.serving.board import ev as _ev  # noqa: E402
from sportsmodel.serving.board import novig as _novig  # noqa: E402

MODEL_VERSION = "props-sim-v1-test"


def _pmf(mapping: dict, length: int) -> dict:
    """Build a {"kind": "pmf", "pmf": [...]} dist from a {index: prob} map."""
    pmf = [0.0] * length
    for k, p in mapping.items():
        pmf[k] = p
    return {"kind": "pmf", "pmf": pmf}


def test_assemble_prop_rows_featured_player_matched_line_emits_higher_ev_side():
    # rec_yds (sim) <-> reception_yds (odds) reconciliation, plus name
    # normalization ("A.J. Brown" sim vs "AJ Brown" odds).
    sim_rows = [{
        "game_pk": 1001, "player_id": "p1", "name": "A.J. Brown",
        "market": "rec_yds", "mean": 70.0,
        "dist": _pmf({50: 0.31, 75: 0.69}, 76),
        "commence_time": "2026-09-21T17:00:00Z", "matchup": "PHI @ DAL",
    }]
    odds_rows = [
        {"game_pk": 1001, "market": "reception_yds", "side": "over",
         "player_name": "AJ Brown", "book": "draftkings", "line": 68.5, "price": -125},
        {"game_pk": 1001, "market": "reception_yds", "side": "over",
         "player_name": "AJ Brown", "book": "pinnacle", "line": 68.5, "price": -130},
        {"game_pk": 1001, "market": "reception_yds", "side": "under",
         "player_name": "AJ Brown", "book": "fanduel", "line": 68.5, "price": 105},
    ]

    rows = assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION)

    assert len(rows) == 1
    row = rows[0]
    p_over = 0.69  # P(X > 68.5) = pmf[75]; kept under EV_CEILING (0.25) at -125
    expected_novig_over = _novig(-125, 105)
    expected_ev_over = _ev(p_over, -125)
    expected_ev_under = _ev(1 - p_over, 105)
    assert expected_ev_over >= expected_ev_under  # over is the +EV side here

    assert row["sport"] == "nfl"
    assert row["game_pk"] == 1001
    assert row["player_id"] == "p1"
    assert row["player_name"] == "A.J. Brown"
    assert row["market"] == "rec_yds"  # stores the SIM market name
    assert row["side"] == "over"
    assert row["line"] == 68.5
    assert row["model_version"] == MODEL_VERSION
    assert row["matchup"] == "PHI @ DAL"
    assert row["commence_time"] == "2026-09-21T17:00:00Z"
    assert row["model_prob"] == pytest.approx(p_over)
    assert row["market_prob"] == pytest.approx(expected_novig_over)
    assert row["edge"] == pytest.approx(p_over - expected_novig_over)
    assert row["ev_best"] == pytest.approx(expected_ev_over)
    assert row["best_book"] == "draftkings"
    assert row["best_price"] == -125
    assert row["pinnacle_price"] == -130
    assert row["open_pinnacle_price"] == -130
    assert row["is_pick"] is True


def test_assemble_prop_rows_excludes_non_featured_player():
    sim_rows = [{
        "game_pk": 1002, "player_id": "p2", "name": "Backup Guy",
        "market": "rec_yds", "mean": 20.0,  # below the 25.0 rec_yds gate
        "dist": _pmf({50: 0.3, 75: 0.7}, 76),
        "commence_time": "2026-09-21T17:00:00Z",
    }]
    odds_rows = [
        {"game_pk": 1002, "market": "reception_yds", "side": "over",
         "player_name": "Backup Guy", "book": "draftkings", "line": 68.5, "price": -125},
        {"game_pk": 1002, "market": "reception_yds", "side": "under",
         "player_name": "Backup Guy", "book": "fanduel", "line": 68.5, "price": 105},
    ]

    assert assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION) == []


def test_assemble_prop_rows_excludes_player_with_no_matching_odds():
    sim_rows = [{
        "game_pk": 1003, "player_id": "p3", "name": "Lonely Player",
        "market": "rec_yds", "mean": 70.0,
        "dist": _pmf({50: 0.3, 75: 0.7}, 76),
        "commence_time": "2026-09-21T17:00:00Z",
    }]
    # Odds exist, but for a different game_pk -- no candidates match.
    odds_rows = [
        {"game_pk": 9999, "market": "reception_yds", "side": "over",
         "player_name": "Lonely Player", "book": "draftkings", "line": 68.5, "price": -125},
        {"game_pk": 9999, "market": "reception_yds", "side": "under",
         "player_name": "Lonely Player", "book": "fanduel", "line": 68.5, "price": 105},
    ]

    assert assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION) == []


def test_assemble_prop_rows_picks_under_side_when_it_is_higher_ev():
    # Low p_over (0.2) plus an attractive under price should make UNDER the
    # +EV side. This discriminates the under branch from the over branch:
    # if assemble_prop_rows wrongly reused p_over (instead of 1 - p_over)
    # or novig_over (instead of 1 - novig_over) for the under side, these
    # assertions would fail.
    sim_rows = [{
        "game_pk": 3003, "player_id": "p5", "name": "Underdog Back",
        "market": "rush_yds", "mean": 40.0,
        "dist": _pmf({30: 0.8, 60: 0.2}, 61),
        "commence_time": "2026-09-21T17:00:00Z",
    }]
    odds_rows = [
        {"game_pk": 3003, "market": "rush_yds", "side": "over",
         "player_name": "Underdog Back", "book": "draftkings", "line": 45.5, "price": -110},
        # -200 (not -110) keeps ev_under under EV_CEILING (0.25) while still
        # clearly the +EV side vs the over's -0.618 EV.
        {"game_pk": 3003, "market": "rush_yds", "side": "under",
         "player_name": "Underdog Back", "book": "fanduel", "line": 45.5, "price": -200},
    ]

    rows = assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION)

    assert len(rows) == 1
    row = rows[0]
    p_over = 0.2  # P(X > 45.5) = pmf[60]
    expected_novig_over = _novig(-110, -200)
    expected_ev_over = _ev(p_over, -110)
    expected_ev_under = _ev(1 - p_over, -200)
    assert expected_ev_under > expected_ev_over  # under is the +EV side here
    assert 0 < expected_ev_under <= EV_CEILING

    assert row["side"] == "under"
    assert row["model_prob"] == pytest.approx(1 - p_over)
    assert row["market_prob"] == pytest.approx(1 - expected_novig_over)
    assert row["edge"] == pytest.approx((1 - p_over) - (1 - expected_novig_over))
    assert row["ev_best"] == pytest.approx(expected_ev_under)
    assert row["best_book"] == "fanduel"
    assert row["best_price"] == -200
    assert row["is_pick"] is True


def test_assemble_prop_rows_picks_main_line_with_most_books():
    sim_rows = [{
        "game_pk": 2002, "player_id": "p4", "name": "Justin Jefferson",
        "market": "receptions", "mean": 8.0,
        "dist": _pmf({7: 0.4, 9: 0.6}, 10),
        "commence_time": "2026-09-21T17:00:00Z",
    }]
    odds_rows = [
        # Line 6.5: only 2 distinct books (draftkings, fanduel).
        {"game_pk": 2002, "market": "receptions", "side": "over",
         "player_name": "Justin Jefferson", "book": "draftkings", "line": 6.5, "price": -110},
        {"game_pk": 2002, "market": "receptions", "side": "under",
         "player_name": "Justin Jefferson", "book": "fanduel", "line": 6.5, "price": -110},
        # Line 7.5: 4 distinct books (draftkings, fanduel, betmgm, caesars) -- main line.
        {"game_pk": 2002, "market": "receptions", "side": "over",
         "player_name": "Justin Jefferson", "book": "draftkings", "line": 7.5, "price": -115},
        {"game_pk": 2002, "market": "receptions", "side": "over",
         "player_name": "Justin Jefferson", "book": "fanduel", "line": 7.5, "price": -110},
        {"game_pk": 2002, "market": "receptions", "side": "under",
         "player_name": "Justin Jefferson", "book": "betmgm", "line": 7.5, "price": -110},
        {"game_pk": 2002, "market": "receptions", "side": "under",
         "player_name": "Justin Jefferson", "book": "caesars", "line": 7.5, "price": 100},
    ]

    rows = assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION)

    assert len(rows) == 1
    row = rows[0]
    assert row["line"] == 7.5  # most-books line wins over the 6.5 line
    assert row["market"] == "receptions"
    p_over = 0.6  # P(X > 7.5) = pmf[9]
    assert row["model_prob"] == pytest.approx(p_over)
    assert row["side"] == "over"
    assert row["best_book"] == "fanduel"
    assert row["best_price"] == -110


# =============================================================================
# EV_CEILING + Pinnacle-excluded-from-soft-shopping (C final-review fixes)
# =============================================================================

def test_assemble_prop_rows_ev_above_ceiling_is_not_a_pick():
    # p_over=0.95 at an ordinary -110/-110 market is a wildly mispriced-line
    # artifact (ev_over ~= 0.814, far above EV_CEILING=0.25) -- the row is
    # still emitted (over is still the higher-EV side) but must NOT be
    # flagged is_pick, same as the game pilot's EV_CEILING convention.
    sim_rows = [{
        "game_pk": 4004, "player_id": "p6", "name": "Way Too Good",
        "market": "rec_yds", "mean": 70.0,
        "dist": _pmf({40: 0.05, 75: 0.95}, 76),
        "commence_time": "2026-09-21T17:00:00Z",
    }]
    odds_rows = [
        {"game_pk": 4004, "market": "reception_yds", "side": "over",
         "player_name": "Way Too Good", "book": "draftkings", "line": 68.5, "price": -110},
        {"game_pk": 4004, "market": "reception_yds", "side": "under",
         "player_name": "Way Too Good", "book": "fanduel", "line": 68.5, "price": -110},
    ]

    rows = assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION)

    assert len(rows) == 1
    row = rows[0]
    p_over = 0.95  # P(X > 68.5) = pmf[75]
    expected_ev_over = _ev(p_over, -110)
    assert expected_ev_over > EV_CEILING  # confirm the scenario is actually above ceiling

    assert row["side"] == "over"
    assert row["ev_best"] == pytest.approx(expected_ev_over)
    assert row["is_pick"] is False


def test_assemble_prop_rows_best_price_shops_soft_book_not_pinnacle():
    # Pinnacle (-105, decimal ~1.9524) is numerically the single best price
    # on the over side, but it's the sharp reference, not a shoppable soft
    # book -- best_book/best_price must come from draftkings (-140) instead,
    # even though Pinnacle would "win" an unfiltered best_price() call.
    sim_rows = [{
        "game_pk": 5005, "player_id": "p7", "name": "Soft Shop Guy",
        "market": "rec_yds", "mean": 70.0,
        "dist": _pmf({50: 0.4, 75: 0.6}, 76),
        "commence_time": "2026-09-21T17:00:00Z",
    }]
    odds_rows = [
        {"game_pk": 5005, "market": "reception_yds", "side": "over",
         "player_name": "Soft Shop Guy", "book": "pinnacle", "line": 68.5, "price": -105},
        {"game_pk": 5005, "market": "reception_yds", "side": "over",
         "player_name": "Soft Shop Guy", "book": "draftkings", "line": 68.5, "price": -140},
        {"game_pk": 5005, "market": "reception_yds", "side": "under",
         "player_name": "Soft Shop Guy", "book": "fanduel", "line": 68.5, "price": -110},
    ]

    rows = assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION)

    assert len(rows) == 1
    row = rows[0]
    # Sanity: Pinnacle really is the numerically-better price here, so this
    # test would pass trivially (and fail to discriminate) if it weren't.
    assert _decimal_odds(-105) > _decimal_odds(-140)

    assert row["side"] == "over"
    assert row["best_book"] == "draftkings"
    assert row["best_price"] == -140
    # The sharp CLV anchor still comes from the Pinnacle entry, unaffected
    # by the soft-book filtering above.
    assert row["pinnacle_price"] == -105
    assert row["open_pinnacle_price"] == -105


def test_assemble_prop_rows_side_with_only_pinnacle_price_is_excluded():
    # The over side has ONLY a Pinnacle quote -- no soft book to shop -- so
    # best_price(soft) is None and the whole (player, market) is skipped,
    # same as if the over side had no odds at all.
    sim_rows = [{
        "game_pk": 6006, "player_id": "p8", "name": "Sharp Only",
        "market": "rec_yds", "mean": 70.0,
        "dist": _pmf({50: 0.3, 75: 0.7}, 76),
        "commence_time": "2026-09-21T17:00:00Z",
    }]
    odds_rows = [
        {"game_pk": 6006, "market": "reception_yds", "side": "over",
         "player_name": "Sharp Only", "book": "pinnacle", "line": 68.5, "price": -110},
        {"game_pk": 6006, "market": "reception_yds", "side": "under",
         "player_name": "Sharp Only", "book": "fanduel", "line": 68.5, "price": -110},
    ]

    assert assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION) == []


# =============================================================================
# assemble_prop_line_rows (Task 4) -- pure prop-LINE builder, every sim
# projection with a book line, no usage gate, no EV filter.
# =============================================================================

import json  # noqa: E402


def _sim(market="rec_yds", mean=40.0, pmf=None, name="A.J. Brown", dist=None):
    if dist is None:
        pmf = pmf or [0.0] * 30 + [0.02] * 50   # mass on 30..79
        dist = {"kind": "pmf", "pmf": pmf}
    return {"game_pk": 1, "player_id": "00-1", "name": name, "team": "PHI",
            "market": market, "mean": mean, "dist": dist,
            "commence_time": "2026-09-27T17:00:00Z"}


def _o(market, side, line, book, price, name="AJ Brown"):
    return {"game_pk": 1, "market": market, "side": side, "player_name": name,
            "book": book, "line": line, "price": price}


def test_line_markets_cover_tracked_markets():
    assert LINE_MARKETS == {"pass_yds": "pass_yds", "rush_yds": "rush_yds",
                            "rec_yds": "reception_yds", "receptions": "receptions",
                            "rush_att": "rush_att"}


def test_line_row_main_line_best_prices_and_lean():
    # Ruling (overrides brief fixture): pmf mass on 30..69 (40 bins of 0.025),
    # so P(over 54.5) = bins 55..69 = 15 * 0.025 = 0.375 exactly in floating
    # point -- the brief's original fixture (25 bins of 0.02 above 54.5) sums
    # to ~0.5000000000000001 in float, which can flip the lean assertion.
    pmf = [0.0] * 30 + [0.025] * 40  # mass on 30..69
    odds = [_o("reception_yds", "over", 54.5, "draftkings", -110),
            _o("reception_yds", "under", 54.5, "draftkings", -110),
            _o("reception_yds", "over", 54.5, "fanduel", -105),
            _o("reception_yds", "under", 54.5, "fanduel", -120),
            _o("reception_yds", "over", 60.5, "fanatics", +100)]
    [r] = assemble_prop_line_rows([_sim(pmf=pmf)], odds)
    assert r["line"] == 54.5 and r["n_books"] == 2
    assert (r["over_book"], r["over_price"]) == ("fanduel", -105)
    assert (r["under_book"], r["under_price"]) == ("draftkings", -110)
    assert r["projection"] == 40.0 and r["market"] == "rec_yds"
    assert abs(r["p_over"] - 0.375) < 1e-9 and r["lean"] == "under"


def test_line_row_lean_over_when_mass_above_line():
    # Mass clearly above the line -> p_over > 0.5 -> lean "over".
    pmf = [0.0] * 30 + [0.025] * 40  # mass on 30..69
    odds = [_o("reception_yds", "over", 34.5, "draftkings", -110),
            _o("reception_yds", "under", 34.5, "draftkings", -110)]
    [r] = assemble_prop_line_rows([_sim(pmf=pmf)], odds)
    # P(over 34.5) = bins 35..69 = 35 * 0.025 = 0.875
    assert abs(r["p_over"] - 0.875) < 1e-9
    assert r["lean"] == "over"


def test_line_row_dist_as_json_string_is_handled():
    pmf = [0.0] * 30 + [0.025] * 40
    dist_str = json.dumps({"kind": "pmf", "pmf": pmf})
    odds = [_o("reception_yds", "over", 54.5, "draftkings", -110),
            _o("reception_yds", "under", 54.5, "draftkings", -110)]
    [r] = assemble_prop_line_rows([_sim(dist=dist_str)], odds)
    assert abs(r["p_over"] - 0.375) < 1e-9 and r["lean"] == "under"


def test_line_row_no_usage_gate_and_pass_yds_included():
    sim = _sim(market="pass_yds", mean=12.0, name="Backup QB")
    odds = [_o("pass_yds", "over", 10.5, "draftkings", -110, "Backup QB"),
            _o("pass_yds", "under", 10.5, "draftkings", -110, "Backup QB")]
    assert len(assemble_prop_line_rows([sim], odds)) == 1


def test_line_row_skips_unmatched_untracked_and_one_sided_ok():
    assert assemble_prop_line_rows([_sim(market="anytime_td")], []) == []
    assert assemble_prop_line_rows([_sim()], []) == []
    [r] = assemble_prop_line_rows([_sim()], [_o("reception_yds", "over", 54.5, "draftkings", -110)])
    assert r["under_price"] is None and r["over_price"] == -110
