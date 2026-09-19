"""Tests for prop market/name reconciliation (sportsmodel.serving.props_ev).

Covers the sim-market -> odds-market mapping, player-name normalization for
joining sim player names to Odds API `player_name`, and the projected-usage
gate relocated here from scripts/backtest_sim_nfl.py (see Task 2 brief,
Ruling C1)."""
import math

from sportsmodel.serving.props_ev import (
    PROJECTED_USAGE_GATE,
    SIM_TO_ODDS_MARKET,
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


def test_sim_to_odds_market_has_exactly_three_entries():
    assert SIM_TO_ODDS_MARKET == {
        "rush_yds": "rush_yds",
        "rec_yds": "reception_yds",
        "receptions": "receptions",
    }


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
    }
