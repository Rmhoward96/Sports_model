"""Tests for the PURE metric helpers in backtest_sim_nfl.py (the NFL sim
walk-forward ship gate). run_backtest()/report()/main() are heavy IO and NOT
unit-tested here -- see the module docstring in scripts/backtest_sim_nfl.py."""
import importlib.util
import math
import pathlib

import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "backtest_sim_nfl.py"
_spec = importlib.util.spec_from_file_location("backtest_sim_nfl", _p)
bsn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsn)


# =============================================================================
# brier
# =============================================================================

def test_brier_perfect_predictions_scores_zero():
    assert bsn.brier([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0


def test_brier_worst_predictions_scores_one():
    assert bsn.brier([1.0, 0.0], [0.0, 1.0]) == 1.0


def test_brier_coin_flip_on_certain_outcomes_scores_quarter():
    assert bsn.brier([0.5, 0.5], [1.0, 0.0]) == 0.25


def test_brier_matches_known_hand_computation():
    # (0.7-1)^2=0.09, (0.4-0)^2=0.16, (0.6-1)^2=0.16 -> mean = 0.41/3
    result = bsn.brier([0.7, 0.4, 0.6], [1.0, 0.0, 1.0])
    assert math.isclose(result, (0.09 + 0.16 + 0.16) / 3, abs_tol=1e-9)


def test_brier_empty_input_is_nan():
    assert math.isnan(bsn.brier([], []))


# =============================================================================
# mae
# =============================================================================

def test_mae_identical_sequences_is_zero():
    assert bsn.mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_matches_known_hand_computation():
    # |3-1| + |5-8| + |0-1| = 2 + 3 + 1 = 6 -> mean 2.0
    assert bsn.mae([3.0, 5.0, 0.0], [1.0, 8.0, 1.0]) == 2.0


def test_mae_is_symmetric_in_sign():
    assert bsn.mae([10.0], [7.0]) == bsn.mae([7.0], [10.0])


def test_mae_empty_input_is_nan():
    assert math.isnan(bsn.mae([], []))


# =============================================================================
# pearson_corr
# =============================================================================

def test_pearson_corr_perfect_positive_correlation():
    result = bsn.pearson_corr([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_pearson_corr_perfect_negative_correlation():
    result = bsn.pearson_corr([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
    assert math.isclose(result, -1.0, abs_tol=1e-9)


def test_pearson_corr_no_correlation_is_near_zero():
    # Symmetric zigzag: deviation products cancel exactly -> corr == 0.
    result = bsn.pearson_corr([1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 4.0, 2.0])
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_pearson_corr_zero_variance_series_is_nan():
    assert math.isnan(bsn.pearson_corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_pearson_corr_single_point_is_nan():
    assert math.isnan(bsn.pearson_corr([1.0], [2.0]))


# =============================================================================
# quantile_from_pmf
# =============================================================================

def _uniform_pmf(n: int) -> dict:
    """A pmf uniform over integers 0..n-1, as {"kind","pmf","mean"}."""
    p = 1.0 / n
    return {"kind": "pmf", "pmf": [p] * n, "mean": (n - 1) / 2.0}


def test_quantile_from_pmf_median_of_uniform_ten_bin_pmf():
    # Uniform over 0..9: cumsum reaches 0.5 exactly at k=4 (5*0.1=0.5).
    dist = _uniform_pmf(10)
    assert bsn.quantile_from_pmf(dist, 0.50) == 4.0


def test_quantile_from_pmf_p90_of_uniform_ten_bin_pmf():
    dist = _uniform_pmf(10)
    assert bsn.quantile_from_pmf(dist, 0.90) == 8.0


def test_quantile_from_pmf_point_mass_returns_that_point():
    dist = {"kind": "pmf", "pmf": [0.0, 0.0, 1.0, 0.0], "mean": 2.0}
    assert bsn.quantile_from_pmf(dist, 0.50) == 2.0
    assert bsn.quantile_from_pmf(dist, 0.99) == 2.0


def test_quantile_from_pmf_q_beyond_total_mass_falls_back_to_top_bin():
    dist = {"kind": "pmf", "pmf": [0.5, 0.3], "mean": 0.3}
    assert bsn.quantile_from_pmf(dist, 0.999) == 1.0


# =============================================================================
# share_below
# =============================================================================

def test_share_below_known_pmf_p50_matches_target_quantile():
    """Build actuals sampled exactly at the deciles of a known uniform pmf's
    quantile function; share_below at p50 should land at 0.5 for actuals
    drawn symmetrically around the median."""
    dist = _uniform_pmf(10)
    p50 = bsn.quantile_from_pmf(dist, 0.50)  # 4.0
    # 10 actuals: 5 at/below the p50 value, 5 above -> coverage should be 0.5
    actuals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    quantile_values = [p50] * len(actuals)
    result = bsn.share_below(quantile_values, actuals)
    assert math.isclose(result, 0.5, abs_tol=1e-9)


def test_share_below_all_actuals_under_quantile_is_one():
    result = bsn.share_below([10.0, 10.0, 10.0], [1.0, 2.0, 3.0])
    assert result == 1.0


def test_share_below_all_actuals_over_quantile_is_zero():
    result = bsn.share_below([1.0, 1.0, 1.0], [10.0, 20.0, 30.0])
    assert result == 0.0


def test_share_below_boundary_equal_values_counts_as_covered():
    result = bsn.share_below([5.0], [5.0])
    assert result == 1.0


def test_share_below_empty_input_is_nan():
    assert math.isnan(bsn.share_below([], []))


# =============================================================================
# is_propable
# =============================================================================

def test_is_propable_pass_yds_at_threshold_attempts_is_propable():
    assert bsn.is_propable("pass_yds", {"attempts": 10.0}) is True


def test_is_propable_pass_yds_below_threshold_attempts_is_not_propable():
    assert bsn.is_propable("pass_yds", {"attempts": 9.0}) is False


def test_is_propable_rush_yds_at_threshold_carries_is_propable():
    assert bsn.is_propable("rush_yds", {"carries": 5.0}) is True


def test_is_propable_rush_yds_below_threshold_carries_is_not_propable():
    assert bsn.is_propable("rush_yds", {"carries": 4.0}) is False


def test_is_propable_rec_yds_at_threshold_targets_is_propable():
    assert bsn.is_propable("rec_yds", {"targets": 3.0}) is True


def test_is_propable_rec_yds_below_threshold_targets_is_not_propable():
    assert bsn.is_propable("rec_yds", {"targets": 2.0}) is False


def test_is_propable_receptions_uses_same_targets_gate_as_rec_yds():
    assert bsn.is_propable("receptions", {"targets": 3.0}) is True
    assert bsn.is_propable("receptions", {"targets": 2.0}) is False


def test_is_propable_missing_usage_key_treated_as_zero_usage():
    # A WR-only player has no "attempts" key at all in their actual dict.
    assert bsn.is_propable("pass_yds", {"carries": 0.0, "targets": 8.0}) is False


def test_is_propable_unknown_market_is_never_propable():
    assert bsn.is_propable("not_a_market", {"attempts": 999.0}) is False


def test_is_propable_ignores_irrelevant_usage_columns():
    # High targets shouldn't make a player propable for a carries-gated market.
    assert bsn.is_propable("rush_yds", {"targets": 20.0, "carries": 0.0}) is False


# =============================================================================
# is_propable_projected
# =============================================================================

def test_is_propable_projected_pass_yds_at_threshold_is_propable():
    assert bsn.is_propable_projected("pass_yds", 150.0) is True


def test_is_propable_projected_pass_yds_below_threshold_is_not_propable():
    assert bsn.is_propable_projected("pass_yds", 149.9) is False


def test_is_propable_projected_rush_yds_at_threshold_is_propable():
    assert bsn.is_propable_projected("rush_yds", 25.0) is True


def test_is_propable_projected_rush_yds_below_threshold_is_not_propable():
    assert bsn.is_propable_projected("rush_yds", 24.9) is False


def test_is_propable_projected_rec_yds_at_threshold_is_propable():
    assert bsn.is_propable_projected("rec_yds", 25.0) is True


def test_is_propable_projected_rec_yds_below_threshold_is_not_propable():
    assert bsn.is_propable_projected("rec_yds", 24.9) is False


def test_is_propable_projected_receptions_at_threshold_is_propable():
    assert bsn.is_propable_projected("receptions", 2.5) is True


def test_is_propable_projected_receptions_below_threshold_is_not_propable():
    assert bsn.is_propable_projected("receptions", 2.4) is False


def test_is_propable_projected_unknown_market_is_never_propable():
    assert bsn.is_propable_projected("not_a_market", 9999.0) is False


# =============================================================================
# run_backtest / report expose the projected-usage gate's pairs dicts
# =============================================================================

def test_run_backtest_source_collects_projected_pairs_dicts():
    """run_backtest is heavy IO (nflverse fetch + simulate_game) and not
    unit-tested directly here (see module docstring), but we can still lock
    in -- via a light source-level seam -- that it builds and returns the
    three projected-usage-gated pairs dicts alongside the existing
    actual-usage ones, so a regression that silently drops them is caught
    without paying for a network-backed run."""
    import inspect

    src = inspect.getsource(bsn.run_backtest)
    for name in ("player_mean_pairs_proj", "player_p50_pairs_proj", "player_p90_pairs_proj"):
        assert name in src, f"run_backtest should build/return {name}"


def test_report_source_prints_projected_usage_gate_section():
    import inspect

    src = inspect.getsource(bsn.report)
    assert "player_mean_pairs_proj" in src
    assert "projected-usage gate" in src.lower()


# =============================================================================
# out_names_by_team_week
# =============================================================================

def _injuries_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2023, "week": 1, "team": "KC", "full_name": "Player Out", "report_status": "Out"},
        {"season": 2023, "week": 1, "team": "KC", "full_name": "Player Doubtful", "report_status": "Doubtful"},
        {"season": 2023, "week": 1, "team": "KC", "full_name": "Player Questionable", "report_status": "Questionable"},
        {"season": 2023, "week": 1, "team": "KC", "full_name": float("nan"), "report_status": "Out"},
        {"season": 2023, "week": 1, "team": "KC", "full_name": "Player NanStatus", "report_status": float("nan")},
        {"season": 2023, "week": 1, "team": "BUF", "full_name": "Other Team Out", "report_status": "OUT"},
        {"season": 2023, "week": 2, "team": "KC", "full_name": "Wrong Week Out", "report_status": "Out"},
    ])


def test_out_names_by_team_week_includes_out_and_doubtful_lowercased():
    result = bsn.out_names_by_team_week(_injuries_df(), 2023, 1)
    assert result["KC"] == {"player out", "player doubtful"}


def test_out_names_by_team_week_excludes_questionable():
    result = bsn.out_names_by_team_week(_injuries_df(), 2023, 1)
    assert "player questionable" not in result["KC"]


def test_out_names_by_team_week_excludes_nan_name_and_nan_status():
    result = bsn.out_names_by_team_week(_injuries_df(), 2023, 1)
    assert "nan" not in result["KC"]
    assert "player nanstatus" not in result["KC"]


def test_out_names_by_team_week_is_case_insensitive_on_status():
    result = bsn.out_names_by_team_week(_injuries_df(), 2023, 1)
    assert result["BUF"] == {"other team out"}


def test_out_names_by_team_week_excludes_other_week():
    result = bsn.out_names_by_team_week(_injuries_df(), 2023, 1)
    assert "wrong week out" not in result.get("KC", set())


def test_out_names_by_team_week_no_matching_rows_returns_empty_dict():
    result = bsn.out_names_by_team_week(_injuries_df(), 2099, 1)
    assert result == {}


# =============================================================================
# actual_qb_pass_yds
# =============================================================================

def test_actual_qb_pass_yds_returns_value_for_present_gsis():
    actual_stats = {"00-001": {"pass_yds": 275.0}}
    assert bsn.actual_qb_pass_yds("00-001", actual_stats) == 275.0


def test_actual_qb_pass_yds_returns_none_for_missing_gsis():
    actual_stats = {"00-001": {"pass_yds": 275.0}}
    assert bsn.actual_qb_pass_yds("00-999", actual_stats) is None


def test_actual_qb_pass_yds_returns_none_when_qb_gsis_is_none():
    actual_stats = {"00-001": {"pass_yds": 275.0}}
    assert bsn.actual_qb_pass_yds(None, actual_stats) is None


# =============================================================================
# abbrev_alignment
# =============================================================================

_KNOWN_TEAMS = {"KC", "BUF", "SF"}


def _depth_df_mismatched() -> pd.DataFrame:
    return pd.DataFrame([
        {"club_code": "KC"},
        {"club_code": "BUF"},
        {"club_code": "KAN"},  # not in _KNOWN_TEAMS -- deliberate mismatch
        {"club_code": float("nan")},
        {"club_code": "  "},
    ])


def _injuries_df_mismatched() -> pd.DataFrame:
    return pd.DataFrame([
        {"team": "KC"},
        {"team": "SF"},
        {"team": "JAC"},  # not in _KNOWN_TEAMS -- deliberate mismatch
        {"team": float("nan")},
        {"team": ""},
    ])


def test_abbrev_alignment_flags_unknown_depth_code():
    result = bsn.abbrev_alignment(
        _depth_df_mismatched(), _injuries_df_mismatched(), {"KC", "BUF"}, _KNOWN_TEAMS
    )
    assert result["depth_unknown"] == ["KAN"]


def test_abbrev_alignment_flags_unknown_injuries_team():
    result = bsn.abbrev_alignment(
        _depth_df_mismatched(), _injuries_df_mismatched(), {"KC", "BUF"}, _KNOWN_TEAMS
    )
    assert result["injuries_unknown"] == ["JAC"]


def test_abbrev_alignment_flags_unknown_game_team():
    result = bsn.abbrev_alignment(
        _depth_df_mismatched(), _injuries_df_mismatched(), {"KC", "XYZ"}, _KNOWN_TEAMS
    )
    assert result["games_unknown"] == ["XYZ"]


def test_abbrev_alignment_matching_codes_not_flagged():
    result = bsn.abbrev_alignment(
        _depth_df_mismatched(), _injuries_df_mismatched(), {"KC", "BUF"}, _KNOWN_TEAMS
    )
    assert "KC" not in result["depth_unknown"]
    assert "BUF" not in result["depth_unknown"]
    assert "KC" not in result["injuries_unknown"]
    assert "SF" not in result["injuries_unknown"]
    assert result["games_unknown"] == []


def test_abbrev_alignment_guards_nan_and_blank_values():
    result = bsn.abbrev_alignment(
        _depth_df_mismatched(), _injuries_df_mismatched(), {"KC", "BUF"}, _KNOWN_TEAMS
    )
    assert "nan" not in result["depth_unknown"]
    assert "" not in result["depth_unknown"]
    assert "nan" not in result["injuries_unknown"]
    assert "" not in result["injuries_unknown"]


def test_abbrev_alignment_all_clean_returns_empty_lists():
    depth_df = pd.DataFrame([{"club_code": "KC"}, {"club_code": "BUF"}])
    injuries_df = pd.DataFrame([{"team": "KC"}, {"team": "SF"}])
    result = bsn.abbrev_alignment(depth_df, injuries_df, {"KC", "BUF"}, _KNOWN_TEAMS)
    assert result == {"depth_unknown": [], "injuries_unknown": [], "games_unknown": []}


# =============================================================================
# run_backtest hooks for the props-ML harness
# =============================================================================

def test_run_backtest_accepts_spec_hook_and_record():
    import inspect

    params = inspect.signature(bsn.run_backtest).parameters
    assert "spec_hook" in params and "record" in params


# =============================================================================
# game_seed -- per-game random streams (game-level alignment across runs)
# =============================================================================

def test_game_seed_is_pure_and_deterministic():
    import zlib
    s = bsn.game_seed(42, 2025, 3, "KC", "BAL")
    assert s == [42, 2025, 3, zlib.crc32(b"KC|BAL")]
    assert s == bsn.game_seed(42, 2025, 3, "KC", "BAL")
    assert all(isinstance(x, int) and x >= 0 for x in s)


def test_game_seed_streams_independent_of_iteration_order():
    import numpy as np

    games = [(2025, 3, "KC", "BAL"), (2025, 3, "DET", "LA"), (2025, 4, "KC", "BAL")]

    def draws(order):
        out = {}
        for g in order:
            rng = np.random.default_rng(bsn.game_seed(42, *g))
            out[g] = rng.random(5).tolist()
        return out

    fwd, rev = draws(games), draws(list(reversed(games)))
    assert fwd == rev
    # distinct games (home/away swapped, other week, other seed) get distinct streams
    assert len({tuple(v) for v in fwd.values()}) == 3
    assert bsn.game_seed(42, 2025, 3, "BAL", "KC") != bsn.game_seed(42, 2025, 3, "KC", "BAL")
    assert bsn.game_seed(7, 2025, 3, "KC", "BAL") != bsn.game_seed(42, 2025, 3, "KC", "BAL")


# =============================================================================
# fetch_backtest_sources / run_backtest(sources=...) -- fetch once, reuse
# =============================================================================

def test_backtest_fetch_seasons_adds_warmup():
    assert bsn.backtest_fetch_seasons([2023, 2025]) == [2022, 2023, 2024, 2025]
    assert bsn.backtest_fetch_seasons([2021]) == [2020, 2021]


def test_run_backtest_with_sources_does_not_fetch(monkeypatch):
    """Given `sources`, run_backtest must not touch any fetcher (the harness
    fetches once and passes the same inputs to every run)."""
    import sportsmodel.nfl.nflverse as nflverse
    import nfl_data_py

    def boom(*a, **k):
        raise AssertionError("fetcher called despite sources")

    for mod, name in ((bsn, "fetch_nflverse"), (bsn, "fetch_usage_sources"), (bsn, "load_schedules"),
                      (bsn, "fetch_backtest_sources"), (nflverse, "load_release"),
                      (nfl_data_py, "import_injuries")):
        monkeypatch.setattr(mod, name, boom)
    sched = pd.DataFrame({"season": [2024], "week": [1], "game_type": ["REG"], "home_team": ["KC"],
                          "away_team": ["BAL"], "home_score": [float("nan")], "away_score": [float("nan")]})
    sources = {"pbp": pd.DataFrame(), "weekly": pd.DataFrame(), "snaps": pd.DataFrame(),
               "pfr2gsis": {}, "schedules": sched,
               "depth": pd.DataFrame({"club_code": ["KC"]}),
               "injuries": pd.DataFrame({"team": ["KC"]})}
    out = bsn.run_backtest([2024], 10, sources=sources)
    assert out["game_probs"] == [] and out["n_empty_active"] == 0


def test_fetch_backtest_sources_keys_match_what_run_backtest_reads():
    import inspect

    src = inspect.getsource(bsn.fetch_backtest_sources)
    for key in ("pbp", "weekly", "snaps", "pfr2gsis", "schedules", "depth", "injuries"):
        assert f'"{key}"' in src


# =============================================================================
# out_ids_by_team_week / questionable_ids_by_team_week
# =============================================================================

def _injuries_ids_df() -> pd.DataFrame:
    nan = float("nan")
    return pd.DataFrame([
        {"season": 2023, "week": 1, "team": "KC", "gsis_id": "g1", "full_name": "Player Out", "report_status": "Out"},
        {"season": 2023, "week": 1, "team": "KC", "gsis_id": "g2", "full_name": nan, "report_status": "doubtful"},
        {"season": 2023, "week": 1, "team": "KC", "gsis_id": "g3", "full_name": "Q", "report_status": "Questionable"},
        {"season": 2023, "week": 1, "team": "KC", "gsis_id": nan, "full_name": "No Id", "report_status": "Out"},
        {"season": 2023, "week": 1, "team": "KC", "gsis_id": " ", "full_name": "Blank Id", "report_status": "Out"},
        {"season": 2023, "week": 1, "team": "KC", "gsis_id": "g4", "full_name": "Nan St", "report_status": nan},
        {"season": 2023, "week": 1, "team": "BUF", "gsis_id": "g5", "full_name": "B", "report_status": "OUT"},
        {"season": 2023, "week": 1, "team": "BUF", "gsis_id": "g6", "full_name": "BQ", "report_status": "QUESTIONABLE"},
        {"season": 2023, "week": 2, "team": "KC", "gsis_id": "g7", "full_name": "W", "report_status": "Out"},
    ])


def test_out_ids_by_team_week_out_and_doubtful_skip_nan_ids():
    result = bsn.out_ids_by_team_week(_injuries_ids_df(), 2023, 1)
    assert result == {"KC": {"g1", "g2"}, "BUF": {"g5"}}   # id kept even with NaN name


def test_questionable_ids_by_team_week():
    result = bsn.questionable_ids_by_team_week(_injuries_ids_df(), 2023, 1)
    assert result == {"KC": {"g3"}, "BUF": {"g6"}}


def test_id_helpers_no_rows_or_no_gsis_column_return_empty():
    assert bsn.out_ids_by_team_week(_injuries_ids_df(), 2099, 1) == {}
    assert bsn.out_ids_by_team_week(_injuries_df(), 2023, 1) == {}      # no gsis_id column
    assert bsn.questionable_ids_by_team_week(_injuries_df(), 2023, 1) == {}


def test_run_backtest_passes_id_sets_to_active_usage():
    import inspect

    src = inspect.getsource(bsn.run_backtest)
    assert "out_ids=" in src and "questionable_ids=" in src
    assert "out_ids_by_team_week(" in src and "questionable_ids_by_team_week(" in src
