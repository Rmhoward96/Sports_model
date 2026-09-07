"""Unit tests for scripts/backtest_cfb_priors.py's PURE metric helpers.

Loads the script module directly (it isn't a package) via importlib, the same
pattern tests/test_grade_predictions.py and tests/cfb/test_generate_cfb.py use.
No network, no DB, no parquet reads -- ats_result/bucket_winrate are pure
functions of plain dicts/numbers. Importing the module itself must not touch
assets/cfb/priors.parquet (it doesn't exist yet in this environment); that
constraint is exercised implicitly by this file collecting/running at all.
"""
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "backtest_cfb_priors.py"
_spec = importlib.util.spec_from_file_location("backtest_cfb_priors", _SCRIPT_PATH)
backtest_cfb_priors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backtest_cfb_priors)

ats_result = backtest_cfb_priors.ats_result
bucket_winrate = backtest_cfb_priors.bucket_winrate
load_decay_config = backtest_cfb_priors.load_decay_config
grade_vs_market = backtest_cfb_priors.grade_vs_market


def test_ats_vs_closing_counts_cover_correctly():
    # model picks home -7; closing home -3; home wins by 5 -> home covers
    # closing -> model (home) wins ATS
    assert ats_result(model_margin=7, closing_home_spread=-3, actual_margin=5) == "win"


def test_ats_vs_closing_model_wrong_side_loses():
    # model picks home -7 (thinks home covers -3); home only wins by 1, so
    # home does NOT cover the -3 closing line -> model's pick loses
    assert ats_result(model_margin=7, closing_home_spread=-3, actual_margin=1) == "loss"


def test_ats_vs_closing_away_pick_wins():
    # model picks away (model_margin=-1, i.e. thinks home favored by less
    # than the -3 close, effectively liking away to cover); away wins
    # outright -> home does not cover -3 -> away covers -> model wins
    assert ats_result(model_margin=-1, closing_home_spread=-3, actual_margin=-2) == "win"


def test_ats_vs_closing_push():
    # actual margin lands exactly on the closing number -> push, regardless
    # of which side the model liked
    assert ats_result(model_margin=7, closing_home_spread=-3, actual_margin=3) == "push"


def test_disagreement_bucket_winrate():
    rows = [{"gap": 8, "ats": "win"}, {"gap": 9, "ats": "loss"}, {"gap": 1, "ats": "win"}]
    wr = bucket_winrate(rows, min_gap=5)
    assert wr == 0.5   # only the two gap>=5 games count


def test_disagreement_bucket_winrate_excludes_pushes_from_denominator():
    rows = [{"gap": 8, "ats": "win"}, {"gap": 9, "ats": "push"}]
    wr = bucket_winrate(rows, min_gap=5)
    assert wr == 1.0   # the push is excluded from both numerator and denominator


def test_disagreement_bucket_winrate_no_qualifying_rows_is_zero():
    rows = [{"gap": 1, "ats": "win"}]
    assert bucket_winrate(rows, min_gap=5) == 0.0


def test_load_decay_config_defaults_when_missing(tmp_path):
    from sportsmodel.cfb.priors import DecayConfig
    missing = tmp_path / "no_such_file.json"
    cfg = load_decay_config(missing)
    assert cfg == DecayConfig(half_life_games=backtest_cfb_priors._DEFAULT_HALF_LIFE_GAMES)


def test_load_decay_config_reads_fitted_values(tmp_path):
    from sportsmodel.cfb.priors import DecayConfig
    path = tmp_path / "priors_decay.json"
    path.write_text('{"half_life_games": 6.0, "prior_floor": 0.05}')
    cfg = load_decay_config(path)
    assert cfg == DecayConfig(half_life_games=6.0, prior_floor=0.05)


def test_load_decay_config_lives_in_sportsmodel_cfb_priors():
    # Task 6's live producer needs DecayConfig at runtime and scripts/ is not
    # an importable package -- the loader (and its default path constant)
    # must live in sportsmodel.cfb.priors, not in this script.
    import sportsmodel.cfb.priors as priors_mod
    assert backtest_cfb_priors.load_decay_config is priors_mod.load_decay_config


def test_grade_vs_market_home_margin_convention_locks_ats_direction():
    # assets/cfb/lines.parquet stores market_spread in HOME-MARGIN convention
    # (positive = home favored), NOT sportsbook convention (negative = home
    # favored). Home favored by 10 -> market_spread=+10. The model likes away
    # (model_margin=+3, i.e. it expects a smaller home margin than the
    # market's +10). Home wins by 15 (actual_margin=+15) -> home covers the
    # sportsbook -10 closing line -> the model's away pick LOSES. This example
    # inverts to "win" if market_spread's convention is mishandled (passed
    # straight into ats_result instead of negated), so it locks the
    # convention with a real assertion rather than only self-consistent
    # ats_result cases.
    result = grade_vs_market(model_margin=3, market_spread=10, actual_margin=15)
    assert result["ats"] == "loss"


def test_clv_proxy_is_outcome_based_and_differs_from_gap():
    # gap is pre-game only: |model_margin - market_spread|, independent of
    # what actually happened. clv_proxy must be a genuinely different,
    # signed, OUTCOME-based quantity -- not just a relabeling of gap.
    #
    # Home favored by 10 (market_spread=+10). Model likes away by a lot
    # (model_margin=-5, well below the +10 threshold) -> gap = 15.
    # Away wins outright (actual_margin=-2), so the model's away pick
    # actually covers the closing number by 12 (threshold - actual_margin
    # = 10 - (-2) = 12) -- a real, decent-sized win, much smaller than the
    # pre-game gap of 15 and with independent meaning.
    result = grade_vs_market(model_margin=-5, market_spread=10, actual_margin=-2)
    assert result["gap"] == 15
    assert result["clv_proxy"] == 12
    assert result["clv_proxy"] != result["gap"]
    assert result["ats"] == "win"  # sign of clv_proxy agrees with the ats grade

    # Same pre-game gap, but the model's away pick LOSES this time (home
    # actually covers) -- clv_proxy flips negative while gap stays positive,
    # proving clv_proxy carries outcome information gap cannot.
    losing = grade_vs_market(model_margin=-5, market_spread=10, actual_margin=20)
    assert losing["gap"] == 15
    assert losing["clv_proxy"] == -10  # threshold - actual_margin = 10 - 20
    assert losing["ats"] == "loss"
