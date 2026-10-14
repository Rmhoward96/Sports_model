"""Closed-form k, grade-row to leg conversion, version fallback, pooling and the 300-leg rules. No IO."""
import random

import pytest

from sportsmodel.serving.props_ev import MLB_SIM_TO_ODDS_MARKET, SIM_TO_ODDS_MARKET
from sportsmodel.shrink import fit
from sportsmodel.shrink.formula import market_prob_over


def leg(p=0.6, m=0.5, y=1, market="rush_yds", version="nfl-sim-ml-v2", season=2026, week=1, game_pk=1):
    return {"season": season, "week": week, "game_pk": game_pk, "market": market, "p": p, "m": m, "y": y,
            "version": version}


def legs_with_k(k, n, **kw):
    """n legs whose closed-form k is exactly k: p - m = 0.1 everywhere and y = m + k (p - m) (an expected outcome)."""
    return [leg(p=0.6, m=0.5, y=0.5 + k * 0.1, **kw) for _ in range(n)]


def grade(**kw):
    base = {"game_pk": 1, "player_id": "00-1", "market": "rush_yds", "p_over": 0.6, "over_price": -110,
            "under_price": -110, "lean": "over", "result": "hit", "season": 2026, "week": 3,
            "model_version": "nfl-sim-ml-v2"}
    return base | kw


# ---- the constants ---------------------------------------------------------------------------------------------

def test_the_300_leg_constants_are_the_specs():
    assert fit.MIN_POOL_LEGS == 300 and fit.PRIOR_LEGS == 300


# ---- closed-form k ---------------------------------------------------------------------------------------------

def test_closed_form_recovers_a_known_k_exactly_on_expected_outcomes():
    assert fit.closed_form_k(legs_with_k(0.3, 50)) == pytest.approx(0.3)
    assert fit.closed_form_k(legs_with_k(0.0, 50)) == pytest.approx(0.0, abs=1e-12)
    assert fit.closed_form_k(legs_with_k(1.0, 50)) == pytest.approx(1.0)


def test_closed_form_recovers_a_known_k_on_simulated_bernoulli_outcomes():
    rng = random.Random(7)
    legs = []
    for _ in range(40_000):
        m = rng.uniform(0.35, 0.65)
        p = min(max(m + rng.uniform(-0.15, 0.15), 0.02), 0.98)
        q = m + 0.30 * (p - m)                       # the market is right 70% of the way to the truth: true k = 0.30
        legs.append(leg(p=p, m=m, y=1 if rng.random() < q else 0))
    assert fit.closed_form_k(legs) == pytest.approx(0.30, abs=0.05)


def test_closed_form_clamps_to_zero_and_one():
    assert fit.closed_form_k([leg(p=0.7, m=0.5, y=0), leg(p=0.3, m=0.5, y=1)]) == 0.0      # the model points the wrong way
    assert fit.closed_form_k(legs_with_k(2.5, 10)) == 1.0                                   # the model is under-confident


def test_zero_variance_and_no_legs_give_zero():
    assert fit.closed_form_k([leg(p=0.5, m=0.5, y=1) for _ in range(10)]) == 0.0
    assert fit.closed_form_k([]) == 0.0


# ---- grade rows to legs ----------------------------------------------------------------------------------------

def test_a_row_becomes_a_leg_with_the_boards_market_key_and_the_over_outcome():
    [l] = fit.legs_from_grades([grade(market="rec_yds", over_price=-120, under_price=100)])
    assert l["market"] == "reception_yds" and l["season"] == 2026 and l["week"] == 3 and l["version"] == "nfl-sim-ml-v2"
    assert l["p"] == 0.6 and l["m"] == pytest.approx(market_prob_over(-120, 100)) and l["y"] == 1


@pytest.mark.parametrize("lean,result,y", [("over", "hit", 1), ("over", "miss", 0), ("under", "hit", 0), ("under", "miss", 1)])
def test_y_is_one_when_the_over_hit_whatever_the_lean(lean, result, y):
    [l] = fit.legs_from_grades([grade(lean=lean, result=result)])
    assert l["y"] == y


def test_pushes_are_excluded():
    assert fit.legs_from_grades([grade(result="push")]) == []


@pytest.mark.parametrize("over,under", [(None, -110), (-110, None), (0, -110), (-110, 0), (-99, -110), (99, -110),
                                        (50, -110), (float("nan"), -110), ("n/a", -110)])
def test_rows_with_a_missing_or_invalid_american_price_are_excluded(over, under):
    assert fit.legs_from_grades([grade(over_price=over, under_price=under)]) == []


@pytest.mark.parametrize("bad", [dict(p_over=None), dict(p_over=float("nan")), dict(lean=None), dict(lean="push"),
                                 dict(result=None), dict(season=None), dict(week=None)])
def test_rows_with_no_probability_lean_result_or_week_are_excluded(bad):
    assert fit.legs_from_grades([grade(**bad)]) == []


def test_markets_the_board_does_not_serve_are_dropped():
    rows = [grade(market="pass_yds"), grade(market="rush_att"), grade(market="made_up"), grade(market="receptions")]
    assert [l["market"] for l in fit.legs_from_grades(rows)] == ["receptions"]


def test_a_missing_model_version_is_kept_as_none():
    [l] = fit.legs_from_grades([grade(model_version=None)])
    assert l["version"] is None


# ---- version selection -----------------------------------------------------------------------------------------

def test_the_current_version_is_used_when_it_has_300_legs():
    legs = legs_with_k(0.2, 300, version="v2") + legs_with_k(0.9, 500, version="v1")
    sample, used = fit.select_sample(legs, "v2")
    assert used == "v2" and len(sample) == 300 and {l["version"] for l in sample} == {"v2"}


def test_299_legs_of_the_current_version_fall_back_to_every_version():
    legs = legs_with_k(0.2, 299, version="v2") + legs_with_k(0.9, 500, version="v1") + legs_with_k(0.5, 5, version=None)
    sample, used = fit.select_sample(legs, "v2")
    assert used == fit.ALL_VERSIONS == "all" and len(sample) == 804


def test_no_current_version_pools_everything():
    assert fit.select_sample(legs_with_k(0.2, 400), None)[1] == "all"


# ---- the table --------------------------------------------------------------------------------------------------

def test_fewer_than_300_pooled_legs_means_k_zero_for_every_market():
    t = fit.fit_table(legs_with_k(0.4, 299, market="rush_yds"), None)
    assert t["pooled"]["n"] == 299 and t["pooled"]["k"] == 0.0 and t["pooled"]["k_raw"] == pytest.approx(0.4)
    assert all(e["k"] == 0.0 for e in t["markets"]["nfl"].values())
    assert t["markets"]["nfl"]["rush_yds"]["n"] == 299 and t["markets"]["nfl"]["rush_yds"]["k_raw"] == pytest.approx(0.4)


def test_exactly_300_pooled_legs_fits():
    t = fit.fit_table(legs_with_k(0.4, 300, market="rush_yds"), None)
    assert t["pooled"]["k"] == pytest.approx(0.4)
    assert t["markets"]["nfl"]["rush_yds"]["k"] == pytest.approx(0.4)             # one market: k_raw == k_pool, so the blend is 0.4


def test_each_market_is_partially_pooled_toward_the_pooled_k():
    legs = legs_with_k(0.10, 100, market="rush_yds") + legs_with_k(0.50, 300, market="receptions") \
        + legs_with_k(0.30, 700, market="reception_yds")
    t = fit.fit_table(legs, None)
    k_pool = fit.closed_form_k(legs)
    assert t["pooled"]["n"] == 1100 and t["pooled"]["k"] == pytest.approx(k_pool)
    rush = t["markets"]["nfl"]["rush_yds"]
    assert rush["n"] == 100 and rush["k_raw"] == pytest.approx(0.10)
    assert rush["k"] == pytest.approx((100 * 0.10 + 300 * k_pool) / (100 + 300))
    rec = t["markets"]["nfl"]["receptions"]
    assert rec["k"] == pytest.approx((300 * 0.50 + 300 * k_pool) / (300 + 300))


def test_a_market_with_no_legs_gets_k_zero_not_the_pooled_k():
    t = fit.fit_table(legs_with_k(0.4, 400, market="rush_yds"), None)
    assert t["markets"]["nfl"]["pass_tds"] == {"k": 0.0, "n": 0, "k_raw": None}
    assert set(t["markets"]["nfl"]) == set(SIM_TO_ODDS_MARKET.values())


def test_a_market_with_zero_variance_has_raw_k_zero_and_is_pulled_toward_the_pool():
    legs = legs_with_k(0.4, 400, market="rush_yds") + [leg(p=0.5, m=0.5, y=1, market="receptions") for _ in range(50)]
    t = fit.fit_table(legs, None)
    rec = t["markets"]["nfl"]["receptions"]
    assert rec["k_raw"] == 0.0 and rec["k"] == pytest.approx((50 * 0.0 + 300 * t["pooled"]["k"]) / 350)


def test_mlb_is_zero_for_every_market_whatever_the_data():
    t = fit.fit_table(legs_with_k(0.9, 2000), None)
    assert set(t["markets"]["mlb"]) == set(MLB_SIM_TO_ODDS_MARKET.values())
    assert all(e["k"] == 0.0 for e in t["markets"]["mlb"].values())


def test_the_table_reports_the_version_it_used_and_falls_back_with_the_sample():
    legs = legs_with_k(0.2, 320, version="v2") + legs_with_k(0.8, 400, version="v1")
    cur = fit.fit_table(legs, "v2")
    assert cur["model_version_used"] == "v2" and cur["pooled"]["n"] == 320 and cur["pooled"]["k"] == pytest.approx(0.2)
    short = fit.fit_table(legs_with_k(0.2, 100, version="v2") + legs_with_k(0.8, 400, version="v1"), "v2")
    assert short["model_version_used"] == "all" and short["pooled"]["n"] == 500


def test_the_clamped_k_values_stay_in_zero_one():
    t = fit.fit_table(legs_with_k(5.0, 400, market="rush_yds") + legs_with_k(-3.0, 400, market="receptions"), None)
    for e in t["markets"]["nfl"].values():
        assert 0.0 <= e["k"] <= 1.0


# ---- the committed document ------------------------------------------------------------------------------------

def test_the_document_records_k_n_k_raw_pooled_version_fitted_at_and_the_data_window():
    legs = legs_with_k(0.3, 200, week=2) + legs_with_k(0.3, 150, week=5, market="receptions") + legs_with_k(0.3, 10, week=1, season=2025)
    doc = fit.build_document(legs, "nfl-sim-ml-v2", "2026-10-20T15:31:00+00:00")
    assert doc["schema"] == 1 and doc["fitted_at"] == "2026-10-20T15:31:00+00:00" and doc["current_version"] == "nfl-sim-ml-v2"
    assert doc["model_version_used"] == "nfl-sim-ml-v2"
    assert doc["data_window"] == {"first": [2025, 1], "last": [2026, 5], "n_legs": 360}
    assert doc["pooled"]["sport"] == "nfl" and doc["pooled"]["n"] == 360 and doc["pooled"]["k"] == pytest.approx(0.3)
    assert doc["markets"]["nfl"]["rush_yds"]["n"] == 210 and set(doc["markets"]) == {"nfl", "mlb"}
    import json
    assert json.loads(json.dumps(doc)) == doc                                   # plain JSON types only


def test_an_empty_fit_is_a_valid_all_zero_document():
    doc = fit.build_document([], None, "2026-10-20T15:31:00+00:00")
    assert doc["data_window"] == {"first": None, "last": None, "n_legs": 0}
    assert doc["pooled"]["k"] == 0.0 and all(e["k"] == 0.0 for s in doc["markets"].values() for e in s.values())


def test_a_fit_on_fewer_eligible_legs_than_the_committed_file_is_flagged():
    old = fit.build_document(legs_with_k(0.3, 500), None, "t")
    assert fit.has_fewer_legs(old, fit.build_document(legs_with_k(0.3, 499), None, "t")) is True
    assert fit.has_fewer_legs(old, fit.build_document(legs_with_k(0.3, 500), None, "t")) is False
    assert fit.has_fewer_legs(old, fit.build_document(legs_with_k(0.3, 900), None, "t")) is False
    for odd in (None, {}, {"data_window": None}, {"data_window": {"n_legs": "x"}}, []):
        assert fit.has_fewer_legs(odd, old) is False                               # no usable previous count: nothing to protect
