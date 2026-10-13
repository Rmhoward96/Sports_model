"""passtest.py: collecting vs judged, per-market minimums, each of the four criteria, the cluster bootstrap. Pure."""
import inspect

import pytest

from sportsmodel.early import passtest

from .builders import many, res, spreads, totals


# ---- frozen numbers ----------------------------------------------------------------------------------------

def test_the_frozen_numbers_match_the_spec():
    assert (passtest.MIN_GRADED_TOTAL, passtest.MIN_CFB_TOTALS, passtest.MIN_OTHER_MARKET) == (150, 60, 40)
    assert (passtest.MIN_MEAN_CLOSE_EV, passtest.MIN_BEAT_RATE) == (0.015, 0.55)
    assert (passtest.BOOTSTRAP_RESAMPLES, passtest.BOOTSTRAP_SEED, passtest.LOWER_PERCENTILE) == (10_000, 20261013, 2.5)
    sig = inspect.signature(passtest.cluster_bootstrap_lower).parameters
    assert sig["resamples"].default == 10_000 and sig["seed"].default == 20261013 and sig["percentile"].default == 2.5


# ---- collecting versus judged ----------------------------------------------------------------------------------

def test_before_the_minimums_the_status_is_collecting_and_nothing_can_pass_or_fail():
    assert passtest.decide([], None, "2026-10-20")["status"] == "collecting"
    out = passtest.decide(totals(59) + spreads(45), None, "2026-10-20")          # 149 usable: one short of 150 and of 60 totals
    assert out["status"] == "collecting" and "markets" not in out
    assert out["graded"] == 59 + 90 and out["cfb_totals"] == 59


def test_both_conditions_are_needed_150_graded_and_60_cfb_totals():
    assert passtest.ready(totals(60) + spreads(45)) is True                      # 60 + 90 = 150
    assert passtest.ready(totals(60) + spreads(44)) is False                     # 148 graded
    assert passtest.ready(totals(59) + spreads(46)) is False                     # 151 graded but only 59 totals
    assert passtest.ready(totals(150)) is True


def test_only_tickets_with_a_pinnacle_close_count_toward_the_minimums():
    missing = many(40, kind="missing", ev=None, game_start=9000)
    assert passtest.ready(totals(60) + spreads(45) + missing) is True
    assert passtest.ready(totals(60) + spreads(44) + missing) is False
    nulls = [res(kind="pinnacle_novig", ev=None, game_pk=g) for g in range(8000, 8010)]
    assert passtest.ready(totals(60) + spreads(44) + nulls) is False


def test_judging_happens_once_and_is_frozen_afterwards():
    first = passtest.decide(totals(60) + spreads(45), None, "2026-12-01")
    assert first["status"] == "judged" and first["judged_on"] == "2026-12-01"
    later = passtest.decide(totals(60, ev=-0.5) + spreads(45, ev=-0.5), first, "2026-12-08")
    assert later is first and later["judged_on"] == "2026-12-01"                  # new, worse data cannot re-judge it
    still = passtest.decide([], {"status": "collecting"}, "2026-12-08")
    assert still["status"] == "collecting"                                         # a collecting previous is not frozen


# ---- per-market minimums -----------------------------------------------------------------------------------------

def test_cfb_spreads_and_nfl_spreads_need_40_graded_tickets_else_insufficient():
    rows = (totals(100) + many(50, sport="cfb", market="spread", game_start=2000)
            + many(39, sport="nfl", market="spread", game_start=3000))
    out = passtest.decide(rows, None, "d")
    assert out["status"] == "judged"
    verdicts = {k: v["verdict"] for k, v in out["markets"].items()}
    assert verdicts == {"cfb_total": "PASS", "cfb_spread": "PASS", "nfl_spread": "INSUFFICIENT"}
    assert "39 graded tickets" in out["markets"]["nfl_spread"]["detail"] and out["markets"]["nfl_spread"]["criteria"] == []


def test_forty_graded_spread_tickets_are_enough_to_be_judged():
    rows = totals(100) + many(40, sport="cfb", market="spread", game_start=2000) + many(40, sport="nfl", market="spread", game_start=3000)
    out = passtest.decide(rows, None, "d")
    assert {v["verdict"] for v in out["markets"].values()} == {"PASS"}


def test_each_market_is_judged_on_its_own_tickets():
    rows = totals(60) + many(45, sport="cfb", market="spread", game_start=2000, ev=-0.05) + many(45, sport="nfl", market="spread", game_start=3000)
    out = passtest.decide(rows, None, "d")
    assert {k: v["verdict"] for k, v in out["markets"].items()} == {"cfb_total": "PASS", "cfb_spread": "FAIL", "nfl_spread": "PASS"}


# ---- the four criteria, one failing at a time --------------------------------------------------------------------------

def verdict(rows, sport="cfb", market="total"):
    v = passtest.judge_market(sport, market, rows)
    return v["verdict"], [c["passed"] for c in v["criteria"]]


def test_all_four_criteria_met_passes():
    assert verdict(totals(60)) == ("PASS", [True, True, True, True])


def test_criterion_1_mean_close_ev_at_least_one_and_a_half_percent():
    assert verdict(totals(60, ev=0.015)) == ("PASS", [True, True, True, True])            # the boundary is inclusive
    assert verdict(totals(60, ev=0.0149)) == ("FAIL", [False, True, True, True])


def test_criterion_2_clustered_lower_bound_above_zero():
    noisy = many(34, ev=0.3, game_start=1000) + many(26, ev=-0.3, game_start=2000)          # mean +4%, 56.7% beat the close
    assert verdict(noisy) == ("FAIL", [True, False, True, True])


def test_criterion_3_beat_close_rate_at_least_55_percent():
    rows = many(29, ev=0.10, game_start=1000) + many(31, ev=-0.002, game_start=2000)         # mean +4.7%, only 48% beat
    assert verdict(rows) == ("FAIL", [True, True, False, True])
    exactly = many(33, ev=0.05, game_start=1000) + many(27, ev=0.02, beat=False, game_start=2000)   # 33 / 60 = 55.0%
    assert verdict(exactly)[1][2] is True


def test_criterion_4_tickets_still_available_a_poll_later_must_also_have_a_positive_mean():
    never = totals(60, avail=False)
    assert verdict(never) == ("FAIL", [True, True, True, False])
    unobserved = totals(60, avail=None)
    assert verdict(unobserved) == ("FAIL", [True, True, True, False])                         # NULL is not evidence of availability
    rows = many(10, ev=-0.2, avail=True, game_start=5000) + many(50, ev=0.2, avail=False, game_start=1000)
    v = passtest.judge_market("cfb", "total", rows)
    assert v["criteria"][3]["passed"] is False and v["criteria"][0]["passed"] is True
    assert "-0.2000 over 10 tickets" in v["criteria"][3]["detail"]


def test_realized_roi_is_reported_never_gated():
    losing = totals(60, units=-1.0, result="L")                      # every ticket lost money, but beat the close
    v = passtest.judge_market("cfb", "total", losing)
    assert v["verdict"] == "PASS" and v["stats"]["roi"] == -1.0 and v["stats"]["l"] == 60 and v["stats"]["units"] == -60.0


def test_market_stats_counts_results_units_and_availability_share():
    rows = [res(avail=True, result="W", units=0.9), res(avail=False, result="L", units=-1.0, game_pk=2),
            res(avail=None, result="P", units=0.0, game_pk=3)]
    s = passtest.market_stats(rows)
    assert (s["n"], s["games"], s["w"], s["l"], s["p"]) == (3, 3, 1, 1, 1)
    assert s["units"] == pytest.approx(-0.1) and s["roi"] == pytest.approx(-0.1 / 3)
    assert s["avail_share"] == 0.5 and s["avail_observed"] == 2 and s["avail_n"] == 1


# ---- the cluster bootstrap ---------------------------------------------------------------------------------------

def test_a_constant_close_ev_has_that_value_as_its_lower_bound():
    assert passtest.cluster_bootstrap_lower(many(20, ev=0.03)) == pytest.approx(0.03)


def test_resampling_whole_games_not_tickets_keeps_the_correlated_tickets_together():
    game_a = [res(game_pk=1, ev=0.5) for _ in range(30)]
    game_b = [res(game_pk=2, ev=-0.5) for _ in range(30)]
    # ticket-level resampling would give a tight interval around 0; two games can both be drawn as B: mean -0.5
    assert passtest.cluster_bootstrap_lower(game_a + game_b) == pytest.approx(-0.5)


def test_same_seed_same_answer_and_a_different_seed_a_different_one():
    rows = [res(game_pk=g, ev=((g * 37) % 11 - 4) / 100) for g in range(80)]
    first = passtest.cluster_bootstrap_lower(rows)
    assert passtest.cluster_bootstrap_lower(rows) == first
    assert passtest.cluster_bootstrap_lower(list(reversed(rows))) == first            # input order does not matter
    assert passtest.cluster_bootstrap_lower(rows, seed=1) != first


def test_the_bootstrap_uses_seed_20261013_and_10000_resamples(monkeypatch):
    seen = {"seeds": [], "rows": 0}
    real = passtest.np.random.default_rng

    class Spy:
        def __init__(self, seed):
            self.rng = real(seed)
            seen["seeds"].append(seed)

        def integers(self, low, high, size):
            seen["rows"] += size[0]
            return self.rng.integers(low, high, size=size)

    monkeypatch.setattr(passtest.np.random, "default_rng", Spy)
    passtest.cluster_bootstrap_lower(many(25, ev=0.01))
    assert seen["seeds"] == [20261013] and seen["rows"] == 10_000


def test_the_lower_bound_is_the_2_5th_percentile_below_the_mean():
    rows = [res(game_pk=g, ev=((g * 37) % 11 - 4) / 100) for g in range(80)]
    mean = passtest.market_stats(rows, bootstrap=False)["mean_close_ev"]
    assert passtest.cluster_bootstrap_lower(rows) < mean


def test_an_empty_sample_has_no_bound():
    assert passtest.cluster_bootstrap_lower([]) is None
