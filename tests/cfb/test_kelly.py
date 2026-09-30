import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb.kelly import (
    american_to_decimal,
    apply_day_cap,
    choose_side,
    flat_pnl,
    roi_ci,
    simulate,
    stake_fraction,
)

D110 = 1 + 100 / 110  # -110


def test_american_to_decimal():
    assert american_to_decimal(-110) == pytest.approx(D110)
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(-100) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        american_to_decimal(50)
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_stake_fraction_hand_computed():
    # edge = .55*1.909-1 = .04995 ; /(.909) = .054950 ; *.25 = .0137376
    assert stake_fraction(0.55, 1.909, 0.25) == pytest.approx(0.0137376, abs=1e-6)


def test_stake_fraction_no_edge_and_cap():
    assert stake_fraction(0.45, 1.909, 0.25) == 0.0
    # huge edge: p=.9, d=2 -> full kelly .8 ; quarter = .2 -> capped at .03
    assert stake_fraction(0.9, 2.0, 0.25) == 0.03
    assert stake_fraction(0.9, 2.0, 0.25, cap=0.05) == 0.05


def test_apply_day_cap():
    assert apply_day_cap([0.03] * 4) == pytest.approx([0.03] * 4)  # 12% <= 15%
    out = apply_day_cap([0.03] * 8)  # 24% -> 15%
    assert sum(out) == pytest.approx(0.15)
    assert out == pytest.approx([0.15 / 8] * 8)
    out = apply_day_cap([0.03, 0.06, 0.09, 0.12])  # 30% -> halve
    assert out == pytest.approx([0.015, 0.03, 0.045, 0.06])
    assert apply_day_cap([]) == []
    assert apply_day_cap([0.0, 0.0]) == [0.0, 0.0]


def test_choose_side_picks_larger_edge():
    # home: .60*1.909-1 = .1454 ; away: .40*1.909-1 = -.2364
    side, edge = choose_side(0.60, 1.909, 1.909, 0.02)
    assert side == "home"
    assert edge == pytest.approx(0.1454, abs=1e-4)
    side, edge = choose_side(0.30, 1.909, 1.909, 0.02)
    assert side == "away"
    assert edge == pytest.approx(0.7 * 1.909 - 1)


def test_choose_side_min_edge_and_labels():
    assert choose_side(0.60, 1.909, 1.909, 0.20) is None
    assert choose_side(0.50, 1.909, 1.909, 0.01) is None
    # exactly at threshold counts (>=)
    p = 0.55
    e = p * 2.0 - 1
    assert choose_side(p, 2.0, 1.5, e - 1e-12) == ("home", pytest.approx(e))
    # asymmetric prices: away price long enough to win with p_away = .45
    side, edge = choose_side(0.55, 1.7, 2.4, 0.02)
    assert side == "away"
    assert edge == pytest.approx(0.45 * 2.4 - 1)
    # over/under labels
    assert choose_side(0.6, 1.909, 1.909, 0.02, sides=("over", "under"))[0] == "over"
    assert choose_side(0.4, 1.909, 1.909, 0.02, sides=("over", "under"))[0] == "under"


def test_choose_side_tie_goes_home():
    # symmetric edges: p=.5, d_home=d_away=2.2 -> both .1
    side, edge = choose_side(0.5, 2.2, 2.2, 0.02)
    assert side == "home"
    assert edge == pytest.approx(0.1)


def _bets(rows):
    return pd.DataFrame(rows, columns=["day", "stake_frac", "dec", "result"])


def test_push_leaves_bankroll_unchanged():
    r = simulate(_bets([("d1", 0.03, D110, "push")]))
    assert r["end_bankroll"] == pytest.approx(100.0)
    assert r["profit"] == pytest.approx(0.0)
    assert r["log_growth_total"] == pytest.approx(0.0)
    assert r["n_bets"] == 1
    assert r["staked"] == pytest.approx(3.0)
    assert r["roi"] == pytest.approx(0.0)
    assert r["max_drawdown"] == pytest.approx(0.0)


def test_simulate_three_day_script():
    bets = _bets([
        ("d1", 0.02, 2.0, "win"),    # stake 2, +2
        ("d1", 0.02, 2.0, "loss"),   # stake 2, -2  -> day end 100
        ("d2", 0.03, D110, "win"),   # stake 3, +3*(100/110)
        ("d3", 0.03, D110, "loss"),  # stake .03*b2
    ])
    b1 = 100.0
    b2 = b1 + 3 * (D110 - 1)
    b3 = b2 - 0.03 * b2
    r = simulate(bets)
    assert r["end_bankroll"] == pytest.approx(b3)
    assert r["n_bets"] == 4
    assert r["staked"] == pytest.approx(2 + 2 + 3 + 0.03 * b2)
    assert r["profit"] == pytest.approx(b3 - 100)
    assert r["roi"] == pytest.approx((b3 - 100) / (7 + 0.03 * b2))
    assert r["log_growth_total"] == pytest.approx(math.log(b3 / 100))
    assert r["log_growth_per_bet"] == pytest.approx(math.log(b3 / 100) / 4)
    # numbers: b2 = 102.72727, b3 = 99.64545
    assert b3 == pytest.approx(99.6455, abs=1e-4)


def test_stakes_use_start_of_day_bankroll():
    # two same-day bets each 10% at dec 2: win then loss. Both staked on 100 -> 10 each.
    r = simulate(_bets([("d1", 0.1, 2.0, "win"), ("d1", 0.1, 2.0, "win")]))
    assert r["end_bankroll"] == pytest.approx(120.0)  # not 121
    assert r["staked"] == pytest.approx(20.0)


def test_drawdown_known_path():
    # day-end path: 100 -> 110 -> 88 -> 99 ; peak 110 trough 88 -> .2
    bets = _bets([
        ("d1", 0.10, 2.0, "win"),
        ("d2", 0.20, 2.0, "loss"),
        ("d3", 0.125, 2.0, "win"),
    ])
    r = simulate(bets)
    assert r["end_bankroll"] == pytest.approx(99.0)
    assert r["max_drawdown"] == pytest.approx(0.2)


def test_drawdown_from_start_and_monotone_up():
    r = simulate(_bets([("d1", 0.1, 2.0, "loss")]))
    assert r["max_drawdown"] == pytest.approx(0.1)
    r = simulate(_bets([("d1", 0.1, 2.0, "win"), ("d2", 0.1, 2.0, "win")]))
    assert r["max_drawdown"] == pytest.approx(0.0)


def test_simulate_empty_and_custom_start():
    r = simulate(_bets([]))
    assert r["end_bankroll"] == 100.0
    assert r["n_bets"] == 0
    assert r["roi"] == 0.0
    assert r["log_growth_per_bet"] == 0.0
    r = simulate(_bets([("d1", 0.1, 2.0, "win")]), start=1000.0)
    assert r["end_bankroll"] == pytest.approx(1100.0)
    assert r["profit"] == pytest.approx(100.0)


def test_flat_pnl():
    bets = _bets([
        ("d1", 0.02, 2.0, "win"),
        ("d1", 0.02, 2.0, "loss"),
        ("d2", 0.02, D110, "win"),
        ("d2", 0.02, D110, "push"),
    ])
    r = flat_pnl(bets)
    assert r["n_bets"] == 4
    assert r["staked"] == pytest.approx(40.0)
    assert r["profit"] == pytest.approx(10 - 10 + 10 * (D110 - 1))
    assert r["roi"] == pytest.approx(r["profit"] / 40.0)
    assert flat_pnl(_bets([])) == {"n_bets": 0, "staked": 0.0, "profit": 0.0, "roi": 0.0}


def _clustered(n_clusters=30, per=5, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        for _ in range(per):
            rows.append((2020 + c % 4, 1 + c // 4, f"d{c}", 0.02, 1.909,
                         "win" if rng.random() < 0.6 else "loss"))
    return pd.DataFrame(rows, columns=["season", "week", "day", "stake_frac", "dec", "result"])


def test_roi_ci_brackets_point_estimate_and_is_seeded():
    bets = _clustered()
    lo, hi = roi_ci(bets, n_boot=500, seed=3)
    assert lo < hi
    win = bets["result"] == "win"
    profit = np.where(win, 0.02 * 0.909, -0.02)
    point = profit.sum() / (0.02 * len(bets))
    assert lo < point < hi
    assert roi_ci(bets, n_boot=500, seed=3) == (lo, hi)
    assert roi_ci(bets, n_boot=500, seed=4) != (lo, hi)


def test_roi_ci_all_winners_positive_and_degenerate():
    rows = [(2023, w, f"d{w}", 0.02, 2.0, "win") for w in range(1, 11)]
    bets = pd.DataFrame(rows, columns=["season", "week", "day", "stake_frac", "dec", "result"])
    lo, hi = roi_ci(bets, n_boot=200)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    lo, hi = roi_ci(bets.iloc[0:0])
    assert math.isnan(lo) and math.isnan(hi)


def test_roi_ci_clusters_by_season_week():
    # Two clusters (2023 wk1 all wins, 2023 wk2 all losses): resamples are only
    # ever (win,win)=+1, (win,loss)=0, (loss,loss)=-1 in ROI terms.
    rows = [(2023, 1, "a", 0.02, 2.0, "win")] * 5 + [(2023, 2, "b", 0.02, 2.0, "loss")] * 5
    bets = pd.DataFrame(rows, columns=["season", "week", "day", "stake_frac", "dec", "result"])
    lo, hi = roi_ci(bets, n_boot=2000, seed=0)
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(1.0)


def test_stake_fraction_non_finite_or_no_odds_is_zero():
    nan, inf = float("nan"), float("inf")
    assert stake_fraction(0.55, nan, 0.25) == 0.0
    assert stake_fraction(nan, 1.909, 0.25) == 0.0
    assert stake_fraction(0.55, inf, 0.25) == 0.0
    assert stake_fraction(inf, 1.909, 0.25) == 0.0
    assert stake_fraction(0.9, 1.0, 0.25) == 0.0
    assert stake_fraction(0.9, 0.5, 0.25) == 0.0
