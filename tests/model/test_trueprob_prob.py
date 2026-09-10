import pytest

from sportsmodel.model.trueprob_prob import (
    MARGIN_OFFSET,
    TOTAL_MAX,
    DESK_MAX_PTS,
    DESK_MAX_PROB_DELTA,
    win_prob,
    cover_prob,
    over_prob,
    desk_margin_shift,
    apply_desk,
)

SIGMA = 13.2


def test_win_prob_at_zero_margin_is_near_half():
    assert win_prob(0.0, SIGMA) == pytest.approx(0.5, abs=0.02)


def test_win_prob_increases_with_positive_margin():
    assert win_prob(7.0, SIGMA) > 0.5


def test_win_prob_strictly_increasing_in_margin():
    margins = [-6, -3, 0, 3, 6, 9]
    probs = [win_prob(m, SIGMA) for m in margins]
    for a, b in zip(probs, probs[1:]):
        assert b > a


def test_cover_prob_in_bounds():
    p = cover_prob(0.0, SIGMA, -3.5)
    assert 0.0 <= p <= 1.0


def test_cover_prob_large_negative_home_line_is_low():
    # home covers iff margin + home_line > 0 (prob_cover's documented semantics).
    # home_line = -50 means home must win by more than 50 to cover -> near 0
    # when the model's expected margin is a pick'em (0).
    p = cover_prob(0.0, SIGMA, -50.0)
    assert p < 0.01


def test_cover_prob_large_positive_home_line_is_high():
    # home_line = +50 means home covers unless it loses by more than 50 -> near 1.
    p = cover_prob(0.0, SIGMA, 50.0)
    assert p > 0.99


def test_over_prob_in_bounds():
    p = over_prob(44.0, 10.0, 40.0)
    assert 0.0 <= p <= 1.0


def test_over_prob_above_low_line_is_high():
    p = over_prob(44.0, 10.0, 5.0)
    assert p > 0.95


def test_over_prob_above_high_line_is_low():
    p = over_prob(44.0, 10.0, 100.0)
    assert p < 0.05


def test_desk_margin_shift_none_is_zero():
    assert desk_margin_shift(None) == 0.0


def test_desk_margin_shift_high_home_positive():
    pick = {"conviction_tier": "high", "spread_side": "home", "agrees_with_model": True}
    shift = desk_margin_shift(pick)
    assert shift == pytest.approx(1.0 * DESK_MAX_PTS)


def test_desk_margin_shift_high_away_negative():
    pick = {"conviction_tier": "high", "spread_side": "away", "agrees_with_model": True}
    shift = desk_margin_shift(pick)
    assert shift == pytest.approx(-1.0 * DESK_MAX_PTS)


def test_desk_margin_shift_scales_with_conviction():
    high = desk_margin_shift({"conviction_tier": "high", "spread_side": "home", "agrees_with_model": True})
    med = desk_margin_shift({"conviction_tier": "medium", "spread_side": "home", "agrees_with_model": True})
    low = desk_margin_shift({"conviction_tier": "low", "spread_side": "home", "agrees_with_model": True})
    assert high > med > low > 0


def test_desk_margin_shift_falls_back_to_ml_pick_side():
    pick = {"conviction_tier": "high", "spread_side": None, "ml_pick": "away", "agrees_with_model": True}
    shift = desk_margin_shift(pick)
    assert shift < 0


def test_desk_margin_shift_uses_desk_side_even_on_disagreement():
    pick = {"conviction_tier": "high", "spread_side": "home", "agrees_with_model": False}
    shift = desk_margin_shift(pick)
    assert shift == pytest.approx(1.0 * DESK_MAX_PTS)


def test_desk_margin_shift_ignores_confidence_uses_tier():
    # Confidence is NOT used to size the nudge -- only conviction_tier is.
    # A pick with confidence but no tier -> zero (no tier weight).
    assert desk_margin_shift({"confidence": 0.9, "spread_side": "home"}) == pytest.approx(0.0)
    # Same tier, different confidence -> identical shift (confidence ignored).
    a = desk_margin_shift({"conviction_tier": "high", "confidence": 0.55, "spread_side": "home"})
    b = desk_margin_shift({"conviction_tier": "high", "confidence": 0.95, "spread_side": "home"})
    assert a == b == pytest.approx(1.0 * DESK_MAX_PTS)


def test_desk_margin_shift_sign_toward_away():
    assert desk_margin_shift({"conviction_tier": "high", "spread_side": "away"}) < 0


def test_desk_margin_shift_missing_confidence_falls_back_to_tier():
    pick = {"conviction_tier": "medium", "spread_side": "home"}
    shift = desk_margin_shift(pick)
    assert shift == pytest.approx(0.6 * DESK_MAX_PTS)  # old tier-based value (_TIER_WEIGHTS["medium"])


def test_apply_desk_none_pick_is_identity():
    margin, base, adj = apply_desk(0.0, SIGMA, None)
    assert margin == 0.0
    assert base == pytest.approx(win_prob(0.0, SIGMA))
    assert adj == pytest.approx(base)


def test_apply_desk_high_conviction_home_raises_prob_but_clamped():
    pick = {"conviction_tier": "high", "spread_side": "home", "agrees_with_model": True}
    _, base, adj = apply_desk(0.0, SIGMA, pick)
    assert adj > base
    assert (adj - base) <= DESK_MAX_PROB_DELTA + 1e-9


def test_apply_desk_clamp_bound_holds_even_with_tiny_sigma():
    # tiny sigma makes the raw win_prob shift huge; clamp must still hold.
    pick = {"conviction_tier": "high", "spread_side": "home", "agrees_with_model": True}
    _, base, adj = apply_desk(0.0, 1.0, pick)
    assert abs(adj - base) <= DESK_MAX_PROB_DELTA + 1e-9


def test_apply_desk_low_conviction_moves_less_than_high():
    high_pick = {"conviction_tier": "high", "spread_side": "home", "agrees_with_model": True}
    low_pick = {"conviction_tier": "low", "spread_side": "home", "agrees_with_model": True}
    _, base_h, adj_h = apply_desk(0.0, SIGMA, high_pick)
    _, base_l, adj_l = apply_desk(0.0, SIGMA, low_pick)
    assert (adj_h - base_h) >= (adj_l - base_l) >= 0


def test_apply_desk_away_pick_lowers_prob():
    pick = {"conviction_tier": "high", "spread_side": "away", "agrees_with_model": True}
    _, base, adj = apply_desk(0.0, SIGMA, pick)
    assert adj < base
    assert (base - adj) <= DESK_MAX_PROB_DELTA + 1e-9


def test_constants():
    assert MARGIN_OFFSET == 75
    assert TOTAL_MAX == 120
    assert DESK_MAX_PTS == 3.5
    assert DESK_MAX_PROB_DELTA == 0.10
