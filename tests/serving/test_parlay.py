"""Tests for the pure auto-parlay builder (sportsmodel.serving.parlay)."""
from __future__ import annotations

import pytest

from sportsmodel.serving.board import decimal_odds, ev
from sportsmodel.serving.parlay import (
    MIN_PARLAY_LEGS,
    american_from_decimal,
    build_parlay,
    parlay_id,
)


def _leg(game_pk, side="home", market="moneyline", true_prob=0.78,
         book_prices=None, matchup="Away @ Home", ev_val=0.04, commence="2026-09-20T17:00:00Z"):
    book_prices = book_prices if book_prices is not None else [("fanduel", -300)]
    return {
        "sport": "cfb", "game_pk": game_pk, "market": market, "side": side,
        "matchup": matchup, "commence_time": commence,
        "true_prob": true_prob, "ev": ev_val, "book_prices": book_prices,
    }


def test_american_from_decimal_roundtrips_common_prices():
    assert american_from_decimal(2.0) == 100
    assert american_from_decimal(1.5) == -200          # -100 / 0.5
    assert american_from_decimal(decimal_odds(-300)) == -300
    assert american_from_decimal(decimal_odds(150)) == 150


def test_none_below_min_legs():
    assert build_parlay([_leg(1)]) is None
    assert build_parlay([]) is None


def test_two_juiced_legs_form_a_plus_ev_parlay():
    legs = [_leg(1, true_prob=0.78), _leg(2, true_prob=0.78)]
    p = build_parlay(legs)
    assert p is not None
    assert p["n_legs"] == 2
    assert p["book"] == "fanduel"
    # decimal is the product of the per-leg decimals at the chosen book
    expected_dec = decimal_odds(-300) ** 2
    assert p["parlay_price"] == american_from_decimal(expected_dec)
    # combined true prob is the product; EV = product(1+ev_i)-1 > each leg
    assert p["true_prob"] == pytest.approx(0.78 * 0.78)
    assert p["ev"] == pytest.approx(ev(0.78 * 0.78, p["parlay_price"]), abs=1e-9)
    assert p["ev"] > 0
    # the combined price is far more palatable than -300
    assert p["parlay_price"] > -300


def test_one_leg_per_game_dedupe_keeps_highest_ev():
    # same game, two markets -> only one leg (the higher-ev one) may appear
    legs = [
        _leg(1, market="moneyline", ev_val=0.03),
        _leg(1, market="spread", ev_val=0.06),
        _leg(2, market="moneyline", ev_val=0.04),
    ]
    p = build_parlay(legs)
    assert p["n_legs"] == 2
    games = {l["game_pk"] for l in p["legs"]}
    assert games == {1, 2}
    leg1 = next(l for l in p["legs"] if l["game_pk"] == 1)
    assert leg1["market"] == "spread"  # the higher-ev market for game 1


def test_caps_at_max_legs_taking_highest_ev():
    legs = [_leg(i, ev_val=0.02 * i) for i in range(1, 6)]  # 5 games, ev 0.02..0.10
    p = build_parlay(legs, max_legs=3)
    assert p["n_legs"] == 3
    # keeps the three highest-ev games (3,4,5)
    assert {l["game_pk"] for l in p["legs"]} == {3, 4, 5}


def test_picks_best_book_across_legs():
    # book A prices both legs at -300; book B prices both better (-250) -> B wins
    legs = [
        _leg(1, book_prices=[("bookA", -300), ("bookB", -250)]),
        _leg(2, book_prices=[("bookA", -300), ("bookB", -250)]),
    ]
    p = build_parlay(legs)
    assert p["book"] == "bookB"
    assert p["parlay_price"] == american_from_decimal(decimal_odds(-250) ** 2)


def test_none_when_no_single_book_covers_min_legs():
    # each leg only at its own book -> no book covers >= 2 legs
    legs = [
        _leg(1, book_prices=[("bookA", -300)]),
        _leg(2, book_prices=[("bookB", -300)]),
    ]
    assert build_parlay(legs) is None


def test_none_when_combined_ev_below_floor():
    # legs individually ~break-even (true_prob barely above implied) -> tiny EV
    legs = [_leg(1, true_prob=0.7505), _leg(2, true_prob=0.7505)]  # -300 implied .75
    p = build_parlay(legs, min_ev=0.05)
    assert p is None


def test_parlay_id_is_stable_and_order_independent():
    a = [{"game_pk": 2, "market": "moneyline", "side": "home"},
         {"game_pk": 1, "market": "spread", "side": "away"}]
    b = list(reversed(a))
    assert parlay_id(a) == parlay_id(b)


def test_min_parlay_legs_is_two():
    assert MIN_PARLAY_LEGS == 2
