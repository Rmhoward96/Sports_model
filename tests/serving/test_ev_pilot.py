"""Tests for the pure +EV pilot edge/EV engine (sportsmodel.serving.ev_pilot).

Sub-project 4 (+EV desk-driven pilot), task 1 -- see
.superpowers/sdd/2026-09-09-plus-ev-4-desk-driven-pilot/task-1-brief.md.
"""
from __future__ import annotations

import pytest

from sportsmodel.model.trueprob_prob import DESK_MAX_PROB_DELTA, win_prob
from sportsmodel.serving.board import EV_CEILING, USER_BOOKS, best_price, ev, implied_prob, novig
from sportsmodel.serving.ev_pilot import (
    MIN_EDGE,
    MIN_EV,
    SIGMA_MARGIN,
    assemble_games,
    ev_rows_for_game,
    market_margin,
)


def _pickem_game(desk=None):
    """A pick'em game: Pinnacle -110/-110 moneyline+spread (home_line=0.0), and a
    44.5 total at -110/-110. No soft books unless a test adds them."""
    return {
        "sport": "nfl",
        "game_pk": 1,
        "matchup": "Away @ Home",
        "commence_time": "2026-09-13T17:00:00Z",
        "desk": desk,
        "pinnacle": {
            "moneyline": {"home": -110, "away": -110},
            "spread": {"home_line": 0.0, "home": -110, "away": -110},
            "total": {"line": 44.5, "over": -110, "under": -110},
        },
        "books": {
            "moneyline": {"home": [], "away": []},
            "spread": {"home": [], "away": []},
            "total": {"over": [], "under": []},
        },
    }


def test_open_pinnacle_price_seeds_from_pinnacle_price():
    # Every row seeds open_pinnacle_price == pinnacle_price; the upsert freezes
    # it on first insert so CLV can measure the true pick-time price later.
    game = _pickem_game(desk=None)
    for row in ev_rows_for_game(game):
        assert row["open_pinnacle_price"] == row["pinnacle_price"]


def test_market_margin():
    assert market_margin(-3.0) == 3.0
    assert market_margin(3.0) == -3.0


def test_no_desk_pick_passes_on_every_market():
    game = _pickem_game(desk=None)
    rows = ev_rows_for_game(game)
    assert len(rows) == 6  # both sides of moneyline, spread, total
    markets = {r["market"] for r in rows}
    assert markets == {"moneyline", "spread", "total"}
    for row in rows:
        assert row["desk_delta"] == 0.0
        assert row["true_prob"] == row["base_prob"]
        assert row["edge"] == 0.0
        assert row["is_pick"] is False


def test_high_conviction_ml_pick_raises_true_prob_within_clamp_and_is_pick():
    desk = {"ml_pick": "home", "conviction_tier": "high"}
    game = _pickem_game(desk=desk)
    # A soft book posting the same -110 as Pinnacle -- the desk's raised
    # true_prob alone (not a soft-line edge) is what should clear MIN_EV here.
    game["books"]["moneyline"]["home"] = [("fanduel", -110)]
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["side"] == "home"
    assert ml["base_prob"] == novig(-110, -110)
    assert ml["true_prob"] > ml["base_prob"]
    assert ml["true_prob"] - ml["base_prob"] <= DESK_MAX_PROB_DELTA + 1e-9
    assert ml["edge"] > 0
    # The pick comes purely from the desk overlay pushing ev_best past MIN_EV --
    # the soft book posts the SAME -110 as Pinnacle, so the price-vs-price
    # soft-vs-sharp gap is 0 (no line-shopping edge); the edge here is the desk.
    assert abs(ml["soft_vs_sharp_gap"]) < 1e-9
    assert ml["ev_best"] is not None and ml["ev_best"] > MIN_EV
    assert ml["is_pick"] is True
    assert ml["edge"] >= MIN_EDGE

    # spread/total on this game carry no desk pick and no soft-book prices ->
    # untouched (pass).
    spread = next(r for r in rows if r["market"] == "spread")
    total = next(r for r in rows if r["market"] == "total")
    assert spread["desk_delta"] == 0.0 and spread["is_pick"] is False
    assert total["desk_delta"] == 0.0 and total["is_pick"] is False


def test_tiered_desk_pick_drives_desk_delta_and_true_prob():
    # A desk pick with a conviction tier moves true_prob/desk_delta and produces
    # a pick (the desk's own numeric confidence is NOT used to size the nudge).
    desk = {"ml_pick": "home", "conviction_tier": "high"}
    game = _pickem_game(desk=desk)
    game["books"]["moneyline"]["home"] = [("fanduel", -110)]
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["side"] == "home"
    assert ml["desk_delta"] > 0
    assert ml["true_prob"] > ml["base_prob"]
    assert ml["true_prob"] - ml["base_prob"] <= DESK_MAX_PROB_DELTA + 1e-9
    assert ml["is_pick"] is True


def test_higher_tier_desk_pick_yields_larger_desk_delta_than_lower():
    game_high = _pickem_game(desk={"ml_pick": "home", "conviction_tier": "high"})
    game_low = _pickem_game(desk={"ml_pick": "home", "conviction_tier": "low"})
    ml_high = next(r for r in ev_rows_for_game(game_high) if r["market"] == "moneyline")
    ml_low = next(r for r in ev_rows_for_game(game_low) if r["market"] == "moneyline")

    assert ml_high["desk_delta"] >= ml_low["desk_delta"] > 0


def test_ev_best_at_least_ev_pinnacle_when_soft_book_prices_better():
    desk = {"ml_pick": "home", "conviction_tier": "high"}
    game = _pickem_game(desk=desk)
    # Soft book (fanduel) posts a friendlier home price than Pinnacle's -110.
    game["books"]["moneyline"]["home"] = [("fanduel", 120)]
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["best_book"] == "fanduel"
    assert ml["best_price"] == 120
    assert ml["ev_best"] is not None
    assert ml["ev_best"] >= ml["ev_pinnacle"]


def test_line_shopping_pick_with_no_desk_pick():
    # No desk pick at all -- but a soft book prices the (market-favored) home
    # side better than Pinnacle's -110/-110. This should surface a pure
    # line-shopping edge and become a pick with zero desk involvement.
    game = _pickem_game(desk=None)
    game["books"]["moneyline"]["home"] = [("fanduel", 120)]
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["desk_delta"] == 0.0
    assert ml["best_book"] == "fanduel"
    assert ml["best_price"] == 120
    assert ml["best_line_implied"] == implied_prob(120)
    assert ml["soft_vs_sharp_gap"] is not None and ml["soft_vs_sharp_gap"] > 0
    assert ml["ev_best"] is not None and ml["ev_best"] > 0
    assert ml["is_pick"] is True


def test_no_desk_pick_and_best_price_matches_pinnacle_is_not_a_pick():
    # No desk pick, and the "best" soft price is identical to Pinnacle's --
    # no line-shopping bargain, no desk overlay -> should not clear MIN_EV.
    game = _pickem_game(desk=None)
    game["books"]["moneyline"]["home"] = [("fanduel", -110)]
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["desk_delta"] == 0.0
    # The soft price matches Pinnacle's own vig-laden price -- no bargain vs.
    # the no-vig base_prob, so the gap is <= 0, not a positive edge.
    assert ml["soft_vs_sharp_gap"] <= 0
    assert ml["ev_best"] is not None and ml["ev_best"] <= 0
    assert ml["is_pick"] is False


def test_best_line_implied_matches_implied_prob_of_best_price():
    game = _pickem_game(desk=None)
    game["books"]["moneyline"]["home"] = [("fanduel", 150), ("draftkings", 130)]
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["best_price"] == 150
    assert ml["best_line_implied"] == implied_prob(150)


def test_ev_ceiling_blocks_contrived_huge_edge():
    # A lopsided two-way Pinnacle price (home is a big underdog priced at +900) makes
    # any positive true_prob translate into a huge EV at that price, well past
    # EV_CEILING -- is_pick must stay False even though desk_delta/edge are nonzero.
    # A soft book matching the same +900 price makes ev_best just as huge, so this
    # also confirms EV_CEILING blocks on ev_best, not just ev_pinnacle.
    desk = {"ml_pick": "home", "conviction_tier": "high"}
    game = _pickem_game(desk=desk)
    game["pinnacle"]["moneyline"] = {"home": 900, "away": -3000}
    game["books"]["moneyline"]["home"] = [("fanduel", 900)]
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["desk_delta"] != 0.0
    assert ml["ev_pinnacle"] > EV_CEILING
    assert ml["ev_best"] > EV_CEILING
    assert ml["is_pick"] is False


def test_assemble_games_joins_desk_and_odds_by_game_pk():
    desk_rows = [{
        "sport": "nfl", "game_pk": 42, "matchup": "Bears @ Packers",
        "commence_time": "2026-09-13T17:00:00Z", "ml_pick": "home",
        "spread_side": None, "total_side": None, "conviction_tier": "medium",
    }]
    odds_rows = [
        {"game_pk": 42, "market": "moneyline", "side": "home", "book": "pinnacle", "price": -130, "line": None},
        {"game_pk": 42, "market": "moneyline", "side": "away", "book": "pinnacle", "price": 110, "line": None},
        {"game_pk": 42, "market": "moneyline", "side": "home", "book": "fanduel", "price": -120, "line": None},
        {"game_pk": 42, "market": "spread", "side": "home", "book": "pinnacle", "price": -110, "line": -3.0},
        {"game_pk": 42, "market": "spread", "side": "away", "book": "pinnacle", "price": -110, "line": 3.0},
        {"game_pk": 42, "market": "total", "side": "over", "book": "pinnacle", "price": -110, "line": 44.5},
        {"game_pk": 42, "market": "total", "side": "under", "book": "pinnacle", "price": -110, "line": 44.5},
        # unrelated game -- must not leak in.
        {"game_pk": 99, "market": "moneyline", "side": "home", "book": "pinnacle", "price": -200, "line": None},
    ]
    games = assemble_games(desk_rows, odds_rows)
    assert len(games) == 1
    game = games[0]
    assert game["game_pk"] == 42
    assert game["desk"]["ml_pick"] == "home"
    assert game["pinnacle"]["moneyline"] == {"home": -130, "away": 110}
    assert game["pinnacle"]["spread"]["home_line"] == -3.0
    assert game["pinnacle"]["total"]["line"] == 44.5
    assert ("fanduel", -120) in game["books"]["moneyline"]["home"]

    rows = ev_rows_for_game(game)
    assert any(r["market"] == "moneyline" for r in rows)


# ---- user's 9 books are the only shoppable pool; both sides of every market are priced ----


def test_user_books_exact_order():
    assert USER_BOOKS == ("draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet",
                          "fanatics", "hardrockbet", "ballybet", "betrivers")


def test_best_price_only_shops_user_books():
    # bet365 / thescore / caesars (alias) / bovada are not books the user can bet.
    junk = [("bet365", 500), ("thescore", 400), ("caesars", 300), ("bovada", 600)]
    assert best_price(junk) is None
    assert best_price(junk + [("betrivers", 120)]) == ("betrivers", 120)
    assert best_price(junk + [("fanduel", 105), ("betrivers", 110), ("draftkings", -105)]) == ("betrivers", 110)
    assert best_price([("pinnacle", 200)]) is None
    for bk in USER_BOOKS:
        assert best_price([(bk, 100)]) == (bk, 100)


def _odds(gpk, market, side, book, price, line=None):
    return {"game_pk": gpk, "market": market, "side": side, "book": book, "price": price, "line": line}


def test_assemble_games_soft_pool_is_user_books_only():
    desk_rows = [{"sport": "nfl", "game_pk": 7, "matchup": "A @ B", "commence_time": "2026-10-18T17:00:00Z"}]
    odds_rows = [
        _odds(7, "moneyline", "home", "pinnacle", -130), _odds(7, "moneyline", "away", "pinnacle", 110),
        _odds(7, "moneyline", "home", "betrivers", -120),
        _odds(7, "moneyline", "home", "bet365", -105),
        _odds(7, "moneyline", "home", "thescore", -105),
    ]
    game = assemble_games(desk_rows, odds_rows)[0]
    assert game["books"]["moneyline"]["home"] == [("betrivers", -120)]


def _full_game(**overrides):
    g = _pickem_game()
    g["pinnacle"] = {
        "moneyline": {"home": -170, "away": 150},
        "spread": {"home_line": -3.5, "home": -110, "away": -110},
        "total": {"line": 44.5, "over": -105, "under": -115},
    }
    g.update(overrides)
    return g


def test_ev_rows_price_both_sides_of_every_market():
    rows = ev_rows_for_game(_full_game())
    assert len(rows) == 6
    assert sorted((r["market"], r["side"]) for r in rows) == [
        ("moneyline", "away"), ("moneyline", "home"), ("spread", "away"), ("spread", "home"),
        ("total", "over"), ("total", "under")]
    by = {(r["market"], r["side"]): r for r in rows}
    assert by["moneyline", "home"]["true_prob"] == pytest.approx(novig(-170, 150))
    assert by["moneyline", "away"]["true_prob"] == pytest.approx(novig(150, -170))
    assert by["moneyline", "away"]["pinnacle_price"] == 150
    assert by["spread", "away"]["true_prob"] == pytest.approx(0.5)
    assert by["total", "over"]["true_prob"] == pytest.approx(novig(-105, -115))
    assert by["total", "under"]["true_prob"] == pytest.approx(novig(-115, -105))
    assert by["total", "under"]["pinnacle_price"] == -115
    for m in ("moneyline", "spread", "total"):
        sides = [r for r in rows if r["market"] == m]
        assert sum(r["true_prob"] for r in sides) == pytest.approx(1.0)


def test_underdog_side_is_a_pick_when_a_user_book_beats_fair():
    # Pinnacle fair for the away dog (+150/-170) is ~0.3784; BetRivers +175 is +EV.
    # The old engine only ever priced the market favourite, so this was never evaluated.
    game = _full_game()
    game["books"]["moneyline"]["away"] = [("betrivers", 175), ("bet365", 400)]
    rows = ev_rows_for_game(game)
    away = next(r for r in rows if r["market"] == "moneyline" and r["side"] == "away")
    assert away["best_book"] == "betrivers" and away["best_price"] == 175
    assert away["ev_best"] == pytest.approx(ev(novig(150, -170), 175))
    assert away["is_pick"] is True
    home = next(r for r in rows if r["market"] == "moneyline" and r["side"] == "home")
    assert home["is_pick"] is False


def test_at_most_one_side_per_market_is_pick_higher_ev_wins():
    game = _full_game()
    game["pinnacle"]["total"] = {"line": 44.5, "over": -110, "under": -110}
    game["books"]["total"] = {"over": [("fanduel", 105)], "under": [("draftkings", 110)]}
    rows = [r for r in ev_rows_for_game(game) if r["market"] == "total"]
    by = {r["side"]: r for r in rows}
    assert by["over"]["ev_best"] > MIN_EV and by["under"]["ev_best"] > MIN_EV  # both clear the rule
    assert by["under"]["ev_best"] > by["over"]["ev_best"]
    assert by["under"]["is_pick"] is True and by["over"]["is_pick"] is False


def test_both_sides_pick_tie_goes_to_home_or_over():
    game = _full_game()
    game["pinnacle"]["total"] = {"line": 44.5, "over": -110, "under": -110}
    game["books"]["total"] = {"over": [("fanduel", 105)], "under": [("draftkings", 105)]}
    by = {r["side"]: r for r in ev_rows_for_game(game) if r["market"] == "total"}
    assert by["over"]["is_pick"] is True and by["under"]["is_pick"] is False
    game["pinnacle"]["spread"] = {"home_line": -3.5, "home": -110, "away": -110}
    game["books"]["spread"] = {"home": [("fanduel", 105)], "away": [("draftkings", 105)]}
    sp = {r["side"]: r for r in ev_rows_for_game(game) if r["market"] == "spread"}
    assert sp["home"]["is_pick"] is True and sp["away"]["is_pick"] is False


def test_desk_delta_sign_is_per_side():
    game = _pickem_game(desk={"ml_pick": "home", "conviction_tier": "high"})
    by = {r["side"]: r for r in ev_rows_for_game(game) if r["market"] == "moneyline"}
    assert by["home"]["desk_delta"] > 0 and by["away"]["desk_delta"] == pytest.approx(-by["home"]["desk_delta"])
    game = _pickem_game(desk={"total_side": "under", "conviction_tier": "high"})
    by = {r["side"]: r for r in ev_rows_for_game(game) if r["market"] == "total"}
    assert by["under"]["desk_delta"] > 0 and by["over"]["desk_delta"] == pytest.approx(-by["under"]["desk_delta"])
