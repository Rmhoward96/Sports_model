"""Tests for the pure +EV pilot edge/EV engine (sportsmodel.serving.ev_pilot).

Sub-project 4 (+EV desk-driven pilot), task 1 -- see
.superpowers/sdd/2026-09-09-plus-ev-4-desk-driven-pilot/task-1-brief.md.
"""
from __future__ import annotations

import pytest

from sportsmodel.model.trueprob_prob import DESK_MAX_PROB_DELTA, win_prob
from sportsmodel.serving.board import EV_CEILING, USER_BOOKS, best_price, ev, implied_prob, novig, novig_power
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
    assert ml["base_prob"] == novig(-110, -110) == 0.5   # power fair at an even pair is exactly 0.5
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
    assert by["moneyline", "home"]["true_prob"] == pytest.approx(novig_power(-170, 150))
    assert by["moneyline", "away"]["true_prob"] == pytest.approx(novig_power(150, -170))
    assert by["moneyline", "away"]["pinnacle_price"] == 150
    assert by["spread", "away"]["true_prob"] == pytest.approx(0.5)
    assert by["total", "over"]["true_prob"] == pytest.approx(novig_power(-105, -115))
    assert by["total", "under"]["true_prob"] == pytest.approx(novig_power(-115, -105))
    assert by["total", "under"]["pinnacle_price"] == -115
    for m in ("moneyline", "spread", "total"):
        sides = [r for r in rows if r["market"] == m]
        assert sum(r["true_prob"] for r in sides) == pytest.approx(1.0)


def test_underdog_side_is_a_pick_when_a_user_book_beats_fair():
    # Pinnacle fair for the away dog (+150/-170) is ~0.3836 (power; 0.3885 proportional); BetRivers +175 is +EV.
    # The old engine only ever priced the market favourite, so this was never evaluated.
    game = _full_game()
    game["books"]["moneyline"]["away"] = [("betrivers", 175), ("bet365", 400)]
    rows = ev_rows_for_game(game)
    away = next(r for r in rows if r["market"] == "moneyline" and r["side"] == "away")
    assert away["best_book"] == "betrivers" and away["best_price"] == 175
    assert away["ev_best"] == pytest.approx(ev(novig_power(150, -170), 175))   # 5.48% (6.83% proportional)
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


# ---- power de-vig on the +EV game-line board --------------------------------------------------


def _dog_game(**overrides):
    """Pinnacle -900/+605 away long shot ML (plus a spread so the ML has its margin anchor)."""
    g = _full_game()
    g["pinnacle"]["moneyline"] = {"home": -900, "away": 605}
    g.update(overrides)
    return g


def test_true_prob_uses_power_devig_not_proportional():
    rows = {(r["market"], r["side"]): r for r in ev_rows_for_game(_dog_game())}
    assert rows["moneyline", "away"]["true_prob"] == pytest.approx(novig_power(605, -900))
    assert rows["moneyline", "away"]["true_prob"] == pytest.approx(0.111578, abs=1e-6)
    assert rows["moneyline", "away"]["true_prob"] < novig(605, -900)
    assert rows["moneyline", "home"]["true_prob"] == pytest.approx(0.888422, abs=1e-6)
    assert rows["moneyline", "home"]["true_prob"] > novig(-900, 605)
    assert rows["moneyline", "away"]["base_prob"] == rows["moneyline", "away"]["true_prob"]  # no desk


def test_longshot_ev_is_smaller_with_power_than_proportional():
    # BetRivers +800 (decimal 9.0) vs Pinnacle -900/+605: ev = p*9 - 1.
    # proportional fair .136147 -> +22.5%; power fair .111578 -> +0.42% (the apparent edge was mostly the de-vig).
    game = _dog_game()
    game["books"]["moneyline"]["away"] = [("betrivers", 800)]
    away = next(r for r in ev_rows_for_game(game) if r["market"] == "moneyline" and r["side"] == "away")
    assert away["ev_best"] == pytest.approx(0.0042, abs=1e-4)
    assert away["ev_best"] == pytest.approx(ev(0.111578, 800), abs=1e-5)
    assert away["ev_best"] < ev(novig(605, -900), 800)
    assert ev(novig(605, -900), 800) == pytest.approx(0.2253, abs=1e-4)
    assert away["is_pick"] is False   # 0.42% is below MIN_EV: no longer a surfaced pick


def test_short_priced_side_changes_only_slightly():
    # -105 vs Pinnacle -116/-104: fair(over) proportional .51301, power .51393 (+0.09 pts); ev +0.16% -> +0.34%.
    g = _full_game()
    g["pinnacle"]["total"] = {"line": 44.5, "over": -116, "under": -104}
    g["books"]["total"] = {"over": [("fanduel", -105)], "under": []}
    over = next(r for r in ev_rows_for_game(g) if r["market"] == "total" and r["side"] == "over")
    assert over["true_prob"] == pytest.approx(novig_power(-116, -104))
    assert abs(novig_power(-116, -104) - novig(-116, -104)) < 0.002
    assert over["ev_best"] == pytest.approx(ev(novig_power(-116, -104), -105))
    assert over["ev_best"] == pytest.approx(0.0034, abs=1e-4)
    assert ev(novig(-116, -104), -105) == pytest.approx(0.0016, abs=1e-4)
    assert over["ev_best"] - ev(novig(-116, -104), -105) < 0.005


# ---- same-pull freshness ----------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

from sportsmodel.serving.ev_pilot import SAME_PULL_TOLERANCE  # noqa: E402

_T0 = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
_DESK = [{"sport": "nfl", "game_pk": 7, "matchup": "A @ B", "commence_time": "2026-10-18T17:00:00Z"}]


def _tsodds(market, side, book, price, line=None, at=_T0):
    d = _odds(7, market, side, book, price, line)
    if at is not None:
        d["captured_at"] = at
    return d


def _pinnacle_rows(at=_T0, away_at=None):
    return [
        _tsodds("moneyline", "home", "pinnacle", -130, at=at), _tsodds("moneyline", "away", "pinnacle", 110, at=away_at or at),
        _tsodds("spread", "home", "pinnacle", -110, -3.0, at=at), _tsodds("spread", "away", "pinnacle", -110, 3.0, at=away_at or at),
    ]


def test_same_pull_tolerance_is_five_minutes():
    assert SAME_PULL_TOLERANCE == timedelta(minutes=5)


def test_soft_price_three_hours_before_pinnacle_is_dropped_and_counted():
    odds = _pinnacle_rows() + [_tsodds("moneyline", "home", "betrivers", -120, at=_T0 - timedelta(hours=3))]
    stats: dict = {}
    game = assemble_games(_DESK, odds, stats)[0]
    assert game["books"]["moneyline"]["home"] == []
    assert stats["stale_dropped"] == 1


def test_soft_price_within_four_minutes_before_or_after_is_kept():
    odds = _pinnacle_rows() + [
        _tsodds("moneyline", "home", "betrivers", -120, at=_T0 - timedelta(minutes=4)),
        _tsodds("moneyline", "away", "fanduel", 115, at=_T0 + timedelta(minutes=4)),
    ]
    stats: dict = {}
    game = assemble_games(_DESK, odds, stats)[0]
    assert game["books"]["moneyline"]["home"] == [("betrivers", -120)]
    assert game["books"]["moneyline"]["away"] == [("fanduel", 115)]
    assert stats.get("stale_dropped", 0) == 0


def test_tolerance_boundary_is_inclusive():
    odds = _pinnacle_rows() + [
        _tsodds("moneyline", "home", "betrivers", -120, at=_T0 - SAME_PULL_TOLERANCE),
        _tsodds("moneyline", "away", "fanduel", 115, at=_T0 + SAME_PULL_TOLERANCE + timedelta(seconds=1)),
    ]
    stats: dict = {}
    game = assemble_games(_DESK, odds, stats)[0]
    assert game["books"]["moneyline"]["home"] == [("betrivers", -120)]
    assert game["books"]["moneyline"]["away"] == []
    assert stats["stale_dropped"] == 1


def test_pinnacle_sides_ten_minutes_apart_means_no_pinnacle_price_for_that_market():
    odds = _pinnacle_rows(away_at=_T0 + timedelta(minutes=10)) + [
        _tsodds("total", "over", "pinnacle", -110, 44.5), _tsodds("total", "under", "pinnacle", -110, 44.5),
        _tsodds("moneyline", "home", "betrivers", -120),
    ]
    game = assemble_games(_DESK, odds)[0]
    assert game["pinnacle"]["moneyline"] == {}
    assert game["pinnacle"]["spread"] == {}
    assert game["pinnacle"]["total"] == {"over": -110, "under": -110, "line": 44.5}
    assert {r["market"] for r in ev_rows_for_game(game)} == {"total"}  # ML needs the spread anchor; total has its own line


def test_pinnacle_sides_within_tolerance_use_later_capture_as_T():
    odds = _pinnacle_rows(away_at=_T0 + timedelta(minutes=3)) + [
        _tsodds("moneyline", "home", "betrivers", -120, at=_T0 + timedelta(minutes=7)),   # 4 min after later side: kept
        _tsodds("moneyline", "away", "fanduel", 115, at=_T0 - timedelta(minutes=1)),      # 4 min before later side: kept
        _tsodds("moneyline", "away", "betmgm", 112, at=_T0 - timedelta(minutes=3)),       # 6 min before later side: dropped
    ]
    stats: dict = {}
    game = assemble_games(_DESK, odds, stats)[0]
    assert game["pinnacle"]["moneyline"] == {"home": -130, "away": 110}
    assert game["books"]["moneyline"]["home"] == [("betrivers", -120)]
    assert game["books"]["moneyline"]["away"] == [("fanduel", 115)]
    assert stats["stale_dropped"] == 1


def test_captured_at_accepts_iso_strings_datetimes_and_naive_utc():
    odds = _pinnacle_rows(at=_T0.isoformat()) + [
        _tsodds("moneyline", "home", "betrivers", -120, at=(_T0 + timedelta(minutes=2))),
        _tsodds("moneyline", "away", "fanduel", 115, at=(_T0 + timedelta(minutes=2)).isoformat().replace("+00:00", "Z")),
        _tsodds("moneyline", "away", "betmgm", 112, at=(_T0 + timedelta(minutes=1)).replace(tzinfo=None)),
        _tsodds("moneyline", "away", "draftkings", 111, at=(_T0 - timedelta(hours=3)).isoformat()),
    ]
    stats: dict = {}
    game = assemble_games(_DESK, odds, stats)[0]
    assert game["pinnacle"]["moneyline"] == {"home": -130, "away": 110}
    assert game["books"]["moneyline"]["home"] == [("betrivers", -120)]
    assert sorted(game["books"]["moneyline"]["away"]) == [("betmgm", 112), ("fanduel", 115)]
    assert stats["stale_dropped"] == 1


def test_fixtures_without_any_captured_at_keep_old_behaviour():
    odds = [{k: v for k, v in o.items() if k != "captured_at"} for o in _pinnacle_rows()
            + [_tsodds("moneyline", "home", "betrivers", -120)]]
    assert all("captured_at" not in o for o in odds)
    stats: dict = {}
    game = assemble_games(_DESK, odds, stats)[0]
    assert game["books"]["moneyline"]["home"] == [("betrivers", -120)]
    assert game["pinnacle"]["moneyline"] == {"home": -130, "away": 110}
    assert stats.get("stale_dropped", 0) == 0


def test_rows_lacking_captured_at_are_excluded_when_the_game_has_timestamps():
    odds = _pinnacle_rows() + [
        _tsodds("moneyline", "home", "betrivers", -120, at=None),
        _tsodds("moneyline", "away", "fanduel", 115, at=_T0),
    ]
    game = assemble_games(_DESK, odds)[0]
    assert game["books"]["moneyline"]["home"] == []
    assert game["books"]["moneyline"]["away"] == [("fanduel", 115)]
    # ...and an undated Pinnacle side leaves that market with no Pinnacle price.
    odds2 = _pinnacle_rows()
    odds2[1] = {k: v for k, v in odds2[1].items() if k != "captured_at"}
    assert assemble_games(_DESK, odds2)[0]["pinnacle"]["moneyline"] == {"home": -130}


def test_stats_out_parameter_is_optional_and_accumulates_across_games():
    desk = _DESK + [{"sport": "nfl", "game_pk": 8, "matchup": "C @ D", "commence_time": "2026-10-18T17:00:00Z"}]
    old = _T0 - timedelta(hours=3)
    odds = _pinnacle_rows() + [_tsodds("moneyline", "home", "betrivers", -120, at=old)]
    odds += [dict(o, game_pk=8) for o in odds]
    stats = {"stale_dropped": 2}
    assemble_games(desk, odds, stats)
    assert stats["stale_dropped"] == 4
    assert len(assemble_games(desk, odds)) == 2   # stats omitted still works


def test_soft_rows_in_an_emptied_market_are_skipped_and_counted():
    # Pinnacle ML sides 10 min apart -> ML emptied; its soft rows must not leak through (and are counted).
    odds = _pinnacle_rows(away_at=_T0 + timedelta(minutes=10)) + [
        _tsodds("moneyline", "home", "betrivers", -120),
        _tsodds("moneyline", "away", "fanduel", 115),
        _tsodds("total", "over", "pinnacle", -110, 44.5), _tsodds("total", "under", "pinnacle", -110, 44.5),
        _tsodds("total", "over", "betmgm", -105, 44.5),
    ]
    stats: dict = {}
    game = assemble_games(_DESK, odds, stats)[0]
    assert game["books"]["moneyline"] == {"home": [], "away": []}
    assert game["books"]["spread"] == {"home": [], "away": []}
    assert game["books"]["total"]["over"] == [("betmgm", -105)]
    assert stats["stale_dropped"] == 2
