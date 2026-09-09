"""Tests for the pure +EV pilot edge/EV engine (sportsmodel.serving.ev_pilot).

Sub-project 4 (+EV desk-driven pilot), task 1 -- see
.superpowers/sdd/2026-09-09-plus-ev-4-desk-driven-pilot/task-1-brief.md.
"""
from __future__ import annotations

from sportsmodel.model.trueprob_prob import DESK_MAX_PROB_DELTA, win_prob
from sportsmodel.serving.board import EV_CEILING, ev, novig
from sportsmodel.serving.ev_pilot import (
    MIN_EDGE,
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


def test_market_margin():
    assert market_margin(-3.0) == 3.0
    assert market_margin(3.0) == -3.0


def test_no_desk_pick_passes_on_every_market():
    game = _pickem_game(desk=None)
    rows = ev_rows_for_game(game)
    assert len(rows) == 3  # moneyline, spread, total
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
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["side"] == "home"
    assert ml["base_prob"] == novig(-110, -110)
    assert ml["true_prob"] > ml["base_prob"]
    assert ml["true_prob"] - ml["base_prob"] <= DESK_MAX_PROB_DELTA + 1e-9
    assert ml["edge"] > 0
    assert ml["is_pick"] is True
    assert ml["edge"] >= MIN_EDGE

    # spread/total on this game carry no desk pick -> untouched (pass).
    spread = next(r for r in rows if r["market"] == "spread")
    total = next(r for r in rows if r["market"] == "total")
    assert spread["desk_delta"] == 0.0 and spread["is_pick"] is False
    assert total["desk_delta"] == 0.0 and total["is_pick"] is False


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


def test_ev_ceiling_blocks_contrived_huge_edge():
    # A lopsided two-way Pinnacle price (home is a big underdog priced at +900) makes
    # any positive true_prob translate into a huge EV at that price, well past
    # EV_CEILING -- is_pick must stay False even though desk_delta/edge are nonzero.
    desk = {"ml_pick": "home", "conviction_tier": "high"}
    game = _pickem_game(desk=desk)
    game["pinnacle"]["moneyline"] = {"home": 900, "away": -3000}
    rows = ev_rows_for_game(game)
    ml = next(r for r in rows if r["market"] == "moneyline")

    assert ml["desk_delta"] != 0.0
    assert ml["ev_pinnacle"] > EV_CEILING
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
