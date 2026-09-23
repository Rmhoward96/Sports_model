from sportsmodel.serving.best_parlays import (
    PARLAY_BOOKS, assemble_game_legs, assemble_prop_legs, build_best_parlays,
    game_leg_result, prop_leg_result, settle_parlay)


def _leg(key, game_pk, prob, prices, kind="game"):
    return {"key": key, "kind": kind, "sport": "nfl", "game_pk": game_pk, "market": "moneyline",
            "side": "home", "line": None, "prob": prob, "label": key, "matchup": "A @ B",
            "commence_time": f"2026-09-27T1{game_pk % 10}:00:00Z", "player_id": None,
            "player_name": None, "book_prices": prices}


def test_pinnacle_is_not_a_parlay_book():
    assert "pinnacle" not in PARLAY_BOOKS and "draftkings" in PARLAY_BOOKS


def test_builds_best_single_book_ticket_one_leg_per_game():
    legs = [_leg("a", 1, 0.60, {"draftkings": -120, "fanduel": -110}),
            _leg("a2", 1, 0.58, {"draftkings": -105}),            # same game as a
            _leg("b", 2, 0.62, {"draftkings": -130, "fanduel": -125}),
            _leg("c", 3, 0.57, {"draftkings": +100, "fanduel": -105}),
            _leg("d", 4, 0.56, {"fanduel": +105})]
    [t, *_] = build_best_parlays(legs)
    assert t["n_legs"] == 3 and len({l["game_pk"] for l in t["legs"]}) == 3
    assert t["book"] in ("draftkings", "fanduel")
    prices = {l["key"]: l["price"] for l in t["legs"]}
    assert all(prices[k] == next(x for x in legs if x["key"] == k)["book_prices"][t["book"]] for k in prices)
    dec = 1.0
    prob = 1.0
    for l in t["legs"]:
        dec *= (1 + (l["price"] / 100 if l["price"] > 0 else 100 / -l["price"]))
        prob *= l["prob"]
    assert abs(t["parlay_dec"] - dec) < 1e-9 and abs(t["true_prob"] - prob) < 1e-9
    assert abs(t["ev"] - (prob * dec - 1)) < 1e-9
    assert t["parlay_id"].startswith(t["book"] + "|")


def test_prob_floor_excluded_keys_and_no_shared_legs():
    legs = [_leg(k, g, 0.60, {"draftkings": +100}) for k, g in
            [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6), ("g", 7)]]
    legs.append(_leg("low", 8, 0.54, {"draftkings": +300}))       # under 55% floor
    tickets = build_best_parlays(legs, excluded_keys=frozenset({"g"}))
    used = [l["key"] for t in tickets for l in t["legs"]]
    assert len(tickets) == 2                                       # 6 usable legs -> 2 tickets
    assert len(used) == len(set(used)) and "low" not in used and "g" not in used


def test_leg_must_be_plus_ev_at_the_book_and_ticket_needs_three_legs():
    legs = [_leg("a", 1, 0.56, {"draftkings": -150}),              # 0.56*1.667-1 < 0
            _leg("b", 2, 0.60, {"draftkings": +100}),
            _leg("c", 3, 0.60, {"draftkings": +100})]
    assert build_best_parlays(legs) == []


def test_mixed_sport_ticket_labelled_mixed():
    legs = [_leg("a", 1, 0.6, {"draftkings": +100}), _leg("b", 2, 0.6, {"draftkings": +100}),
            {**_leg("c", 3, 0.6, {"draftkings": +100}), "sport": "cfb"}]
    [t] = build_best_parlays(legs)
    assert t["sport"] == "mixed"


def test_assemble_game_legs_matches_line_and_skips_pinnacle():
    picks = [{"sport": "nfl", "game_pk": 1, "market": "spread", "side": "home",
              "matchup": "Falcons @ Packers", "commence_time": "t", "true_prob": 0.57, "line": -3.5}]
    odds = [{"game_pk": 1, "market": "spread", "side": "home", "book": "draftkings", "line": -3.5, "price": -105},
            {"game_pk": 1, "market": "spread", "side": "home", "book": "fanduel", "line": -3.0, "price": -120},
            {"game_pk": 1, "market": "spread", "side": "home", "book": "pinnacle", "line": -3.5, "price": +100}]
    [leg] = assemble_game_legs(picks, odds)
    assert leg["book_prices"] == {"draftkings": -105}
    assert leg["label"] == "Packers -3.5" and leg["key"] == "g:1:spread:home" and leg["prob"] == 0.57


def test_assemble_game_legs_moneyline_and_total_labels():
    picks = [{"sport": "cfb", "game_pk": 2, "market": "moneyline", "side": "away", "matchup": "Oregon Ducks @ USC Trojans",
              "commence_time": "t", "true_prob": 0.6, "line": None},
             {"sport": "cfb", "game_pk": 3, "market": "total", "side": "under", "matchup": "X @ Y",
              "commence_time": "t", "true_prob": 0.55, "line": 49.5}]
    odds = [{"game_pk": 2, "market": "moneyline", "side": "away", "book": "fanduel", "line": None, "price": -139},
            {"game_pk": 3, "market": "total", "side": "under", "book": "betmgm", "line": 49.5, "price": -104}]
    legs = assemble_game_legs(picks, odds)
    assert [l["label"] for l in legs] == ["Oregon Ducks ML", "Under 49.5"]


def test_assemble_prop_legs_matches_player_market_side_line():
    picks = [{"game_pk": 9, "player_id": "00-1", "player_name": "A.J. Brown", "market": "rec_yds",
              "side": "under", "line": 66.5, "model_prob": 0.61, "matchup": "M", "commence_time": "t"}]
    odds = [{"game_pk": 9, "market": "reception_yds", "side": "under", "player_name": "AJ Brown", "book": "fanduel", "line": 66.5, "price": -112},
            {"game_pk": 9, "market": "reception_yds", "side": "under", "player_name": "AJ Brown", "book": "draftkings", "line": 64.5, "price": -110},
            {"game_pk": 9, "market": "reception_yds", "side": "over", "player_name": "AJ Brown", "book": "fanduel", "line": 66.5, "price": -108}]
    [leg] = assemble_prop_legs(picks, odds)
    assert leg["book_prices"] == {"fanduel": -112}
    assert leg["key"] == "p:9:00-1:rec_yds:under" and leg["kind"] == "prop" and leg["sport"] == "nfl"
    assert leg["label"] == "A.J. Brown Rec Yds Under 66.5"


def _g(market, side, line=None):
    return {"kind": "game", "market": market, "side": side, "line": line}


def test_game_leg_results():
    final = {"actual_margin": 7, "actual_total": 44}           # home won by 7, 44 total
    assert game_leg_result(_g("moneyline", "home"), final) == "win"
    assert game_leg_result(_g("moneyline", "away"), final) == "loss"
    assert game_leg_result(_g("moneyline", "home"), {"actual_margin": 0, "actual_total": 40}) == "push"
    assert game_leg_result(_g("spread", "home", -7.0), final) == "push"
    assert game_leg_result(_g("spread", "home", -6.5), final) == "win"
    assert game_leg_result(_g("spread", "away", 6.5), final) == "loss"
    assert game_leg_result(_g("spread", "away", 7.5), final) == "win"
    assert game_leg_result(_g("total", "over", 43.5), final) == "win"
    assert game_leg_result(_g("total", "under", 44.0), final) == "push"
    assert game_leg_result(_g("total", "under", 43.5), None) is None   # not final yet


def test_prop_leg_results_and_void():
    leg = {"kind": "prop", "side": "under", "line": 66.5}
    assert prop_leg_result(leg, 50.0, True) == "win"
    assert prop_leg_result(leg, 70.0, True) == "loss"
    assert prop_leg_result({**leg, "side": "over", "line": 5.0}, 5.0, True) == "push"
    assert prop_leg_result(leg, None, True) == "push"        # didn't play -> void
    assert prop_leg_result(leg, None, False) is None         # game not captured yet


def test_settle_parlay():
    legs = [{"price": -110}, {"price": +100}, {"price": -120}]
    assert settle_parlay(legs, ["win", "loss", None]) == {"result": "loss", "pnl": -10.0, "payout_dec": None}
    assert settle_parlay(legs, ["win", "win", None]) is None
    win = settle_parlay(legs, ["win", "win", "win"])
    dec = (1 + 100 / 110) * 2.0 * (1 + 100 / 120)
    assert win["result"] == "win" and abs(win["pnl"] - 10 * (dec - 1)) < 1e-9
    one_push = settle_parlay(legs, ["win", "push", "win"])
    assert abs(one_push["payout_dec"] - (1 + 100 / 110) * (1 + 100 / 120)) < 1e-9
    assert settle_parlay(legs, ["push", "push", "push"]) == {"result": "push", "pnl": 0.0, "payout_dec": 1.0}
