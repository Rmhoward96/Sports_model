"""Adapters from each writer's rows to ledger observations. Pure."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel.ledger import sources

KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)
_gp = importlib.util.spec_from_file_location(
    "grade_predictions", Path(__file__).resolve().parents[2] / "scripts" / "grade_predictions.py")
grade_predictions = importlib.util.module_from_spec(_gp)
_gp.loader.exec_module(grade_predictions)


# ---- ev_game -------------------------------------------------------------------------------------

def _game():
    return {"game_pk": 7, "pinnacle": {"moneyline": {"home": -150, "away": 130},
                                       "spread": {"home": -110, "away": -110, "home_line": -3.5},
                                       "total": {"over": -108, "under": -112, "line": 47.5}}}


def _ev(market, side, **kw):
    return {"sport": "nfl", "game_pk": 7, "market": market, "side": side, "commence_time": KICK,
            "model_version": "ev-pilot-v1", "base_prob": 0.55, "best_price": 105, "best_book": "fanduel",
            "ev_best": 0.032, "is_pick": True, **kw}


def test_ev_game_lines_follow_the_side():
    obs = {(o["market"], o["side"]): o for o in sources.ev_game_observations(
        [_ev("spread", "home"), _ev("spread", "away"), _ev("total", "under"), _ev("moneyline", "home")], [_game()])}
    assert obs[("spread", "home")]["line"] == -3.5 and obs[("spread", "away")]["line"] == 3.5
    assert obs[("total", "under")]["line"] == 47.5 and obs[("moneyline", "home")]["line"] is None


def test_ev_game_observation_fields():
    (o,) = sources.ev_game_observations([_ev("spread", "away")], [_game()])
    assert o["source"] == "ev_game" and o["model_prob"] is None
    assert (o["ref_prob"], o["ref_kind"], o["ref_books"]) == (0.55, "pinnacle_novig", 1)
    assert (o["best_price"], o["best_book"]) == (105, "fanduel") and o["ev_pct"] == pytest.approx(3.2)


def test_ev_game_skips_a_spread_whose_line_cannot_be_read():
    assert sources.ev_game_observations([_ev("spread", "home")], [{"game_pk": 7, "pinnacle": {}}]) == []
    assert len(sources.ev_game_observations([_ev("moneyline", "home")], [{"game_pk": 7, "pinnacle": {}}])) == 1


# ---- ev_prop -------------------------------------------------------------------------------------

def test_ev_prop_observation_fields_and_skips():
    row = {"sport": "nfl", "game_pk": 7, "player_id": "00-1", "player_name": "A. Brown", "market": "rec_yds",
           "side": "over", "line": 64.5, "model_version": "props-sim-v1", "commence_time": KICK,
           "model_prob": 0.58, "market_prob": 0.51, "ev_best": 0.07, "best_book": "dk", "best_price": -105}
    (o,) = sources.ev_prop_observations([row, {**row, "line": None}, {**row, "player_id": None}])
    assert o["source"] == "ev_prop" and o["player_id"] == "00-1" and o["line"] == 64.5
    assert (o["model_prob"], o["ref_prob"], o["ref_kind"]) == (0.58, 0.51, None) and o["ev_pct"] == pytest.approx(7.0)
    assert o["pick_key"] == "ev_prop|nfl|7|rec_yds|over|00-1||"


# ---- parlay legs -----------------------------------------------------------------------------------

def test_parlay_legs_become_one_observation_each():
    leg_g = {"key": "g:7:spread:home", "kind": "game", "sport": "nfl", "game_pk": 7, "market": "spread",
             "side": "home", "line": -3.5, "prob": 0.6, "price": -110, "commence_time": KICK,
             "player_id": None, "player_name": None}
    leg_p = {**leg_g, "key": "p:8:00-1:rec_yds:over", "kind": "prop", "game_pk": 8, "market": "rec_yds",
             "side": "over", "line": 64.5, "price": 100, "player_id": "00-1", "player_name": "A. Brown"}
    ticket = {"parlay_id": "draftkings|g:7:spread:home|p:8:00-1:rec_yds:over", "book": "draftkings",
              "model_version": "ev-parlays-v1", "legs": [leg_g, leg_p]}
    a, b = sources.parlay_leg_observations([ticket])
    assert a["source"] == "parlay_leg" and a["parlay_id"] == ticket["parlay_id"] and a["best_book"] == "draftkings"
    assert a["model_prob"] == 0.6 and a["best_price"] == -110
    assert a["ev_pct"] == pytest.approx((0.6 * (1 + 100 / 110) - 1) * 100)
    assert b["player_id"] == "00-1" and b["pick_key"] != a["pick_key"]
    assert a["pick_key"].endswith("|" + ticket["parlay_id"])


# ---- model_game --------------------------------------------------------------------------------------

def _pmf_margin(mean_home_by, sd=13.0, half=60):
    import math
    pmf = [math.exp(-0.5 * ((i - half - mean_home_by) / sd) ** 2) for i in range(2 * half + 1)]
    s = sum(pmf)
    return json.dumps({"kind": "margin", "offset": half, "pmf": [p / s for p in pmf]})


def _pred(**kw):
    base = {"sport": "cfb", "game_pk": 5, "model_version": "cfb-v2", "commence_time": KICK,
            "home_win_prob": 0.7, "pred_home_score": 30.0, "pred_away_score": 20.0,
            "market_spread": -7.0, "market_total": 47.0,
            "margin_dist": _pmf_margin(10.0),
            "total_dist": json.dumps({"kind": "pmf", "pmf": [1.0 / 100] * 100})}
    return {**base, **kw}


def test_model_game_picks_use_prediction_accuracys_side_logic():
    obs = {o["market"]: o for o in sources.model_game_observations([_pred()])}
    assert obs["moneyline"]["side"] == "home" and obs["moneyline"]["model_prob"] == 0.7
    assert obs["spread"]["side"] == "home" and obs["spread"]["line"] == -7.0          # 10 + (-7) > 0
    assert obs["total"]["side"] == "over" and obs["total"]["line"] == 47.0             # 50 > 47
    assert 0.5 < obs["spread"]["model_prob"] < 1.0
    assert all(o["best_price"] is None and o["source"] == "model_game" for o in obs.values())


def test_model_game_away_lean_and_under_lean():
    obs = {o["market"]: o for o in sources.model_game_observations(
        [_pred(home_win_prob=0.4, pred_home_score=24.0, pred_away_score=23.0, market_spread=-7.0, market_total=52.0)])}
    assert obs["moneyline"]["side"] == "away" and obs["moneyline"]["model_prob"] == pytest.approx(0.6)
    assert obs["spread"]["side"] == "away" and obs["spread"]["line"] == 7.0            # 1 + (-7) < 0 -> away +7
    assert obs["total"]["side"] == "under"


def test_model_game_no_pick_on_exact_ties_or_missing_market():
    obs = sources.model_game_observations([_pred(pred_home_score=27.0, pred_away_score=20.0, market_spread=-7.0,
                                                 market_total=None)])
    assert {o["market"] for o in obs} == {"moneyline"}                                  # margin 7 + -7 == 0 -> none
    tie_total = sources.model_game_observations([_pred(pred_home_score=25.0, pred_away_score=22.0, market_total=47.0)])
    assert {o["market"] for o in tie_total} == {"moneyline", "spread"}                  # total 47 == market 47 -> none
    assert {o["market"] for o in sources.model_game_observations([_pred(market_spread=None, market_total=None)])} == {"moneyline"}


def test_model_game_prob_is_null_without_a_distribution():
    obs = {o["market"]: o for o in sources.model_game_observations([_pred(margin_dist=None, total_dist=None)])}
    assert obs["spread"]["model_prob"] is None and obs["total"]["model_prob"] is None


def test_model_game_falls_back_to_pred_margin_and_total_columns():
    row = _pred(pred_home_score=None, pred_away_score=None, pred_margin=10.0, pred_total=50.0)
    assert {o["market"] for o in sources.model_game_observations([row])} == {"moneyline", "spread", "total"}


@pytest.mark.parametrize("pred_margin, pred_total, ms, mt, final", [
    (10.0, 50.0, -7.0, 47.0, (31, 17)),          # home covers, over
    (10.0, 50.0, -7.0, 47.0, (20, 17)),          # away covers, under
    (-3.0, 41.0, 2.5, 44.5, (14, 24)),           # home +2.5, model margin -3 -> away lean; under lean
    (2.0, 55.0, -9.5, 49.5, (30, 24)),
])
def test_side_logic_agrees_with_grade_predictions(pred_margin, pred_total, ms, mt, final):
    """The ledger's lean and prediction_accuracy's lean are the same bet: the side we record wins exactly when
    `_accuracy_row` says spread_pick_correct / total_pick_correct (pushes excluded)."""
    ph, pa = (pred_total + pred_margin) / 2, (pred_total - pred_margin) / 2
    pred = _pred(pred_home_score=ph, pred_away_score=pa, market_spread=ms, market_total=mt, home_win_prob=0.6)
    row = grade_predictions._accuracy_row(
        {**pred, "home_team_name": "H", "away_team_name": "A"},
        {"home_score": final[0], "away_score": final[1], "market_spread": ms, "market_total": mt})
    obs = {o["market"]: o for o in sources.model_game_observations([pred])}
    margin = final[0] - final[1]
    side_covers = (margin + ms > 0) if obs["spread"]["side"] == "home" else (margin + ms < 0)
    if row["spread_pick_correct"] is not None:
        assert row["spread_pick_correct"] == side_covers
    tot = final[0] + final[1]
    side_hits = (tot > mt) if obs["total"]["side"] == "over" else (tot < mt)
    if row["total_pick_correct"] is not None:
        assert row["total_pick_correct"] == side_hits


# ---- alternate prop lines: one observation per pick key ---------------------------------------------------

def _prop(**kw):
    return {"sport": "nfl", "game_pk": 7, "player_id": "00-1", "player_name": "A. Brown", "market": "rec_yds",
            "side": "over", "line": 64.5, "model_version": "props-sim-v1", "commence_time": KICK,
            "model_prob": 0.58, "market_prob": 0.51, "ev_best": 0.07, "best_book": "dk", "best_price": -105, **kw}


def test_alternate_lines_of_one_prop_are_one_observation():
    """`line` is not part of the pick key, so alternates of the same (game, player, market, side) must not become
    several observations of one pick -- exactly one, the last row (the line the board upserted)."""
    rows = [_prop(line=62.5), _prop(line=64.5), _prop(line=66.5, best_price=130)]
    (o,) = sources.ev_prop_observations(rows)
    assert o["line"] == 66.5 and o["best_price"] == 130
    assert len({o["pick_key"] for o in sources.ev_prop_observations(rows + [_prop(side="under")])}) == 2


def test_distinct_players_and_markets_are_not_collapsed():
    rows = [_prop(), _prop(player_id="00-2"), _prop(market="rush_yds"), _prop(side="under")]
    assert len(sources.ev_prop_observations(rows)) == 4


# ---- one bad candidate never sinks the batch ----------------------------------------------------------------------

def test_candidates_are_deferred_callables_and_a_bad_one_is_skipped():
    cands = sources.ev_prop_candidates([_prop(), _prop(player_id="00-2", game_pk=None), _prop(player_id="00-3")])
    assert len(cands) == 3 and all(callable(c) for c in cands)
    built = sources.build_all(cands)
    assert [o["player_id"] for o in built] == ["00-1", "00-3"]


@pytest.mark.parametrize("bad", [{"game_pk": None}, {"sport": None}, {"sport": ""}])
def test_model_game_bad_row_is_skipped_per_candidate(bad):
    obs = sources.model_game_observations([_pred(**bad), _pred(game_pk=6)])
    assert {o["game_pk"] for o in obs} == {6}


def test_model_game_non_numeric_fields_skip_only_that_game():
    obs = sources.model_game_observations([_pred(pred_home_score="x"), _pred(game_pk=6)])
    assert {o["game_pk"] for o in obs} == {6}


def test_ev_game_and_parlay_bad_rows_are_skipped():
    good = _ev("moneyline", "home")
    obs = sources.ev_game_observations([{**good, "game_pk": None}, good, {**good, "market": "moneyline|x"}], [_game()])
    assert len(obs) == 1
    leg = {"key": "k", "kind": "game", "sport": "nfl", "game_pk": 7, "market": "spread", "side": "home",
           "line": -3.5, "prob": 0.6, "price": -110, "commence_time": KICK}
    t = {"parlay_id": "p", "book": "dk", "model_version": "v", "legs": [leg, {**leg, "game_pk": None}, {**leg, "price": 0}]}
    assert len(sources.parlay_leg_observations([t])) == 1                # price 0 divides by zero -> skipped too
