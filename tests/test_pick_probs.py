"""Pick-side cover / over probabilities from stored distributions (prediction_accuracy.spread_pick_prob /
total_pick_prob). Pure maths: no network, no DB.

Sign conventions (must match site/js/pages/game.js gmDists / gmSplit): margin = HOME minus AWAY as
{"kind": "margin", "offset": o, "pmf": [...]} (pmf[i] = P(margin == i - o)); the home side covers a HOME line L
when margin > -L, the away side when margin < -L; pushes (margin == -L) are excluded and the rest renormalised."""
import json
import math
from pathlib import Path

import pytest

from sportsmodel.model import distributions as D

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "pick_prob_parity.json").read_text())

# margins -2..2 with mass .1 .2 .3 .25 .15
MARGIN = {"kind": "margin", "offset": 2, "pmf": [0.1, 0.2, 0.3, 0.25, 0.15]}
TOTAL = {"kind": "pmf", "pmf": [0, 0.1, 0.2, 0.4, 0.2, 0.1]}   # totals 0..5


def test_split_excludes_pushes_and_renormalises():
    # threshold 0: under = .1+.2 = .3, push .3, over = .25+.15 = .4 -> /0.7
    under, over = D.dist_split(MARGIN, 0)
    assert math.isclose(under, 0.3 / 0.7) and math.isclose(over, 0.4 / 0.7)
    assert math.isclose(under + over, 1.0)


def test_split_half_point_has_no_push():
    under, over = D.dist_split(MARGIN, 0.5)
    assert math.isclose(under, 0.6) and math.isclose(over, 0.4)


def test_split_none_without_usable_distribution():
    assert D.dist_split(None, 0) is None
    assert D.dist_split({"kind": "normal", "mean": 1, "sd": 2}, 0) is None   # no pmf: never fabricate
    assert D.dist_split({"kind": "margin", "offset": 0, "pmf": []}, 0) is None
    assert D.dist_split(MARGIN, None) is None
    # all the mass sits on the line: only a push, nothing to split
    assert D.dist_split({"kind": "pmf", "pmf": [0, 0, 1.0]}, 2) is None


def test_split_accepts_json_text():
    assert D.dist_split(json.dumps(MARGIN), 0.5) == D.dist_split(MARGIN, 0.5)
    assert D.dist_split("not json", 0.5) is None


def test_spread_home_pick_covers_when_margin_beats_minus_line():
    # home -1.5 (L = -1.5): home covers when margin > 1.5 -> only margin 2 (.15) of {-2..2}: over=.15/1
    # pred_margin 3 + (-1.5) > 0 -> home pick
    p = D.spread_pick_prob(MARGIN, pred_margin=3.0, market_spread=-1.5)
    assert math.isclose(p, 0.15)


def test_spread_away_pick_covers_when_margin_below_minus_line():
    # home -1.5, model margin only 0.4 -> edge < 0 -> away pick; away covers when margin < 1.5 -> .85
    p = D.spread_pick_prob(MARGIN, pred_margin=0.4, market_spread=-1.5)
    assert math.isclose(p, 0.85)


def test_spread_home_underdog_line():
    # home +1.5 (L = +1.5): home covers when margin > -1.5 -> margins -1..2 = .9; model margin 0 -> edge +1.5 -> home
    assert math.isclose(D.spread_pick_prob(MARGIN, 0.0, 1.5), 0.9)
    # the away side of the same line: model margin -3 -> edge -1.5 -> away covers when margin < -1.5 -> .1
    assert math.isclose(D.spread_pick_prob(MARGIN, -3.0, 1.5), 0.1)


def test_spread_pushes_excluded_on_whole_number_line():
    # home -1 (L = -1): margin == 1 is a push (.25); home covers margin 2 (.15); away covers -2..0 (.6); /0.75
    assert math.isclose(D.spread_pick_prob(MARGIN, 3.0, -1.0), 0.15 / 0.75)
    assert math.isclose(D.spread_pick_prob(MARGIN, 0.0, -1.0), 0.6 / 0.75)


def test_spread_none_cases():
    assert D.spread_pick_prob(None, 3.0, -1.5) is None            # no distribution
    assert D.spread_pick_prob(MARGIN, None, -1.5) is None         # no model margin -> no pick
    assert D.spread_pick_prob(MARGIN, 3.0, None) is None          # no market line
    assert D.spread_pick_prob(MARGIN, 1.5, -1.5) is None          # model exactly on the line: not a bet


def test_total_over_and_under_picks():
    # market 2.5: P(total > 2.5) = .4+.2+.1 = .7 ; under = .3
    assert math.isclose(D.total_pick_prob(TOTAL, pred_total=3.4, market_total=2.5), 0.7)
    assert math.isclose(D.total_pick_prob(TOTAL, pred_total=1.9, market_total=2.5), 0.3)


def test_total_push_excluded():
    # market 3: under = .1+.2 = .3, push .4, over = .3 ; /0.6
    assert math.isclose(D.total_pick_prob(TOTAL, 4.0, 3.0), 0.5)
    assert math.isclose(D.total_pick_prob(TOTAL, 2.0, 3.0), 0.5)


def test_total_none_cases():
    assert D.total_pick_prob(None, 4, 3) is None
    assert D.total_pick_prob(TOTAL, None, 3) is None
    assert D.total_pick_prob(TOTAL, 4, None) is None
    assert D.total_pick_prob(TOTAL, 3.0, 3.0) is None             # model on the market total


def test_pick_dist_prefers_prediction_over_sim():
    pred = {"margin_dist": MARGIN}
    sim = {"margin_dist": {"kind": "margin", "offset": 0, "pmf": [1.0]}}
    assert D.pick_dist("margin_dist", pred, sim) == MARGIN
    # a prediction without a usable pmf falls through to the sim row (game page gmDists order)
    assert D.pick_dist("margin_dist", {"margin_dist": None}, sim)["pmf"] == [1.0]
    assert D.pick_dist("margin_dist", {"margin_dist": json.dumps(MARGIN)}, sim) == MARGIN
    assert D.pick_dist("margin_dist", {}, None) is None
    assert D.pick_dist("margin_dist", {"margin_dist": {"kind": "normal", "mean": 0, "sd": 1}}, None) is None


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: f"{c['dist']}@{c['threshold']}")
def test_split_matches_game_page_gmSplit(case):
    """The same pmf and line as the site's gmSplit (expected values generated by running game.js)."""
    dist = FIXTURE["dists"][case["dist"]]
    got = D.dist_split(dist, case["threshold"])
    assert got is not None
    under, over = got
    assert math.isclose(over, case["over"], abs_tol=1e-12)
    assert math.isclose(under, case["under"], abs_tol=1e-12)


@pytest.mark.parametrize("line", [10, 3.5, 3, 0, -2.5, -7, -7.5])
def test_spread_pick_prob_matches_game_page_cover_probs(line):
    """gmOdds: home covers = gmSplit(margin, -L).over, away covers = gmSplit(margin, L_away=-L).under. Same numbers here."""
    dist = FIXTURE["dists"]["margin_normal"]
    expect = {c["threshold"]: c for c in FIXTURE["cases"] if c["dist"] == "margin_normal"}
    # the fixture holds thresholds t; a HOME line L reads threshold -L
    t = -line
    assert t in expect
    home = D.spread_pick_prob(dist, pred_margin=-line + 1, market_spread=line)   # edge = +1 -> home pick
    away = D.spread_pick_prob(dist, pred_margin=-line - 1, market_spread=line)   # edge = -1 -> away pick
    assert math.isclose(home, expect[t]["over"], abs_tol=1e-12)
    assert math.isclose(away, expect[t]["under"], abs_tol=1e-12)


# --- malformed stored distributions: never raise, never invent a number -------------------------------------------
BAD_DISTS = [
    {"kind": "margin", "offset": None, "pmf": [0.5, 0.5]},          # offset missing / null
    {"kind": "margin", "offset": "x", "pmf": [0.5, 0.5]},           # non-numeric offset
    {"kind": "margin", "offset": 1, "pmf": [0.5, None, 0.5]},       # a null entry
    {"kind": "pmf", "pmf": ["a", "b"]},                              # strings
    {"kind": "pmf", "pmf": [[0.5], [0.5]]},                          # nested lists
    {"kind": "pmf", "pmf": [float("nan"), 0.5]},                     # NaN mass
    {"kind": "pmf", "pmf": [float("inf"), 0.5]},                     # infinite mass
    {"kind": "pmf", "pmf": [-1.0, -2.0]},                            # no positive mass
    {"kind": "margin", "offset": float("nan"), "pmf": [0.5, 0.5]},   # NaN offset
]


@pytest.mark.parametrize("bad", BAD_DISTS, ids=lambda d: json.dumps(d, default=str)[:60])
def test_dist_split_returns_none_on_a_malformed_distribution(bad):
    assert D.dist_split(bad, 0.5) is None
    assert D.spread_pick_prob(bad, 3.0, -1.5) is None
    assert D.total_pick_prob(bad, 4.0, 3.5) is None


def test_dist_split_returns_none_on_a_non_numeric_threshold_or_line():
    assert D.dist_split(MARGIN, "abc") is None
    assert D.spread_pick_prob(MARGIN, "x", -1.5) is None
    assert D.spread_pick_prob(MARGIN, 3.0, "y") is None
    assert D.total_pick_prob(TOTAL, "x", 3.5) is None
    assert D.total_pick_prob(TOTAL, 4.0, object()) is None


def test_dist_split_handles_json_that_is_not_an_object():
    for text in ("[1, 2]", "3", "null", '"s"', b"\xff\xfe"):
        assert D.dist_split(text, 0.5) is None
    assert D.pick_dist("margin_dist", {"margin_dist": "[1,2]"}, None) is None
