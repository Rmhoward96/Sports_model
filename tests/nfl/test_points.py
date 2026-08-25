import math
import pandas as pd
from sportsmodel.nfl.points import compute_points_ratings, expected_total

def _g(h, a, hs, as_):
    return {"home_team": h, "away_team": a, "home_score": hs, "away_score": as_}

def test_off_def_zero_mean_and_convergence():
    games = pd.DataFrame([_g("A","B",30,20), _g("C","D",17,14), _g("A","C",24,21),
                          _g("B","D",20,20), _g("A","D",28,10), _g("B","C",21,24)])
    ratings, lg = compute_points_ratings(games, k_points=0.0)  # no shrink -> pure solve
    offs = [r["off"] for r in ratings.values()]
    defs = [r["def"] for r in ratings.values()]
    assert abs(sum(offs)) < 1e-6 and abs(sum(defs)) < 1e-6
    assert lg > 0

def test_expected_total_uses_off_and_def():
    games = pd.DataFrame([_g("A","B",30,20), _g("A","B",28,24), _g("B","A",21,17)])
    ratings, lg = compute_points_ratings(games, k_points=0.0)
    et = expected_total(ratings, lg, "A", "B")
    manual = ((lg + ratings["A"]["off"] + ratings["B"]["def"])
              + (lg + ratings["B"]["off"] + ratings["A"]["def"]))
    assert math.isclose(et, manual, rel_tol=1e-9)

def test_early_season_shrinkage_pulls_toward_zero():
    games = pd.DataFrame([_g("A","B",40,10)])  # 1 game each
    r0, _ = compute_points_ratings(games, k_points=0.0)
    r4, _ = compute_points_ratings(games, k_points=4.0)
    # n=1 -> factor 1/5; shrunk magnitude strictly smaller
    assert abs(r4["A"]["off"]) < abs(r0["A"]["off"])
    assert math.isclose(r4["A"]["off"], r0["A"]["off"] * (1 / 5), rel_tol=1e-9)

def test_off_rewards_scoring_against_strong_defenses():
    # A and D score the SAME raw points (24 ppg over 2 games each), so any
    # gap in their "off" rating must come purely from opponent defensive
    # strength, not from one team simply outscoring the other. Same design
    # discipline as P1 SRS's strength-of-schedule test.
    #
    # S is a demonstrably strong defense: it held F to just 3 points.
    # W is a demonstrably weak defense: it allowed F 30 points.
    # A's only opponents are S and F; D's only opponents are W and F -- F is
    # common to both, so it contributes equally to each side's average
    # opponent strength and cancels out, leaving S vs. W as the only source
    # of any "off" gap between A and D.
    games = pd.DataFrame([
        _g("S", "F", 30, 3),    # S: strong defense, allowed only 3
        _g("W", "F", 3, 30),    # W: weak defense, allowed 30
        _g("S", "W", 20, 20),   # closes the S/F/W triangle (neutral tie)
        _g("A", "S", 24, 20),   # A scores 24 against strong-D team S
        _g("A", "F", 24, 20),   # A scores 24 against common opponent F
        _g("D", "W", 24, 20),   # D scores 24 against weak-D team W
        _g("D", "F", 24, 20),   # D scores 24 against common opponent F
    ])
    ratings, lg = compute_points_ratings(games, k_points=0.0)
    offs = [r["off"] for r in ratings.values()]
    defs = [r["def"] for r in ratings.values()]
    assert abs(sum(offs)) < 1e-6 and abs(sum(defs)) < 1e-6
    # equal raw points-for: A and D each scored 24 in both of their games
    assert ratings["A"]["off"] > ratings["D"]["off"]
