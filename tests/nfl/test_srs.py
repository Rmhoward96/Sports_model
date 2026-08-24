import math
import pandas as pd
from sportsmodel.nfl.srs import compute_srs

def _g(h, a, hs, as_):
    return {"home_team": h, "away_team": a, "home_score": hs, "away_score": as_}

def test_two_team_single_game():
    # A beats B by 10 at home -> ratings +/-5, zero mean
    games = pd.DataFrame([_g("A", "B", 20, 10)])
    r = compute_srs(games)
    assert math.isclose(r["A"], 5.0, abs_tol=1e-6)
    assert math.isclose(r["B"], -5.0, abs_tol=1e-6)
    assert math.isclose(sum(r.values()), 0.0, abs_tol=1e-6)

def test_strength_of_schedule_ranking():
    # A and D have the *same* average point margin (0: one close win, one
    # close loss each), so any rating gap between them must come purely
    # from opponent strength. B/C are anchored strong (blowout wins over
    # Z) and E/F are anchored weak (blowout losses to W); Z-W is a single
    # connecting game so the whole schedule is one component.
    games = pd.DataFrame([
        _g("B", "Z", 30, 0),    # B crushes weak anchor Z
        _g("C", "Z", 30, 0),    # C crushes weak anchor Z
        _g("W", "E", 30, 0),    # strong anchor W crushes E
        _g("W", "F", 30, 0),    # strong anchor W crushes F
        _g("A", "B", 24, 21),   # A edges strong B by 3
        _g("C", "A", 20, 17),   # strong C edges A by 3
        _g("D", "E", 13, 10),   # D edges weak E by 3
        _g("F", "D", 10, 7),    # weak F edges D by 3
        _g("Z", "W", 10, 9),    # connects the two halves of the schedule
    ])
    r = compute_srs(games)
    assert math.isclose(sum(r.values()), 0.0, abs_tol=1e-6)
    assert r["A"] > r["D"]      # same record & same avg margin, tougher schedule -> higher rating
