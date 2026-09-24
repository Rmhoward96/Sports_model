import numpy as np
import pandas as pd
from sportsmodel.model.props_eval import (
    decile_ece,
    paired_frame,
    relative_skill,
    cluster_bootstrap,
    rung_decision,
    population_from_baseline,
    pit_pmf,
    pit_uniform,
    rps_pmf,
)


def test_rps_zero_for_point_mass_on_actual():
    assert rps_pmf([0, 0, 1, 0], 2) == 0.0


def test_rps_penalizes_distance():
    near, far = [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]
    assert rps_pmf(near, 2) < rps_pmf(far, 2)


def test_pit_bounds_and_determinism():
    u = pit_uniform(2024, 3, "p", "rec_yds")
    assert u == pit_uniform(2024, 3, "p", "rec_yds") and 0 <= u < 1
    assert pit_pmf([0.25, 0.25, 0.5], 1, 0.0) == 0.25
    assert pit_pmf([0.25, 0.25, 0.5], 1, 1.0) == 0.5


def test_decile_ece_uniform_is_small():
    rng = np.random.default_rng(0)
    assert decile_ece(rng.uniform(size=20000)) < 0.01
    assert decile_ece(np.full(1000, 0.05)) > 0.15


def _recs(rps, market="rec_yds", n=200, shift=0.0):
    return [
        {
            "season": 2024,
            "week": 1 + i % 17,
            "home": f"T{i % 8}",
            "player_id": f"p{i}",
            "market": market,
            "rps": rps + shift * (i % 3),
            "pit": (i % 10) / 10 + 0.05,
            "mean": 50.0,
        }
        for i in range(n)
    ]


def test_rung_passes_on_clear_improvement():
    base, cand = _recs(10.0), _recs(9.0)
    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    d = rung_decision(paired_frame(base, cand, pop))
    assert d["pass"] and d["lo"] > 0


def test_rung_fails_when_one_market_worsens():
    base = _recs(10.0) + _recs(2.0, market="receptions")
    cand = _recs(8.0) + _recs(2.1, market="receptions")  # receptions 5% worse
    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    d = rung_decision(paired_frame(base, cand, pop))
    assert not d["pass"] and any("receptions" in r for r in d["reasons"])


def test_population_uses_projected_gate():
    recs = [
        {"season": 2024, "week": 1, "player_id": "a", "market": "rec_yds", "mean": 60.0},
        {"season": 2024, "week": 1, "player_id": "b", "market": "rec_yds", "mean": 1.0},
    ]
    assert population_from_baseline(recs) == {(2024, 1, "a", "rec_yds")}
