import numpy as np
from sportsmodel.model.props_eval import pit_pmf, pit_uniform, rps_pmf

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
