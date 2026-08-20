import pytest
from sportsmodel.sim.mlb.advancement import AdvancementTable, TTO_MULT

_ROWS = [
    # outcome p_1b, bases empty (occ 0): always -> runner on 1st (occ 1), 0 runs
    {"outcome": "p_1b", "occ": 0, "end_occ": 1, "runs": 0, "prob": 1.0},
    # outcome p_1b, runner on 2nd (occ 2): 60% score (occ 1, 1 run), 40% -> 1st&3rd (occ 5, 0)
    {"outcome": "p_1b", "occ": 2, "end_occ": 1, "runs": 1, "prob": 0.6},
    {"outcome": "p_1b", "occ": 2, "end_occ": 5, "runs": 0, "prob": 0.4},
]


def test_sample_deterministic_single_outcome():
    t = AdvancementTable.from_rows(_ROWS)
    # S=2 in the kernel's encoding
    assert t.sample(2, 0, 0.01) == (1, 0)
    assert t.sample(2, 0, 0.99) == (1, 0)


def test_sample_respects_cumulative_probability():
    t = AdvancementTable.from_rows(_ROWS)
    assert t.sample(2, 2, 0.3) == (1, 1)   # in first 0.6
    assert t.sample(2, 2, 0.7) == (5, 0)   # in last 0.4


def test_tto_mult_worsens_for_starter():
    # times through the order 2 and 3 should raise hit/hr, lower K vs pass 1
    assert TTO_MULT["p_hr"][1] > TTO_MULT["p_hr"][0]
    assert TTO_MULT["p_k"][2] < TTO_MULT["p_k"][0]
