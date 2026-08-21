import importlib.util
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts" / "validate_sim_dist.py"
_s = importlib.util.spec_from_file_location("validate_sim_dist", _p)
vsd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(vsd)


def test_tail_metrics_basic():
    home = np.array([5, 0, 8, 2])
    away = np.array([3, 6, 8, 1])
    m = vsd.tail_metrics(home, away)
    # totals: 8, 6, 16, 3
    assert abs(m["mean_total"] - 8.25) < 1e-9
    assert abs(m["p_ge11"] - 0.25) < 1e-9   # only 16
    assert abs(m["p_le5"] - 0.25) < 1e-9    # only 3
    assert abs(m["p_shutout"] - 0.25) < 1e-9  # the 0-6 game
    # margins: 2, -6, 0, 1 -> |.|>=5 : one (-6)
    assert abs(m["p_blowout"] - 0.25) < 1e-9
