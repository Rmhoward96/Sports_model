import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "tune_dispersion.py"
_s = importlib.util.spec_from_file_location("tune_dispersion", _p)
td = importlib.util.module_from_spec(_s)
_s.loader.exec_module(td)


def test_objective_zero_when_matching():
    emp = {"sd_total": 4.59, "p_ge11": 0.329, "p_le5": 0.253, "p_blowout": 0.287, "sd_margin": 4.58}
    assert td.objective(dict(emp), emp) < 1e-12


def test_objective_penalizes_deviation():
    emp = {"sd_total": 4.59, "p_ge11": 0.329, "p_le5": 0.253, "p_blowout": 0.287, "sd_margin": 4.58}
    worse = dict(emp); worse["sd_total"] = 3.5
    assert td.objective(worse, emp) > 0.0


def test_coord_search_finds_min_of_toy():
    best = td.coord_search(lambda p: (p["sigma_shared"] - 0.15) ** 2,
                           {"sigma_shared": [0.0, 0.1, 0.15, 0.2]})
    assert best["sigma_shared"] == 0.15
