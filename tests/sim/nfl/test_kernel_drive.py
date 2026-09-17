import numpy as np
from sportsmodel.sim.nfl.kernel import sample_drive
from sportsmodel.sim.nfl.spec import TeamRates


def _tr(**o):
    base = {
        "td": 0.2,
        "fg": 0.15,
        "punt": 0.4,
        "turnover": 0.15,
        "downs": 0.05,
        "end": 0.05,
    }
    base.update(o)
    return TeamRates(base, 0.58, 11.0, 0.6)


def test_sample_drive_points_map_and_dist():
    rng = np.random.default_rng(0)
    off = _tr()
    deff = _tr()
    outs = [sample_drive(off, deff, rng) for _ in range(20000)]
    td = sum(1 for o, _ in outs if o == "td") / 20000
    assert 0.15 < td < 0.25
    assert all((p == 7) == (o == "td") and (p == 3) == (o == "fg") for o, p in outs)
