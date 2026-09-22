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


def test_sample_drive_stout_defense_lowers_scoring():
    # A defense that allows almost no TD/FG (mostly punts) should pull the
    # combined scoring rate well below facing an average defense.
    rng = np.random.default_rng(1)
    off = _tr()
    stout = _tr(td=0.03, fg=0.05, punt=0.72, turnover=0.15, downs=0.03, end=0.02)
    avg = _tr()
    n = 40000
    score = lambda d: sum(1 for _ in range(n) if sample_drive(off, d, rng)[1] > 0) / n
    assert score(stout) < score(avg) - 0.05


def test_sample_drive_score_tilt_raises_and_lowers_scoring():
    rng = np.random.default_rng(2)
    off, deff = _tr(), _tr()
    n = 40000
    frac = lambda tilt: sum(1 for _ in range(n) if sample_drive(off, deff, rng, score_tilt=tilt)[1] > 0) / n
    assert frac(1.3) > frac(1.0) > frac(0.7)   # home tilt scores more, away less
