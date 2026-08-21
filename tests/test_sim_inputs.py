# tests/test_sim_inputs.py
from sportsmodel.sim.mlb import inputs, kernel
from sportsmodel.sim.mlb.advancement import AdvancementTable

_L = {"p_bb": .08, "p_k": .22, "p_1b": .15, "p_2b": .045, "p_3b": .004, "p_hr": .03, "p_out": .471}


def _vec(**o):
    v = dict(_L); v.update(o); s = sum(v.values()); return {k: x / s for k, x in v.items()}


def test_build_game_spec_shapes_and_context():
    home = [(100 + i, _vec()) for i in range(9)]
    away = [(200 + i, _vec()) for i in range(9)]
    spec = inputs.build_game_spec(
        home, away, _vec(p_k=.28), _vec(p_k=.26), _vec(), _vec(),
        workload={1: (16.0, 5.0), 2: (15.0, 5.5)},
        context={"home_pf": 1.05, "hr_mult": 1.0, "home_def": 1.0, "away_def": 1.0},
        league=_L, adv=AdvancementTable.from_rows([]),
        home_starter_id=1, away_starter_id=2,
    )
    assert isinstance(spec, kernel.GameSpec)
    assert len(spec.home_order) == 9 and len(spec.away_order) == 9
    # each batter has BOTH matchup vectors, and they are valid distributions
    b = spec.home_order[0]
    assert abs(sum(b.vec_vs_sp.values()) - 1.0) < 1e-9
    assert abs(sum(b.vec_vs_bp.values()) - 1.0) < 1e-9
    assert spec.home_starter.avg_outs == 16.0


def test_build_game_spec_falls_back_to_starter_when_bullpen_vec_missing():
    """A missing bullpen profile (e.g. thin/no-relief-innings sample in a
    backtest month) must not crash build_game_spec -- it should fall back to
    the opposing starter's vector for vec_vs_bp, matching backtest_game.py's
    `vec_bp = ... if opp_bp else vec_sp` guard."""
    home = [(100 + i, _vec()) for i in range(9)]
    away = [(200 + i, _vec()) for i in range(9)]
    spec = inputs.build_game_spec(
        home, away, _vec(p_k=.28), _vec(p_k=.26), None, None,
        workload={1: (16.0, 5.0), 2: (15.0, 5.5)},
        context={"home_pf": 1.05, "hr_mult": 1.0, "home_def": 1.0, "away_def": 1.0},
        league=_L, adv=AdvancementTable.from_rows([]),
        home_starter_id=1, away_starter_id=2,
    )
    for b in (*spec.home_order, *spec.away_order):
        assert abs(sum(b.vec_vs_sp.values()) - 1.0) < 1e-9
        assert abs(sum(b.vec_vs_bp.values()) - 1.0) < 1e-9
