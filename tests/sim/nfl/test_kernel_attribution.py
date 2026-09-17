import numpy as np
from sportsmodel.sim.nfl.kernel import attribute_offense
from sportsmodel.sim.nfl.spec import PlayerInput


def _player(**o):
    base = {
        "player_id": "p1",
        "name": "Player",
        "pos": "WR",
        "target_share": 0.0,
        "carry_share": 0.0,
        "ypt": 8.0,
        "ypc": 4.0,
        "catch_rate": 1.0,
        "td_share": 0.0,
    }
    base.update(o)
    return PlayerInput(**base)


def test_target_shares_respected_over_many_draws():
    rng = np.random.default_rng(0)
    high = _player(player_id="high", target_share=0.7, catch_rate=1.0)
    low = _player(player_id="low", target_share=0.3, catch_rate=1.0)
    result = attribute_offense([high, low], n_pass=10000, n_rush=0, n_off_tds=0, rng=rng)

    # catch_rate=1.0 => receptions == targets assigned, so this checks share allocation.
    assert result["high"]["receptions"] > result["low"]["receptions"]
    ratio = result["high"]["receptions"] / (result["high"]["receptions"] + result["low"]["receptions"])
    assert 0.6 < ratio < 0.8


def test_carry_shares_respected_over_many_draws():
    rng = np.random.default_rng(0)
    high = _player(player_id="high", carry_share=0.8, target_share=0.0)
    low = _player(player_id="low", carry_share=0.2, target_share=0.0)
    result = attribute_offense([high, low], n_pass=0, n_rush=10000, n_off_tds=0, rng=rng)

    # rush_yds is noisy but carry counts drive magnitude; use abs(rush_yds) as a rough proxy
    # via checking that the high-carry player accumulated far more total rushing volume.
    assert result["high"]["rush_yds"] > result["low"]["rush_yds"]


def test_receptions_never_exceed_targets_thrown():
    rng = np.random.default_rng(1)
    players = [
        _player(player_id="a", target_share=0.5, catch_rate=0.5),
        _player(player_id="b", target_share=0.5, catch_rate=0.5),
    ]
    n_pass = 1000
    result = attribute_offense(players, n_pass=n_pass, n_rush=0, n_off_tds=0, rng=rng)
    total_receptions = sum(r["receptions"] for r in result.values())
    assert total_receptions <= n_pass


def test_qb_pass_yds_equals_sum_of_team_rec_yds():
    rng = np.random.default_rng(2)
    qb = _player(player_id="qb", name="QB", pos="QB", target_share=0.0, carry_share=0.05, catch_rate=0.0)
    wr = _player(player_id="wr", name="WR", pos="WR", target_share=0.6, catch_rate=0.8)
    rb = _player(player_id="rb", name="RB", pos="RB", target_share=0.4, catch_rate=0.6, carry_share=0.5)

    result = attribute_offense([qb, wr, rb], n_pass=300, n_rush=200, n_off_tds=0, rng=rng)

    team_rec_yds = sum(r["rec_yds"] for r in result.values())
    assert result["qb"]["pass_yds"] == team_rec_yds


def test_td_allocation_is_conserved():
    rng = np.random.default_rng(3)
    players = [
        _player(player_id="a", td_share=0.5),
        _player(player_id="b", td_share=0.3),
        _player(player_id="c", td_share=0.2),
    ]
    n_off_tds = 7
    result = attribute_offense(players, n_pass=0, n_rush=0, n_off_tds=n_off_tds, rng=rng)
    total_td = sum(r["td"] for r in result.values())
    assert total_td == n_off_tds


def test_deterministic_with_seeded_rng():
    players = [
        _player(player_id="a", target_share=0.5, carry_share=0.5, td_share=0.5),
        _player(player_id="b", target_share=0.5, carry_share=0.5, td_share=0.5),
    ]

    rng1 = np.random.default_rng(42)
    result1 = attribute_offense(players, n_pass=50, n_rush=50, n_off_tds=3, rng=rng1)

    rng2 = np.random.default_rng(42)
    result2 = attribute_offense(players, n_pass=50, n_rush=50, n_off_tds=3, rng=rng2)

    assert result1 == result2


def test_all_yardage_values_are_ints():
    rng = np.random.default_rng(4)
    players = [_player(player_id="a", target_share=1.0, carry_share=1.0, td_share=1.0)]
    result = attribute_offense(players, n_pass=20, n_rush=20, n_off_tds=1, rng=rng)
    for field in ("pass_yds", "rush_yds", "rec_yds", "receptions", "td"):
        assert isinstance(result["a"][field], int)
