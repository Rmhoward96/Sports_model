import numpy as np
import pytest
from sportsmodel.sim.nfl.kernel import (
    _REC_YDS_SHAPE,
    _apply_usage_dispersion,
    _skewed_play_yards,
    attribute_offense,
)
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
        "ypr": 8.0,
        "catch_rate": 1.0,
        "td_share": 0.0,
    }
    base.update(o)
    return PlayerInput(**base)


def test_target_shares_respected_over_many_draws():
    # FIX 3 disperses each player's share by its own per-CALL Gamma(mean=1)
    # draw, so a single huge-n_pass call is no longer a reliable share check
    # (one call = one dispersion draw, which itself can land far from 1.0).
    # Average across many independent team-games instead -- that's exactly
    # the regime the long-run share is supposed to hold in, since the
    # dispersion multiplier's mean is 1 and averages out across sims.
    rng = np.random.default_rng(0)
    high = _player(player_id="high", target_share=0.7, catch_rate=1.0)
    low = _player(player_id="low", target_share=0.3, catch_rate=1.0)

    high_total = low_total = 0
    for _ in range(300):
        result = attribute_offense([high, low], n_pass=100, n_rush=0, n_off_tds=0, rng=rng)
        high_total += result["high"]["receptions"]
        low_total += result["low"]["receptions"]

    # catch_rate=1.0 => receptions == targets assigned, so this checks share allocation.
    assert high_total > low_total
    ratio = high_total / (high_total + low_total)
    assert 0.6 < ratio < 0.8


def test_carry_shares_respected_over_many_draws():
    # Same reasoning as test_target_shares_respected_over_many_draws: average
    # across many independent calls so the per-call usage-dispersion draw
    # (FIX 3) averages out to the underlying carry_share.
    rng = np.random.default_rng(0)
    high = _player(player_id="high", carry_share=0.8, target_share=0.0)
    low = _player(player_id="low", carry_share=0.2, target_share=0.0)

    high_carries_total = low_carries_total = 0
    for _ in range(300):
        result = attribute_offense([high, low], n_pass=0, n_rush=100, n_off_tds=0, rng=rng)
        # rush_yds is noisy per play but its sign/magnitude still tracks carry
        # volume in aggregate; use it as a rough proxy the same way the
        # original test did, just averaged over many independent calls.
        high_carries_total += result["high"]["rush_yds"]
        low_carries_total += result["low"]["rush_yds"]

    assert high_carries_total > low_carries_total


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


def test_multiple_qbs_attribute_pass_yds_to_a_qb_not_a_skill_player():
    rng = np.random.default_rng(5)
    # A high-target WR should NOT be mistaken for the passer just because
    # it has more targets than either QB.
    starter_qb = _player(player_id="qb1", name="Starter QB", pos="QB", target_share=0.0, catch_rate=0.0)
    backup_qb = _player(player_id="qb2", name="Backup QB", pos="QB", target_share=0.0, catch_rate=0.0)
    wr = _player(player_id="wr", name="WR1", pos="WR", target_share=1.0, catch_rate=0.8)

    result = attribute_offense([starter_qb, backup_qb, wr], n_pass=300, n_rush=0, n_off_tds=0, rng=rng)

    team_rec_yds = sum(r["rec_yds"] for r in result.values())
    assert result["qb1"]["pass_yds"] == team_rec_yds or result["qb2"]["pass_yds"] == team_rec_yds
    assert result["wr"]["pass_yds"] == 0


def test_empty_players_list_returns_empty_dict():
    rng = np.random.default_rng(6)
    result = attribute_offense([], n_pass=10, n_rush=10, n_off_tds=1, rng=rng)
    assert result == {}


def test_all_zero_shares_fall_back_to_uniform_without_crashing():
    rng = np.random.default_rng(7)
    players = [
        _player(player_id="a", target_share=0.0, carry_share=0.0, td_share=0.0),
        _player(player_id="b", target_share=0.0, carry_share=0.0, td_share=0.0),
    ]
    n_pass, n_rush, n_off_tds = 100, 100, 4
    result = attribute_offense(players, n_pass=n_pass, n_rush=n_rush, n_off_tds=n_off_tds, rng=rng)

    total_receptions = sum(r["receptions"] for r in result.values())
    total_td = sum(r["td"] for r in result.values())
    assert total_receptions <= n_pass
    assert total_td == n_off_tds
    # Uniform fallback: neither player should be starved of carries/targets.
    assert result["a"]["receptions"] > 0
    assert result["b"]["receptions"] > 0


def test_rec_yds_tracks_ypr_not_ypt():
    # FIX 1: a reception's yards are drawn around ypr, not ypt. Use a single
    # player at catch_rate=1.0 so every target becomes a reception and the
    # allocation is deterministic (one player => prob is always 1.0 even
    # after usage dispersion), isolating the yardage-draw mean.
    rng = np.random.default_rng(0)
    player = _player(
        player_id="a", target_share=1.0, catch_rate=1.0, ypt=5.0, ypr=15.0,
    )
    n_pass = 8000
    result = attribute_offense([player], n_pass=n_pass, n_rush=0, n_off_tds=0, rng=rng)

    assert result["a"]["receptions"] == n_pass
    mean_yds_per_rec = result["a"]["rec_yds"] / result["a"]["receptions"]

    # Should land near ypr (15.0), nowhere near the old ypt-based mean (5.0).
    assert abs(mean_yds_per_rec - 15.0) < 1.5
    assert abs(mean_yds_per_rec - 5.0) > 5.0


def test_play_yardage_has_heavy_right_tail():
    # FIX 2: per-play yardage is a right-skewed Gamma, not a thin Normal --
    # real boom/bust receptions have a much fatter upper tail than a Normal
    # with the same mean would produce. Draw the actual per-play generator
    # directly (rather than reimplementing/duplicating it) and check shape
    # properties that a Normal distribution centered on the same mean would
    # not exhibit: p90 well above p50, and a max far beyond the mean.
    rng = np.random.default_rng(1)
    mean = 10.0
    samples = np.array(
        [_skewed_play_yards(mean, _REC_YDS_SHAPE, 0.0, rng) for _ in range(20000)]
    )

    p50 = np.percentile(samples, 50)
    p90 = np.percentile(samples, 90)

    assert samples.mean() == pytest.approx(mean, rel=0.1)
    # A Normal(mean, sd) has p90/p50 close to 1 + O(sd/mean); for any
    # reasonable per-play sd this stays well under 2. The Gamma tail blows
    # well past that.
    assert p90 / p50 > 2.5
    # A single play breaking for several multiples of the mean should show
    # up across 20000 draws -- a thin Normal would essentially never do this.
    assert samples.max() > 8.0 * mean


def test_usage_dispersion_widens_across_sim_target_variance():
    # FIX 3: usage dispersion should make a player's per-sim target count
    # noisier across independent team-games than a plain fixed-share
    # multinomial would produce (variance = n*p*(1-p)) for the same n/p.
    rng = np.random.default_rng(2)
    high = _player(player_id="high", target_share=0.6, catch_rate=1.0)
    low = _player(player_id="low", target_share=0.4, catch_rate=1.0)

    n_pass = 100
    n_sims = 400
    receptions = np.array(
        [
            attribute_offense([high, low], n_pass=n_pass, n_rush=0, n_off_tds=0, rng=rng)[
                "high"
            ]["receptions"]
            for _ in range(n_sims)
        ]
    )

    empirical_var = receptions.var()
    fixed_share_var = n_pass * 0.6 * (1 - 0.6)

    # Dispersion should push variance clearly above the fixed-multinomial floor
    # (proving the Dirichlet redraw is active), but only moderately: B.2 retuned
    # _USAGE_CONCENTRATION 20 -> 150 precisely because the old, larger dispersion
    # was over-boom/bust and mis-calibrated the player-prop medians. At the
    # calibrated concentration the redraw still widens variance ~1.5-1.9x the
    # floor rather than the >2x it produced when over-dispersed.
    assert empirical_var > 1.3 * fixed_share_var


def test_usage_dispersion_is_mean_preserving():
    # Fix round 1: a naive `p_i * Gamma(k, 1/k)` then renormalize regresses
    # shares toward equal (confirmed 0.90 -> ~0.88, 0.80 -> ~0.78 in review),
    # understating workhorse usage -- the wrong direction, since receivers
    # already under-project. The true-Dirichlet replacement is exactly
    # mean-preserving: E[dispersed_share_i] == p_i. Check that directly by
    # averaging many independent dispersion draws for a skewed share vector.
    rng = np.random.default_rng(3)
    probs = np.array([0.7, 0.2, 0.1])

    n_draws = 4000
    dispersed = np.array([_apply_usage_dispersion(probs, rng) for _ in range(n_draws)])
    mean_dispersed = dispersed.mean(axis=0)

    assert mean_dispersed == pytest.approx(probs, abs=0.02)


def test_usage_dispersion_zero_shares_stay_zero_and_fall_back_when_all_zero():
    # A player with a true zero share must never receive a positive dispersed
    # share (a hard zero, not just "unlikely" under a tiny Dirichlet weight).
    rng = np.random.default_rng(4)
    probs = np.array([0.8, 0.2, 0.0])
    for _ in range(200):
        dispersed = _apply_usage_dispersion(probs, rng)
        assert dispersed[2] == 0.0
        assert dispersed.sum() == pytest.approx(1.0)

    # All-zero input has no positive alpha to draw from; falls back to the
    # (all-zero) input rather than crashing or fabricating a distribution.
    all_zero = np.array([0.0, 0.0])
    assert np.array_equal(_apply_usage_dispersion(all_zero, rng), all_zero)


def test_td_split_credits_qb_passing_and_conserves_scorers():
    # TD feature: each offensive TD is split into passing (receiving) vs rushing.
    # Passing TDs credit the receiver's anytime `td` AND the QB's `pass_tds`;
    # rushing TDs credit the rusher's `td`. Every TD is credited to exactly one
    # scorer (conservation), and QB pass_tds ~ n_off_tds * pass_td_share.
    qb = _player(player_id="qb", pos="QB", target_share=0.0, rec_td_share=0.0, rush_td_share=0.1)
    wr = _player(player_id="wr", pos="WR", target_share=0.6, catch_rate=0.65, rec_td_share=0.7, rush_td_share=0.0)
    rb = _player(player_id="rb", pos="RB", carry_share=0.7, rec_td_share=0.3, rush_td_share=0.9)
    rng = np.random.default_rng(7)
    n, pass_tds, wr_td, rb_td, qb_td = 6000, [], 0, 0, 0
    for _ in range(n):
        box = attribute_offense([qb, wr, rb], n_pass=30, n_rush=25, n_off_tds=3, rng=rng, pass_td_share=0.6)
        assert sum(box[p]["td"] for p in box) == 3   # conservation: 3 TDs -> exactly 3 scorer credits
        pass_tds.append(box["qb"]["pass_tds"])
        wr_td += box["wr"]["td"]; rb_td += box["rb"]["td"]; qb_td += box["qb"]["td"]
    assert abs(np.mean(pass_tds) - 3 * 0.6) < 0.12     # QB passing TDs track pass_td_share
    assert wr_td > 0 and rb_td > 0                      # receivers and rushers both score
    # WR (receiving-only) never gets a rushing TD credited beyond its share; the
    # QB's own anytime `td` (rushing) is small vs its passing credit.
    assert np.mean(pass_tds) > qb_td / n               # QB scores through the air far more than on the ground


def test_no_offensive_tds_means_no_pass_tds():
    qb = _player(player_id="qb", pos="QB")
    wr = _player(player_id="wr", pos="WR", target_share=1.0, rec_td_share=1.0)
    box = attribute_offense([qb, wr], n_pass=20, n_rush=10, n_off_tds=0, rng=np.random.default_rng(1), pass_td_share=0.6)
    assert box["qb"]["pass_tds"] == 0
    assert sum(box[p]["td"] for p in box) == 0
