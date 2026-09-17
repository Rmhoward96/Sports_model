import numpy as np

from sportsmodel.sim.engine import home_win_prob
from sportsmodel.sim.nfl.kernel import simulate_game
from sportsmodel.sim.nfl.spec import NflGameSpec, PlayerInput, TeamRates


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
    drive_outcomes = {
        "td": base["td"],
        "fg": base["fg"],
        "punt": base["punt"],
        "turnover": base["turnover"],
        "downs": base["downs"],
        "end": base["end"],
    }
    return TeamRates(drive_outcomes, base.get("pass_rate", 0.58),
                      base.get("drives_per_game", 11.0), base.get("rz_td_rate", 0.6))


def _roster(prefix: str) -> list[PlayerInput]:
    return [
        PlayerInput(player_id=f"{prefix}_qb", name=f"{prefix} QB", pos="QB",
                    target_share=0.0, carry_share=0.05, ypt=0.0, ypc=2.0,
                    catch_rate=0.0, td_share=0.05),
        PlayerInput(player_id=f"{prefix}_wr1", name=f"{prefix} WR1", pos="WR",
                    target_share=0.55, carry_share=0.0, ypt=8.5, ypc=0.0,
                    catch_rate=0.65, td_share=0.4),
        PlayerInput(player_id=f"{prefix}_wr2", name=f"{prefix} WR2", pos="WR",
                    target_share=0.45, carry_share=0.0, ypt=7.0, ypc=0.0,
                    catch_rate=0.6, td_share=0.2),
        PlayerInput(player_id=f"{prefix}_rb", name=f"{prefix} RB", pos="RB",
                    target_share=0.0, carry_share=0.95, ypt=0.0, ypc=4.2,
                    catch_rate=0.0, td_share=0.35),
    ]


def _spec(home=None, away=None) -> NflGameSpec:
    return NflGameSpec(
        home_team="Home",
        away_team="Away",
        home=home or _tr(),
        away=away or _tr(),
        home_players=_roster("home"),
        away_players=_roster("away"),
    )


def test_shapes_match_n_sims():
    rng = np.random.default_rng(0)
    n_sims = 500
    sims = simulate_game(_spec(), n_sims, rng)
    assert sims.home_score.shape == (n_sims,)
    assert sims.away_score.shape == (n_sims,)
    for pid, stats in sims.player_stats.items():
        for market, arr in stats.items():
            assert arr.shape == (n_sims,), f"{pid}.{market}"


def test_scores_non_negative_and_sane_total_range():
    rng = np.random.default_rng(1)
    n_sims = 3000
    sims = simulate_game(_spec(), n_sims, rng)
    assert np.all(sims.home_score >= 0)
    assert np.all(sims.away_score >= 0)
    mean_total = float(np.mean(sims.home_score + sims.away_score))
    assert 30.0 < mean_total < 60.0


def test_home_win_prob_in_bounds_via_engine_helper():
    rng = np.random.default_rng(2)
    sims = simulate_game(_spec(), 2000, rng)
    p = home_win_prob(sims)
    assert 0.0 <= p <= 1.0


def test_dominant_offense_beats_weak_offense_on_average():
    neutral_defense = _tr()
    dominant_off = _tr(td=0.45, fg=0.1, punt=0.25, turnover=0.1, downs=0.05, end=0.05)
    weak_off = _tr(td=0.05, fg=0.05, punt=0.5, turnover=0.3, downs=0.05, end=0.05)

    n_sims = 3000
    dominant_sims = simulate_game(
        _spec(home=dominant_off, away=neutral_defense), n_sims, np.random.default_rng(3)
    )
    weak_sims = simulate_game(
        _spec(home=weak_off, away=neutral_defense), n_sims, np.random.default_rng(3)
    )

    assert dominant_sims.home_score.mean() > weak_sims.home_score.mean()


def test_player_stats_keys_equal_full_roster():
    rng = np.random.default_rng(4)
    spec = _spec()
    sims = simulate_game(spec, 200, rng)
    expected_ids = {p.player_id for p in (*spec.home_players, *spec.away_players)}
    assert set(sims.player_stats.keys()) == expected_ids
    for stats in sims.player_stats.values():
        assert set(stats.keys()) == {"pass_yds", "rush_yds", "rec_yds", "receptions", "td"}


def test_reproducible_with_fixed_seed():
    spec = _spec()
    sims1 = simulate_game(spec, 500, np.random.default_rng(42))
    sims2 = simulate_game(spec, 500, np.random.default_rng(42))

    assert np.array_equal(sims1.home_score, sims2.home_score)
    assert np.array_equal(sims1.away_score, sims2.away_score)
    for pid in sims1.player_stats:
        for market in sims1.player_stats[pid]:
            assert np.array_equal(sims1.player_stats[pid][market], sims2.player_stats[pid][market])


def test_all_td_offense_reconciles_player_tds_with_score():
    # Every drive is a TD for both teams, so each team's score is exactly
    # 7 * (number of TD drives), and attribute_offense conserves TD counts
    # across players -- so 7 * sum(player tds) must equal the team's score,
    # per sim, exactly (no averaging/tolerance needed).
    all_td = _tr(td=1.0, fg=0.0, punt=0.0, turnover=0.0, downs=0.0, end=0.0)
    spec = _spec(home=all_td, away=all_td)
    n_sims = 200
    sims = simulate_game(spec, n_sims, np.random.default_rng(10))

    home_ids = [p.player_id for p in spec.home_players]
    away_ids = [p.player_id for p in spec.away_players]

    home_td_totals = sum(sims.player_stats[pid]["td"] for pid in home_ids)
    away_td_totals = sum(sims.player_stats[pid]["td"] for pid in away_ids)

    assert np.array_equal(sims.home_score, home_td_totals * 7)
    assert np.array_equal(sims.away_score, away_td_totals * 7)
    assert np.all(sims.home_score % 7 == 0)
    assert np.all(sims.away_score % 7 == 0)


def test_all_fg_offense_scores_are_multiples_of_three_with_no_player_tds():
    # Every drive is a FG for both teams: scores are multiples of 3, and
    # since no drive scores a TD, attribute_offense is called with
    # n_off_tds=0 for both teams -- no player should ever be credited a TD.
    all_fg = _tr(td=0.0, fg=1.0, punt=0.0, turnover=0.0, downs=0.0, end=0.0)
    spec = _spec(home=all_fg, away=all_fg)
    n_sims = 200
    sims = simulate_game(spec, n_sims, np.random.default_rng(11))

    assert np.all(sims.home_score % 3 == 0)
    assert np.all(sims.away_score % 3 == 0)
    for stats in sims.player_stats.values():
        assert np.all(stats["td"] == 0)
