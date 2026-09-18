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
    return TeamRates(
        drive_outcomes,
        base.get("pass_rate", 0.58),
        base.get("drives_per_game", 11.0),
        base.get("rz_td_rate", 0.6),
        pass_att_pg=base.get("pass_att_pg", 0.0),
        rush_att_pg=base.get("rush_att_pg", 0.0),
        sack_rate=base.get("sack_rate", 0.0),
        completion_pct=base.get("completion_pct", 0.0),
    )


def _roster(prefix: str) -> list[PlayerInput]:
    return [
        PlayerInput(player_id=f"{prefix}_qb", name=f"{prefix} QB", pos="QB",
                    target_share=0.0, carry_share=0.05, ypt=0.0, ypc=2.0,
                    ypr=0.0, catch_rate=0.0, td_share=0.05),
        PlayerInput(player_id=f"{prefix}_wr1", name=f"{prefix} WR1", pos="WR",
                    target_share=0.55, carry_share=0.0, ypt=8.5, ypc=0.0,
                    ypr=13.1, catch_rate=0.65, td_share=0.4),
        PlayerInput(player_id=f"{prefix}_wr2", name=f"{prefix} WR2", pos="WR",
                    target_share=0.45, carry_share=0.0, ypt=7.0, ypc=0.0,
                    ypr=11.7, catch_rate=0.6, td_share=0.2),
        PlayerInput(player_id=f"{prefix}_rb", name=f"{prefix} RB", pos="RB",
                    target_share=0.0, carry_share=0.95, ypt=0.0, ypc=4.2,
                    ypr=0.0, catch_rate=0.0, td_share=0.35),
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


def test_shared_game_env_correlates_home_and_away_scoring():
    # FIX 4: a single shared per-sim game_env multiplier scales both teams'
    # drive counts together, so home and away scoring should be positively
    # correlated within a sim (a "shootout" sim runs hot for both offenses;
    # a "slog" runs cold for both) -- fully independent team simulations
    # would produce ~0 correlation here.
    rng = np.random.default_rng(20)
    sims = simulate_game(_spec(), 800, rng)

    corr = np.corrcoef(sims.home_score, sims.away_score)[0, 1]
    assert corr > 0.1


def _single_receiver_rusher_roster(prefix: str) -> list[PlayerInput]:
    # A roster engineered so box-score counts are directly observable:
    # catch_rate=1.0 + a single receiver with target_share=1.0 means
    # receptions == targets assigned == n_pass exactly (no usage-dispersion
    # ambiguity, since it's the only player with a positive target share);
    # a single rusher with carry_share=1.0 and ypc=1.0 means mean rush_yds
    # tracks n_rush 1-for-1.
    return [
        PlayerInput(player_id=f"{prefix}_qb", name=f"{prefix} QB", pos="QB",
                    target_share=0.0, carry_share=0.0, ypt=0.0, ypc=0.0,
                    ypr=0.0, catch_rate=0.0, td_share=0.0),
        PlayerInput(player_id=f"{prefix}_wr", name=f"{prefix} WR", pos="WR",
                    target_share=1.0, carry_share=0.0, ypt=8.0, ypc=0.0,
                    ypr=8.0, catch_rate=1.0, td_share=0.0),
        PlayerInput(player_id=f"{prefix}_rb", name=f"{prefix} RB", pos="RB",
                    target_share=0.0, carry_share=1.0, ypt=0.0, ypc=1.0,
                    ypr=0.0, catch_rate=0.0, td_share=0.0),
    ]


def test_drive_box_volume_uses_real_per_game_attempts():
    # B.3 Task 3: box-score volume should be anchored to the team's real
    # per-game attempts (pass_att_pg/rush_att_pg), not the old
    # n_drives * _PLAYS_PER_DRIVE * pass_rate derivation -- that derivation
    # over-counted pass attempts (+63 pass_yds bias) and under-counted rush
    # attempts (-6 rush_yds bias) in the walk-forward.
    home_rates = _tr(pass_att_pg=34.0, rush_att_pg=27.0)
    away_rates = _tr()  # no volume fields -> legacy path, doesn't matter here
    home_players = _single_receiver_rusher_roster("home")
    away_players = _roster("away")
    spec = NflGameSpec(
        home_team="Home", away_team="Away",
        home=home_rates, away=away_rates,
        home_players=home_players, away_players=away_players,
    )

    n_sims = 4000
    sims = simulate_game(spec, n_sims, np.random.default_rng(50))

    mean_receptions = sims.player_stats["home_wr"]["receptions"].mean()
    mean_rush_yds = sims.player_stats["home_rb"]["rush_yds"].mean()

    # catch_rate=1.0, sole receiver -> team receptions == n_pass exactly
    # each sim, so the mean over many sims should land tightly on
    # pass_att_pg (game_env has mean 1.0).
    assert abs(mean_receptions - 34.0) < 1.5
    # ypc=1.0, sole rusher -> mean rush_yds tracks n_rush * ypc == rush_att_pg.
    assert abs(mean_rush_yds - 27.0) < 3.0

    # B.3 dispersion: n_pass is a Poisson draw around pass_att_pg*game_env, so
    # the per-game target count must VARY across sims (a deterministic round()
    # left only game_env as noise and under-dispersed every marginal). With
    # catch_rate=1.0 the sole receiver's receptions == n_pass, so its std over
    # sims should be at least ~sqrt(34) (Poisson) inflated by game_env spread,
    # comfortably above a small floor.
    assert sims.player_stats["home_wr"]["receptions"].std() > 4.0


def test_drive_box_volume_falls_back_to_legacy_without_volume_fields():
    # Back-compat: a TeamRates without the B.3 volume fields (pass_att_pg=0.0
    # / rush_att_pg=0.0, the dataclass default) must still run via the
    # pre-B.3 n_drives/pass_rate derivation instead of collapsing to 0 plays.
    drives_per_game = 11.0
    pass_rate = 0.58
    rates = _tr(drives_per_game=drives_per_game, pass_rate=pass_rate)
    assert rates.pass_att_pg == 0.0
    assert rates.rush_att_pg == 0.0

    players = _single_receiver_rusher_roster("home")
    spec = NflGameSpec(
        home_team="Home", away_team="Away",
        home=rates, away=_tr(),
        home_players=players, away_players=_roster("away"),
    )

    n_sims = 3000
    sims = simulate_game(spec, n_sims, np.random.default_rng(51))

    mean_receptions = sims.player_stats["home_wr"]["receptions"].mean()
    # Legacy formula: n_plays = round(n_drives * _PLAYS_PER_DRIVE),
    # n_pass = round(n_plays * pass_rate); with drives_per_game=11 well
    # above the 6-drive floor, this lands close to
    # drives_per_game * 6.0 * pass_rate = 38.28. A generous band confirms
    # the legacy path (not a collapse to 0, and not the unrelated
    # pass_att_pg-anchored value) is still exercised.
    legacy_expected = drives_per_game * 6.0 * pass_rate
    assert abs(mean_receptions - legacy_expected) < 5.0


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
