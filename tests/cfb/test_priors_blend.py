from sportsmodel.cfb.priors import DecayConfig, prior_weight, blend_rating


def test_prior_dominates_at_zero_games():
    assert prior_weight(0, DecayConfig(half_life_games=3)) == 1.0
    assert blend_rating(1800.0, 1500.0, 0, DecayConfig(half_life_games=3)) == 1800.0


def test_prior_halves_at_half_life():
    w = prior_weight(3, DecayConfig(half_life_games=3))
    assert abs(w - 0.5) < 1e-9


def test_in_season_dominates_late():
    r = blend_rating(1800.0, 1500.0, 20, DecayConfig(half_life_games=3))
    assert abs(r - 1500.0) < 3.5   # prior almost gone by 20 games
