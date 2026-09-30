import numpy as np
import pandas as pd
import pytest

from sportsmodel.context.results_power import (
    ResultsParams, center, chain, played_games, predict_errors, season_walk,
    strength_of_victory, venue_records)

P0 = ResultsParams(cap=None, hfa=0.0, beta=0.0, sigma=14.0, rho=0.5, k=1.0)


def _g(rows, season=2024):
    return played_games(pd.DataFrame([
        dict(season=season, week=w, home_team=h, away_team=a, home_score=hs, away_score=as_,
             neutral=neu) for w, h, a, hs, as_, neu in rows]))


def test_one_game_is_half_this_season_half_prior():
    g = _g([(1, "A", "B", 24, 14, False)])
    w = season_walk(g, {"A": 2.0, "B": -2.0}, P0)
    r = w.ratings()
    # r_A = (10 + r_B + 2) / 2, r_B = (-10 + r_A - 2) / 2
    ra = (10 + 2 + (-10 - 2) / 2) / (2 - 0.5)
    assert r["A"] == pytest.approx(ra)
    assert r["B"] == pytest.approx((-10 + ra - 2) / 2)
    assert w.games()["A"] == 1


def test_prior_only_before_any_game_and_leak_free():
    rows = [(1, "A", "B", 30, 10, False), (2, "B", "C", 20, 17, False), (3, "C", "A", 10, 13, False)]
    g = _g(rows)
    w = season_walk(g, {"A": 1.0}, P0)
    assert w.ratings(1)["A"] == pytest.approx(1.0) and w.ratings(1)["B"] == 0.0
    later = _g(rows[:2] + [(3, "C", "A", 50, 0, False)])
    w2 = season_walk(later, {"A": 1.0}, P0)
    for wk in (1, 2, 3):
        pd.testing.assert_series_equal(w.ratings(wk), w2.ratings(wk))
    assert not np.allclose(w.final, w2.final)


def test_road_win_counts_more_than_home_win():
    p = ResultsParams(cap=None, hfa=3.0, beta=0.0, sigma=14.0, rho=0.5)
    home = season_walk(_g([(1, "A", "B", 17, 10, False)]), {}, p).ratings()
    road = season_walk(_g([(1, "B", "A", 10, 17, False)]), {}, p).ratings()
    neutral = season_walk(_g([(1, "A", "B", 17, 10, True)]), {}, p).ratings()
    assert road["A"] > neutral["A"] > home["A"]


def test_cap_limits_blowouts():
    p = ResultsParams(cap=21.0, hfa=0.0, beta=0.0, sigma=14.0, rho=0.5)
    a = season_walk(_g([(1, "A", "B", 70, 0, False)]), {}, p).ratings()
    b = season_walk(_g([(1, "A", "B", 21, 0, False)]), {}, p).ratings()
    assert a["A"] == pytest.approx(b["A"])


def test_win_credit_rewards_the_win_and_upsets():
    p = ResultsParams(cap=None, hfa=0.0, beta=10.0, sigma=14.0, rho=0.5)
    win = season_walk(_g([(1, "A", "B", 11, 10, True)]), {}, p).ratings()
    loss = season_walk(_g([(1, "A", "B", 10, 11, True)]), {}, p).ratings()
    assert win["A"] - loss["A"] > 2 * 1.0          # beyond the 2-point margin swing
    # same 1-point win: beating a team expected to win (prior +10) earns more
    upset = season_walk(_g([(1, "A", "B", 11, 10, True)]), {"B": 10.0}, p)
    plain = season_walk(_g([(1, "A", "B", 11, 10, True)]), {"B": 0.0}, p)
    assert upset.pwin[0] < 0.5 < plain.pwin[0] + 1e-9
    assert (upset.ratings()["A"] - upset.ratings()["B"] / 2) > (plain.ratings()["A"] - plain.ratings()["B"] / 2)


def test_strength_of_schedule_is_built_in():
    # A and C both beat B by 7; B's other game: B beats D by 20 -> B stronger than D
    rows = [(1, "A", "B", 17, 10, True), (1, "C", "D", 17, 10, True), (2, "B", "D", 30, 10, True)]
    r = season_walk(_g(rows), {}, P0).ratings()
    assert r["A"] > r["C"]


def test_chain_uses_shrunk_previous_final_and_given_prior_overrides():
    g = pd.concat([_g([(1, "A", "B", 30, 0, True)], 2023), _g([(1, "A", "B", 3, 0, True)], 2024)])
    walks = chain(g, P0, [2023, 2024])
    prev = center(walks[2023].ratings())
    assert walks[2024].ratings(1)["A"] == pytest.approx(0.5 * prev["A"])
    w2 = chain(g, P0, [2023, 2024], preseason={2024: {"A": 9.0}})
    assert w2[2024].ratings(1)["A"] == pytest.approx(9.0)
    assert w2[2024].ratings(1)["B"] == pytest.approx(0.5 * prev["B"])


def test_predict_errors_uses_ratings_entering_the_week():
    g = _g([(1, "A", "B", 20, 10, False), (2, "A", "B", 30, 10, False)])
    p = ResultsParams(cap=None, hfa=2.0, beta=0.0, sigma=14.0, rho=0.5)
    walks = {2024: season_walk(g, {}, p)}
    e = predict_errors(g, walks, p, [2024])
    assert e.loc[e["week"] == 1, "pred"].iloc[0] == pytest.approx(2.0)
    r2 = walks[2024].ratings(2)
    assert e.loc[e["week"] == 2, "pred"].iloc[0] == pytest.approx(r2["A"] - r2["B"] + 2.0)


def test_sov_and_venue_records():
    log = pd.DataFrame([
        dict(team="A", opponent="B", su="W", venue="home"),
        dict(team="A", opponent="C", su="W", venue="away"),
        dict(team="A", opponent="D", su="L", venue="neutral"),
        dict(team="B", opponent="A", su="L", venue="away")])
    r = pd.Series({"A": 5.0, "B": 2.0, "C": -4.0, "D": 1.0})
    sov = strength_of_victory(log, r)
    assert sov["A"] == pytest.approx(-1.0) and "B" not in sov.index
    rec = venue_records(log)
    assert rec.loc["A", "home_record"] == "1-0" and rec.loc["A", "road_record"] == "1-0"
    assert rec.loc["B", "road_record"] == "0-1" and rec.loc["B", "home_record"] == "0-0"


def test_power_asof_current_and_previous_week_centered():
    from sportsmodel.context.results_power import power_asof
    rows23 = [(1, "A", "B", 30, 0, True), (2, "C", "A", 0, 10, True)]
    rows24 = [(1, "A", "B", 21, 14, False), (2, "B", "C", 24, 10, False), (3, "C", "A", 7, 3, False)]
    g = pd.concat([_g(rows23, 2023), _g(rows24, 2024)])
    cur, prev = power_asof(g, P0, 2024, 4)
    assert set(cur["week"]) == {4} and set(prev["week"]) == {3}
    assert cur["rating"].mean() == pytest.approx(0.0)
    walks = chain(g, P0, [2023])
    prior = {t: 0.5 * x for t, x in center(walks[2023].ratings()).items()}
    full = season_walk(g[g["season"] == 2024], prior, P0)
    exp = center(full.ratings(None))
    assert cur.set_index("team")["rating"].to_dict() == pytest.approx(exp.to_dict())
    assert cur.set_index("team").loc["A", "games"] == 2
    fbs_only, _ = power_asof(g, P0, 2024, 4, members={"A", "B"})
    f = fbs_only.set_index("team")
    assert f.loc[["A", "B"], "rating"].mean() == pytest.approx(0.0)
    assert not f.loc["C", "is_fbs"] and f.loc["A", "is_fbs"]
