"""Efficiency ratings: ridge recovery on a synthetic league, NaN handling, prior blend, leak invariant."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import efficiency as eff
from sportsmodel.cfb.priors import DecayConfig, prior_weight


def synthetic_obs(n_teams=16, n_games=3000, noise=0.02, seed=0, hfa=0.05, mu=0.1):
    rng = np.random.default_rng(seed)
    off, dfn = rng.normal(0, 0.2, n_teams), rng.normal(0, 0.2, n_teams)
    rows = []
    for _ in range(n_games):
        a, b = rng.choice(n_teams, 2, replace=False)
        h = int(rng.choice([-1, 0, 1]))
        rows.append((str(a), str(b), h, mu + off[a] - dfn[b] + hfa * h + rng.normal(0, noise), 60.0))
    return pd.DataFrame(rows, columns=["team", "opponent", "home", "y", "w"]), off, dfn


def test_ridge_recovers_known_offsets_and_home_edge():
    obs, off, dfn = synthetic_obs()
    r = eff.ridge_adjust(obs, ridge=0.5)
    t = len(off)
    assert np.corrcoef([r.off[str(i)] for i in range(t)], off)[0, 1] > 0.99
    assert np.corrcoef([r.deff[str(i)] for i in range(t)], dfn)[0, 1] > 0.99
    assert r.hfa == pytest.approx(0.05, abs=0.01)
    assert r.mu == pytest.approx(obs["y"].mean(), abs=1e-9)         # equal weights -> plain mean


def test_bigger_ridge_shrinks_ratings_toward_zero():
    obs, *_ = synthetic_obs(n_games=200)
    lo, hi = eff.ridge_adjust(obs, 0.5), eff.ridge_adjust(obs, 50.0)
    assert np.mean(np.abs(list(hi.off.values()))) < np.mean(np.abs(list(lo.off.values())))


def test_nan_rows_are_dropped_not_zeroed_and_empty_is_nan_mu():
    obs, *_ = synthetic_obs(n_games=300)
    base = eff.ridge_adjust(obs, 2.0)
    junk = pd.DataFrame([("0", "1", 1, np.nan, 60.0), ("0", "1", 1, 0.5, np.nan), ("0", "1", 1, 0.5, 0.0)],
                        columns=obs.columns)
    got = eff.ridge_adjust(pd.concat([obs, junk], ignore_index=True), 2.0)
    assert got.off == base.off and got.mu == base.mu
    empty = eff.ridge_adjust(obs.iloc[0:0], 2.0)
    assert math.isnan(empty.mu) and empty.off == {}


def test_weights_pull_the_fit_toward_heavy_observations():
    obs = pd.DataFrame([("a", "b", 0, 1.0, 100.0), ("a", "b", 0, 0.0, 1.0)],
                       columns=["team", "opponent", "home", "y", "w"])
    r = eff.ridge_adjust(obs, 0.01)
    assert r.mu > 0.95                                                # heavy row dominates the league mean


# ----------------------------------------------------------- eff_games frames --

def league_games(seasons=(2022, 2023), weeks=4, n_teams=8, seed=3):
    """team-game rows (one per team per game) on a tiny league with real column names."""
    rng = np.random.default_rng(seed)
    off = {s: rng.normal(0, 0.2, n_teams) for s in seasons}
    rows, gid = [], 1000
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(n_teams)
            for i in range(0, n_teams, 2):
                h, a = int(order[i]), int(order[i + 1])
                gid += 1
                for team, opp, hs in ((h, a, 1), (a, h, -1)):
                    y = 0.1 + off[s][team] - 0.5 * off[s][opp] + 0.03 * hs
                    rows.append({"season": s, "game_id": gid, "team": str(team), "opponent": str(opp),
                                 "season_type": "regular", "week": float(w), "home": float(hs),
                                 "plays": 70.0, "rush_plays": 35.0, "pass_plays": 35.0,
                                 **{f"y_{m}": y for m in eff.METRICS}, **{f"w_{m}": 60.0 for m in eff.METRICS}})
    return pd.DataFrame(rows)


def test_state_before_uses_only_earlier_weeks_and_ignores_postseason():
    g = league_games()
    cfg = eff.EffConfig()
    base = eff.state_before(g, 2023, 3, None, cfg)
    assert set(base.games.values()) == {2}                            # weeks 1-2 only
    later = g[(g.season == 2023) & (g.week == 3)].copy()
    later["y_ppa"] += 5.0                                             # week 3 data must not matter
    post = g[(g.season == 2023) & (g.week == 1)].copy()
    post["season_type"], post["week"], post["y_ppa"] = "postseason", np.nan, 9.0
    for extra in (later, post):
        got = eff.state_before(pd.concat([g, extra], ignore_index=True), 2023, 3, None, cfg)
        assert got.ratings["ppa"].off == base.ratings["ppa"].off
        assert got.games == base.games


def test_state_before_empty_week_one_is_prior_only():
    g = league_games()
    cfg = eff.EffConfig(k0=0.5)
    final = eff.raw_state(g[g.season == 2022], cfg.ridge)
    prior = eff.build_prior(final, {t: 0.5 for t in final.games})
    st = eff.state_before(g, 2023, 1, prior, cfg)
    t = "0"
    assert st.ratings["ppa"].off[t] == pytest.approx(0.5 * final.ratings["ppa"].off[t])
    assert st.games == {}


def test_blend_weight_follows_prior_weight_curve():
    cfg = eff.EffConfig(half_life_games=2.0, prior_floor=0.1)
    prior = eff.EffState({m: eff.MetricRatings({"a": 1.0}, {"a": 0.0}, 0.0, 0.0) for m in eff.METRICS},
                         {}, {"a": 80.0}, {"a": 0.4}, 70.0, 0.5)
    cur = eff.EffState({m: eff.MetricRatings({"a": 0.0}, {"a": 0.0}, 0.0, 0.0) for m in eff.METRICS},
                       {"a": 4}, {"a": 60.0}, {"a": 0.6}, 70.0, 0.5)
    w = prior_weight(4, DecayConfig(2.0, 0.1))
    out = eff.blend_state(cur, prior, cfg)
    assert out.ratings["ppa"].off["a"] == pytest.approx(w * 1.0)
    assert out.pace["a"] == pytest.approx(w * 80.0 + (1 - w) * 60.0)
    assert out.rush_share["a"] == pytest.approx(w * 0.4 + (1 - w) * 0.6)
    assert eff.blend_state(cur, None, cfg) is cur


def test_retention_is_clipped_and_scaled_by_returning_and_talent():
    cfg = eff.EffConfig(k0=0.6, k_ret=0.2, k_tal=0.1)
    k = eff.retention_for(["a", "b", "c"], {"a": 3.0, "b": -5.0}, {"a": 1.0}, cfg)
    assert k["a"] == 1.0 and k["b"] == 0.0 and k["c"] == pytest.approx(0.6)


def test_season_prior_uses_previous_season_final_and_z_scored_inputs():
    g = league_games()
    cfg = eff.EffConfig(k0=0.5, k_ret=0.1, k_tal=0.0)
    rows = [{"team_espn_id": str(t), "returning_pct": 0.2 + 0.1 * t, "recruiting_points": None,
             "portal_net": None, "prior_sos": None} for t in range(8)]
    prior = eff.season_prior(g, 2023, rows, None, cfg)
    final = eff.raw_state(g[g.season == 2022], cfg.ridge)
    vals = np.array([0.2 + 0.1 * t for t in range(8)])
    z7 = (vals[7] - vals.mean()) / vals.std()
    assert prior.ratings["ppa"].off["7"] == pytest.approx(
        min(1.0, max(0.0, 0.5 + 0.1 * z7)) * final.ratings["ppa"].off["7"])
    assert eff.season_prior(g, 2022, rows, None, cfg) is None          # no 2021 data -> no prior


# -------------------------------------------------------- matchup features --

def test_side_features_are_deviations_and_missing_metrics_are_neutral():
    r = eff.MetricRatings({"x": 0.3}, {"y": 0.1}, 0.1, 0.02)
    st = eff.EffState({"rush_ppa": r, "pass_ppa": r, "success": r}, {"x": 5, "y": 5},
                      {"x": 80.0, "y": 60.0}, {"x": 0.25}, 70.0, 0.5)
    f = eff.side_features(st, "x", "y", 1.0)
    assert f["success"] == pytest.approx(0.3 - 0.1 + 0.02)
    plays = 70.0 + 0.5 * ((80 - 70) + (60 - 70))
    assert f["plays_dev"] == pytest.approx(plays - 70.0)
    assert f["ppa_plays"] == pytest.approx((0.3 - 0.1 + 0.02) * plays)
    assert f["havoc"] == 0.0 and f["ppo"] == 0.0 and f["explosiveness"] == 0.0   # absent metrics: 0, not NaN
    assert eff.side_features(st, "x", "y", -1.0)["success"] == pytest.approx(0.3 - 0.1 - 0.02)


# ----------------------------------------------------------------- the frame --

def test_build_eff_games_joins_home_week_and_nan_for_missing_metrics():
    adv = pd.DataFrame({
        "season": [2023] * 4, "week": [1, 1, 1, 1], "season_type": ["regular", "regular", "postseason", "regular"],
        "game_id": [1, 1, 2, 3], "team": ["a", "b", "a", "a"], "opponent": ["b", "a", "b", "b"],
        "off_plays": [70.0, 60.0, 65.0, 50.0], "off_ppa": [0.2, 0.1, 0.3, 0.0],
        "off_success": [0.5, 0.4, 0.5, 0.4], "off_explosiveness": [1.2, 1.1, 1.3, 1.0],
        "off_pass_ppa": [0.3, 0.2, 0.3, 0.0], "off_rush_ppa": [0.1, 0.0, 0.2, 0.0],
        "off_pass_plays": [40.0, np.nan, 30.0, 20.0], "off_rush_plays": [30.0, 30.0, 35.0, 30.0]})
    havoc = pd.DataFrame({"season": [2023], "game_id": [1], "team": ["a"], "off_havoc": [0.11], "off_plays": [72.0]})
    meta = pd.DataFrame({"game_id": [1, 2], "home_team": ["a", "b"], "neutral_site": [False, True]})
    sched = pd.DataFrame({"game_pk": [1, 3], "week": [2, 5], "game_type": ["REG", "REG"]})
    g = eff.build_eff_games(adv, havoc, None, meta, sched)
    assert len(g) == 3                                               # game 3 unknown to meta -> dropped
    r_a, r_b, r_post = g.iloc[0], g.iloc[1], g.iloc[2]
    assert (r_a["home"], r_b["home"], r_post["home"]) == (1.0, -1.0, 0.0)   # neutral bowl -> 0
    assert r_a["week"] == 2.0 and math.isnan(r_post["week"])         # schedule week; postseason NaN
    assert r_a["y_havoc"] == pytest.approx(1.0) and r_a["w_havoc"] == 72.0   # H1: ratio to its week's league mean
    assert math.isnan(r_b["y_havoc"]) and r_b["w_havoc"] == 60.0     # no havoc row: NaN y, plays as weight
    assert math.isnan(r_a["y_ppo"])                                   # no drives asset: NaN
    assert r_b["w_pass_ppa"] == 30.0                                  # missing pass count -> half of plays
    assert r_a["w_pass_ppa"] == 40.0


def test_load_eff_config_roundtrip_and_default(tmp_path):
    assert eff.load_eff_config(tmp_path / "missing.json") == eff.EffConfig()
    p = tmp_path / "eff_config.json"
    p.write_text(json.dumps({"ridge": 8.0, "half_life_games": 2.0, "prior_floor": 0.1,
                             "k0": 0.7, "k_ret": 0.05, "k_tal": 0.0}))
    assert eff.load_eff_config(p) == eff.EffConfig(8.0, 2.0, 0.1, 0.7, 0.05, 0.0)


# ------------------------------------------------- ruling H1: relative havoc --

def _havoc_league(scale_by_week, rates=(0.10, 0.20, 0.15, 0.25), plays=(60.0, 80.0, 70.0, 50.0)):
    """4 teams, one game per (season 2019, week) among a-b / c-d; havoc rate scaled per week."""
    adv_rows, hv_rows, gid = [], [], 0
    for wk, sc in scale_by_week.items():
        for (t, o, i) in (("a", "b", 0), ("b", "a", 1), ("c", "d", 2), ("d", "c", 3)):
            g = gid + (0 if t in "ab" else 1) + 100 * wk
            adv_rows.append({"season": 2019, "week": wk, "season_type": "regular", "game_id": g, "team": t,
                             "opponent": o, "off_plays": plays[i], "off_ppa": 0.1, "off_success": 0.4,
                             "off_explosiveness": 1.1, "off_pass_ppa": 0.1, "off_rush_ppa": 0.1,
                             "off_pass_plays": 30.0, "off_rush_plays": 30.0})
            hv_rows.append({"season": 2019, "game_id": g, "team": t, "off_havoc": rates[i] * sc,
                            "off_plays": plays[i]})
    meta = pd.DataFrame({"game_id": sorted({r["game_id"] for r in adv_rows}),
                         "home_team": [("a" if g % 100 == 0 else "c") for g in sorted({r["game_id"] for r in adv_rows})],
                         "neutral_site": False})
    sched = pd.DataFrame({"game_pk": meta["game_id"], "week": meta["game_id"] // 100, "game_type": "REG"})
    return pd.DataFrame(adv_rows), pd.DataFrame(hv_rows), meta, sched


def test_havoc_is_expressed_relative_to_the_weeks_play_weighted_league_mean():
    adv, hv, meta, sched = _havoc_league({1: 1.0})
    g = eff.build_eff_games(adv, hv, None, meta, sched)
    mean = np.average([0.10, 0.20, 0.15, 0.25], weights=[60.0, 80.0, 70.0, 50.0])
    got = g.sort_values("team").set_index("team")
    assert got.loc["a", "y_havoc"] == pytest.approx(0.10 / mean)
    assert got.loc["b", "y_havoc"] == pytest.approx(0.20 / mean)
    assert np.average(g["y_havoc"], weights=g["w_havoc"]) == pytest.approx(1.0)


def test_constant_factor_shift_in_a_weeks_raw_havoc_gives_identical_relative_ratings():
    ratings = {}
    for scale in (1.0, 1.45):                                       # the 2019 weeks 3-9 shift
        adv, hv, meta, sched = _havoc_league({1: scale})
        g = eff.build_eff_games(adv, hv, None, meta, sched)
        ratings[scale] = eff.ridge_adjust(eff.metric_obs(g, "havoc"), 1.0)
    lo, hi = ratings[1.0], ratings[1.45]
    for t in lo.off:
        assert hi.off[t] == pytest.approx(lo.off[t]) and hi.deff[t] == pytest.approx(lo.deff[t])
    assert hi.mu == pytest.approx(lo.mu)
    # mixing shifted and unshifted weeks: per-week normalisation, so y matches week by week
    adv, hv, meta, sched = _havoc_league({1: 1.0, 2: 1.45})
    g = eff.build_eff_games(adv, hv, None, meta, sched)
    w1, w2 = g[g.week == 1].sort_values("team"), g[g.week == 2].sort_values("team")
    assert w1["y_havoc"].to_numpy() == pytest.approx(w2["y_havoc"].to_numpy())


def test_relative_havoc_is_nan_without_havoc_data_and_other_metrics_are_untouched():
    adv, hv, meta, sched = _havoc_league({1: 1.0})
    none = eff.build_eff_games(adv, None, None, meta, sched)
    assert none["y_havoc"].isna().all()
    g = eff.build_eff_games(adv, hv, None, meta, sched)
    assert (g["y_ppa"] == 0.1).all() and (g["y_success"] == 0.4).all()


# ------------------------------------------------------ review fix round 1 --

def _hfa_states(cur_hfa, prior_hfa, games):
    mk = lambda hfa: {m: eff.MetricRatings({"a": 0.0, "b": 0.0}, {"a": 0.0, "b": 0.0}, 0.1, hfa)   # noqa: E731
                      for m in eff.METRICS}
    prior = eff.EffState(mk(prior_hfa), {}, {}, {}, 70.0, 0.5)
    cur = eff.EffState(mk(cur_hfa), games, {}, {}, 70.0, 0.5)
    return cur, prior


def test_blend_state_blends_home_field_with_the_mean_prior_weight():
    cfg = eff.EffConfig(half_life_games=2.0, prior_floor=0.0)
    cur, prior = _hfa_states(0.08, 0.02, {"a": 2, "b": 4})
    w_bar = (prior_weight(2, cfg.decay) + prior_weight(4, cfg.decay)) / 2
    out = eff.blend_state(cur, prior, cfg)
    assert out.ratings["ppa"].hfa == pytest.approx(w_bar * 0.02 + (1 - w_bar) * 0.08)
    assert 0.02 < out.ratings["ppa"].hfa < 0.08


def test_blend_state_hfa_fallbacks_no_prior_and_no_current_games():
    cfg = eff.EffConfig()
    cur, prior = _hfa_states(0.08, 0.02, {"a": 3, "b": 3})
    assert eff.blend_state(cur, None, cfg).ratings["ppa"].hfa == 0.08
    empty_cur = eff.raw_state(league_games().iloc[0:0], cfg.ridge)
    assert eff.blend_state(empty_cur, prior, cfg).ratings["ppa"].hfa == 0.02


def test_season_prior_only_reads_priors_rows_of_the_target_season():
    g = league_games()
    cfg = eff.EffConfig(k0=0.5, k_ret=0.1, k_tal=0.0)
    mk = lambda season, f: [{"season": season, "team_espn_id": str(t), "returning_pct": f(t),   # noqa: E731
                             "recruiting_points": None, "portal_net": None, "prior_sos": None}
                            for t in range(8)]
    target = mk(2023, lambda t: 0.2 + 0.1 * t)
    other = mk(2022, lambda t: 0.9 - 0.1 * t)           # reversed pattern; must not overwrite by team id
    clean = eff.season_prior(g, 2023, target, None, cfg)
    mixed = eff.season_prior(g, 2023, other + target, None, cfg)
    mixed_rev = eff.season_prior(g, 2023, target + other, None, cfg)
    for st in (mixed, mixed_rev):
        assert st.ratings["ppa"].off == clean.ratings["ppa"].off
    assert eff.season_prior(g, 2023, other, None, cfg).ratings["ppa"].off != clean.ratings["ppa"].off


def test_season_prior_ignores_current_and_future_season_games():
    g = league_games()
    cfg = eff.EffConfig(k0=0.5)
    full = eff.season_prior(g, 2023, [], None, cfg)
    past = eff.season_prior(g[g.season < 2023], 2023, [], None, cfg)
    assert full.ratings["ppa"].off == past.ratings["ppa"].off
    assert full.pace == past.pace and full.rush_share == past.rush_share
    assert full.lg_plays == past.lg_plays
