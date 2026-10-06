"""v3 feature table: v2 parity, leak invariant, FCS skip, live frame. Tiny synthetic league."""
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import context as cx, efficiency as eff, v3, v3_table as vt
from sportsmodel.cfb.priors import DecayConfig, PriorWeights
from sportsmodel.cfb.walkforward import raw_model_predictions
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig

ELO = EloConfig(k=40, hfa_elo=70, carryover=0.9, base=1500.0)
BLEND = BlendConfig(w_sos=0.45, srs_min_games=3)
TEAMS = [str(i) for i in range(1, 9)]


def world(seasons=(2021, 2022, 2023), weeks=5, seed=5, with_fcs=True):
    """(schedule frame, eff_games) over a tiny league; game_pk == eff game_id."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 8, len(TEAMS))
    sched, effrows, pk = [], [], 1000
    for s in seasons:
        for w in range(1, weeks + 1):
            order = list(rng.permutation(len(TEAMS)))
            for i in range(0, len(order), 2):
                hi, ai = order[i], order[i + 1]
                h, a = TEAMS[hi], TEAMS[ai]
                margin = strength[hi] - strength[ai] + 2.5 + rng.normal(0, 10)
                hs = int(max(0, round(28 + margin / 2 + rng.normal(0, 4))))
                as_ = int(max(0, round(28 - margin / 2 + rng.normal(0, 4))))
                pk += 1
                sched.append({"season": s, "week": w, "home_team": h, "away_team": a, "home_score": hs,
                              "away_score": as_, "game_type": "REG", "game_pk": pk,
                              "start_date": f"{s}-09-{w * 7 - 3:02d}T17:00Z", "neutral_site": bool(i == 2),
                              "market_spread": float(round(margin)) + 0.5, "market_total": 55.5})
                for team, opp, sgn in ((hi, ai, 1), (ai, hi, -1)):
                    y = 0.1 + 0.01 * (strength[team] - strength[opp]) + rng.normal(0, 0.03)
                    effrows.append({"season": s, "game_id": pk, "team": TEAMS[team], "opponent": TEAMS[opp],
                                    "season_type": "regular", "week": float(w), "home": float(sgn),
                                    "plays": 70.0, "rush_plays": 35.0, "pass_plays": 35.0,
                                    **{f"y_{m}": y for m in eff.METRICS}, **{f"w_{m}": 60.0 for m in eff.METRICS}})
        if with_fcs:                                           # an FCS game every season, week 1
            pk += 1
            sched.append({"season": s, "week": 1, "home_team": "1", "away_team": "FCS", "home_score": 45,
                          "away_score": 7, "game_type": "REG", "game_pk": pk, "start_date": f"{s}-09-01T17:00Z",
                          "neutral_site": False, "market_spread": np.nan, "market_total": np.nan})
    return pd.DataFrame(sched), pd.DataFrame(effrows)


def ctx_assets(sched):
    """A small but non-trivial context: every team has its own venue (three time zones), neutral games
    are played at a far-away ninth venue, each game has a forecast, and every team-season has a talent."""
    zones = [("America/New_York", 40.7, -74.0), ("America/Chicago", 41.9, -87.6),
             ("America/Los_Angeles", 34.0, -118.2)]
    venues = pd.DataFrame({"venue_id": [100 + i for i in range(1, 10)], "name": [f"v{i}" for i in range(1, 10)],
                           "timezone": [zones[i % 3][0] for i in range(9)],
                           "latitude": [zones[i % 3][1] + i for i in range(9)],
                           "longitude": [zones[i % 3][2] for i in range(9)],
                           "elevation": [0.0] * 9, "dome": [0.0] * 8 + [1.0]})
    meta = sched[sched["away_team"] != "FCS"][["game_pk", "season", "home_team", "neutral_site"]].copy()
    meta["game_id"] = meta["game_pk"]
    meta["venue_id"] = [109 if n else 100 + int(h) for h, n in zip(meta["home_team"], meta["neutral_site"])]
    rng = np.random.default_rng(11)
    wx = pd.DataFrame({"game_id": meta["game_id"], "venue_id": meta["venue_id"],
                       "wind_speed": rng.uniform(0, 25, len(meta)), "temperature": rng.uniform(25, 85, len(meta)),
                       "precipitation": rng.choice([0.0, 0.0, 0.3], len(meta)), "game_indoors": False})
    talent = pd.DataFrame([{"season": s, "team": t, "talent": 600.0 + 40.0 * i}
                           for s in sorted(sched["season"].unique()) for i, t in enumerate(TEAMS)])
    return cx.build_context_assets(venues, meta, wx, talent)


def inputs(eff_games, priors_rows=None, ctx=None, weights=None):
    return vt.V3Inputs(ELO, BLEND, weights or PriorWeights(sp_scale=17.5, sp_offset=1500.0), DecayConfig(3.0, 0.35),
                       eff.EffConfig(), eff_games, priors_rows or {}, None, ctx or cx.ContextAssets())


def test_table_margin_and_total_match_the_v2_walk_when_no_prior():
    sched, eg = world()
    table = vt.build_table(sched, inputs(eg))
    raw = pd.DataFrame(raw_model_predictions(sched, ELO, BLEND))
    raw = raw[(raw.home_team != "FCS") & (raw.away_team != "FCS")].set_index(["season", "week", "home_team"])
    got = table.set_index(["season", "week", "home_team"])
    assert len(got) == len(raw)
    assert np.allclose(got["margin_v2"], raw.loc[got.index, "model_margin"])
    assert np.allclose(got["total_v2"], raw.loc[got.index, "model_total"])
    assert not table[["home_team", "away_team"]].isin(["FCS"]).any().any()


def test_prior_margin_zero_without_priors_and_blends_into_v2_with_them():
    sched, eg = world()
    rows = {s: [{"team_espn_id": t, "sp_rating": float(i), "returning_pct": 0.5, "recruiting_points": 100.0,
                 "portal_net": 0.0, "prior_sos": 0.0, "qb_returning": True} for i, t in enumerate(TEAMS)]
            for s in (2020, 2021, 2022, 2023)}
    assert (vt.build_table(sched, inputs(eg)).prior_margin == 0).all()
    with_p = vt.build_table(sched, inputs(eg, rows))
    wk1 = with_p[(with_p.season == 2022) & (with_p.week == 1)]
    assert (wk1.prior_margin != 0).any()                     # week 1: prior carries full weight
    late = with_p[(with_p.season == 2022) & (with_p.week == 5)]
    assert late.prior_margin.abs().mean() < wk1.prior_margin.abs().mean()   # decays toward the floor


def test_appending_a_later_game_never_changes_earlier_rows():
    sched, eg = world()
    ctx = ctx_assets(sched)
    base = vt.build_table(sched, inputs(eg, ctx=ctx))
    assert base[["travel_far_diff", "wind_excess", "talent_gap"]].abs().to_numpy().sum() > 0   # context is live
    extra = sched[(sched.season == 2023) & (sched.week == 5)].copy()
    extra["week"], extra["game_pk"] = 6, extra["game_pk"] + 5000
    extra["home_score"], extra["away_score"] = 70, 0
    eg_extra = eg[(eg.season == 2023) & (eg.week == 5)].copy()
    eg_extra["week"], eg_extra["game_id"], eg_extra["y_ppa"] = 6.0, eg_extra["game_id"] + 5000, 9.0
    later = vt.build_table(pd.concat([sched, extra], ignore_index=True),
                           inputs(pd.concat([eg, eg_extra], ignore_index=True), ctx=ctx))
    keep = later[later.game_pk.isin(base.game_pk)].sort_values("game_pk").reset_index(drop=True)
    pd.testing.assert_frame_equal(base.sort_values("game_pk").reset_index(drop=True), keep[base.columns],
                                  check_dtype=False)


def test_perturbing_later_weeks_results_and_efficiency_never_changes_earlier_rows():
    sched, eg = world()
    ctx = ctx_assets(sched)
    base = vt.build_table(sched, inputs(eg, ctx=ctx))
    late = (sched.season == 2023) & (sched.week >= 4)
    s2 = sched.copy()
    s2.loc[late, "home_score"], s2.loc[late, "away_score"] = 63, 3
    eg2 = eg.copy()
    late_eff = (eg2.season == 2023) & (eg2.week >= 4)
    for m in eff.METRICS:
        eg2.loc[late_eff, f"y_{m}"] = 7.0
    got = vt.build_table(s2, inputs(eg2, ctx=ctx))
    feat = [c for c in base.columns if c not in ("actual_margin", "actual_total")]
    # a week-W row reads only games before W: weeks 1-4 of 2023 (week 4 rows see weeks 1-3) are untouched
    cut = lambda t: t[(t.season < 2023) | (t.week <= 4)].sort_values("game_pk").reset_index(drop=True)
    pd.testing.assert_frame_equal(cut(base)[feat], cut(got)[feat], check_dtype=False)
    assert not cut(base)[["actual_margin"]].equals(cut(got)[["actual_margin"]])   # the perturbation did bite
    after = lambda t: t[(t.season == 2023) & (t.week == 5)].sort_values("game_pk")
    assert not np.allclose(after(base)["margin_v2"], after(got)["margin_v2"])      # ...and reaches week 5


def test_margin_v2_with_priors_equals_the_independent_prior_seeded_v2_margin():
    """With live priors the table's margin_v2 is the backtest's prior_seeded_margin (an independent
    implementation of the same blend), game for game."""
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backtest_cfb_priors.py"
    spec = importlib.util.spec_from_file_location("backtest_cfb_priors", p)
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    sched, eg = world()
    rng = np.random.default_rng(3)
    rows = {s: [{"team_espn_id": t, "sp_rating": float(rng.normal(0, 8)), "returning_pct": float(rng.random()),
                 "recruiting_points": float(rng.uniform(50, 300)), "portal_net": float(rng.normal(0, 3)),
                 "prior_sos": float(rng.normal(0, 1)), "qb_returning": bool(rng.random() > 0.5)}
                for t in TEAMS] for s in (2020, 2021, 2022, 2023)}
    pw = PriorWeights(sp_scale=17.5, sp_offset=1500.0, w_returning=25.0, w_recruiting=15.0, w_portal=-10.0,
                      w_qb=30.0, w_sos_prior=12.0)
    decay = DecayConfig(3.0, 0.35)
    table = vt.build_table(sched, inputs(eg, rows, weights=pw))
    assert (table.prior_margin != 0).any()
    r_pre = bt.r_pre_table({s: bt.season_prior_inputs(rows, s) for s in (2021, 2022, 2023)}, pw)
    ref = {(r["season"], r["week"], r["home_team"]): bt.prior_seeded_margin(r, r_pre, decay, ELO, BLEND)
           for r in bt.raw_walk_forward(sched, ELO, BLEND) if not r["fcs"]}
    got = table.set_index(["season", "week", "home_team"])["margin_v2"]
    assert len(got) == len(ref)
    assert np.allclose(got.to_numpy(), [ref[k] for k in got.index])
    plain = vt.build_table(sched, inputs(eg))
    assert not np.allclose(plain["margin_v2"], table["margin_v2"])             # the priors really moved it


def test_predict_v3_identity_weights_reproduce_the_v2_margin_and_total():
    sched, eg = world()
    t = vt.build_table(sched, inputs(eg, ctx=ctx_assets(sched)))
    zero_m = {k: 0.0 for k in v3.ALL_MARGIN_FEATURES}
    zero_t = {k: 0.0 for k in v3.ALL_TOTAL_FEATURES}
    w = v3.V3Weights(v3.LinearBlend(25.0, {"ppa_plays": 3.0}),
                     v3.LinearBlend(0.0, {**zero_m, "margin_v2": 1.0}),      # a1 = 1, everything else 0
                     v3.LinearBlend(0.0, {**zero_t, "total_v2": 1.0}))       # b2 = 1, everything else 0
    pred = np.array([v3.predict_v3(r, w) for r in t.to_dict("records")])
    assert np.allclose(pred[:, 0], t["margin_v2"]) and np.allclose(pred[:, 1], t["total_v2"])


def test_features_are_finite_and_columns_complete():
    sched, eg = world()
    t = vt.build_table(sched, inputs(eg))
    assert list(t.columns) == vt.TABLE_COLUMNS
    feat = [c for c in vt.TABLE_COLUMNS if c not in ("market_spread", "market_total", "start_date",
                                                      "home_team", "away_team", "neutral")]
    assert np.isfinite(t[feat].to_numpy(float)).all()
    assert set(t["non_neutral"]) <= {0.0, 1.0} and (t["neutral"] == (t["non_neutral"] == 0.0)).all()


def test_unscored_games_emit_only_when_asked_and_have_nan_actuals():
    sched, eg = world()
    sched.loc[sched.index[-3:], ["home_score", "away_score"]] = np.nan
    assert vt.build_table(sched, inputs(eg)).actual_margin.notna().all()
    live = vt.build_table(sched, inputs(eg), include_unscored=True)
    assert live.actual_margin.isna().sum() >= 2 and live.actual_total.isna().sum() >= 2


def test_live_frame_has_only_prior_games_plus_unscored_slate():
    sched, _ = world()
    slate = [{"game_pk": 9001, "home_team": "1", "away_team": "2", "neutral_site": False,
              "start_date": "2023-10-07T17:00Z"}]
    f = vt.live_frame(sched, 2023, 4, slate)
    assert (f[f.game_pk != 9001][["season", "week"]].apply(tuple, axis=1)
            .map(lambda sw: sw < (2023, 4))).all()
    up = f[f.game_pk == 9001].iloc[0]
    assert up.week == 4 and pd.isna(up.home_score) and up.start_date == "2023-10-07T17:00Z"
    assert (f.game_pk != 9001).sum() == ((sched.season < 2023) | ((sched.season == 2023) & (sched.week < 4))).sum()
