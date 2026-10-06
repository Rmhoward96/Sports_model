"""v3 feature table: v2 parity, leak invariant, FCS skip, live frame. Tiny synthetic league."""
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import context as cx, efficiency as eff, v3_table as vt
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


def inputs(eff_games, priors_rows=None):
    return vt.V3Inputs(ELO, BLEND, PriorWeights(sp_scale=17.5, sp_offset=1500.0), DecayConfig(3.0, 0.35),
                       eff.EffConfig(), eff_games, priors_rows or {}, None, cx.ContextAssets())


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
    base = vt.build_table(sched, inputs(eg))
    extra = sched[(sched.season == 2023) & (sched.week == 5)].copy()
    extra["week"], extra["game_pk"] = 6, extra["game_pk"] + 5000
    extra["home_score"], extra["away_score"] = 70, 0
    eg_extra = eg[(eg.season == 2023) & (eg.week == 5)].copy()
    eg_extra["week"], eg_extra["game_id"], eg_extra["y_ppa"] = 6.0, eg_extra["game_id"] + 5000, 9.0
    later = vt.build_table(pd.concat([sched, extra], ignore_index=True),
                           inputs(pd.concat([eg, eg_extra], ignore_index=True)))
    keep = later[later.game_pk.isin(base.game_pk)].sort_values("game_pk").reset_index(drop=True)
    pd.testing.assert_frame_equal(base.sort_values("game_pk").reset_index(drop=True), keep[base.columns],
                                  check_dtype=False)


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
