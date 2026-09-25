"""Tests for scripts/fit_props_ml_final.py (props-ML Props-2 final fit, model
artifacts and the weekly quick gate). Synthetic tables, stubbed backtest,
tiny real fits for the artifact round trip -- no network."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel.model import props_eval
from sportsmodel.model.props_eval import pit_pmf, pit_uniform, rps_pmf
from sportsmodel.model.props_ml.dist_models import ROLE_SUBSETS, fit_market, predict_pmfs
from sportsmodel.model.props_ml.pit_calibration import IDENTITY_KNOTS, IDENTITY_PLATT
from sportsmodel.sim.nfl import learned

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fit_props_ml_final.py"
_spec = importlib.util.spec_from_file_location("fit_props_ml_final", _p)
fpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpf)

MARKETS = ("pass_yds", "rush_yds", "rec_yds", "receptions", "rush_att", "pass_tds", "anytime_td")
KEPT = ["volume", "efficiency", "context", "market"]
A_GATE = {"kept": sorted(KEPT),
          "tuned": {"2021": [1.0, 300], "2024": [1.0, 150], "2025": [0.8, 300]}}
KMAX = 8
MEANS = {"pass_yds": 220.0, "rush_yds": 45.0, "rec_yds": 45.0, "receptions": 4.0,
         "rush_att": 9.0, "pass_tds": 1.6, "anytime_td": 0.4}
IDENT = {"git_head": "deadbeef", "player_features": {"size": 1, "sha256": "a" * 64},
         "team_features": {"size": 2, "sha256": "b" * 64}}


def _pipeline(w=1.0, calibrate=False, baseline=("pass_tds",)):
    return {"kept_a_toggles": list(KEPT),
            "tuned": {"2024": [1.0, 150], "2025": [0.8, 300]},
            "markets": {m: {"source": "baseline" if m in baseline else "ml",
                            "w_final": float(w[m] if isinstance(w, dict) else w),
                            "calibrate": bool(calibrate)} for m in MARKETS}}


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    real = props_eval.cluster_bootstrap
    monkeypatch.setattr(props_eval, "cluster_bootstrap",
                        lambda df, stat, n_boot=1000, seed=0: real(df, stat, n_boot=40, seed=seed))


# ---- trained_through / holdout weeks ----------------------------------------------------

def _label_tbl():
    nan = float("nan")
    return pd.DataFrame({
        "season":  [2025, 2025, 2025, 2026, 2026, 2026, 2026],
        "week":    [17, 18, 18, 1, 2, 2, 3],
        "player_id": ["a", "a", "b", "a", "a", "b", "a"],
        "is_stub": [False, False, True, False, True, False, False],
        "y_targets": [3.0, 1.0, nan, 2.0, nan, nan, nan],   # (2026, 2): stub + unplayed
        "y_rec_yds": [30.0, 5.0, nan, nan, nan, nan, nan],  # (2026, 3): future row
    })


def test_trained_through_is_last_week_with_played_labels():
    assert fpf.trained_through(_label_tbl()) == (2026, 1)
    assert fpf.labelled_weeks(_label_tbl()) == [(2025, 17), (2025, 18), (2026, 1)]
    assert fpf.next_week((2026, 1)) == (2026, 2)
    with pytest.raises(ValueError, match="no played"):
        fpf.trained_through(_label_tbl().assign(y_targets=float("nan"), y_rec_yds=float("nan")))


def test_holdout_weeks_are_the_last_n_completed_weeks():
    assert fpf.holdout_weeks(_label_tbl(), 2) == [(2025, 18), (2026, 1)]
    assert fpf.holdout_weeks(_label_tbl(), 1) == [(2026, 1)]
    with pytest.raises(ValueError):
        fpf.holdout_weeks(_label_tbl(), 0)
    with pytest.raises(ValueError, match="train"):
        fpf.holdout_weeks(_label_tbl(), 3)  # nothing left to train on


def test_a_fit_params_use_the_most_recent_tuned_season_and_kept_must_match():
    assert fpf.a_fit_params(A_GATE) == {"season": 2025, "decay": 0.8, "max_iter": 300}
    assert fpf.kept_toggles(_pipeline(), A_GATE) == frozenset(KEPT)
    with pytest.raises(ValueError, match="kept"):
        fpf.kept_toggles({**_pipeline(), "kept_a_toggles": ["volume"]}, A_GATE)
    with pytest.raises(ValueError, match="kept"):
        fpf.kept_toggles({**_pipeline(), "kept_a_toggles": []}, {**A_GATE, "kept": []})


# ---- quick-gate decision ------------------------------------------------------------------

def _pm(**ratios):
    return {m: {"n": 10, "rps_b": 2.0, "rps_c": 2.0 * r, "ece_b": 0.01, "ece_c": 0.01}
            for m, r in ratios.items()}


def test_quick_gate_decision_passes_on_nonnegative_skill_and_no_market_over_1_05():
    assert fpf.QUICK_MAX_RPS_RATIO == 1.05 and fpf.QUICK_MIN_SKILL == 0.0
    assert fpf.quick_gate_decision(_pm(rec_yds=0.98, pass_yds=1.049), 0.001) == \
        {"pass": True, "reasons": []}
    assert fpf.quick_gate_decision(_pm(rec_yds=1.0), 0.0)["pass"] is True  # skill >= 0


def test_quick_gate_decision_fails_with_market_named_reasons():
    d = fpf.quick_gate_decision(_pm(rec_yds=1.06, pass_yds=0.9, rush_att=1.2), 0.02)
    assert d["pass"] is False and len(d["reasons"]) == 2
    assert "rec_yds" in d["reasons"][0] and "1.05" in d["reasons"][0]
    assert "rush_att" in d["reasons"][1]
    d = fpf.quick_gate_decision(_pm(rec_yds=1.0), -0.001)
    assert d["pass"] is False and "skill" in d["reasons"][0]
    d = fpf.quick_gate_decision(_pm(rec_yds=1.0), 0.1, ["run=quick_ml: game coverage differs"])
    assert d == {"pass": False, "reasons": ["run=quick_ml: game coverage differs"]}
    assert fpf.quick_gate_decision({}, 0.0)["pass"] is False


# ---- serving weights / calibration maps from OOF ------------------------------------------

def _oof(seasons=(2024, 2025), weeks=(1, 2, 3, 4), n=3, seed=0, markets=MARKETS):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for m in markets:
                k = 2 if m == "anytime_td" else KMAX + 1
                for i in range(n):
                    a, b = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
                    rows.append({"season": s, "week": w, "home": "KC", "player_id": f"p{i}",
                                 "market": m, "w": 1.0, "pit_pre": 0.5, "p_pre": 0.5,
                                 "actual": float(rng.integers(0, k)),
                                 "pmf_a": a.astype(np.float32),
                                 "pmf_b": None if i == 2 else b.astype(np.float32),
                                 "in_role": i != 1})
    return pd.DataFrame(rows)


def test_serving_weights_final_is_w_final_and_quick_refits_before_holdout(monkeypatch):
    pipe = _pipeline(w={m: (1.0 if m == "rec_yds" else 0.4) for m in MARKETS})
    oof = _oof()
    assert fpf.serving_weights(pipe["markets"], oof) == \
        {m: (1.0 if m == "rec_yds" else 0.4) for m in MARKETS}
    seen = []

    def spy(rows, market):
        rows = list(rows)
        seen.append((market, len(rows)))
        return 0.7

    monkeypatch.setattr(fpf, "choose_weight", spy)
    got = fpf.serving_weights(pipe["markets"], oof, before=(2025, 3))
    # blend not active (w_final 1.0) stays pure A; others re-chosen on OOF rows
    # strictly before (2025, 3), in role, with a B pmf: 2024 wk1-4 + 2025 wk1-2 = 6 weeks x 1
    assert got == {m: (1.0 if m == "rec_yds" else 0.7) for m in MARKETS}
    assert sorted(seen) == sorted((m, 6) for m in MARKETS if m != "rec_yds")
    # no OOF row before the holdout -> pure A
    assert fpf.serving_weights(pipe["markets"], oof, before=(2024, 1))["pass_yds"] == 1.0


def test_calibration_maps_recompute_pre_calibration_at_w_final(monkeypatch):
    pipe = _pipeline(w=0.0, calibrate=True)   # w_final 0 -> pre-calibration pmf = B
    oof = _oof()
    seen = {}

    def spy(rows, market):
        seen[market] = list(rows)
        return (0.9, 0.1) if market == "anytime_td" else np.vstack([np.linspace(0, 1, 3)] * 2)

    monkeypatch.setattr(fpf.tpb, "fit_calibration", spy)
    cal = fpf.calibration_maps(pipe["markets"], oof, {m: 0.0 for m in MARKETS})
    assert set(cal) == set(MARKETS)
    assert cal["anytime_td"] == {"calibrate": True, "kind": "platt", "map": [0.9, 0.1],
                                 "n": len(oof[oof.market == "anytime_td"])}
    assert cal["rec_yds"]["kind"] == "pit" and cal["rec_yds"]["n"] == 24
    # every OOF row of the market; pit_pre recomputed from pmf_b (w = 0), not the stored 0.5
    rows = seen["rec_yds"]
    assert len(rows) == 24
    by_key = {(r.season, r.week, r.player_id): r for r in oof[oof.market == "rec_yds"].itertuples()}
    for r in rows:
        o = by_key[(r["season"], r["week"], r["player_id"])]
        pmf = o.pmf_a if o.pmf_b is None else o.pmf_b     # no B pmf -> A kept
        want = pit_pmf(np.asarray(pmf, float), o.actual,
                       pit_uniform(o.season, o.week, o.player_id, "rec_yds"))
        assert r["pit_pre"] == pytest.approx(want, abs=1e-5)
    # quick mode: only OOF rows strictly before the holdout's first week
    seen.clear()
    fpf.calibration_maps(pipe["markets"], oof, {m: 0.0 for m in MARKETS}, before=(2025, 1))
    assert {(r["season"]) for r in seen["rec_yds"]} == {2024} and len(seen["rec_yds"]) == 12


def test_calibration_maps_identity_when_not_calibrating_and_apply_form():
    cal = fpf.calibration_maps(_pipeline(calibrate=False)["markets"], None, {m: 1.0 for m in MARKETS})
    assert cal["anytime_td"] == {"calibrate": False, "kind": "platt",
                                 "map": list(IDENTITY_PLATT), "n": 0}
    assert cal["rush_att"]["map"] == IDENTITY_KNOTS.tolist()
    assert fpf.calib_for_apply(cal) is None
    on = fpf.calibration_maps(_pipeline(calibrate=True)["markets"], _oof(n=1),
                              {m: 1.0 for m in MARKETS})
    ap = fpf.calib_for_apply(on)
    assert ap["anytime_td"] == IDENTITY_PLATT            # < 100 per class -> identity
    np.testing.assert_array_equal(ap["rush_att"], IDENTITY_KNOTS)
    with pytest.raises(ValueError, match="OOF"):
        fpf.calibration_maps(_pipeline(calibrate=True)["markets"], None, {m: 1.0 for m in MARKETS})


def test_holdout_sources_filters_schedules_to_the_holdout_weeks():
    sched = pd.DataFrame({"season": [2025, 2025, 2025, 2026], "week": [17, 18, 18, 1],
                          "home_team": ["A", "B", "C", "D"]})
    pbp = pd.DataFrame({"x": [1]})
    src = {"schedules": sched, "pbp": pbp}
    got = fpf.holdout_sources(src, [(2025, 18), (2026, 1)])
    assert got["pbp"] is pbp and list(got["schedules"]["home_team"]) == ["B", "C", "D"]
    assert len(src["schedules"]) == 4  # input untouched


# ---- artifacts: real (tiny) fits, round trip -----------------------------------------------

POS = ("QB", "RB", "WR", "WR", "TE", "RB")
TEAMS = ("KC", "BUF", "DAL", "SF")


def _real_tables(seasons=(2023, 2024, 2025), weeks=8, future=(9, 10)):
    rng = np.random.default_rng(7)
    rows, trows = [], []
    lam_t, lam_c = (1, 3, 7, 6, 4, 2), (3, 14, 0.2, 0.2, 0.1, 6)
    for s in seasons:
        for w in list(range(1, weeks + 1)) + (list(future) if s == seasons[-1] else []):
            played = w <= weeks
            for t in TEAMS:
                trows.append({"team": t, "season": s, "week": w, "opponent": "X",
                              "tm_pass_rate": rng.random(), "op_def": rng.random(),
                              "cx_temp": rng.random(), "mk_total": 40 + 10 * rng.random(),
                              "y_team_pass_att": float(rng.poisson(34)) if played else np.nan,
                              "y_team_rush_att": float(rng.poisson(26)) if played else np.nan})
                for k, pos in enumerate(POS):
                    tg, ca = float(rng.poisson(lam_t[k])), float(rng.poisson(lam_c[k]))
                    rec = float(rng.binomial(int(tg), 0.65))
                    pa = float(rng.poisson(33)) if pos == "QB" else 0.0
                    lab = {"y_targets": tg, "y_carries": ca, "y_receptions": rec,
                           "y_rec_yds": rec * 10 + float(rng.integers(0, 9)),
                           "y_rush_yds": ca * 4 + float(rng.integers(0, 5)), "y_pass_att": pa,
                           "y_pass_yds": pa * 7, "y_pass_tds": float(rng.poisson(1.5)) if pa else 0.0,
                           "y_anytime_td": float(rng.random() < 0.3)}
                    rows.append({"player_id": f"{t}{k}", "season": s, "week": w, "team": t,
                                 "opponent": "X", "position": pos, "is_stub": False,
                                 "st_questionable": float(rng.random() < 0.1),
                                 "p_y_targets_ewm": lam_t[k] + rng.random(),
                                 "p_y_carries_ewm": lam_c[k] + rng.random(),
                                 "p_y_pass_att_ewm": 33.0 if pos == "QB" else 0.0,
                                 "p_target_share_ewm": lam_t[k] / 23 + 0.01 * rng.random(),
                                 "p_carry_share_ewm": lam_c[k] / 23.5 + 0.01 * rng.random(),
                                 "cx_temp": rng.random(), "mk_total": 40 + 10 * rng.random(),
                                 **{c: (v if played else np.nan) for c, v in lab.items()}})
    return pd.DataFrame(rows), pd.DataFrame(trows)


def _fit_small(p, t, markets=("rec_yds", "receptions", "anytime_td")):
    kept = frozenset(KEPT)
    lm = learned.fit_models(p, t, kept, upto=(2025, 9), test_season=2025, decay=0.8, max_iter=5)
    cols = learned.feature_columns(p, kept)
    b = {m: fit_market(p, m, cols, upto=(2025, 9), test_season=2025, decay=1.0, max_iter=3)
         for m in markets}
    return lm, b


@pytest.fixture(scope="module")
def small_fit():
    p, t = _real_tables()
    lm, b = _fit_small(p, t)
    return p, t, lm, b


def _config(p, lm, b, pipeline=None):
    return fpf.build_config(pipeline or _pipeline(), learned_models=lm, b_models=b,
                            a_fit={"season": 2025, "decay": 0.8, "max_iter": 300},
                            trained_through=(2025, 8), market_max={"rec_yds": 200, "receptions": 15},
                            role_cols=fpf.role_columns(p, b), identity=IDENT,
                            created_at="2026-09-25T00:00:00+00:00")


def test_save_load_round_trip_gives_identical_predictions(tmp_path, small_fit):
    p, t, lm, b = small_fit
    cal = fpf.calibration_maps(_pipeline()["markets"], None, {m: 1.0 for m in MARKETS})
    cfg = _config(p, lm, b)
    fpf.save_artifacts(tmp_path, lm, b, cal, cfg)
    assert sorted(x.name for x in tmp_path.iterdir()) == sorted(
        ["learned.joblib", "b_rec_yds.joblib", "b_receptions.joblib", "b_anytime_td.joblib",
         "calibration.json", "props_ml_config.json"])
    art = fpf.load_artifacts(tmp_path, p, t)
    X, T = p[p.season == 2025], t[t.season == 2025]
    for name in ("targets", "carries"):
        np.testing.assert_array_equal(getattr(lm, name).predict(X), getattr(art.learned, name).predict(X))
    for name in ("team_pass", "team_rush"):
        np.testing.assert_array_equal(getattr(lm, name).predict(T), getattr(art.learned, name).predict(T))
    for name, m in lm.eff.items():
        np.testing.assert_array_equal(m.predict(X), art.learned.eff[name].predict(X))
    for m, model in b.items():
        kmax = 1 if m == "anytime_td" else 30
        for x, y in zip(predict_pmfs(model, X, kmax), predict_pmfs(art.b_models[m], X, kmax)):
            np.testing.assert_array_equal(x, y)
    assert art.config == json.loads(json.dumps(fpf.tpm.to_jsonable(cfg)))
    assert art.calibration == json.loads(json.dumps(cal))
    assert art.learned.share_fallbacks == 0


def test_config_records_everything_serving_needs(tmp_path, small_fit):
    p, t, lm, b = small_fit
    cfg = _config(p, lm, b)
    for k in ("kept_a_toggles", "tuned", "markets"):
        assert cfg[k] == _pipeline()[k]
    assert cfg["trained_through"] == [2025, 8] and cfg["fit_upto"] == [2025, 9]
    assert cfg["git"] == "deadbeef"
    assert cfg["features"] == {"player": IDENT["player_features"], "team": IDENT["team_features"]}
    assert cfg["created_at"] == "2026-09-25T00:00:00+00:00"
    assert cfg["q_weight"] == 1.0
    assert cfg["a_fit"] == {"season": 2025, "decay": 0.8, "max_iter": 300}
    assert cfg["b_fit"] == {"decay": 1.0, "max_iter": 150}
    assert cfg["role_subsets"] == ROLE_SUBSETS
    assert cfg["market_max"] == {"rec_yds": 200, "receptions": 15, "anytime_td": 1}
    fc = cfg["feature_columns"]
    assert fc["learned"]["player"] == lm.player_cols and fc["learned"]["team"] == lm.team_cols
    assert fc["learned"]["fitted"]["targets"] == lm.targets.cols
    assert fc["learned"]["fitted"]["team_pass"] == lm.team_pass.cols
    assert fc["learned"]["fitted"]["eff"]["ypr"] == lm.eff["ypr"].cols
    assert fc["b"] == {m: model.cols for m, model in b.items()}
    assert fc["role"] == ["p_y_targets_ewm", "position"]      # rec_yds / receptions roles
    assert cfg["artifact_files"] == {"learned": "learned.joblib", "calibration": "calibration.json",
                                     "b": {m: f"b_{m}.joblib" for m in b}}
    json.dumps(fpf.tpm.to_jsonable(cfg))


def test_load_verifies_feature_columns_exist_in_the_tables(tmp_path, small_fit):
    p, t, lm, b = small_fit
    fpf.save_artifacts(tmp_path, lm, b, {}, _config(p, lm, b))
    req = fpf.required_columns(json.loads((tmp_path / "props_ml_config.json").read_text()))
    assert {"p_target_share_ewm", "mk_total", "position", "player_id", "team"} <= set(req["player"])
    assert {"tm_pass_rate", "team"} <= set(req["team"])
    fpf.load_artifacts(tmp_path)  # no tables -> no check
    with pytest.raises(ValueError, match="mk_total"):
        fpf.load_artifacts(tmp_path, p.drop(columns=["mk_total"]), t)
    with pytest.raises(ValueError, match="tm_pass_rate"):
        fpf.load_artifacts(tmp_path, p, t.drop(columns=["tm_pass_rate"]))
    with pytest.raises(ValueError, match="position"):
        fpf.load_artifacts(tmp_path, p.drop(columns=["position"]), t)


def test_save_artifacts_removes_stale_models_but_keeps_quick_gate(tmp_path, small_fit):
    p, t, lm, b_all = small_fit
    b = {m: b_all[m] for m in ("receptions", "anytime_td")}
    fpf.save_artifacts(tmp_path, lm, b, {}, _config(p, lm, b))
    (tmp_path / "quick_gate.json").write_text("{}")
    b2 = {"receptions": b["receptions"]}
    fpf.save_artifacts(tmp_path, lm, b2, {}, _config(p, lm, b2))
    assert not (tmp_path / "b_anytime_td.joblib").exists()
    assert (tmp_path / "quick_gate.json").exists()
    assert set(fpf.load_artifacts(tmp_path).b_models) == {"receptions"}


# ---- run_final_fit (real tiny fits behind spies) --------------------------------------------

class _Bsn:
    SIM_SEED = 42
    MARKET_MAX = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 200, "receptions": 15,
                  "pass_tds": 6, "rush_att": 40}


def _spy_fits(monkeypatch):
    calls = {"a": [], "b": []}
    real_a, real_b = learned.fit_models, fpf.fit_market

    def fa(p, t, toggles, *, upto, test_season, decay, max_iter):
        calls["a"].append({"toggles": toggles, "upto": upto, "test_season": test_season,
                           "decay": decay, "max_iter": max_iter})
        return real_a(p, t, toggles, upto=upto, test_season=test_season, decay=decay, max_iter=3)

    def fb(df, market, cols, *, upto, test_season, decay, max_iter):
        calls["b"].append({"market": market, "cols": list(cols), "upto": upto, "decay": decay,
                           "max_iter": max_iter})
        return real_b(df, market, cols, upto=upto, test_season=test_season, decay=decay,
                      max_iter=2)

    monkeypatch.setattr(learned, "fit_models", fa)
    monkeypatch.setattr(fpf, "fit_market", fb)
    return calls


def test_run_final_fit_fits_on_every_completed_week_and_writes_artifacts(tmp_path, monkeypatch):
    calls = _spy_fits(monkeypatch)
    p, t = _real_tables()
    pipe = _pipeline(w=0.5, calibrate=True)
    oof = _oof(seasons=(2023, 2024), n=2)
    cfg = fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                            oof=oof, identity=IDENT, out_dir=tmp_path, log=lambda m: None,
                            created_at="2026-09-25T00:00:00+00:00")
    # A: kept toggles, most recent tuned season, all rows before trained_through + 1 week
    assert calls["a"] == [{"toggles": frozenset(KEPT), "upto": (2025, 9), "test_season": 2025,
                           "decay": 0.8, "max_iter": 300}]
    # B: served (source "ml") markets only, unweighted, max_iter 150, kept-A columns
    cols = learned.feature_columns(p, frozenset(KEPT))
    assert sorted(c["market"] for c in calls["b"]) == sorted(m for m in MARKETS if m != "pass_tds")
    assert all(c["upto"] == (2025, 9) and c["decay"] == 1.0 and c["max_iter"] == 150
               and c["cols"] == cols for c in calls["b"])
    assert cfg["trained_through"] == [2025, 8]
    assert cfg["market_max"] == {**_Bsn.MARKET_MAX, "anytime_td": 1}
    art = fpf.load_artifacts(tmp_path, p, t)
    assert set(art.b_models) == set(MARKETS) - {"pass_tds"}
    assert art.config["markets"]["rec_yds"] == {"source": "ml", "w_final": 0.5, "calibrate": True}
    assert art.calibration["anytime_td"]["kind"] == "platt"
    assert art.calibration["rec_yds"]["n"] == len(oof[oof.market == "rec_yds"])
    assert not (tmp_path / "quick_gate.json").exists()


def test_run_final_fit_needs_oof_when_calibrating(tmp_path, monkeypatch):
    _spy_fits(monkeypatch)
    p, t = _real_tables()
    with pytest.raises(ValueError, match="OOF"):
        fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=_pipeline(calibrate=True),
                          a_gate=A_GATE, oof=None, identity=IDENT, out_dir=tmp_path,
                          log=lambda m: None)
    assert not (tmp_path / "props_ml_config.json").exists()


# ---- run_quick_gate with a stubbed backtest --------------------------------------------------

GAMES, PLAYERS, WEEKS = 3, 2, 6   # labelled weeks 1..6 of 2025; week 7 is upcoming


def _truth(season, week, g, k, m):
    rng = np.random.default_rng([season, week, g, k, MARKETS.index(m)])
    n = 2 if m == "anytime_td" else KMAX + 1
    return rng.dirichlet(np.ones(n)), int(rng.integers(0, n))


def _schedule():
    rows = [{"season": 2025, "week": w, "game_type": "REG", "home_team": f"H{g}",
             "away_team": f"A{g}", "home_score": 20.0 if w <= WEEKS else np.nan,
             "away_score": 17.0 if w <= WEEKS else np.nan}
            for w in range(1, WEEKS + 2) for g in range(GAMES)]
    return pd.DataFrame(rows)


class _FakeBsn:
    SIM_SEED = 42
    MARKET_MAX = {m: KMAX for m in MARKETS if m != "anytime_td"}

    def __init__(self, worse=(), drop_game=False):
        self.worse, self.drop_game, self.runs, self.fetches = worse, drop_game, [], []
        self.src = {"schedules": _schedule(), "pbp": pd.DataFrame({"x": [1]})}

    def backtest_fetch_seasons(self, seasons):
        return [min(seasons) - 1] + list(seasons)

    def fetch_backtest_sources(self, fetch_seasons):
        self.fetches.append(list(fetch_seasons))
        return self.src

    def run_backtest(self, seasons, n_sims, *, seed, on_game, spec_hook, record, sources,
                     record_pmf=False, **prod):
        sch = sources["schedules"]
        games = sch[sch["season"].isin(seasons) & (sch["game_type"] == "REG")
                    & sch["home_score"].notna()]
        self.runs.append({"hook": spec_hook is not None, "record_pmf": record_pmf, "prod": prod,
                          "n_sims": n_sims, "seed": seed,
                          "weeks": sorted({(int(s), int(w)) for s, w in
                                           zip(games["season"], games["week"])})})
        for row in games.itertuples(index=False):
            s, w, home, away = int(row.season), int(row.week), row.home_team, row.away_team
            g = int(home[1:])
            if spec_hook is not None:
                if self.drop_game and g == 0 and w == WEEKS:
                    continue
                spec_hook(s, w, home, away, "spec")
            on_game(s, w, home, away, None)
            for k in range(PLAYERS):
                pid = f"{home}p{k}"
                for m in MARKETS:
                    base, actual = _truth(s, w, g, k, m)
                    pmf = base
                    if spec_hook is not None:
                        n = len(base)
                        tgt = (actual + n // 2) % n if m in self.worse else actual
                        pmf = 0.7 * base + 0.3 * np.eye(n)[tgt]
                    rec = {"season": s, "week": w, "home": home, "player_id": pid, "market": m,
                           "mean": MEANS[m], "p50": 0.0, "p90": 0.0, "rps": rps_pmf(pmf, actual),
                           "pit": pit_pmf(pmf, actual, pit_uniform(s, w, pid, m)),
                           "actual": float(actual)}
                    if record_pmf:
                        rec["pmf"] = np.asarray(pmf, dtype=np.float32)
                    record.append(rec)


class _Models:
    share_fallbacks = 0


_ROLE = {0: {"position": "QB", "p_y_pass_att_ewm": 30.0, "p_y_carries_ewm": 5.0,
             "p_y_targets_ewm": 0.0},
         1: {"position": "WR", "p_y_pass_att_ewm": 0.0, "p_y_carries_ewm": 0.0,
             "p_y_targets_ewm": 6.0}}


def _qtbl():
    rows = [{"player_id": f"H{g}p{k}", "season": 2025, "week": w, "team": f"H{g}",
             "is_stub": False, "st_questionable": 0.0, "p_signal": 1.0, **_ROLE[k],
             "y_targets": 3.0 if w <= WEEKS else np.nan}
            for w in range(1, WEEKS + 2) for g in range(GAMES) for k in range(PLAYERS)]
    return pd.DataFrame(rows)


def _stub_quick(monkeypatch):
    calls = {"a": [], "b": [], "hook_models": set()}
    fitted = _Models()

    def fa(p, t, toggles, *, upto, test_season, decay, max_iter):
        calls["a"].append({"toggles": toggles, "upto": upto, "test_season": test_season,
                           "decay": decay, "max_iter": max_iter})
        return fitted

    def apply(spec, models, prow, trow, questionable, q_weight):
        calls["hook_models"].add(id(models))
        assert q_weight == 1.0
        return spec

    def fb(df, market, cols, *, upto, test_season, decay, max_iter):
        calls["b"].append({"market": market, "upto": upto, "decay": decay, "max_iter": max_iter})
        return {"market": market, "upto": upto}

    def fp(model, rows, kmax):
        m = model["market"]
        assert all((int(s), int(w)) >= model["upto"] for s, w in zip(rows["season"], rows["week"]))
        assert fpf.in_role(rows, m).all()
        out = []
        for s, w, pid in zip(rows["season"], rows["week"], rows["player_id"]):
            n = 2 if m == "anytime_td" else kmax + 1
            truth, actual = _truth(int(s), int(w), int(pid[1]), int(pid[3]), m)
            out.append(0.5 * truth + 0.5 * np.eye(n)[actual])
        return out

    monkeypatch.setattr(learned, "fit_models", fa)
    monkeypatch.setattr(learned, "apply_to_spec", apply)
    monkeypatch.setattr(fpf, "fit_market", fb)
    monkeypatch.setattr(fpf, "predict_pmfs", fp)
    calls["fitted"] = fitted
    return calls


def _quick(tmp_path, bsn, pipeline, oof=None, n_weeks=2):
    return fpf.run_quick_gate(n_weeks, bsn=bsn, player_tbl=_qtbl(),
                              team_tbl=pd.DataFrame({"team": ["H0"], "season": [2025],
                                                     "week": [1], "tm_x": [1.0]}),
                              pipeline=pipeline, a_gate=A_GATE, oof=oof, identity=IDENT,
                              out_dir=tmp_path, n_sims=17, log=lambda m: None,
                              created_at="2026-09-25T00:00:00+00:00")


def test_quick_gate_fits_before_the_holdout_and_scores_only_holdout_weeks(tmp_path, monkeypatch):
    calls = _stub_quick(monkeypatch)
    bsn = _FakeBsn()
    oof = _oof(seasons=(2024, 2025), weeks=(1, 4, 5, 6), n=2)
    cal_seen = []
    real_cal = fpf.calibration_maps
    monkeypatch.setattr(fpf, "calibration_maps",
                        lambda markets, o, w, before=None: cal_seen.append(before)
                        or real_cal(markets, o, w, before=before))
    gate = _quick(tmp_path, bsn, _pipeline(w=0.5, calibrate=True), oof=oof)

    # models: fit strictly before the first holdout week (2025, 5)
    assert calls["a"] == [{"toggles": frozenset(KEPT), "upto": (2025, 5), "test_season": 2025,
                           "decay": 0.8, "max_iter": 300}]
    assert sorted(c["market"] for c in calls["b"]) == sorted(m for m in MARKETS if m != "pass_tds")
    assert all(c["upto"] == (2025, 5) and c["decay"] == 1.0 and c["max_iter"] == 150
               for c in calls["b"])
    assert calls["hook_models"] == {id(calls["fitted"])}
    assert cal_seen == [(2025, 5)]
    # two runs (baseline, pipeline A) on the holdout weeks only, one fetch, pmfs recorded
    assert bsn.fetches == [[2024, 2025]]
    assert [r["hook"] for r in bsn.runs] == [False, True]
    assert all(r["weeks"] == [(2025, 5), (2025, 6)] and r["record_pmf"] and r["n_sims"] == 17
               and r["prod"] == fpf.tpm.PROD and r["seed"] == 42 for r in bsn.runs)

    assert gate["pass"] is True and gate["reasons"] == []
    assert gate["holdout_weeks"] == [[2025, 5], [2025, 6]] and gate["fit_upto"] == [2025, 5]
    assert gate["skill"] > 0 and gate["n_games"] == 2 * GAMES
    assert gate["per_market"]["pass_tds"]["rps_c"] == gate["per_market"]["pass_tds"]["rps_b"]
    assert gate["sources"]["pass_tds"] == "baseline"
    assert set(gate["weights"]) == set(MARKETS)
    saved = json.loads((tmp_path / "quick_gate.json").read_text())
    assert saved["pass"] is True and saved["holdout_weeks"] == [[2025, 5], [2025, 6]]
    assert saved["thresholds"] == {"min_skill": 0.0, "max_rps_ratio": 1.05}
    assert not (tmp_path / "props_ml_config.json").exists()  # quick gate writes no models


def test_quick_gate_fails_with_market_named_reason_and_still_writes_json(tmp_path, monkeypatch):
    _stub_quick(monkeypatch)
    gate = _quick(tmp_path, _FakeBsn(worse=("rec_yds",)), _pipeline())
    assert gate["pass"] is False
    assert any("rec_yds" in r and "1.05" in r for r in gate["reasons"])
    assert json.loads((tmp_path / "quick_gate.json").read_text())["pass"] is False


def test_quick_gate_coverage_failure_is_a_reason_not_a_crash(tmp_path, monkeypatch):
    _stub_quick(monkeypatch)
    gate = _quick(tmp_path, _FakeBsn(drop_game=True), _pipeline())
    assert gate["pass"] is False and gate["skill"] is None
    assert any("coverage" in r for r in gate["reasons"])
    assert json.loads((tmp_path / "quick_gate.json").read_text())["pass"] is False


def test_main_exit_codes(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(fpf, "_load_inputs", lambda args: {"x": 1})
    monkeypatch.setattr(fpf, "run_quick_gate",
                        lambda n, **kw: seen.append(("quick", n, kw["n_sims"])) or {"pass": False,
                                                                                   "reasons": ["r"]})
    monkeypatch.setattr(fpf, "run_final_fit", lambda **kw: seen.append(("final",)) or {})
    assert fpf.main(["--holdout-weeks", "2", "--out-dir", str(tmp_path)]) == 1
    assert fpf.main(["--out-dir", str(tmp_path)]) == 0
    assert seen == [("quick", 2, 1000), ("final",)]
