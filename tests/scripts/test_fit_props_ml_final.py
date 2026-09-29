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
from sportsmodel.model.props_ml.dist_models import (ROLE_SUBSETS, ROLE_SUBSETS_V2, fit_market,
                                                     predict_pmfs)
from sportsmodel.model.props_ml.pit_calibration import IDENTITY_PLATT, fit_pit_map
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


def _pipeline(w=1.0, calibrate=False, baseline=("pass_tds",), data_end=(2025, 4), final_pass=True):
    return {"kept_a_toggles": list(KEPT),
            "tuned": {"2024": [1.0, 150], "2025": [0.8, 300]},
            "markets": {m: {"source": "baseline" if m in baseline else "ml",
                            "w_final": float(w[m] if isinstance(w, dict) else w),
                            "calibrate": bool(calibrate)} for m in MARKETS},
            "data_end": list(data_end), "final_pass": final_pass, "unselected_pass": False,
            "run_tag": "s2021-2025__every4__b7", "git": "feedface"}


def _pipeline_b(b_markets, **kw):
    """A pipeline whose markets reading B (ml, w_final < 1) are exactly ``b_markets``."""
    return _pipeline(w={m: (0.5 if m in b_markets else 1.0) for m in MARKETS}, **kw)


def _calib(pipeline, rows=()):
    """calibration.json for a pipeline (``train_props_ml_b.final_calibration``)."""
    return {"data_end": list(pipeline["data_end"]),
            "markets": fpf.tpb.final_calibration(list(rows), pipeline["markets"])}


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


def test_holdout_weeks_are_the_last_n_completed_weeks_after_data_end():
    t = _label_tbl()
    assert fpf.holdout_weeks(t, 2, after=(2025, 16)) == [(2025, 18), (2026, 1)]
    assert fpf.holdout_weeks(t, 1, after=(2025, 16)) == [(2026, 1)]
    # never a week <= data_end; fewer than n remain -> the ones that do
    assert fpf.holdout_weeks(t, 2, after=(2025, 18)) == [(2026, 1)]
    assert fpf.holdout_weeks(t, 2, after=(2026, 1)) == []
    with pytest.raises(ValueError):
        fpf.holdout_weeks(t, 0, after=(2025, 16))
    with pytest.raises(ValueError, match="train"):
        fpf.holdout_weeks(t, 3, after=(2024, 18))  # nothing left to train on


def test_a_fit_params_use_the_most_recent_tuned_season_and_kept_must_match():
    assert fpf.a_fit_params(A_GATE) == {"season": 2025, "decay": 0.8, "max_iter": 300}
    assert fpf.kept_toggles(_pipeline(), A_GATE) == frozenset(KEPT)
    with pytest.raises(ValueError, match="kept"):
        fpf.kept_toggles({**_pipeline(), "kept_a_toggles": ["volume"]}, A_GATE)
    with pytest.raises(ValueError, match="kept"):
        fpf.kept_toggles({**_pipeline(), "kept_a_toggles": []}, {**A_GATE, "kept": []})


def test_b_markets_are_ml_markets_with_an_active_blend():
    pipe = _pipeline(w={m: (1.0 if m == "rec_yds" else 0.4) for m in MARKETS})
    assert fpf.b_markets(pipe) == [m for m in MARKETS if m not in ("rec_yds", "pass_tds")]
    assert fpf.b_markets(_pipeline(w=1.0)) == []


def test_check_calibration_needs_the_same_ladder_run():
    pipe = _pipeline(calibrate=True)
    fpf.check_calibration(pipe, _calib(pipe))
    with pytest.raises(ValueError, match="data_end"):
        fpf.check_calibration(pipe, {**_calib(pipe), "data_end": [2025, 3]})
    with pytest.raises(ValueError, match="calibrate"):
        fpf.check_calibration(pipe, _calib(_pipeline(calibrate=False)))
    with pytest.raises(ValueError, match="data_end"):
        fpf.pipeline_data_end({k: v for k, v in pipe.items() if k != "data_end"})


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
    d = fpf.quick_gate_decision(_pm(rec_yds=1.0), None)          # no skill -> fail
    assert d["pass"] is False and "skill" in d["reasons"][0]


# ---- calibration.json -> apply_pipeline -----------------------------------------------------

def test_calib_for_apply_none_without_calibration_and_maps_otherwise():
    assert fpf.calib_for_apply(_calib(_pipeline(calibrate=False))) is None
    ap = fpf.calib_for_apply(_calib(_pipeline(calibrate=True)))
    assert ap["anytime_td"] == IDENTITY_PLATT and ap["rush_att"].shape == (2, 2)


def test_non_identity_calibration_round_trips_through_calibration_json(tmp_path, small_fit):
    p, t, lm, b = small_fit
    rng = np.random.default_rng(3)
    knots = fit_pit_map(rng.beta(0.5, 0.5, 400))            # overconfident PITs -> widening
    assert not np.allclose(knots[0], knots[1])
    pipe = _pipeline_b(b, calibrate=True)
    cal = _calib(pipe)
    for m, v in cal["markets"].items():
        v["map"] = [0.8, 0.3] if m == "anytime_td" else knots.tolist()
    fpf.save_artifacts(tmp_path / "models", lm, b, cal, _config(p, lm, b, pipe))
    loaded = fpf.load_artifacts(tmp_path / "models").calibration
    assert loaded == json.loads(json.dumps(cal))
    recs = [{"season": 2025, "week": 3, "home": "KC", "player_id": "x", "market": "rec_yds",
             "pmf": rng.dirichlet(np.ones(12)), "actual": 4.0},
            {"season": 2025, "week": 3, "home": "KC", "player_id": "x", "market": "anytime_td",
             "pmf": np.array([0.7, 0.3]), "actual": 1.0}]
    w = {m: 1.0 for m in MARKETS}
    got = fpf.tpb.apply_pipeline(recs, {}, w, fpf.calib_for_apply(loaded))
    want = fpf.tpb.apply_pipeline(recs, {}, w, {"rec_yds": knots, "anytime_td": (0.8, 0.3)})
    plain = fpf.tpb.apply_pipeline(recs, {}, w)
    for g, x, y in zip(got, want, plain):
        np.testing.assert_allclose(g["pmf"], x["pmf"], atol=1e-15)
        assert not np.allclose(g["pmf"], y["pmf"], atol=1e-3)   # the map really moved it


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
B3 = ("rec_yds", "receptions", "anytime_td")


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


def _fit_small(p, t, markets=B3):
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
    return fpf.build_config(pipeline or _pipeline_b(b), learned_models=lm, b_models=b,
                            a_fit={"season": 2025, "decay": 0.8, "max_iter": 300},
                            trained_through=(2025, 8), market_max={"rec_yds": 200, "receptions": 15},
                            role_cols=fpf.role_columns(p, b), identity=IDENT,
                            created_at="2026-09-25T00:00:00+00:00")


def test_save_load_round_trip_gives_identical_predictions(tmp_path, small_fit):
    p, t, lm, b = small_fit
    out = tmp_path / "models"
    cal = _calib(_pipeline_b(b))
    cfg = _config(p, lm, b)
    fpf.save_artifacts(out, lm, b, cal, cfg)
    assert sorted(x.name for x in out.iterdir()) == sorted(
        ["learned.joblib", "b_rec_yds.joblib", "b_receptions.joblib", "b_anytime_td.joblib",
         "calibration.json", "props_ml_config.json"])
    assert sorted(x.name for x in tmp_path.iterdir()) == ["models"]   # no temp dirs left
    art = fpf.load_artifacts(out, p, t)
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


def test_config_records_everything_serving_needs(small_fit):
    p, t, lm, b = small_fit
    cfg = _config(p, lm, b)
    for k in ("kept_a_toggles", "tuned", "markets", "data_end"):
        assert cfg[k] == _pipeline_b(b)[k]
    assert cfg["trained_through"] == [2025, 8] and cfg["fit_upto"] == [2025, 9]
    assert cfg["git"] == "deadbeef"
    assert cfg["features"] == {"player": IDENT["player_features"], "team": IDENT["team_features"]}
    assert cfg["created_at"] == "2026-09-25T00:00:00+00:00"
    assert cfg["q_weight"] == 1.0
    assert cfg["a_fit"] == {"season": 2025, "decay": 0.8, "max_iter": 300}
    assert cfg["b_fit"] == {"decay": 1.0, "max_iter": 150}
    assert cfg["b_markets"] == list(B3)
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
    out = tmp_path / "models"
    fpf.save_artifacts(out, lm, b, _calib(_pipeline_b(b)), _config(p, lm, b))
    req = fpf.required_columns(json.loads((out / "props_ml_config.json").read_text()))
    assert {"p_target_share_ewm", "mk_total", "position", "player_id", "team", "season", "week",
            "st_questionable", "opponent"} <= set(req["player"])
    assert {"tm_pass_rate", "team", "season", "week", "opponent"} <= set(req["team"])
    fpf.load_artifacts(out)  # no tables -> no column check
    for tbl, col in ((p, "mk_total"), (p, "position"), (p, "st_questionable"), (t, "tm_pass_rate")):
        args = (p.drop(columns=[col]), t) if tbl is p else (p, t.drop(columns=[col]))
        with pytest.raises(ValueError, match=col):
            fpf.load_artifacts(out, *args)


def test_load_verifies_role_subsets_and_the_b_model_set(tmp_path, small_fit):
    p, t, lm, b = small_fit
    out = tmp_path / "models"
    cfg = _config(p, lm, b)
    bad_roles = {**cfg, "role_subsets": {**ROLE_SUBSETS, "anytime_td": {"positions": None,
                                                                      "col": "x", "min": 1.0}}}
    fpf.save_artifacts(out, lm, b, _calib(_pipeline_b(b)), bad_roles)
    with pytest.raises(ValueError, match="role_subsets"):
        fpf.load_artifacts(out)
    # config says rush_att reads B too, but no rush_att model was saved
    pipe = _pipeline_b(B3 + ("rush_att",))
    fpf.save_artifacts(out, lm, b, _calib(pipe), _config(p, lm, b, pipe))
    with pytest.raises(ValueError, match="rush_att"):
        fpf.load_artifacts(out)


def test_save_artifacts_swaps_atomically_keeps_quick_gate_and_drops_stale(tmp_path, small_fit,
                                                                         monkeypatch):
    p, t, lm, b_all = small_fit
    out = tmp_path / "models"
    b = {m: b_all[m] for m in ("receptions", "anytime_td")}
    fpf.save_artifacts(out, lm, b, _calib(_pipeline_b(b)), _config(p, lm, b))
    (out / "quick_gate.json").write_text('{"pass": true}')
    # a crash mid-save leaves the previous artifacts loadable, and no temp dir behind
    real_dump, calls = fpf.joblib.dump, []

    def flaky(obj, path, *a, **k):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        return real_dump(obj, path, *a, **k)

    monkeypatch.setattr(fpf.joblib, "dump", flaky)
    b1 = {"receptions": b["receptions"]}
    with pytest.raises(OSError, match="disk full"):
        fpf.save_artifacts(out, lm, b1, _calib(_pipeline_b(b1)), _config(p, lm, b1))
    assert set(fpf.load_artifacts(out, p, t).b_models) == {"receptions", "anytime_td"}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["models"]
    monkeypatch.setattr(fpf.joblib, "dump", real_dump)
    fpf.save_artifacts(out, lm, b1, _calib(_pipeline_b(b1)), _config(p, lm, b1))
    assert not (out / "b_anytime_td.joblib").exists()
    assert (out / "quick_gate.json").read_text() == '{"pass": true}'
    assert set(fpf.load_artifacts(out).b_models) == {"receptions"}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["models"]


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

    def fb(df, market, cols, *, upto, test_season, decay, max_iter, subsets=ROLE_SUBSETS):
        calls["b"].append({"market": market, "cols": list(cols), "upto": upto, "decay": decay,
                           "max_iter": max_iter, "subsets": subsets})
        return real_b(df, market, cols, upto=upto, test_season=test_season, decay=decay,
                      max_iter=2, subsets=subsets)

    monkeypatch.setattr(learned, "fit_models", fa)
    monkeypatch.setattr(fpf, "fit_market", fb)
    return calls


def test_run_final_fit_fits_on_every_completed_week_and_writes_artifacts(tmp_path, monkeypatch):
    calls = _spy_fits(monkeypatch)
    p, t = _real_tables()
    pipe = _pipeline(w={m: (1.0 if m == "rush_att" else 0.5) for m in MARKETS}, calibrate=True)
    cal = _calib(pipe)
    cal["markets"]["anytime_td"]["map"] = [0.8, 0.3]
    out = tmp_path / "models"
    cfg = fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                            calibration=cal, identity=IDENT, out_dir=out, log=lambda m: None,
                            created_at="2026-09-25T00:00:00+00:00")
    # A: kept toggles, most recent tuned season, all rows before trained_through + 1 week
    assert calls["a"] == [{"toggles": frozenset(KEPT), "upto": (2025, 9), "test_season": 2025,
                           "decay": 0.8, "max_iter": 300}]
    # B: only markets that read it (ml AND w_final < 1), unweighted, max_iter 150, kept-A cols
    want_b = [m for m in MARKETS if m not in ("pass_tds", "rush_att")]
    cols = learned.feature_columns(p, frozenset(KEPT))
    assert [c["market"] for c in calls["b"]] == want_b
    assert all(c["upto"] == (2025, 9) and c["decay"] == 1.0 and c["max_iter"] == 150
               and c["cols"] == cols for c in calls["b"])
    assert cfg["trained_through"] == [2025, 8] and cfg["b_markets"] == want_b
    assert cfg["market_max"] == {**_Bsn.MARKET_MAX, "anytime_td": 1}
    art = fpf.load_artifacts(out, p, t)
    assert set(art.b_models) == set(want_b)
    assert art.config["markets"]["rec_yds"] == {"source": "ml", "w_final": 0.5, "calibrate": True}
    assert art.calibration == cal                     # the committed maps, copied
    assert not (out / "quick_gate.json").exists()


def test_run_final_fit_refuses_a_calibration_from_another_ladder_run(tmp_path, monkeypatch):
    _spy_fits(monkeypatch)
    p, t = _real_tables()
    pipe = _pipeline(calibrate=True)
    with pytest.raises(ValueError, match="data_end"):
        fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                          calibration={**_calib(pipe), "data_end": [2024, 18]}, identity=IDENT,
                          out_dir=tmp_path / "models", log=lambda m: None)
    assert not (tmp_path / "models").exists()


# ---- run_quick_gate with a stubbed backtest --------------------------------------------------

GAMES, PLAYERS = 3, 2
PLAYED = [(2025, w) for w in range(1, 7)]          # completed weeks
UPCOMING = [(2025, 7)]


def _truth(season, week, g, k, m):
    rng = np.random.default_rng([season, week, g, k, MARKETS.index(m)])
    n = 2 if m == "anytime_td" else KMAX + 1
    return rng.dirichlet(np.ones(n)), int(rng.integers(0, n))


def _schedule(played, upcoming):
    return pd.DataFrame([{"season": s, "week": w, "game_type": "REG", "home_team": f"H{g}",
                          "away_team": f"A{g}",
                          "home_score": 20.0 if (s, w) in played else np.nan,
                          "away_score": 17.0 if (s, w) in played else np.nan}
                         for s, w in played + upcoming for g in range(GAMES)])


class _FakeBsn:
    SIM_SEED = 42
    MARKET_MAX = {m: KMAX for m in MARKETS if m != "anytime_td"}

    def __init__(self, worse=(), drop_game=None, played=PLAYED, upcoming=UPCOMING, fail=False):
        self.worse, self.drop_game, self.fail, self.runs, self.fetches = worse, drop_game, fail, [], []
        self.src = {"schedules": _schedule(played, upcoming), "pbp": pd.DataFrame({"x": [1]})}

    def backtest_fetch_seasons(self, seasons):
        return [min(seasons) - 1] + list(seasons)

    def fetch_backtest_sources(self, fetch_seasons):
        if self.fail:
            raise ConnectionError("nflverse unreachable")
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
                if self.drop_game == (s, w) and g == 0:
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


def _qtbl(played=PLAYED, upcoming=UPCOMING):
    return pd.DataFrame([{"player_id": f"H{g}p{k}", "season": s, "week": w, "team": f"H{g}",
                          "is_stub": False, "st_questionable": 0.0, "p_signal": 1.0, **_ROLE[k],
                          "y_targets": 3.0 if (s, w) in played else np.nan}
                         for s, w in played + upcoming for g in range(GAMES)
                         for k in range(PLAYERS)])


def _stub_quick(monkeypatch):
    calls = {"a": [], "b": [], "hook": []}
    fitted = _Models()

    def fa(p, t, toggles, *, upto, test_season, decay, max_iter):
        calls["a"].append({"toggles": toggles, "upto": upto, "test_season": test_season,
                           "decay": decay, "max_iter": max_iter})
        return fitted

    def apply(spec, models, prow, trow, questionable, q_weight):
        calls["hook"].append((id(models), sorted({(int(s), int(w)) for s, w in
                                                  zip(prow["season"], prow["week"])})))
        assert q_weight == 1.0
        return spec

    def fb(df, market, cols, *, upto, test_season, decay, max_iter, subsets=ROLE_SUBSETS):
        calls["b"].append({"market": market, "upto": upto, "decay": decay, "max_iter": max_iter,
                           "subsets": subsets})
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


def _quick(tmp_path, bsn, pipeline, calibration=None, n_weeks=2, ptbl=None):
    return fpf.run_quick_gate(n_weeks, bsn=bsn, player_tbl=_qtbl() if ptbl is None else ptbl,
                              team_tbl=pd.DataFrame({"team": ["H0"], "season": [2025],
                                                     "week": [1], "tm_x": [1.0]}),
                              pipeline=pipeline, a_gate=A_GATE,
                              calibration=_calib(pipeline) if calibration is None else calibration,
                              identity=IDENT, out_dir=tmp_path, n_sims=17, log=lambda m: None,
                              created_at="2026-09-25T00:00:00+00:00")


def test_quick_gate_fits_before_the_holdout_and_scores_only_holdout_weeks(tmp_path, monkeypatch):
    calls = _stub_quick(monkeypatch)
    bsn = _FakeBsn()
    pipe = _pipeline(w={m: (1.0 if m == "rush_att" else 0.5) for m in MARKETS}, calibrate=True,
                     data_end=(2025, 4))
    gate = _quick(tmp_path, bsn, pipe)

    # models: fit strictly before the first holdout week (2025, 5)
    assert calls["a"] == [{"toggles": frozenset(KEPT), "upto": (2025, 5), "test_season": 2025,
                           "decay": 0.8, "max_iter": 300}]
    assert [c["market"] for c in calls["b"]] == [m for m in MARKETS
                                                 if m not in ("pass_tds", "rush_att")]
    assert all(c["upto"] == (2025, 5) and c["decay"] == 1.0 and c["max_iter"] == 150
               for c in calls["b"])
    assert {i for i, _ in calls["hook"]} == {id(calls["fitted"])}
    # two runs (baseline, pipeline A) on the holdout weeks only, one fetch, pmfs recorded
    assert bsn.fetches == [[2024, 2025]]
    assert [r["hook"] for r in bsn.runs] == [False, True]
    assert all(r["weeks"] == [(2025, 5), (2025, 6)] and r["record_pmf"] and r["n_sims"] == 17
               and r["prod"] == fpf.tpm.PROD and r["seed"] == 42 for r in bsn.runs)

    assert gate["pass"] is True and gate["reasons"] == []
    assert gate["holdout_weeks"] == [[2025, 5], [2025, 6]] and gate["fit_upto"] == [2025, 5]
    assert gate["data_end"] == [2025, 4]
    assert gate["skill"] > 0 and gate["n_games"] == 2 * GAMES
    assert gate["per_market"]["pass_tds"]["rps_c"] == gate["per_market"]["pass_tds"]["rps_b"]
    assert gate["sources"]["pass_tds"] == "baseline"
    assert gate["weights"] == {m: v["w_final"] for m, v in pipe["markets"].items()}  # committed
    saved = json.loads((tmp_path / "quick_gate.json").read_text())
    assert saved["pass"] is True and saved["holdout_weeks"] == [[2025, 5], [2025, 6]]
    assert saved["thresholds"] == {"min_skill": 0.0, "max_rps_ratio": 1.05}
    assert not (tmp_path / "props_ml_config.json").exists()  # quick gate writes no models


def test_quick_gate_holdout_never_includes_weeks_up_to_data_end(tmp_path, monkeypatch):
    calls = _stub_quick(monkeypatch)
    bsn = _FakeBsn()
    gate = _quick(tmp_path, bsn, _pipeline(data_end=(2025, 5)))
    assert gate["holdout_weeks"] == [[2025, 6]] and calls["a"][0]["upto"] == (2025, 6)
    assert all(r["weeks"] == [(2025, 6)] for r in bsn.runs)


def test_quick_gate_fails_when_no_completed_week_follows_data_end(tmp_path, monkeypatch):
    calls = _stub_quick(monkeypatch)
    bsn = _FakeBsn()
    gate = _quick(tmp_path, bsn, _pipeline(data_end=(2025, 6)))
    assert gate["pass"] is False
    assert gate["reasons"] == ["no completed weeks after ladder data_end (2025, 6)"]
    assert bsn.runs == [] and calls["a"] == []
    saved = json.loads((tmp_path / "quick_gate.json").read_text())
    assert saved["pass"] is False and saved["reasons"] == gate["reasons"]


def test_quick_gate_holdout_can_span_two_seasons(tmp_path, monkeypatch):
    calls = _stub_quick(monkeypatch)
    played = [(2025, 16), (2025, 17), (2025, 18), (2026, 1)]
    bsn = _FakeBsn(played=played, upcoming=[(2026, 2)])
    gate = _quick(tmp_path, bsn, _pipeline(data_end=(2025, 17)),
                  ptbl=_qtbl(played, [(2026, 2)]))
    assert gate["holdout_weeks"] == [[2025, 18], [2026, 1]] and gate["pass"] is True
    assert calls["a"][0]["upto"] == (2025, 18) and calls["a"][0]["test_season"] == 2025
    assert bsn.fetches == [[2024, 2025, 2026]]
    assert all(r["weeks"] == [(2025, 18), (2026, 1)] for r in bsn.runs)
    # the hook serves both seasons' games with the one fitted A, on that week's feature rows
    assert {i for i, _ in calls["hook"]} == {id(calls["fitted"])}
    assert sorted({wk for _, wks in calls["hook"] for wk in wks}) == [(2025, 18), (2026, 1)]


def test_quick_gate_fails_with_market_named_reason_and_still_writes_json(tmp_path, monkeypatch):
    _stub_quick(monkeypatch)
    gate = _quick(tmp_path, _FakeBsn(worse=("rec_yds",)), _pipeline())
    assert gate["pass"] is False
    assert any("rec_yds" in r and "1.05" in r for r in gate["reasons"])
    assert json.loads((tmp_path / "quick_gate.json").read_text())["pass"] is False


def test_quick_gate_coverage_failure_is_a_reason_not_a_crash(tmp_path, monkeypatch):
    _stub_quick(monkeypatch)
    gate = _quick(tmp_path, _FakeBsn(drop_game=(2025, 6)), _pipeline())
    assert gate["pass"] is False and gate["skill"] is None
    assert any("coverage" in r for r in gate["reasons"])
    assert json.loads((tmp_path / "quick_gate.json").read_text())["pass"] is False


def test_quick_gate_crash_writes_a_failing_record_and_no_stale_pass_survives(tmp_path,
                                                                             monkeypatch):
    _stub_quick(monkeypatch)
    (tmp_path / "quick_gate.json").write_text('{"pass": true}')
    with pytest.raises(ConnectionError):
        _quick(tmp_path, _FakeBsn(fail=True), _pipeline())
    saved = json.loads((tmp_path / "quick_gate.json").read_text())
    assert saved["pass"] is False
    assert saved["reasons"] == ["error: ConnectionError: nflverse unreachable"]
    assert saved["holdout_weeks"] == [[2025, 5], [2025, 6]] and saved["identity"] == IDENT
    # an error before the holdout is known (mismatched calibration) is recorded too
    (tmp_path / "quick_gate.json").write_text('{"pass": true}')
    pipe = _pipeline()
    with pytest.raises(ValueError):
        _quick(tmp_path, _FakeBsn(), pipe, calibration={**_calib(pipe), "data_end": [2020, 1]})
    saved = json.loads((tmp_path / "quick_gate.json").read_text())
    assert saved["pass"] is False and saved["reasons"][0].startswith("error: ValueError:")


def test_main_exit_codes(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(fpf, "_load_inputs", lambda args: {"pipeline": _pipeline()})
    monkeypatch.setattr(fpf, "run_quick_gate",
                        lambda n, **kw: seen.append(("quick", n, kw["n_sims"])) or {"pass": False,
                                                                                   "reasons": ["r"]})
    monkeypatch.setattr(fpf, "run_final_fit", lambda **kw: seen.append(("final",)) or {})
    assert fpf.main(["--holdout-weeks", "2", "--out-dir", str(tmp_path)]) == 1
    assert fpf.main(["--out-dir", str(tmp_path)]) == 0
    assert seen == [("quick", 2, 1000), ("final",)]
    with pytest.raises(SystemExit):
        fpf.main(["--oof", "x.parquet"])      # no OOF input any more


# ---- I1: a failed B gate is never fit or published -----------------------------------------

@pytest.mark.parametrize("pipe,needle", [
    ({**_pipeline(), "final_pass": False}, "final_pass"),
    ({k: v for k, v in _pipeline().items() if k != "final_pass"}, "final_pass"),
    ({**_pipeline(), "final_pass": "true"}, "final_pass"),          # only a real JSON true
    (_pipeline(baseline=MARKETS), 'no market with source "ml"'),
])
def test_pipeline_refusal_names_the_reason(pipe, needle):
    reason = fpf.pipeline_refusal(pipe)
    assert reason is not None and needle in reason


def test_pipeline_refusal_none_for_a_passing_pipeline_with_an_ml_market():
    assert fpf.pipeline_refusal(_pipeline()) is None


def test_run_final_fit_refuses_a_failed_b_gate(tmp_path, monkeypatch):
    calls = _spy_fits(monkeypatch)
    p, t = _real_tables()
    pipe = _pipeline(final_pass=False)
    with pytest.raises(RuntimeError, match="final_pass"):
        fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                          calibration=_calib(pipe), identity=IDENT, out_dir=tmp_path / "models",
                          log=lambda m: None)
    assert calls["a"] == [] and calls["b"] == []
    assert not (tmp_path / "models").exists()


def test_run_final_fit_refuses_when_no_market_is_ml(tmp_path, monkeypatch):
    _spy_fits(monkeypatch)
    p, t = _real_tables()
    pipe = _pipeline(baseline=MARKETS)
    with pytest.raises(RuntimeError, match='no market with source "ml"'):
        fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                          calibration=_calib(pipe), identity=IDENT, out_dir=tmp_path / "models",
                          log=lambda m: None)


def test_quick_gate_refuses_a_failed_b_gate(tmp_path, monkeypatch):
    calls = _stub_quick(monkeypatch)
    bsn = _FakeBsn()
    gate = _quick(tmp_path, bsn, _pipeline(final_pass=False))
    assert gate["pass"] is False and "final_pass" in gate["reasons"][0]
    assert bsn.runs == [] and calls["a"] == []
    assert json.loads((tmp_path / "quick_gate.json").read_text())["pass"] is False


@pytest.mark.parametrize("argv", [["--holdout-weeks", "2"], []])
def test_main_refuses_a_failed_b_gate_in_both_modes(monkeypatch, tmp_path, capsys, argv):
    monkeypatch.setattr(fpf, "_load_inputs", lambda args: {"pipeline": _pipeline(final_pass=False)})

    def boom(*a, **k):
        raise AssertionError("fit / quick gate ran on a failed B gate")

    monkeypatch.setattr(fpf, "run_quick_gate", boom)
    monkeypatch.setattr(fpf, "run_final_fit", boom)
    assert fpf.main([*argv, "--out-dir", str(tmp_path)]) == 1
    assert "REFUSED" in capsys.readouterr().out


# ---- I2: --check-new-weeks (the weekly job's freshness guard) ---------------------------

def _labelled_tbl(tmp_path, weeks, stub_weeks=()):
    rows = [{"season": s, "week": w, "player_id": "p", "y_rec_yds": 10.0, "is_stub": False}
            for s, w in weeks]
    rows += [{"season": s, "week": w, "player_id": "p", "y_rec_yds": None, "is_stub": True}
             for s, w in stub_weeks]
    path = tmp_path / "player_features.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("tt,expected", [([2026, 2], "true"), ([2026, 3], "false"),
                                         ([2025, 18], "true")])
def test_main_check_new_weeks(tmp_path, capsys, tt, expected):
    ptbl = _labelled_tbl(tmp_path, [(2025, 18), (2026, 1), (2026, 2), (2026, 3)],
                         stub_weeks=[(2026, 4)])        # the upcoming stub week is not labelled
    cfg = tmp_path / "props_ml_config.json"
    cfg.write_text(json.dumps({"trained_through": tt}))

    def boom(args):
        raise AssertionError("the freshness check must not load the fit inputs")

    import unittest.mock as um
    with um.patch.object(fpf, "_load_inputs", boom):
        assert fpf.main(["--check-new-weeks", str(cfg), "--player-table", str(ptbl)]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == f"new_weeks={expected}"


def test_new_labelled_weeks_after():
    tbl = pd.DataFrame({"season": [2026] * 3, "week": [1, 2, 3], "y_rec_yds": [1.0, 2.0, None]})
    assert fpf.new_labelled_weeks(tbl, (2026, 1)) == [(2026, 2)]
    assert fpf.new_labelled_weeks(tbl, (2026, 2)) == []


# ---- Task 8: per-version artifacts (v1 unchanged, v2 via --gate-name _v2) ----------------

SERVING_QB = {"H": 1.5, "k": 200.0, "fit_seasons": [2021, 2022, 2023, 2024, 2025, 2026],
              "created_at": "2026-09-29T00:00:00+00:00"}


def _qb_params_file(tmp_path, serving=True):
    path = tmp_path / "qb_profile_params.json"
    doc = {"gate": {"H": 1.0, "k": 100.0}}
    if serving:
        doc["serving"] = SERVING_QB
    path.write_text(json.dumps(doc))
    return path


def test_version_paths_per_gate_name():
    assets = fpf.tpb.A_GATE_PATH.parent
    assert fpf.gate_version("") == "nfl-sim-ml-v1" and fpf.gate_version("_v2") == "nfl-sim-ml-v2"
    with pytest.raises(ValueError):
        fpf.gate_version("_v1cmp")      # the comparison run is never fit / served
    assert fpf.input_paths("") == (assets / "pipeline.json", assets / "calibration.json",
                                   assets / "a_gate.json")
    assert fpf.input_paths("_v2") == (assets / "pipeline_v2.json", assets / "calibration_v2.json",
                                      assets / "a_gate_v2.json")
    root = fpf.tpm.DATA_DIR / "models"
    assert fpf.model_dir("nfl-sim-ml-v1") == root / "nfl-sim-ml-v1" == fpf.MODEL_DIR
    assert fpf.model_dir("nfl-sim-ml-v2") == root / "nfl-sim-ml-v2"
    assert fpf.gate_subsets("") is ROLE_SUBSETS and fpf.gate_subsets("_v2") is ROLE_SUBSETS_V2


def test_role_columns_read_every_condition_of_the_version_table():
    p = pd.DataFrame({"position": ["QB"], "p_y_pass_att_ewm": [30.0], "p_y_pass_att_r10": [30.0]})
    assert fpf.role_columns(p, ["pass_yds"]) == ["p_y_pass_att_ewm", "position"]
    assert fpf.role_columns(p, ["pass_yds", "anytime_td"], ROLE_SUBSETS_V2) == [
        "p_y_pass_att_ewm", "p_y_pass_att_r10", "position"]


def test_serving_qb_block_reads_the_serving_params(tmp_path):
    assert fpf.serving_qb_block(_qb_params_file(tmp_path)) == SERVING_QB
    with pytest.raises(RuntimeError, match="serving"):
        fpf.serving_qb_block(_qb_params_file(tmp_path, serving=False))


def _v2_config(p, lm, b, qb=SERVING_QB):
    return fpf.build_config(_pipeline_b(b), learned_models=lm, b_models=b,
                            a_fit={"season": 2025, "decay": 0.8, "max_iter": 300},
                            trained_through=(2025, 8), market_max={"rec_yds": 200, "receptions": 15},
                            role_cols=fpf.role_columns(p, b, ROLE_SUBSETS_V2), identity=IDENT,
                            created_at="2026-09-25T00:00:00+00:00", gate_name="_v2",
                            qb_profile_params=qb)


def test_v1_config_is_unchanged_no_version_keys(small_fit):
    p, t, lm, b = small_fit
    cfg = _config(p, lm, b)
    assert "model_version" not in cfg and "qb_profile_params" not in cfg
    assert cfg["role_subsets"] is ROLE_SUBSETS


def test_v2_config_round_trips_and_loads_against_the_v2_role_table(tmp_path, small_fit):
    p, t, lm, b = small_fit
    cfg = _v2_config(p, lm, b)
    assert cfg["model_version"] == "nfl-sim-ml-v2" and cfg["qb_profile_params"] == SERVING_QB
    assert cfg["role_subsets"] is ROLE_SUBSETS_V2
    out = tmp_path / "nfl-sim-ml-v2"
    fpf.save_artifacts(out, lm, b, _calib(_pipeline_b(b)), cfg)
    saved = json.loads((out / "props_ml_config.json").read_text())
    # JSON turned the V2 "conds" tuples into lists: the load check must still match
    assert isinstance(saved["role_subsets"]["pass_yds"]["conds"][0], list)
    art = fpf.load_artifacts(out, p, t)
    assert art.config["model_version"] == "nfl-sim-ml-v2"
    assert fpf.artifacts.role_subsets_for(art.config) is ROLE_SUBSETS_V2
    with pytest.raises(ValueError):
        fpf.build_config(**{**_v2_kwargs(p, lm, b), "qb_profile_params": None})


def _v2_kwargs(p, lm, b):
    return dict(pipeline=_pipeline_b(b), learned_models=lm, b_models=b,
                a_fit={"season": 2025, "decay": 0.8, "max_iter": 300}, trained_through=(2025, 8),
                market_max={"rec_yds": 200}, role_cols=[], identity=IDENT,
                created_at="2026-09-25T00:00:00+00:00", gate_name="_v2")


@pytest.mark.parametrize("version,table,ok", [
    (None, ROLE_SUBSETS, True),                       # the live v1 artifacts: no model_version
    ("nfl-sim-ml-v1", ROLE_SUBSETS, True),
    ("nfl-sim-ml-v2", ROLE_SUBSETS_V2, True),
    ("nfl-sim-ml-v2", ROLE_SUBSETS, False),           # v2 config with the v1 table
    (None, ROLE_SUBSETS_V2, False),                   # v1 config with the v2 table
    ("nfl-sim-ml-v9", ROLE_SUBSETS, False),           # unknown version
])
def test_load_checks_role_subsets_against_the_config_model_version(tmp_path, small_fit, version,
                                                                    table, ok):
    p, t, lm, b = small_fit
    cfg = {**_config(p, lm, b), "role_subsets": table}
    if version is not None:
        cfg["model_version"] = version
    out = tmp_path / "m"
    fpf.save_artifacts(out, lm, b, _calib(_pipeline_b(b)), cfg)
    if ok:
        fpf.load_artifacts(out, p, t)
    else:
        with pytest.raises(ValueError, match="role_subsets|model_version"):
            fpf.load_artifacts(out, p, t)


def test_run_final_fit_v2_fits_b_on_the_v2_roles_and_records_version(tmp_path, monkeypatch):
    calls = _spy_fits(monkeypatch)
    p, t = _real_tables()
    p["p_y_pass_att_r10"] = p["p_y_pass_att_ewm"]
    pipe = _pipeline(w={m: (1.0 if m == "rush_att" else 0.5) for m in MARKETS})
    out = tmp_path / "nfl-sim-ml-v2"
    cfg = fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                            calibration=_calib(pipe), identity=IDENT, out_dir=out,
                            log=lambda m: None, created_at="2026-09-25T00:00:00+00:00",
                            gate_name="_v2", qb_profile_params=SERVING_QB)
    assert calls["b"] and all(c["subsets"] is ROLE_SUBSETS_V2 for c in calls["b"])
    assert cfg["model_version"] == "nfl-sim-ml-v2" and cfg["qb_profile_params"] == SERVING_QB
    assert "p_y_pass_att_r10" in cfg["feature_columns"]["role"]
    art = fpf.load_artifacts(out, p, t)
    assert art.config["model_version"] == "nfl-sim-ml-v2"


def test_run_final_fit_v1_uses_v1_roles(tmp_path, monkeypatch):
    calls = _spy_fits(monkeypatch)
    p, t = _real_tables()
    pipe = _pipeline(w={m: (1.0 if m == "rush_att" else 0.5) for m in MARKETS})
    cfg = fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                            calibration=_calib(pipe), identity=IDENT, out_dir=tmp_path / "v1",
                            log=lambda m: None)
    assert calls["b"] and all(c["subsets"] is ROLE_SUBSETS for c in calls["b"])
    assert "model_version" not in cfg


def test_run_final_fit_v2_without_serving_qb_params_writes_nothing(tmp_path, monkeypatch):
    calls = _spy_fits(monkeypatch)
    p, t = _real_tables()
    pipe = _pipeline()
    with pytest.raises(RuntimeError, match="serving"):
        fpf.run_final_fit(bsn=_Bsn(), player_tbl=p, team_tbl=t, pipeline=pipe, a_gate=A_GATE,
                          calibration=_calib(pipe), identity=IDENT, out_dir=tmp_path / "v2",
                          log=lambda m: None, gate_name="_v2", qb_profile_params=None)
    assert calls["a"] == [] and not (tmp_path / "v2").exists()


def test_quick_gate_v2_scores_b_on_the_v2_roles(tmp_path, monkeypatch):
    calls = _stub_quick(monkeypatch)
    pipe = _pipeline(w={m: (1.0 if m == "rush_att" else 0.5) for m in MARKETS})
    ptbl = _qtbl().assign(p_y_pass_att_r10=lambda d: d["p_y_pass_att_ewm"])
    gate = fpf.run_quick_gate(2, bsn=_FakeBsn(), player_tbl=ptbl,
                              team_tbl=pd.DataFrame({"team": ["H0"], "season": [2025],
                                                     "week": [1], "tm_x": [1.0]}),
                              pipeline=pipe, a_gate=A_GATE, calibration=_calib(pipe),
                              identity=IDENT, out_dir=tmp_path, n_sims=17, log=lambda m: None,
                              gate_name="_v2")
    assert gate["gate_name"] == "_v2"
    assert calls["b"] and all(c["subsets"] is ROLE_SUBSETS_V2 for c in calls["b"])


def test_predict_b_uses_the_given_role_table():
    rows = pd.DataFrame({"season": [2025, 2025], "week": [5, 5], "player_id": ["q1", "q2"],
                         "position": ["QB", "QB"], "p_y_pass_att_ewm": [30.0, 30.0],
                         "p_y_pass_att_r10": [30.0, 0.0]})       # q2: stale QB (v2: out of role)
    recs = [{"season": 2025, "week": 5, "home": "H", "player_id": pid, "market": "pass_yds"}
            for pid in ("q1", "q2")]

    class M:
        pass

    import unittest.mock as um
    with um.patch.object(fpf, "predict_pmfs", lambda model, r, k: [np.ones(3) / 3] * len(r)):
        _, oor_v1 = fpf.predict_b({"pass_yds": M()}, rows, recs, {"pass_yds": 2})
        _, oor_v2 = fpf.predict_b({"pass_yds": M()}, rows, recs, {"pass_yds": 2}, ROLE_SUBSETS_V2)
    assert oor_v1 == set() and oor_v2 == {(2025, 5, "q2", "pass_yds")}


def test_main_gate_name_picks_the_version_inputs_and_dir(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(fpf, "QB_PARAMS_PATH", _qb_params_file(tmp_path))

    def load(args):
        seen["load"] = (args.gate_name, args.pipeline, args.calibration)
        return {"pipeline": _pipeline()}

    monkeypatch.setattr(fpf, "_load_inputs", load)
    monkeypatch.setattr(fpf, "run_final_fit", lambda **kw: seen.setdefault("fit", kw) and {})
    monkeypatch.setattr(fpf, "run_quick_gate",
                        lambda n, **kw: seen.setdefault("quick", kw) and {"pass": True})
    assert fpf.main(["--gate-name", "_v2"]) == 0
    pipe, cal, _ = fpf.input_paths("_v2")
    assert seen["load"] == ("_v2", pipe, cal)
    assert seen["fit"]["out_dir"] == fpf.model_dir("nfl-sim-ml-v2")
    assert seen["fit"]["gate_name"] == "_v2" and seen["fit"]["qb_profile_params"] == SERVING_QB
    assert fpf.main(["--gate-name", "_v2", "--holdout-weeks", "2"]) == 0
    assert seen["quick"]["gate_name"] == "_v2"
    assert seen["quick"]["out_dir"] == fpf.model_dir("nfl-sim-ml-v2")
    seen.clear()
    assert fpf.main([]) == 0                                       # no gate name: v1 as today
    assert seen["load"] == ("", *fpf.input_paths("")[:2])
    assert seen["fit"]["out_dir"] == fpf.model_dir("nfl-sim-ml-v1")
    assert seen["fit"]["gate_name"] == "" and seen["fit"]["qb_profile_params"] is None


def test_main_v2_final_fit_without_serving_block_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(fpf, "QB_PARAMS_PATH", _qb_params_file(tmp_path, serving=False))
    monkeypatch.setattr(fpf, "_load_inputs", lambda args: {"pipeline": _pipeline()})

    def boom(**kw):
        raise AssertionError("fit ran without the serving QB params")

    monkeypatch.setattr(fpf, "run_final_fit", boom)
    assert fpf.main(["--gate-name", "_v2"]) == 1
    assert "serving" in capsys.readouterr().out
