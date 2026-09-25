"""Tests for scripts/train_props_ml_b.py (props-ML Props-2 offline B / blend /
calibration ladder). Stubbed backtest, stubbed B fits, tiny tables -- no
network, no real model fits."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel.model import props_eval
from sportsmodel.model.props_eval import pit_pmf, pit_uniform, rps_pmf
from sportsmodel.model.props_ml.pit_calibration import IDENTITY_KNOTS, IDENTITY_PLATT

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "train_props_ml_b.py"
_spec = importlib.util.spec_from_file_location("train_props_ml_b", _p)
tpb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpb)

MARKETS = ("pass_yds", "rush_yds", "rec_yds", "receptions", "rush_att", "pass_tds", "anytime_td")
KMAX = 8
MEANS = {"pass_yds": 220.0, "rush_yds": 45.0, "rec_yds": 45.0, "receptions": 4.0,
         "rush_att": 9.0, "pass_tds": 1.6, "anytime_td": 0.4}
A_GATE = {"kept": ["context", "efficiency", "market", "volume"],
          "tuned": {"2021": [1.0, 300], "2022": [1.0, 150], "2023": [0.8, 150],
                    "2024": [1.0, 150], "2025": [0.8, 300]}}
WEEKS, GAMES, PLAYERS = 6, 3, 2


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    real = props_eval.cluster_bootstrap
    monkeypatch.setattr(props_eval, "cluster_bootstrap",
                        lambda df, stat, n_boot=1000, seed=0: real(df, stat, n_boot=40, seed=seed))


def test_constants():
    assert tpb.MARKETS == MARKETS
    assert tpb.B_DECAY == 1.0 and tpb.B_MAX_ITER == 150
    assert tpb.A_WORSE_TOL == 1.01


# ---- pure: A config / sources / config shape ------------------------------------------

def test_a_config_reads_kept_toggles_and_tuned_per_season():
    kept, tuned = tpb.a_config(A_GATE, [2021, 2025])
    assert kept == frozenset({"volume", "efficiency", "context", "market"})
    assert tuned == {2021: (1.0, 300), 2025: (0.8, 300)}
    with pytest.raises(ValueError, match="2026"):
        tpb.a_config(A_GATE, [2026])


def test_market_sources_final_rule_also_needs_ece_within_tolerance():
    pm = {"rec_yds": {"rps_b": 1.0, "rps_c": 0.9, "ece_b": 0.010, "ece_c": 0.016},
          "receptions": {"rps_b": 1.0, "rps_c": 0.9, "ece_b": 0.010, "ece_c": 0.015},
          "rush_att": {"rps_b": 1.0, "rps_c": 1.1, "ece_b": 0.010, "ece_c": 0.0}}
    got = tpb.market_sources(pm, ("rec_yds", "receptions", "rush_att"), 1.0, ece_tol=0.005)
    assert got == {"rec_yds": "baseline", "receptions": "ml", "rush_att": "baseline"}


def test_market_sources_marks_markets_worse_beyond_tolerance_as_baseline():
    pm = {"rec_yds": {"rps_b": 1.0, "rps_c": 1.009}, "pass_tds": {"rps_b": 1.0, "rps_c": 1.02},
          "receptions": {"rps_b": 1.0, "rps_c": 0.9}}
    got = tpb.market_sources(pm, ("rec_yds", "pass_tds", "receptions", "rush_att"), 1.01)
    assert got == {"rec_yds": "ml", "pass_tds": "baseline", "receptions": "ml",
                   "rush_att": "baseline"}  # no scored rows -> baseline, never silently ml
    final = tpb.market_sources(pm, ("rec_yds", "receptions"), 1.0)
    assert final == {"rec_yds": "baseline", "receptions": "ml"}


def test_pipeline_config_shape():
    cfg = tpb.pipeline_config(frozenset({"volume", "market", "efficiency", "context"}),
                              {2025: (0.8, 300), 2024: (1.0, 150)},
                              {m: ("baseline" if m == "pass_tds" else "ml") for m in MARKETS},
                              {m: 0.7 for m in MARKETS}, True, (2025, 18))
    assert set(cfg) == {"kept_a_toggles", "tuned", "markets", "data_end"}
    assert cfg["data_end"] == [2025, 18]
    assert cfg["kept_a_toggles"] == ["volume", "efficiency", "context", "market"]
    assert cfg["tuned"] == {"2024": [1.0, 150], "2025": [0.8, 300]}
    assert set(cfg["markets"]) == set(MARKETS)
    assert cfg["markets"]["pass_tds"] == {"source": "baseline", "w_final": 0.7, "calibrate": True}
    json.dumps(cfg)  # native types only


def test_output_paths_default_and_tagged(tmp_path):
    kw = dict(gate_path=tmp_path / "b_gate.json", pipeline_path=tmp_path / "pipeline.json",
              report_dir=tmp_path / "r")
    assert tpb.output_paths("s2021-2025__every4", "2026-09-25", **kw) == (
        tmp_path / "b_gate.json", tmp_path / "pipeline.json", tmp_path / "calibration.json",
        tmp_path / "r" / "2026-09-25-props-ml-b-gate.md")
    assert tpb.output_paths("s2025__every4", "2026-09-25", **kw) == (
        tmp_path / "b_gate__s2025__every4.json", tmp_path / "pipeline__s2025__every4.json",
        tmp_path / "calibration__s2025__every4.json",
        tmp_path / "r" / "2026-09-25-props-ml-b-gate__s2025__every4.md")
    assert tpb.CALIBRATION_PATH == tpb.PIPELINE_PATH.with_name("calibration.json")


# ---- pure: apply_pipeline ---------------------------------------------------------------

def _rec(market, pmf, actual, season=2023, week=3, pid="p1"):
    return {"season": season, "week": week, "home": "KC", "player_id": pid, "market": market,
            "pmf": np.asarray(pmf, dtype=np.float32), "actual": float(actual)}


def test_apply_pipeline_pure_a_and_pure_b_and_rescoring():
    a, b = [0.5, 0.3, 0.2], [0.1, 0.2, 0.7]
    r = _rec("receptions", a, 2)
    key = (2023, 3, "p1", "receptions")
    u = pit_uniform(*key)
    out = tpb.apply_pipeline([r], {key: np.array(b)}, {"receptions": 1.0})[0]
    np.testing.assert_allclose(out["pmf"], a, atol=1e-7)
    assert out["rps"] == pytest.approx(rps_pmf(a, 2), abs=1e-6)
    assert out["pit"] == pytest.approx(pit_pmf(a, 2, u), abs=1e-6)
    assert out["pit_pre"] == out["pit"] and out["b_missing"] is False and out["w"] == 1.0
    out = tpb.apply_pipeline([r], {key: np.array(b)}, {"receptions": 0.0})[0]
    np.testing.assert_allclose(out["pmf"], b, atol=1e-12)
    assert out["rps"] == pytest.approx(rps_pmf(b, 2))
    for k in ("season", "week", "home", "player_id", "market", "actual"):
        assert out[k] == r[k]


def test_apply_pipeline_does_not_mutate_inputs():
    import copy
    recs = [_rec("receptions", [0.5, 0.3, 0.2], 2), _rec("anytime_td", [0.7, 0.3], 1)]
    b = {(2023, 3, "p1", "receptions"): np.array([0.1, 0.2, 0.7]),
         (2023, 3, "p1", "anytime_td"): np.array([0.4, 0.6])}
    w = {"receptions": 0.4, "anytime_td": 0.4}
    cal = {"receptions": np.vstack([np.linspace(0, 1, 5), np.linspace(0, 1, 5) ** 2]),
           "anytime_td": (1.0, 0.5)}
    before = copy.deepcopy((recs, b, w, cal))
    tpb.apply_pipeline(recs, b, w, cal)
    for x, y in zip(recs, before[0]):
        assert x.keys() == y.keys()
        for k in x:
            np.testing.assert_array_equal(np.asarray(x[k]), np.asarray(y[k]))
    for k in b:
        np.testing.assert_array_equal(b[k], before[1][k])
    assert w == before[2] and cal["anytime_td"] == before[3]["anytime_td"]
    np.testing.assert_array_equal(cal["receptions"], before[3]["receptions"])


def test_apply_pipeline_missing_b_keeps_a_and_flags_it():
    out = tpb.apply_pipeline([_rec("rec_yds", [0.2, 0.8], 1)], {}, {"rec_yds": 0.3})[0]
    np.testing.assert_allclose(out["pmf"], [0.2, 0.8], atol=1e-7)
    assert out["b_missing"] is True and out["pmf_b"] is None and np.isnan(out["w"])


def test_apply_pipeline_anytime_td_logit_blend_of_one_minus_p0_and_platt():
    key = (2023, 3, "p1", "anytime_td")
    r = _rec("anytime_td", [0.7, 0.3], 1)
    b = np.array([0.4, 0.5, 0.1])  # any shape: P(>=1) = 1 - pmf[0] = 0.6
    out = tpb.apply_pipeline([r], {key: b}, {"anytime_td": 0.5})[0]
    lg = 0.5 * np.log(0.3 / 0.7) + 0.5 * np.log(0.6 / 0.4)
    p = 1 / (1 + np.exp(-lg))
    np.testing.assert_allclose(out["pmf"], [1 - p, p], atol=1e-6)
    assert out["p_pre"] == pytest.approx(p, abs=1e-6)
    assert out["rps"] == pytest.approx((1 - p) ** 2, abs=1e-6)  # Brier
    cal = tpb.apply_pipeline([r], {key: b}, {"anytime_td": 0.5}, {"anytime_td": (1.0, 1.0)})[0]
    assert cal["p_pre"] == pytest.approx(p, abs=1e-6)
    assert cal["pmf"][1] > p                     # Platt b=1 pushes P(TD) up
    assert cal["pit_pre"] == pytest.approx(out["pit"])


def test_apply_pipeline_pit_calibration_changes_post_not_pre():
    rng = np.random.default_rng(1)
    knots = np.vstack([np.linspace(0, 1, 201), np.linspace(0, 1, 201) ** 2])
    r = _rec("rush_att", rng.dirichlet(np.ones(9)), 4)
    plain = tpb.apply_pipeline([r], {}, {"rush_att": 1.0})[0]
    cal = tpb.apply_pipeline([r], {}, {"rush_att": 1.0}, {"rush_att": knots})[0]
    assert cal["pit_pre"] == pytest.approx(plain["pit"])
    assert cal["pit"] != pytest.approx(plain["pit"])
    assert cal["pmf"].sum() == pytest.approx(1.0)


# ---- pure: weights / calibration per season (leakage) -------------------------------------

def _oof_rows(seasons=(2021, 2022, 2023), n=4):
    rng = np.random.default_rng(0)
    rows = []
    for s in seasons:
        for m in MARKETS:
            k = 2 if m == "anytime_td" else KMAX + 1
            for i in range(n):
                a, b = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
                rows.append({"season": s, "week": 1 + i, "player_id": f"p{i}", "market": m,
                             "pmf_a": a, "pmf_b": b, "actual": float(rng.integers(0, k)),
                             "pit_pre": float(rng.random()), "p_pre": float(1 - a[0])})
    return rows


def test_season_weights_first_season_is_pure_a_and_never_sees_its_own_season(monkeypatch):
    seen = []

    def spy(rows, market):
        seen.append((market, sorted({r["season"] for r in rows})))
        return 0.3

    monkeypatch.setattr(tpb, "choose_weight", spy)
    w = tpb.season_weights(_oof_rows(), [2021, 2022, 2023], MARKETS)
    assert w[2021] == {m: 1.0 for m in MARKETS}
    assert w[2022] == {m: 0.3 for m in MARKETS} and w[2023] == {m: 0.3 for m in MARKETS}
    assert sorted(seen) == sorted([(m, [2021]) for m in MARKETS] + [(m, [2021, 2022]) for m in MARKETS])


def test_season_weights_skip_rows_without_b():
    rows = [{**r, "pmf_b": None} if r["season"] == 2021 else r for r in _oof_rows()]
    w = tpb.season_weights(rows, [2021, 2022], MARKETS)
    assert w[2022] == {m: 1.0 for m in MARKETS}  # no prior row with a B pmf


def test_season_calibrations_first_season_identity_and_prior_seasons_only(monkeypatch):
    seen = []
    monkeypatch.setattr(tpb, "fit_pit_map",
                        lambda pits: seen.append(("pit", len(pits))) or np.array([[0, 1], [0, 1.0]]))
    monkeypatch.setattr(tpb, "fit_platt",
                        lambda p, y: seen.append(("platt", len(p))) or (0.9, 0.1))
    c = tpb.season_calibrations(_oof_rows(n=4), [2021, 2022, 2023], MARKETS)
    for m in MARKETS:
        want = IDENTITY_PLATT if m == "anytime_td" else IDENTITY_KNOTS
        np.testing.assert_array_equal(np.asarray(c[2021][m]), np.asarray(want))
    assert c[2022]["anytime_td"] == (0.9, 0.1)
    # 2022 fits on 2021's 4 rows per market, 2023 on 2021+2022's 8
    assert sorted(seen) == sorted([("pit", 4)] * 6 + [("pit", 8)] * 6 + [("platt", 4), ("platt", 8)])


def test_fit_calibration_uses_pre_calibration_pits_and_p():
    rows = _oof_rows(seasons=(2021,), n=400)
    knots = tpb.fit_calibration([r for r in rows if r["market"] == "rec_yds"], "rec_yds")
    assert np.asarray(knots).shape == (2, 201)
    ab = tpb.fit_calibration([r for r in rows if r["market"] == "anytime_td"], "anytime_td")
    assert len(ab) == 2


def _pre_rows(seasons=(2024, 2025), weeks=(1, 2, 3, 4), n=3, seed=0):
    """apply_pipeline-shaped OOF rows (pmf_b None = B missing)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in weeks:
            for m in MARKETS:
                k = 2 if m == "anytime_td" else KMAX + 1
                for i in range(n):
                    a, b = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
                    rows.append({"season": s, "week": w, "home": "KC", "player_id": f"p{i}",
                                 "market": m, "w": 1.0, "pit_pre": 0.5, "p_pre": 0.5,
                                 "actual": float(rng.integers(0, k)), "pmf_a": a,
                                 "pmf_b": None if i == 2 else b})
    return rows


def _markets_cfg(w=1.0, calibrate=True):
    return {m: {"source": "ml", "w_final": w, "calibrate": calibrate} for m in MARKETS}


def test_final_calibration_refits_every_oof_row_at_w_final(monkeypatch):
    rows = _pre_rows()
    seen = {}

    def spy(rs, market):
        seen[market] = list(rs)
        return (0.9, 0.1) if market == "anytime_td" else np.vstack([np.linspace(0, 1, 3)] * 2)

    monkeypatch.setattr(tpb, "fit_calibration", spy)
    cal = tpb.final_calibration(rows, _markets_cfg(w=0.0))  # w_final 0 -> pre pmf = B
    assert set(cal) == set(MARKETS)
    assert cal["anytime_td"] == {"calibrate": True, "kind": "platt", "map": [0.9, 0.1], "n": 24}
    assert cal["rec_yds"]["kind"] == "pit" and cal["rec_yds"]["n"] == 24
    assert cal["rec_yds"]["map"] == np.vstack([np.linspace(0, 1, 3)] * 2).tolist()
    by_key = {(r["season"], r["week"], r["player_id"]): r for r in rows if r["market"] == "rec_yds"}
    assert len(seen["rec_yds"]) == 24
    for r in seen["rec_yds"]:
        o = by_key[(r["season"], r["week"], r["player_id"])]
        pmf = o["pmf_a"] if o["pmf_b"] is None else o["pmf_b"]   # stored pit_pre (0.5) ignored
        want = pit_pmf(pmf, o["actual"], pit_uniform(o["season"], o["week"], o["player_id"],
                                                     "rec_yds"))
        assert r["pit_pre"] == pytest.approx(want, abs=1e-9)


def test_final_calibration_identity_when_not_calibrating_and_guards():
    off = tpb.final_calibration(_pre_rows(), _markets_cfg(calibrate=False))
    assert off["anytime_td"] == {"calibrate": False, "kind": "platt",
                                 "map": list(IDENTITY_PLATT), "n": 0}
    assert off["rush_att"]["map"] == IDENTITY_KNOTS.tolist()
    few = tpb.final_calibration(_pre_rows(n=1), _markets_cfg())   # < 300 PITs / < 100 per class
    assert few["anytime_td"]["map"] == list(IDENTITY_PLATT) and few["anytime_td"]["n"] == 8
    assert few["rush_att"]["map"] == IDENTITY_KNOTS.tolist()


def test_final_weights_skip_rows_without_b(monkeypatch):
    seen = []
    monkeypatch.setattr(tpb, "choose_weight", lambda rs, m: seen.append(list(rs)) or 0.6)
    w = tpb.final_weights(_pre_rows(), MARKETS, blend_kept=True)
    assert w == {m: 0.6 for m in MARKETS}
    assert all(r["pmf_b"] is not None for rs in seen for r in rs)
    assert all(len(rs) == 16 for rs in seen)          # 2 of 3 players have B, 8 weeks
    assert tpb.final_weights(_pre_rows(), MARKETS, blend_kept=False) == {m: 1.0 for m in MARKETS}
    none_b = [{**r, "pmf_b": None} for r in _pre_rows()]
    assert tpb.final_weights(none_b, MARKETS, blend_kept=True) == {m: 1.0 for m in MARKETS}


# ---- run_b_ladder with a stubbed backtest ---------------------------------------------------

def _truth(season, week, g, k, m):
    rng = np.random.default_rng([season, week, g, k, MARKETS.index(m)])
    n = 2 if m == "anytime_td" else KMAX + 1
    return rng.dirichlet(np.ones(n)), int(rng.integers(0, n))


class _FakeBsn:
    SIM_SEED = 42
    MARKET_MAX = {m: KMAX for m in MARKETS if m != "anytime_td"}
    RECORD_MARKETS = MARKETS

    def __init__(self, worse=("pass_tds",)):
        self.worse, self.fetches, self.runs = worse, [], []
        self.src = {"pbp": pd.DataFrame({"season": [2021], "week": [1], "epa": [0.1]})}

    def backtest_fetch_seasons(self, seasons):
        return [min(seasons) - 1] + list(seasons)

    def fetch_backtest_sources(self, fetch_seasons):
        self.fetches.append(list(fetch_seasons))
        return self.src

    def run_backtest(self, seasons, n_sims, *, seed, on_game, spec_hook, record, sources,
                     record_pmf=False, **prod):
        self.runs.append({"hook": spec_hook is not None, "record_pmf": record_pmf,
                          "sources": sources})
        for s in seasons:
            for w in range(1, WEEKS + 1):
                for g in range(GAMES):
                    home = f"H{g}"
                    on_game(s, w, home, f"A{g}", None)
                    for k in range(PLAYERS):
                        pid = f"{home}p{k}"
                        for m in MARKETS:
                            base, actual = _truth(s, w, g, k, m)
                            pmf = base
                            if spec_hook is not None:  # A: sharper around the actual ...
                                n = len(base)
                                tgt = (actual + n // 2) % n if m in self.worse else actual
                                pmf = 0.7 * base + 0.3 * np.eye(n)[tgt]  # ... or away from it
                            rec = {"season": s, "week": w, "home": home, "player_id": pid,
                                   "market": m, "mean": MEANS[m], "p50": 0.0, "p90": 0.0,
                                   "rps": rps_pmf(pmf, actual),
                                   "pit": pit_pmf(pmf, actual, pit_uniform(s, w, pid, m)),
                                   "actual": float(actual)}
                            if record_pmf:
                                rec["pmf"] = np.asarray(pmf, dtype=np.float32)
                            record.append(rec)


class _Models:
    share_fallbacks = 0


# p0: a QB-like runner (pass + rush roles, out of the receiving role);
# p1: a WR (receiving role only). anytime_td has no role subset.
_ROLE = {0: {"position": "QB", "p_y_pass_att_ewm": 30.0, "p_y_carries_ewm": 5.0,
             "p_y_targets_ewm": 0.0},
         1: {"position": "WR", "p_y_pass_att_ewm": 0.0, "p_y_carries_ewm": 0.0,
             "p_y_targets_ewm": 6.0}}
FEATURE_COLS = ["st_questionable", "p_signal", "p_y_pass_att_ewm", "p_y_carries_ewm",
                "p_y_targets_ewm"]


def _in_role(k, m):
    if m in ("pass_yds", "pass_tds", "rush_yds", "rush_att"):
        return k == 0
    if m in ("rec_yds", "receptions"):
        return k == 1
    return True


def _ptbl(seasons):
    rows = [{"player_id": f"H{g}p{k}", "season": s, "week": w, "team": f"H{g}",
             "is_stub": False, "st_questionable": 0.0, "p_signal": 1.0, **_ROLE[k]}
            for s in seasons for w in range(1, WEEKS + 1) for g in range(GAMES)
            for k in range(PLAYERS)]
    return pd.DataFrame(rows)


def _ttbl():
    return pd.DataFrame({"team": ["H0"], "season": [2021], "week": [1], "tm_x": [1.0]})


def _stub_b(monkeypatch, mode):
    """fit_market/predict_pmfs stubs. mode 'bad': point mass away from the
    actual (A always wins -> w = 1.0); 'good': sharper around the actual than A."""
    fits = []
    monkeypatch.setattr(tpb.learned, "fit_models", lambda *a, **k: _Models())

    def fake_fit(df, market, cols, *, upto, test_season, decay, max_iter):
        fits.append({"market": market, "cols": list(cols), "upto": upto, "test_season": test_season,
                     "decay": decay, "max_iter": max_iter})
        return {"market": market, "upto": upto}

    def fake_predict(model, rows, kmax):
        m = model["market"]
        s0, r0 = model["upto"]  # predicted rows lie in the fitted block: season S, r <= week < next r
        nxt = min([r for r in tpb.tpm.REFIT_WEEKS if r > r0], default=99)
        assert all(int(s) == s0 and r0 <= int(w) < nxt for s, w in zip(rows["season"], rows["week"]))
        assert tpb.in_role(rows, m).all()  # B predicts only in-role rows
        out = []
        for s, w, pid in zip(rows["season"], rows["week"], rows["player_id"]):
            n = 2 if m == "anytime_td" else kmax + 1
            truth, actual = _truth(int(s), int(w), int(pid[1]), int(pid[3]), m)
            out.append(np.eye(n)[(actual + n // 2) % n] if mode == "bad"
                       else 0.5 * truth + 0.5 * np.eye(n)[actual])  # sharper than A
        return out

    monkeypatch.setattr(tpb, "fit_market", fake_fit)
    monkeypatch.setattr(tpb, "predict_pmfs", fake_predict)
    return fits


def _run(tmp_path, bsn, seasons=(2021, 2022), env_extra=None, player_tbl=None):
    env = {"PROPS_ML_SEASONS": ",".join(str(s) for s in seasons), "PROPS_ML_N_SIMS": "10",
           **(env_extra or {})}
    return tpb.run_b_ladder(env, bsn=bsn, player_tbl=_ptbl(seasons) if player_tbl is None else player_tbl,
                            team_tbl=_ttbl(),
                            identity={"git_head": "deadbeef",
                                      "player_features": {"size": 1, "sha256": "a" * 64},
                                      "team_features": {"size": 1, "sha256": "b" * 64}},
                            a_gate=A_GATE, data_dir=tmp_path / "data",
                            gate_path=tmp_path / "b_gate.json",
                            pipeline_path=tmp_path / "pipeline.json",
                            report_dir=tmp_path / "reports", log=lambda m: None,
                            run_date="2026-09-25")


def test_ladder_failing_blend_keeps_pipeline_at_a(tmp_path, monkeypatch):
    fits = _stub_b(monkeypatch, "bad")
    bsn = _FakeBsn()
    gate = _run(tmp_path, bsn)

    # one fetch; two runs (baseline, kept A), both recording pmfs, sharing the sources
    assert bsn.fetches == [[2020, 2021, 2022]]
    assert [r["hook"] for r in bsn.runs] == [False, True]
    assert all(r["record_pmf"] and r["sources"] is bsn.src for r in bsn.runs)

    # B: 7 markets per (season, block with games), unweighted, fixed max_iter, kept-A columns
    assert {(f["test_season"], f["upto"]) for f in fits} == {(s, (s, r)) for s in (2021, 2022)
                                                              for r in (1, 5)}
    assert len(fits) == 2 * 2 * 7
    assert all(f["decay"] == 1.0 and f["max_iter"] == 150 for f in fits)
    assert all(f["cols"] == FEATURE_COLS for f in fits)

    # 2021 is pure A with identity calibration; weights exist for every market
    assert gate["blend"]["weights"]["2021"] == {m: 1.0 for m in MARKETS}
    assert gate["calibration"]["maps"]["2021"] == {m: "identity" for m in MARKETS}
    assert gate["blend"]["weights"]["2022"] == {m: 1.0 for m in MARKETS}  # bad B: A wins

    # A re-check: pass_tds is worse under A -> starts with source baseline
    assert gate["a_recheck"]["sources"]["pass_tds"] == "baseline"
    assert gate["a_recheck"]["sources"]["rec_yds"] == "ml"

    # blend candidate == A -> rung fails; the kept pipeline stays at A
    assert gate["blend"]["pass"] is False
    assert "blend" not in gate["kept_rungs"]
    pipe = json.loads((tmp_path / "pipeline__s2021-2022__every4.json").read_text())
    assert set(pipe) == {"kept_a_toggles", "tuned", "markets", "data_end"}
    assert pipe["data_end"] == [2022, WEEKS]
    cal = json.loads((tmp_path / "calibration__s2021-2022__every4.json").read_text())
    assert cal["data_end"] == [2022, WEEKS] and set(cal["markets"]) == set(MARKETS)
    assert all(v["calibrate"] is False and v["n"] == 0 for v in cal["markets"].values())
    assert pipe["kept_a_toggles"] == ["volume", "efficiency", "context", "market"]
    assert pipe["tuned"] == {"2021": [1.0, 300], "2022": [1.0, 150]}
    assert set(pipe["markets"]) == set(MARKETS)
    for v in pipe["markets"].values():
        assert set(v) == {"source", "w_final", "calibrate"}
        assert v["w_final"] == 1.0 and v["calibrate"] is False
    assert pipe["markets"]["pass_tds"]["source"] == "baseline"
    # final source rule: ml iff RPS <= baseline AND ECE <= baseline + 0.005
    pm = gate["final"]["ml_per_market"]
    for m in MARKETS:
        ok = pm[m]["rps_c"] <= pm[m]["rps_b"] and pm[m]["ece_c"] <= pm[m]["ece_b"] + 0.005
        assert pipe["markets"][m]["source"] == ("ml" if ok else "baseline"), m
    assert pm["rec_yds"]["rps_c"] < pm["rec_yds"]["rps_b"]
    assert {v["source"] for v in pipe["markets"].values()} == {"ml", "baseline"}

    # OOF (pre-calibration) = pure A: its PIT is the A record's PIT
    oof = pd.read_parquet(tmp_path / "data" / "oof_b7__s2021-2022__every4.parquet")
    assert {"season", "week", "home", "player_id", "market", "pit_pre", "p_pre", "actual",
            "pmf_a", "pmf_b", "w"} <= set(oof.columns)
    assert len(oof) == 2 * WEEKS * GAMES * PLAYERS * 7
    a_pit = {(r["season"], r["week"], r["player_id"], r["market"]): r["pit"]
             for r in pd.read_parquet(tmp_path / "data" /
                                      "records_kept_a__s2021-2022__every4__b7.parquet")
             .to_dict("records")}
    for r in oof.to_dict("records"):
        assert r["pit_pre"] == pytest.approx(a_pit[(r["season"], r["week"], r["player_id"],
                                                    r["market"])], abs=1e-5)

    md = (tmp_path / "reports" / "2026-09-25-props-ml-b-gate__s2021-2022__every4.md").read_text()

    # role subsets: out-of-role records are B := A, never "missing"
    assert gate["b_oof"]["missing"] == {m: 0 for m in MARKETS}
    assert gate["b_oof"]["out_of_role"]["pass_yds"] == 2 * WEEKS * GAMES  # every p1 row
    assert gate["b_oof"]["out_of_role"]["anytime_td"] == 0
    assert set(oof.loc[oof.market == "rec_yds", "in_role"]) == {True, False}

    # final decisions: selected (per-market sources) and unselected, side by side
    un = gate["final"]["unselected"]
    assert set(un) >= {"all", "season_2025", "pass", "sources"}
    assert un["sources"] == gate["a_recheck"]["sources"]
    assert "optimistic by construction" in md

    # outputs + tagged checkpoints (never colliding with Props-1's)
    assert (tmp_path / "b_gate__s2021-2022__every4.json").exists()
    assert md.split("\n\n")[1].startswith("**Verdict:")
    assert "anytime_td" in md and "Brier" in md and "not an independent holdout" in md
    names = {p.name for p in (tmp_path / "data").glob("*.parquet")}
    assert {"records_baseline__s2021-2022__every4__b7.parquet",
            "records_kept_a__s2021-2022__every4__b7.parquet",
            "b_oof__s2021-2022__every4__b7.parquet"} <= names
    meta = json.loads((tmp_path / "data" /
                       "records_kept_a__s2021-2022__every4__b7.meta.json").read_text())["meta"]
    assert meta["record_pmf"] is True and meta["identity"]["seed"] == 42
    assert meta["identity"]["git_head"] == "deadbeef" and "pbp" in meta["identity"]["backtest_sources"]
    assert meta["identity"]["prod"] == tpb.tpm.PROD
    b_meta = json.loads((tmp_path / "data" /
                         "b_oof__s2021-2022__every4__b7.meta.json").read_text())["meta"]
    assert b_meta["role_subsets"]["pass_yds"] == {"positions": ["QB"], "col": "p_y_pass_att_ewm",
                                                  "min": 10.0}


def test_ladder_resume_reuses_all_checkpoints(tmp_path, monkeypatch):
    _stub_b(monkeypatch, "bad")
    first = _run(tmp_path, _FakeBsn())
    fits = _stub_b(monkeypatch, "bad")
    bsn = _FakeBsn()
    again = _run(tmp_path, bsn, env_extra={"PROPS_ML_RESUME": "1"})
    assert bsn.runs == [] and fits == []
    assert again["final"]["all"]["skill"] == first["final"]["all"]["skill"]


def test_passing_blend_is_kept_and_feeds_calibration_and_w_final(tmp_path, monkeypatch):
    _stub_b(monkeypatch, "good")
    real_step = tpb.rung_step

    def forced(name, *a, **k):
        step = real_step(name, *a, **k)
        if name == "blend":
            step["pass"] = True
        return step

    monkeypatch.setattr(tpb, "rung_step", forced)
    cal_rows = []
    real_cal = tpb.season_calibrations
    monkeypatch.setattr(tpb, "season_calibrations",
                        lambda rows, *a: cal_rows.extend(rows) or real_cal(rows, *a))
    gate = _run(tmp_path, _FakeBsn())
    assert "blend" in gate["kept_rungs"]
    w22 = gate["blend"]["weights"]["2022"]
    assert any(w < 1.0 for w in w22.values())          # good B earns weight
    # calibration fits on the BLENDED pre-calibration records
    assert {r["w"] for r in cal_rows if r["season"] == 2022} != {1.0}
    pipe = json.loads((tmp_path / "pipeline__s2021-2022__every4.json").read_text())
    assert any(v["w_final"] < 1.0 for v in pipe["markets"].values())
    cal = json.loads((tmp_path / "calibration__s2021-2022__every4.json").read_text())
    assert cal["data_end"] == pipe["data_end"] == [2022, WEEKS]
    for m, v in cal["markets"].items():   # calibrate follows the calibration rung
        assert v["calibrate"] is pipe["markets"][m]["calibrate"]
        assert v["n"] == (2 * WEEKS * GAMES * PLAYERS if v["calibrate"] else 0)
    # an out-of-role population record's blended pmf is its A pmf (B := A)
    out = [r for r in cal_rows if r["season"] == 2022 and r["market"] == "rec_yds"
           and r["player_id"].endswith("p0")]
    inr = [r for r in cal_rows if r["season"] == 2022 and r["market"] == "rec_yds"
           and r["player_id"].endswith("p1")]
    assert out and inr and w22["rec_yds"] < 1.0
    for r in out:
        np.testing.assert_allclose(r["pmf"], np.asarray(r["pmf_a"], dtype=float), atol=1e-6)
    assert any(not np.allclose(r["pmf"], np.asarray(r["pmf_a"], dtype=float), atol=1e-3)
               for r in inr)


def test_ladder_aborts_when_b_features_missing_for_many_records(tmp_path, monkeypatch):
    _stub_b(monkeypatch, "bad")
    ptbl = _ptbl((2021, 2022))
    ptbl = ptbl[ptbl["player_id"] != "H0p0"]           # 1 of 6 players: ~17% missing
    with pytest.raises(RuntimeError, match=r"B feature rows missing.*rec_yds"):
        _run(tmp_path, _FakeBsn(), player_tbl=ptbl)


def test_missing_guard_is_per_market_and_counts_only_in_role(tmp_path, monkeypatch):
    _stub_b(monkeypatch, "bad")
    seasons = (2021, 2022, 2023)
    ptbl = _ptbl(seasons)
    # rec_yds: 54 in-role records (p1) + 54 out-of-role (p0, never "missing").
    # Drop ONE p1 feature row: 1 of 54 = 1.9% < 2% -> runs. (A record without a
    # feature row cannot be placed in a role, so it counts as missing in each
    # of its markets.)
    keep = ~((ptbl.player_id == "H0p1") & (ptbl.season == 2022) & (ptbl.week == 6))
    gate = _run(tmp_path, _FakeBsn(), seasons=seasons, player_tbl=ptbl[keep])
    assert gate["b_oof"]["missing"]["rec_yds"] == 1
    assert gate["b_oof"]["out_of_role"]["rec_yds"] == 54
    # two such rows (2 of 54 = 3.7% for rec_yds) abort, naming the market
    keep2 = keep & ~((ptbl.player_id == "H1p1") & (ptbl.season == 2022) & (ptbl.week == 6))
    with pytest.raises(RuntimeError, match=r"B feature rows missing.*rec_yds"):
        _run(tmp_path / "b", _FakeBsn(), seasons=seasons, player_tbl=ptbl[keep2])


def test_kept_blend_with_a_missing_b_row_chooses_w_final_without_it(tmp_path, monkeypatch):
    _stub_b(monkeypatch, "good")
    real_step = tpb.rung_step

    def forced(name, *a, **k):
        step = real_step(name, *a, **k)
        if name == "blend":
            step["pass"] = True
        return step

    monkeypatch.setattr(tpb, "rung_step", forced)
    seasons = (2021, 2022, 2023)
    ptbl = _ptbl(seasons)
    keep = ~((ptbl.player_id == "H0p1") & (ptbl.season == 2022) & (ptbl.week == 6))
    gate = _run(tmp_path, _FakeBsn(), seasons=seasons, player_tbl=ptbl[keep])
    assert gate["b_oof"]["missing"]["rec_yds"] == 1 and "blend" in gate["kept_rungs"]
    assert any(v["w_final"] < 1.0 for v in gate["pipeline"]["markets"].values())
