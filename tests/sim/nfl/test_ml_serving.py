"""Tests for the props-ML serving pipeline (``sim.nfl.ml_serving``):
load_artifacts -> build_ml_spec (A) -> sim -> B -> blend -> calibrate -> map.

Tiny REAL artifacts: A and B are fit on a synthetic table with a small
``max_iter`` and written with the real ``save_artifacts`` / ``build_config``
of ``scripts/fit_props_ml_final.py``. No real data, no network.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn.base

from sportsmodel.model.props_ml.blend import blend_binary, blend_pmf
from sportsmodel.model.props_ml.dist_models import fit_market, in_role, predict_pmfs
from sportsmodel.model.props_ml.pit_calibration import apply_pit_map, apply_platt, fit_pit_map
from sportsmodel.sim.nfl import learned, ml_serving
from sportsmodel.sim.nfl.aggregate import nfl_player_prop_dists
from sportsmodel.sim.nfl.kernel import simulate_game
from sportsmodel.sim.nfl.spec import NflGameSims, NflGameSpec, PlayerInput, TeamRates

_p = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "fit_props_ml_final.py"
_spec = importlib.util.spec_from_file_location("fit_props_ml_final_srv", _p)
fpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpf)

N_SIMS = 4000
KEPT = frozenset({"volume", "efficiency", "context", "market"})
MARKET_MAX = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 200, "receptions": 15,
              "pass_tds": 6, "rush_att": 40}
B_MARKETS = ("rec_yds", "receptions", "anytime_td")
BASELINE = ("pass_yds", "pass_tds")
ML = ("rush_yds", "rush_att", "rec_yds", "receptions", "anytime_td")
TEAMS = ("HOM", "AWY", "NYA", "NYB")
# (suffix, position, targets rate, carries rate); "5" is a fringe WR: targets EWM
# 1.0 < 2 -> out of the rec_yds / receptions role subsets.
ROSTER = (("1", "QB", 1, 3), ("2", "WR", 7, 0.2), ("3", "RB", 3, 14), ("4", "TE", 4, 0.1),
          ("5", "WR", 1, 0.1))
UPCOMING = (2025, 9)
IDENT = {"git_head": "deadbeef", "player_features": None, "team_features": None}


def _tables():
    rng = np.random.default_rng(11)
    rows, trows = [], []
    for s in (2023, 2024, 2025):
        for w in list(range(1, 9)) + ([9] if s == 2025 else []):
            played = w <= 8
            for t in TEAMS:
                trows.append({"team": t, "season": s, "week": w, "opponent": "X",
                              "tm_pass_rate": rng.random(), "op_def": rng.random(),
                              "cx_temp": rng.random(), "mk_total": 40 + 10 * rng.random(),
                              "y_team_pass_att": float(rng.poisson(34)) if played else np.nan,
                              "y_team_rush_att": float(rng.poisson(26)) if played else np.nan})
                for suf, pos, lt, lc in ROSTER:
                    tg, ca = float(rng.poisson(lt)), float(rng.poisson(lc))
                    rec = float(rng.binomial(int(tg), 0.65))
                    pa = float(rng.poisson(33)) if pos == "QB" else 0.0
                    lab = {"y_targets": tg, "y_carries": ca, "y_receptions": rec,
                           "y_rec_yds": rec * 10 + float(rng.integers(0, 9)),
                           "y_rush_yds": ca * 4 + float(rng.integers(0, 5)), "y_pass_att": pa,
                           "y_pass_yds": pa * 7,
                           "y_pass_tds": float(rng.poisson(1.5)) if pa else 0.0,
                           "y_anytime_td": float(rng.random() < 0.3)}
                    rows.append({"player_id": f"{t}{suf}", "season": s, "week": w, "team": t,
                                 "opponent": "X", "position": pos, "is_stub": False,
                                 "st_questionable": 1.0 if f"{t}{suf}" == "HOM2" else 0.0,
                                 "p_y_targets_ewm": lt + (0.0 if suf == "5" else rng.random()),
                                 "p_y_carries_ewm": lc + rng.random(),
                                 "p_y_pass_att_ewm": 33.0 if pos == "QB" else 0.0,
                                 "p_target_share_ewm": lt / 16 + 0.01 * rng.random(),
                                 "p_carry_share_ewm": lc / 17.5 + 0.01 * rng.random(),
                                 "cx_temp": rng.random(), "mk_total": 40 + 10 * rng.random(),
                                 **{c: (v if played else np.nan) for c, v in lab.items()}})
    return pd.DataFrame(rows), pd.DataFrame(trows)


def _pipeline():
    return {"kept_a_toggles": sorted(KEPT), "tuned": {"2025": [0.8, 5]},
            "markets": {m: {"source": "baseline" if m in BASELINE else "ml",
                            "w_final": 0.5 if m in B_MARKETS else 1.0, "calibrate": True}
                        for m in (*BASELINE, *ML)},
            "data_end": [2025, 4]}


KNOTS = fit_pit_map(np.random.default_rng(3).beta(0.5, 0.5, 600))  # overconfident -> widen
PLATT = (0.8, 0.3)


def _calibration(pipe):
    cal = {"data_end": pipe["data_end"],
           "markets": fpf.tpb.final_calibration([], pipe["markets"])}
    cal["markets"]["rec_yds"]["map"] = KNOTS.tolist()
    cal["markets"]["anytime_td"]["map"] = list(PLATT)
    return cal


def _save(out, p, lm, b, pipe=None, cal=None):
    pipe = pipe or _pipeline()
    cfg = fpf.build_config(pipe, learned_models=lm, b_models=b,
                           a_fit={"season": 2025, "decay": 0.8, "max_iter": 5},
                           trained_through=(2025, 8), market_max=MARKET_MAX,
                           role_cols=fpf.role_columns(p, b), identity=IDENT,
                           created_at="2026-09-25T00:00:00+00:00")
    fpf.save_artifacts(out, lm, b, cal or _calibration(pipe), cfg)
    return out


def _rates():
    return TeamRates(drive_outcomes={"td": 0.2, "fg": 0.15, "punt": 0.4, "turnover": 0.12,
                                     "downs": 0.05, "end": 0.08},
                     pass_rate=0.58, drives_per_game=11.0, rz_td_rate=0.55, pass_att_pg=34.0,
                     rush_att_pg=26.0, sack_rate=0.06, completion_pct=0.64, pass_td_share=0.6)


def _players(team):
    return [PlayerInput(player_id=f"{team}{suf}", name=f"{team}{suf}", pos=pos,
                        target_share=0.2, carry_share=0.2, ypt=7.0, ypc=4.0, ypr=10.5,
                        catch_rate=0.66, td_share=0.1, rec_td_share=0.2, rush_td_share=0.2)
            for suf, pos, _, _ in ROSTER]


def _game_spec():
    return NflGameSpec(home_team="HOM", away_team="AWY", home=_rates(), away=_rates(),
                       home_players=_players("HOM"), away_players=_players("AWY"))


def _copy(sims):
    return NflGameSims(sims.home_score, sims.away_score,
                       {pid: dict(s) for pid, s in sims.player_stats.items()})


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """Fitted + saved + loaded artifacts, the upcoming rows, the ML spec and its sims."""
    p, t = _tables()
    lm = learned.fit_models(p, t, KEPT, upto=(2025, 9), test_season=2025, decay=0.8, max_iter=5)
    cols = learned.feature_columns(p, KEPT)
    b = {m: fit_market(p, m, cols, upto=(2025, 9), test_season=2025, decay=1.0, max_iter=3)
         for m in B_MARKETS}
    out = _save(tmp_path_factory.mktemp("art") / "models", p, lm, b)
    art = ml_serving.load_artifacts(out, p, t)
    assert art is not None
    teams = ("HOM", "AWY")
    rows = p[(p.season == UPCOMING[0]) & (p.week == UPCOMING[1]) & p.team.isin(teams)]
    trows = t[(t.season == UPCOMING[0]) & (t.week == UPCOMING[1]) & t.team.isin(teams)]
    spec = ml_serving.build_ml_spec(_game_spec(), art, rows, trows)
    sims = simulate_game(spec, N_SIMS, np.random.default_rng(5))
    return {"p": p, "t": t, "lm": lm, "b": b, "dir": out, "art": art, "rows": rows,
            "trows": trows, "spec": spec, "sims": sims}


def _expected_targets(served, sims, rows):
    """The final pmfs, computed independently of ml_serving."""
    art = served["art"]
    a = nfl_player_prop_dists(sims, MARKET_MAX)
    feats = rows.drop_duplicates("player_id").set_index("player_id", drop=False)
    out = {}
    for pid in sorted(a):
        for m in ML:
            pa = np.asarray(a[pid][m]["pmf"], dtype=float)
            pb = None
            if m in B_MARKETS and pid in feats.index:
                row = feats.loc[[pid]]
                if bool(in_role(row, m).iloc[0]):
                    kmax = 1 if m == "anytime_td" else MARKET_MAX[m]
                    pb = predict_pmfs(art.b_models[m], row, kmax)[0]
            if m == "anytime_td":
                p1 = pa[1] if pb is None else blend_binary(pa[1], pb[1], 0.5)
                q = apply_platt(p1, PLATT)
                out[(pid, m)] = np.array([1 - q, q])
                continue
            pre = pa if pb is None else blend_pmf(pa, pb, 0.5)
            out[(pid, m)] = apply_pit_map(pre, KNOTS) if m == "rec_yds" else pre
    return out


def _tv(x, y):
    n = max(len(x), len(y))
    return 0.5 * float(np.abs(np.pad(np.asarray(x, float), (0, n - len(x)))
                              - np.pad(np.asarray(y, float), (0, n - len(y)))).sum())


# ---- load_artifacts ------------------------------------------------------------------------

def test_load_artifacts_returns_the_saved_models(served):
    art = served["art"]
    assert set(art.b_models) == set(B_MARKETS)
    assert art.config["markets"]["pass_yds"]["source"] == "baseline"
    assert art.config["market_max"]["anytime_td"] == 1


def test_load_artifacts_none_with_a_reason_for_a_missing_or_empty_dir(tmp_path, capsys):
    assert ml_serving.load_artifacts(tmp_path / "nope") is None
    assert "props-ML" in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    assert ml_serving.load_artifacts(tmp_path / "empty") is None
    assert "props_ml_config.json" in capsys.readouterr().out


def test_load_artifacts_none_when_feature_columns_do_not_match_the_table(served, capsys):
    p, t = served["p"], served["t"]
    assert ml_serving.load_artifacts(served["dir"], p.drop(columns=["mk_total"]), t) is None
    assert "mk_total" in capsys.readouterr().out
    assert ml_serving.load_artifacts(served["dir"], p, t.drop(columns=["tm_pass_rate"])) is None
    assert "tm_pass_rate" in capsys.readouterr().out


def test_load_artifacts_none_for_incompatible_or_unloadable_files(served, tmp_path, capsys):
    p, lm, b = served["p"], served["lm"], served["b"]
    out = _save(tmp_path / "fmt", p, lm, b)
    cfg = json.loads((out / "props_ml_config.json").read_text())
    (out / "props_ml_config.json").write_text(json.dumps({**cfg, "format_version": 99}))
    assert ml_serving.load_artifacts(out) is None
    assert "format_version" in capsys.readouterr().out

    out = _save(tmp_path / "corrupt", p, lm, b)
    (out / "b_rec_yds.joblib").write_bytes(b"not a pickle")
    assert ml_serving.load_artifacts(out) is None
    assert "b_rec_yds.joblib: cannot load" in capsys.readouterr().out

    out = _save(tmp_path / "missing", p, lm, b)
    (out / "learned.joblib").unlink()
    assert ml_serving.load_artifacts(out) is None
    assert "learned.joblib" in capsys.readouterr().out


def test_load_artifacts_none_for_models_pickled_under_another_sklearn(served, tmp_path, capsys,
                                                                      monkeypatch):
    out = _save(tmp_path / "ver", served["p"], served["lm"], served["b"])
    monkeypatch.setattr(sklearn.base, "__version__", "0.0.1")
    joblib.dump(served["b"]["receptions"], out / "b_receptions.joblib")
    monkeypatch.undo()
    assert ml_serving.load_artifacts(out) is None
    assert "0.0.1" in capsys.readouterr().out


# ---- build_ml_spec -------------------------------------------------------------------------

def test_build_ml_spec_applies_learned_a_with_questionable_and_q_weight(served, monkeypatch):
    calls = []
    real = learned.apply_to_spec

    def spy(spec, models, player_rows, team_rows, questionable, q_weight):
        calls.append((models, questionable, q_weight))
        return real(spec, models, player_rows, team_rows, questionable, q_weight)

    monkeypatch.setattr(learned, "apply_to_spec", spy)
    base = _game_spec()
    got = ml_serving.build_ml_spec(base, served["art"], served["rows"], served["trows"])
    assert calls == [(served["art"].learned, {"HOM2"}, 1.0)]
    assert got.home.pass_att_pg != base.home.pass_att_pg          # learned volume applied
    assert got == served["spec"]


# ---- ml_player_dists -----------------------------------------------------------------------

def test_dists_omit_baseline_markets_and_have_the_market_support(served):
    sims = _copy(served["sims"])
    dists = ml_serving.ml_player_dists(served["spec"], sims, served["rows"], served["trows"],
                                       served["art"], np.random.default_rng(0))
    assert set(dists) == {p.player_id for p in (*served["spec"].home_players,
                                                 *served["spec"].away_players)}
    for pid, md in dists.items():
        assert set(md) == set(ML), pid
        for m, d in md.items():
            assert d["kind"] == "pmf"
            assert len(d["pmf"]) == (2 if m == "anytime_td" else MARKET_MAX[m] + 1)
            assert abs(sum(d["pmf"]) - 1.0) < 1e-9 and min(d["pmf"]) >= 0.0
            assert np.isfinite(d["mean"])


def test_mapped_sims_carry_the_returned_dists_and_they_match_the_final_pmfs(served):
    sims = _copy(served["sims"])
    orig = served["sims"]
    want = _expected_targets(served, orig, served["rows"])
    dists = ml_serving.ml_player_dists(served["spec"], sims, served["rows"], served["trows"],
                                       served["art"], np.random.default_rng(1))
    marg = nfl_player_prop_dists(sims, MARKET_MAX)
    moved = 0
    for (pid, m), target in want.items():
        assert dists[pid][m] == marg[pid][m]                       # sims' marginal exactly
        assert _tv(dists[pid][m]["pmf"], target) < 0.03, (pid, m)  # = the final pmf
        moved += _tv(nfl_player_prop_dists(orig, MARKET_MAX)[pid][m]["pmf"], target) > 0.05
    assert moved > 0                                               # the pipeline did something
    # baseline-market stats and the scores are the unmapped sim's
    for pid in orig.player_stats:
        for stat in ("pass_yds", "pass_tds"):
            assert sims.player_stats[pid][stat] is orig.player_stats[pid][stat]
    assert sims.home_score is orig.home_score
    # correlation kept: a QB's pass yards still move with his top WR's receiving yards
    hq, hw = sims.player_stats["HOM1"]["pass_yds"], sims.player_stats["HOM2"]["rec_yds"]
    assert np.corrcoef(hq, hw)[0, 1] > 0.2


def test_b_is_predicted_only_for_b_markets_and_in_role_players(served, monkeypatch):
    calls = []
    real = ml_serving.predict_pmfs

    def spy(model, rows, kmax):
        calls.append((model.market, sorted(rows["player_id"]), kmax))
        return real(model, rows, kmax)

    monkeypatch.setattr(ml_serving, "predict_pmfs", spy)
    ml_serving.ml_player_dists(served["spec"], _copy(served["sims"]), served["rows"],
                               served["trows"], served["art"], np.random.default_rng(0))
    got = {m: (pids, k) for m, pids, k in calls}
    assert set(got) == set(B_MARKETS)
    rec_role = ["AWY2", "AWY3", "AWY4", "HOM2", "HOM3", "HOM4"]   # no QBs, no fringe WR
    assert got["rec_yds"] == (rec_role, 200) and got["receptions"] == (rec_role, 15)
    assert got["anytime_td"] == (sorted(served["rows"]["player_id"]), 1)


def test_a_player_without_a_feature_row_gets_a_for_b(served):
    rows = served["rows"][served["rows"]["player_id"] != "HOM2"]
    sims = _copy(served["sims"])
    dists = ml_serving.ml_player_dists(served["spec"], sims, rows, served["trows"],
                                       served["art"], np.random.default_rng(2))
    want = _expected_targets(served, served["sims"], rows)
    for m in ML:
        assert _tv(dists["HOM2"][m]["pmf"], want[("HOM2", m)]) < 0.03


def test_deterministic_for_a_fixed_rng_and_input_arrays_untouched(served):
    before = {p: {m: a.copy() for m, a in s.items()}
              for p, s in served["sims"].player_stats.items()}
    runs = []
    for _ in range(2):
        sims = _copy(served["sims"])
        d = ml_serving.ml_player_dists(served["spec"], sims, served["rows"], served["trows"],
                                       served["art"], np.random.default_rng(7))
        runs.append((d, sims))
    assert runs[0][0] == runs[1][0]
    for pid, s in runs[0][1].player_stats.items():
        for m, a in s.items():
            np.testing.assert_array_equal(a, runs[1][1].player_stats[pid][m])
    for pid, s in before.items():
        for m, a in s.items():
            np.testing.assert_array_equal(served["sims"].player_stats[pid][m], a)


def test_no_ml_markets_leaves_the_sims_alone(served, tmp_path):
    pipe = _pipeline()
    for v in pipe["markets"].values():
        v.update(source="baseline", w_final=1.0)
    out = _save(tmp_path / "base", served["p"], served["lm"], {}, pipe=pipe)
    art = ml_serving.load_artifacts(out)
    sims = _copy(served["sims"])
    assert ml_serving.ml_player_dists(served["spec"], sims, served["rows"], served["trows"], art,
                                      np.random.default_rng(0)) == {}
    for pid, s in sims.player_stats.items():
        for m, a in s.items():
            assert a is served["sims"].player_stats[pid][m]


def test_dataclass_spec_not_mutated(served):
    spec = served["spec"]
    snap = dataclasses.asdict(spec)
    ml_serving.ml_player_dists(spec, _copy(served["sims"]), served["rows"], served["trows"],
                               served["art"], np.random.default_rng(0))
    assert dataclasses.asdict(spec) == snap



# ---- I4: gated population only ----------------------------------------------------------------

def test_ml_player_dists_maps_only_the_gated_player_markets(served):
    orig = served["sims"]
    sims = _copy(orig)
    gate = {("HOM2", "rec_yds"), ("HOM2", "receptions"), ("AWY3", "rush_yds")}
    dists = ml_serving.ml_player_dists(served["spec"], sims, served["rows"], served["trows"],
                                       served["art"], np.random.default_rng(3), gate=gate)
    assert {(pid, m) for pid, md in dists.items() for m in md} == gate
    # every ungated player's stats are the unmapped sims' (same array objects)
    for pid in orig.player_stats:
        if pid in ("HOM2", "AWY3"):
            continue
        for stat, arr in orig.player_stats[pid].items():
            assert sims.player_stats[pid][stat] is arr, (pid, stat)
    # the gated targets are the same pipeline pmfs as without a gate
    want = _expected_targets(served, orig, served["rows"])
    for pid, m in gate:
        assert _tv(dists[pid][m]["pmf"], want[(pid, m)]) < 0.03, (pid, m)


def test_ml_player_dists_empty_gate_maps_nothing(served):
    sims = _copy(served["sims"])
    assert ml_serving.ml_player_dists(served["spec"], sims, served["rows"], served["trows"],
                                      served["art"], np.random.default_rng(0), gate=set()) == {}
    for pid, s in sims.player_stats.items():
        for m, a in s.items():
            assert a is served["sims"].player_stats[pid][m]
