"""Tests for scripts/gate_ml_game_lines.py (walk-forward ML-sim vs Elo game gate).

The backtest module and the Elo walk-forward are stubbed; no network, no DB.
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.shrink import ShrinkParams
from sportsmodel.sim.engine import GameSims

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gate_ml_game_lines.py"
_spec = importlib.util.spec_from_file_location("gate_ml_game_lines", _p)
gml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gml)

NAN = float("nan")


def _sims(home, away):
    return GameSims(home_score=np.asarray(home, dtype=int), away_score=np.asarray(away, dtype=int),
                    batter_stats={}, pitcher_stats={})


def _game(**kw):
    base = {"away": "NYJ", "spread_line": 3.5, "total_line": 44.5,
            "actual_margin": 7.0, "actual_total": 41.0}
    return {**base, **kw}


# ---- per-game ML record ---------------------------------------------------------------

def test_ml_record_uses_engine_pmfs_with_negated_offset():
    # home always wins by 10 (20-10): with the wrong offset sign the margin
    # would read -10 and the home win prob would be 0.
    rec = gml.ml_record(2025, 3, "BUF", _sims([20] * 4, [10] * 4), _game(spread_line=9.5))
    assert rec["win_prob"] == 1.0
    assert rec["cover_prob"] == 1.0          # margin 10 > 9.5
    assert rec["over_prob"] == 0.0           # total 30 < 44.5
    assert rec["pred_margin"] == 10.0 and rec["pred_total"] == 30.0
    assert (rec["season"], rec["week"], rec["home"], rec["away"]) == (2025, 3, "BUF", "NYJ")
    assert rec["actual_margin"] == 7.0 and rec["actual_total"] == 41.0
    assert rec["spread_line"] == 9.5 and rec["total_line"] == 44.5


def test_ml_record_away_win_push_and_mixture():
    # sims: margins -3, -3, +3, +7 ; line -3 (away favored by 3): pushes excluded
    rec = gml.ml_record(2025, 1, "BUF", _sims([17, 17, 20, 24], [20, 20, 17, 17]),
                        _game(spread_line=-3.0, total_line=37.0))
    assert rec["win_prob"] == pytest.approx(0.5)
    assert rec["cover_prob"] == pytest.approx(1.0)   # 2 pushes dropped, 2 of 2 above -3
    assert rec["over_prob"] == pytest.approx(1.0)    # totals 37,37,37,41: 3 pushes dropped, 1 over


def test_ml_record_wide_margins_and_high_totals_are_not_clipped():
    # engine defaults (half_range 25, max_total 30) would clip these onto the end
    # bins and flip the answers; the gate builds wider pmfs.
    rec = gml.ml_record(2025, 1, "BUF", _sims([45] * 3, [10] * 3),
                        _game(spread_line=28.5, total_line=54.5))
    assert rec["cover_prob"] == 1.0   # margin 35 > 28.5
    assert rec["over_prob"] == 1.0    # total 55 > 54.5
    assert gml.TOTAL_MAX >= 120


def test_ml_record_missing_line_gives_none():
    rec = gml.ml_record(2025, 1, "BUF", _sims([20], [10]), _game(spread_line=NAN, total_line=NAN))
    assert rec["cover_prob"] is None and rec["over_prob"] is None
    assert math.isnan(rec["spread_line"]) and math.isnan(rec["total_line"])


# ---- schedule / config helpers -----------------------------------------------------------

def _row(season, week, home, away, hs, as_, sl=3.0, tl=44.0, gt="REG"):
    return {"season": season, "week": week, "game_type": gt, "home_team": home,
            "away_team": away, "home_score": hs, "away_score": as_,
            "spread_line": sl, "total_line": tl}


def test_schedule_games_reg_scored_test_seasons_normalized():
    sched = pd.DataFrame([
        _row(2025, 1, "LAR", "SF", 24, 20, sl=-1.5, tl=NAN),
        _row(2025, 2, "SF", "LA", NAN, NAN),                 # unscored
        _row(2025, 19, "KC", "BUF", 30, 27, gt="WC"),        # postseason
        _row(2024, 1, "KC", "BUF", 30, 27),                  # not a test season
    ])
    games = gml.schedule_games(sched, [2025])
    assert list(games) == [(2025, 1, "LA")]
    g = games[(2025, 1, "LA")]
    assert g["away"] == "SF" and g["spread_line"] == -1.5 and math.isnan(g["total_line"])
    assert g["actual_margin"] == 4.0 and g["actual_total"] == 44.0


def test_a_config_reads_kept_toggles_and_tuned_values():
    a_gate = {"kept": ["market", "volume"], "tuned": {"2024": [1.0, 150], "2025": [0.8, 300]}}
    toggles, tuned = gml.a_config(a_gate, [2025])
    assert toggles == frozenset({"market", "volume"})
    assert tuned == {2025: (0.8, 300)}
    with pytest.raises(ValueError, match="2023"):
        gml.a_config(a_gate, [2023])
    with pytest.raises(ValueError, match="kept"):
        gml.a_config({"kept": [], "tuned": {"2025": [0.8, 300]}}, [2025])


def test_elo_history_uses_committed_before_fetch_window_and_fetched_inside():
    history = pd.DataFrame([_row(2023, 1, "KC", "BUF", 1, 0), _row(2024, 1, "KC", "BUF", 1, 0, sl=99.0),
                            _row(2026, 1, "KC", "BUF", 1, 0),
                            _row(2023, 19, "KC", "BUF", 1, 0, gt="DIV")])
    fetched = pd.DataFrame([_row(2024, 1, "KC", "BUF", 1, 0, sl=3.0), _row(2025, 1, "KC", "BUF", 1, 0)])
    out = gml.elo_history(history, fetched, [2024, 2025])
    assert list(zip(out["season"], out["week"])) == [(2023, 1), (2024, 1), (2025, 1)]
    assert out.loc[out["season"] == 2024, "spread_line"].tolist() == [3.0]
    assert set(out["game_type"]) == {"REG"}


def test_elo_records_normal_probs_from_gameline_sigmas():
    gl = GameLineConfig(sigma_margin=10.0, sigma_total=12.0)
    preds = [
        {"season": 2025, "week": 2, "home_team": "LAR", "away_team": "SF", "spread_line": 3.0,
         "total_line": None, "pred_margin": 3.0, "pred_total": 40.0, "win_prob": 0.6,
         "actual_margin": 7.0, "actual_total": 41.0},
        {"season": 2024, "week": 2, "home_team": "KC", "away_team": "SF", "spread_line": 3.0,
         "total_line": 40.0, "pred_margin": 3.0, "pred_total": 40.0, "win_prob": 0.6,
         "actual_margin": 7.0, "actual_total": 41.0},
    ]
    recs = gml.elo_records(preds, gl, [2025])
    assert len(recs) == 1
    r = recs[0]
    assert (r["season"], r["week"], r["home"], r["away"]) == (2025, 2, "LA", "SF")
    assert r["win_prob"] == 0.6 and r["cover_prob"] == pytest.approx(0.5)
    assert r["over_prob"] is None and math.isnan(r["total_line"])
    assert r["pred_margin"] == 3.0 and r["actual_total"] == 41.0


def test_served_gl_cfg_turns_off_market_shrink_only():
    gl = GameLineConfig(sigma_margin=13.0, sigma_total=14.0, w_margin=ShrinkParams(0.9, 0.3, 0.1),
                        w_total=ShrinkParams(0.9, 0.3, 0.1), bias_margin=0.5)
    s = gml.served_gl_cfg(gl)
    assert s.w_margin == ShrinkParams(0.0, 0.0, 0.0) and s.w_total == ShrinkParams(0.0, 0.0, 0.0)
    assert (s.sigma_margin, s.sigma_total, s.bias_margin) == (13.0, 14.0, 0.5)


def _rec(week, home, sl=3.0, tl=44.0, am=7.0, at=41.0):
    return {"season": 2025, "week": week, "home": home, "away": "NYJ", "spread_line": sl,
            "total_line": tl, "actual_margin": am, "actual_total": at}


def test_check_lines_accepts_equal_and_nan_lines():
    ml = [_rec(1, "BUF"), _rec(2, "KC", sl=NAN), _rec(3, "SF")]
    elo = [_rec(1, "BUF"), _rec(2, "KC", sl=None)]
    assert gml.check_lines(ml, elo) == 2


@pytest.mark.parametrize("field,value", [("spread_line", 3.5), ("total_line", NAN),
                                         ("actual_margin", 6.0), ("away", "MIA")])
def test_check_lines_aborts_on_mismatch(field, value):
    ml = [_rec(1, "BUF")]
    elo = [{**_rec(1, "BUF"), field: value}]
    with pytest.raises(RuntimeError, match="BUF"):
        gml.check_lines(ml, elo)


def test_coverage_counts_and_missing_games():
    keys = {(2025, 1, "BUF"), (2025, 1, "KC"), (2025, 2, "SF")}
    ml = [_rec(1, "BUF"), _rec(1, "KC")]
    elo = [_rec(1, "BUF"), _rec(2, "SF")]
    cov = gml.coverage(keys, ml, elo)
    assert cov["schedule_games"] == 3 and cov["ml_games"] == 2 and cov["elo_games"] == 2
    assert cov["paired_games"] == 1
    assert cov["ml_missing"] == [[2025, 2, "SF"]] and cov["elo_missing"] == [[2025, 1, "KC"]]


def test_output_paths_default_and_tagged(tmp_path):
    g, r = gml.output_paths(gml.DEFAULT_TAG, "2026-09-28", gate_path=tmp_path / "game_gate.json",
                            report_dir=tmp_path / "rep")
    assert g == tmp_path / "game_gate.json" and r == tmp_path / "rep" / "2026-09-28-ml-game-gate.md"
    g, r = gml.output_paths("s2025__every4", "2026-09-28", gate_path=tmp_path / "game_gate.json",
                            report_dir=tmp_path / "rep")
    assert g.name == "game_gate__s2025__every4.json"
    assert r.name == "2026-09-28-ml-game-gate__s2025__every4.md"


def test_gate_json_writes_nan_as_null():
    txt = gml.gate_json({"a": NAN, "b": np.float64(1.5), "c": [np.int64(2), None], "d": np.bool_(True)})
    assert json.loads(txt) == {"a": None, "b": 1.5, "c": [2, None], "d": True}


# ---- run_gate with a stubbed backtest + Elo walk -------------------------------------------

_TEAMS = ["BUF", "KC", "SF", "LA", "DAL", "PHI", "DET", "GB"]


def _fetched_schedule(lines_at_actual=False):
    """12 completed 2025 REG games. ``lines_at_actual``: closing lines sit 0.5
    below the actual margin/total (so a comparator shrunk onto the line is
    nearly exact and never pushes)."""
    rows = [_row(2024, 1, "KC", "BUF", 20, 17)]
    for w in range(1, 7):
        for g in range(2):
            home, away = _TEAMS[(w + 2 * g) % 8], _TEAMS[(w + 2 * g + 1) % 8]
            hs, as_ = (24, 17) if (w + g) % 2 else (13, 20)
            sl = NAN if (w, g) == (2, 0) else (3.5 if (w + g) % 2 else -2.5)
            tl = 41.5
            if lines_at_actual:
                sl, tl = hs - as_ - 0.5, hs + as_ - 0.5
            rows.append(_row(2025, w, home, away, hs, as_, sl=sl, tl=tl))
    rows.append(_row(2025, 19, "KC", "BUF", 30, 27, gt="WC"))
    return pd.DataFrame(rows)


_ML_SKIP = (2025, 3, "LA")    # the fake sim "fails" this game (run_backtest skips it)
_ELO_SKIP = (2025, 4, "DAL")  # the stub Elo walk drops this one


class _FakeBsn:
    SIM_SEED = 42

    def __init__(self, perfect=True, margin_shift=0, lines_at_actual=False):
        self.perfect, self.margin_shift = perfect, margin_shift
        self.fetches, self.runs = [], []
        self.src = {"schedules": _fetched_schedule(lines_at_actual),
                    "pbp": pd.DataFrame({"season": [2025], "week": [1], "epa": [0.1]})}

    def backtest_fetch_seasons(self, seasons):
        return [min(seasons) - 1] + list(seasons)

    def fetch_backtest_sources(self, fetch_seasons):
        self.fetches.append(list(fetch_seasons))
        return self.src

    def run_backtest(self, seasons, n_sims, *, seed, on_game, spec_hook, sources, **prod):
        self.runs.append({"seasons": list(seasons), "n_sims": n_sims, "seed": seed,
                          "hook": spec_hook is not None, "sources": sources, "prod": prod})
        s = sources["schedules"]
        s = s[s["season"].isin(seasons) & (s["game_type"] == "REG")]
        for r in s.itertuples(index=False):
            key = (int(r.season), int(r.week), r.home_team)
            try:
                if spec_hook is not None:
                    spec_hook(key[0], key[1], r.home_team, r.away_team, object())
                if key == _ML_SKIP:
                    raise ValueError("no QB")
            except Exception as exc:  # noqa: BLE001 -- mirrors run_backtest's skip
                print(f"skipping {key}: {exc}")
                continue
            hs, as_ = (int(r.home_score), int(r.away_score)) if self.perfect else (10, 30)
            # margin_shift: sim margin off by that many points, total unchanged
            hs, as_ = hs + self.margin_shift // 2, as_ - self.margin_shift // 2
            on_game(key[0], key[1], r.home_team, r.away_team, _sims([hs] * 5, [as_] * 5))


class _Models:
    share_fallbacks = 0


def _elo_walk_stub(calls, served_bad=False):
    """Stub of _raw_model_predictions. ``served_bad``: the model-only margin is
    the negated actual and the model-only total 30 points high."""
    def walk(sched):
        calls.append(sched.copy())
        out = []
        for r in sched.itertuples(index=False):
            if pd.isna(r.home_score) or (int(r.season), int(r.week), r.home_team) == _ELO_SKIP:
                continue
            am, at = float(r.home_score - r.away_score), float(r.home_score + r.away_score)
            mm, mt = (-am, at + 30.0) if served_bad else (1.0, 44.0)
            out.append({"season": int(r.season), "week": int(r.week), "home_team": r.home_team,
                        "away_team": r.away_team, "model_margin": mm, "model_total": mt,
                        "spread_line": None if pd.isna(r.spread_line) else float(r.spread_line),
                        "total_line": None if pd.isna(r.total_line) else float(r.total_line),
                        "actual_margin": float(r.home_score - r.away_score),
                        "actual_total": float(r.home_score + r.away_score)})
        return out
    return walk


_A_GATE = {"kept": ["context", "efficiency", "market", "volume"],
           "tuned": {"2025": [0.8, 300]},
           "identity": {"player_features": {"size": 10, "sha256": "a" * 64},
                        "team_features": {"size": 5, "sha256": "b" * 64}}}
_IDENTITY = {"git_head": "deadbeef", "player_features": {"size": 10, "sha256": "a" * 64},
             "team_features": {"size": 5, "sha256": "b" * 64}}


def _kw(tmp_path, walk_calls, **over):
    history = pd.DataFrame([_row(2023, 1, "KC", "BUF", 20, 17), _row(2024, 1, "KC", "BUF", 1, 0)])
    kw = dict(player_tbl=pd.DataFrame({"season": [2025], "week": [1], "team": ["BUF"],
                                       "player_id": ["p"], "st_questionable": [0]}),
              team_tbl=pd.DataFrame({"season": [2025], "week": [1], "team": ["BUF"]}),
              identity=dict(_IDENTITY), a_gate=_A_GATE, history_sched=history,
              elo_walk=_elo_walk_stub(walk_calls), gl_cfg=GameLineConfig(),
              elo_config={"rating": {"k": 16}, "gameline": {"sigma_margin": 13.2}},
              data_dir=tmp_path / "data", gate_path=tmp_path / "game_gate.json",
              report_dir=tmp_path / "reports", log=lambda m: None, run_date="2026-09-28")
    kw.update(over)
    return kw


def test_run_gate_end_to_end(tmp_path, monkeypatch):
    fits = []
    monkeypatch.setattr(gml.learned, "fit_models",
                        lambda p, t, toggles, **k: fits.append((toggles, k)) or _Models())
    monkeypatch.setattr(gml.learned, "apply_to_spec", lambda spec, *a: spec)
    bsn, walk_calls, logs = _FakeBsn(), [], []
    env = {"PROPS_ML_SEASONS": "2025", "PROPS_ML_N_SIMS": "10"}
    gate = gml.run_gate(env, bsn=bsn, **_kw(tmp_path, walk_calls, log=logs.append))

    # one fetch reused; one ML backtest with the hook, production settings, n_sims from env
    assert bsn.fetches == [[2024, 2025]]
    assert len(bsn.runs) == 1 and bsn.runs[0]["hook"] and bsn.runs[0]["n_sims"] == 10
    assert bsn.runs[0]["sources"] is bsn.src and bsn.runs[0]["prod"] == gml.tpm.PROD
    # A models: kept toggles, tuned values, one fit per refit block strictly before it
    assert {t for t, _ in fits} == {frozenset(_A_GATE["kept"])}
    assert sorted(k["upto"] for _, k in fits) == [(2025, r) for r in gml.tpm.REFIT_WEEKS]
    assert all(k["test_season"] == 2025 and k["decay"] == 0.8 and k["max_iter"] == 300
               for _, k in fits)
    # Elo ran on committed history before the fetch window + the fetched schedule
    sched = walk_calls[0]
    assert sorted(set(zip(sched["season"], sched["week"])))[:2] == [(2023, 1), (2024, 1)]
    assert sched.loc[sched["season"] == 2024, "home_score"].tolist() == [20]  # fetched, not history
    # coverage: 12 scheduled games; each side misses one; both excluded from pairing
    cov = gate["coverage"]
    assert (cov["schedule_games"], cov["ml_games"], cov["elo_games"], cov["paired_games"]) == (12, 11, 11, 10)
    assert cov["ml_missing"] == [list(_ML_SKIP)] and cov["elo_missing"] == [list(_ELO_SKIP)]
    m = gate["decision"]["metrics"]
    assert m["win_brier"]["n"] == 10 and m["cover_brier"]["n"] == 9  # one paired game lacks a spread
    # a perfect ML sim beats the Elo stub on every metric
    assert gate["pass"] is True and gate["decision"]["pass"] is True
    assert m["margin_mae"]["ml"] == 0.0 and m["margin_mae"]["elo"] > 0
    # verdict comparator = Elo as served (model-only): pred margin is the stub's
    # unshrunk 1.0 vs actual +-7 -> |error| 6 or 8, five of each among paired games
    assert m["margin_mae"]["elo"] == pytest.approx(7.0)
    assert "elo_served" not in gate
    assert gate["elo_closing_shrink"]["metrics"]["win_brier"]["n"] == 10
    assert gate["elo_closing_shrink"]["metrics"]["margin_mae"]["elo"] != pytest.approx(7.0)
    # standalone Elo metrics are the served ones too (11 Elo games: six +7, five -7)
    assert gate["elo_metrics"]["margin_mae"] == pytest.approx((6 * 6 + 5 * 8) / 11)
    assert gate["toggles"] == sorted(_A_GATE["kept"]) and gate["tuned"] == {"2025": [0.8, 300]}
    assert gate["identity"]["git_head"] == "deadbeef" and "schedules" in gate["identity"]["backtest_sources"]
    assert gate["identity"]["a_gate_features_match"] is True

    out = tmp_path / "game_gate__s2025__every4.json"
    assert json.loads(out.read_text())["pass"] is True
    report = (tmp_path / "reports" / "2026-09-28-ml-game-gate__s2025__every4.md").read_text()
    first_para = report.split("\n\n")[1]
    assert first_para.startswith("**Verdict: PASS.**")
    for needle in ("in-sample", "win_brier", "cover_brier", "over_brier", "margin_mae", "total_mae",
                   "Coverage", "deadbeef", "a" * 12, "2025 wk3 LA", "2025 wk4 DAL"):
        assert needle in report, needle
    assert (tmp_path / "data" / "game_records__s2025__every4.parquet").exists()
    # observable progress lines: stage, season, block
    assert any("stage=fit" in l and "season=2025" in l and "block=5" in l for l in logs)
    assert any("stage=backtest" in l and "block=5" in l for l in logs)


def test_verdict_follows_the_served_comparator_when_the_two_disagree(tmp_path, monkeypatch):
    # Closing lines sit 0.5 below the actuals and the shrink curve is w = 1, so the
    # closing-line-shrink Elo is nearly exact (margin MAE 0.5) and beats the ML
    # sim's margin (always 2 points off). The served (model-only) Elo is awful.
    monkeypatch.setattr(gml.learned, "fit_models", lambda *a, **k: _Models())
    monkeypatch.setattr(gml.learned, "apply_to_spec", lambda spec, *a: spec)
    full = ShrinkParams(1.0, 1.0, 0.0)
    gate = gml.run_gate({"PROPS_ML_SEASONS": "2025"},
                        bsn=_FakeBsn(margin_shift=2, lines_at_actual=True),
                        **_kw(tmp_path, [], elo_walk=_elo_walk_stub([], served_bad=True),
                              gl_cfg=GameLineConfig(w_margin=full, w_total=full)))
    shrink = gate["elo_closing_shrink"]
    assert shrink["pass"] is False and any(r.startswith("margin_mae") for r in shrink["reasons"])
    assert shrink["metrics"]["margin_mae"]["elo"] == pytest.approx(0.5)
    assert gate["decision"]["pass"] is True and gate["pass"] is True
    assert gate["decision"]["metrics"]["margin_mae"]["elo"] == pytest.approx(14.0)  # -actual vs +-7
    assert gate["decision"]["metrics"]["margin_mae"]["ml"] == pytest.approx(2.0)
    report = (tmp_path / "reports" / "2026-09-28-ml-game-gate__s2025__every4.md").read_text()
    assert report.split("\n\n")[1].startswith("**Verdict: PASS.**")
    info = report.split("## Informational: Elo with closing-line shrink")[1]
    assert "| margin_mae |" in info and "FAIL" in info.split("## Coverage")[0]


def test_verdict_flags_feature_files_that_differ_from_a_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(gml.learned, "fit_models", lambda *a, **k: _Models())
    monkeypatch.setattr(gml.learned, "apply_to_spec", lambda spec, *a: spec)
    other = {**_A_GATE, "identity": {"player_features": {"size": 1, "sha256": "c" * 64},
                                     "team_features": {"size": 5, "sha256": "b" * 64}}}
    gate = gml.run_gate({"PROPS_ML_SEASONS": "2025"}, bsn=_FakeBsn(),
                        **_kw(tmp_path, [], a_gate=other))
    assert gate["identity"]["a_gate_features_match"] is False
    report = (tmp_path / "reports" / "2026-09-28-ml-game-gate__s2025__every4.md").read_text()
    assert "a_gate_features_match=False" in report.split("\n\n")[1]


def test_run_gate_fail_verdict_names_failing_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(gml.learned, "fit_models", lambda *a, **k: _Models())
    monkeypatch.setattr(gml.learned, "apply_to_spec", lambda spec, *a: spec)
    gate = gml.run_gate({"PROPS_ML_SEASONS": "2025"}, bsn=_FakeBsn(perfect=False),
                        **_kw(tmp_path, []))
    assert gate["pass"] is False and gate["decision"]["reasons"]
    report = (tmp_path / "reports" / "2026-09-28-ml-game-gate__s2025__every4.md").read_text()
    assert report.split("\n\n")[1].startswith("**Verdict: FAIL.**")
    assert "win_brier" in report.split("\n\n")[1]


def test_run_gate_resume_reuses_matching_checkpoint(tmp_path, monkeypatch):
    fits = []
    monkeypatch.setattr(gml.learned, "fit_models", lambda *a, **k: fits.append(1) or _Models())
    monkeypatch.setattr(gml.learned, "apply_to_spec", lambda spec, *a: spec)
    env = {"PROPS_ML_SEASONS": "2025", "PROPS_ML_N_SIMS": "10"}
    first = gml.run_gate(env, bsn=_FakeBsn(), **_kw(tmp_path, []))
    n_fits = len(fits)
    bsn2 = _FakeBsn()
    again = gml.run_gate({**env, "PROPS_ML_RESUME": "1"}, bsn=bsn2, **_kw(tmp_path, []))
    assert bsn2.runs == [] and len(fits) == n_fits          # no backtest, no refits
    assert again["decision"] == first["decision"] and again["coverage"] == first["coverage"]
    # a different identity (e.g. new code) does not match: the run executes again
    bsn3 = _FakeBsn()
    gml.run_gate({**env, "PROPS_ML_RESUME": "1"}, bsn=bsn3,
                 **_kw(tmp_path, [], identity={**_IDENTITY, "git_head": "cafef00d"}))
    assert len(bsn3.runs) == 1


def test_run_gate_aborts_when_the_hook_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(gml.learned, "fit_models", lambda *a, **k: _Models())

    def boom(spec, *a):
        raise KeyError("feature row")

    monkeypatch.setattr(gml.learned, "apply_to_spec", boom)
    with pytest.raises(RuntimeError, match="spec_hook raised"):
        gml.run_gate({"PROPS_ML_SEASONS": "2025"}, bsn=_FakeBsn(), **_kw(tmp_path, []))
    assert not (tmp_path / "game_gate__s2025__every4.json").exists()
