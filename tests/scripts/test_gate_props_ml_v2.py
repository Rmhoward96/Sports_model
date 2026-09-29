"""Tests for scripts/gate_props_ml_v2.py (v2-vs-v1 props-ML gate: served composites,
QB-change and two-way matchup sub-checks, game level). Stubbed backtest and model
fits; tiny synthetic tables -- no network, no DB, nothing under data/."""
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel.model import game_gate, props_eval
from sportsmodel.model.props_ml import v2_checks as vc
from sportsmodel.sim.engine import GameSims

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gate_props_ml_v2.py"
_spec = importlib.util.spec_from_file_location("gate_props_ml_v2", _p)
gv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gv2)


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    real = props_eval.cluster_bootstrap
    monkeypatch.setattr(props_eval, "cluster_bootstrap",
                        lambda df, stat, n_boot=1000, seed=0: real(df, stat, n_boot=40, seed=seed))


# ---- verdict (spec §6 ship rule) -----------------------------------------------------------

MARKETS = ("pass_yds", "rush_yds", "rec_yds")


def _paired(improve_by_season):
    """Paired v1 (b) / v2 (c) frame; v2's RPS = v1's x (1 - improve[season])."""
    rows = []
    for s, imp in improve_by_season.items():
        for w in range(1, 13):
            for k in range(4):
                for m in MARKETS:
                    b = 10.0 + k + (w % 3)
                    pit = ((w * 7 + k * 3 + MARKETS.index(m)) % 20) / 20 + 0.025
                    rows.append({"season": s, "week": w, "home": f"H{k % 2}",
                                 "player_id": f"p{k}", "market": m, "rps_b": b,
                                 "rps_c": b * (1 - imp), "pit_b": pit, "pit_c": pit,
                                 "cluster": f"{s}-{w}-H{k % 2}"})
    return pd.DataFrame(rows)


def _game_diffs(v2_worse=0.0):
    recs = []
    for s in (2024, 2025):
        for w in range(1, 9):
            recs.append({"season": s, "week": w, "home": "KC", "away": "BUF",
                         "win_prob": 0.6, "cover_prob": 0.5, "over_prob": 0.5,
                         "pred_margin": 3.0, "pred_total": 44.0, "actual_margin": 7.0,
                         "actual_total": 41.0 + w, "spread_line": 2.5, "total_line": 44.5})
    v2 = [{**r, "pred_margin": r["pred_margin"] - v2_worse} for r in recs]  # margin MAE 4 -> 4 + x
    return game_gate.paired_diffs(v2, recs)


QB_OK = {"n": 50, "diff": -0.4, "lo": -0.7, "hi": -0.1, "pass": True}
MATCHUP_OK = {"pass_top": True, "pass_bottom": True, "corr_v1": 0.1, "corr_v2": 0.6,
              "pass": True}
ECE = {m: 0.02 for m in MARKETS}


def test_verdict_passes_when_every_part_passes():
    v = vc.verdict(_paired({2024: 0.05, 2025: 0.05}), ECE, ECE, _game_diffs(), QB_OK, MATCHUP_OK)
    assert v["pass"] is True and v["reasons"] == []
    assert v["pooled"]["pass"] and v["season_2025"]["pass"]
    assert set(v) >= {"pooled", "season_2025", "ece", "game", "qb_change", "matchup", "pass",
                      "reasons"}


def test_verdict_fails_when_2025_alone_is_worse_even_if_pooled_is_better():
    df = _paired({2024: 0.20, 2025: -0.02})
    pooled = props_eval.rung_decision(df)
    assert pooled["pass"] is True                     # pooled: v2 clearly better
    v = vc.verdict(df, ECE, ECE, _game_diffs(), QB_OK, MATCHUP_OK)
    assert v["pooled"]["pass"] is True and v["season_2025"]["pass"] is False
    assert v["pass"] is False and any("2025" in r for r in v["reasons"])


def test_verdict_fails_without_2025_records():
    v = vc.verdict(_paired({2024: 0.05}), ECE, ECE, _game_diffs(), QB_OK, MATCHUP_OK)
    assert v["season_2025"] is None and v["pass"] is False


def test_verdict_ece_tolerance_per_market():
    worse = {**ECE, "rec_yds": 0.02 + 0.0051}
    v = vc.verdict(_paired({2024: 0.05, 2025: 0.05}), ECE, worse, _game_diffs(), QB_OK,
                   MATCHUP_OK)
    assert v["pass"] is False and any("rec_yds" in r and "ECE" in r for r in v["reasons"])
    ok = {**ECE, "rec_yds": 0.02 + 0.0049}
    assert vc.verdict(_paired({2024: 0.05, 2025: 0.05}), ECE, ok, _game_diffs(), QB_OK,
                      MATCHUP_OK)["pass"] is True


def test_verdict_game_level_within_half_a_percent():
    v = vc.verdict(_paired({2024: 0.05, 2025: 0.05}), ECE, ECE, _game_diffs(v2_worse=1.0), QB_OK,
                   MATCHUP_OK)
    assert v["game"]["metrics"]["margin_mae"]["pass"] is False
    assert v["pass"] is False and any("margin_mae" in r for r in v["reasons"])
    tiny = _game_diffs(v2_worse=0.01)                 # MAE +0.1 % -> within +0.5 %
    assert vc.verdict(_paired({2024: 0.05, 2025: 0.05}), ECE, ECE, tiny, QB_OK,
                      MATCHUP_OK)["game"]["pass"] is True


def test_verdict_needs_both_required_sub_checks():
    good = _paired({2024: 0.05, 2025: 0.05})
    v = vc.verdict(good, ECE, ECE, _game_diffs(), {**QB_OK, "pass": False}, MATCHUP_OK)
    assert v["pass"] is False and any("QB-change" in r for r in v["reasons"])
    v = vc.verdict(good, ECE, ECE, _game_diffs(), QB_OK, {**MATCHUP_OK, "pass": False})
    assert v["pass"] is False and any("matchup" in r for r in v["reasons"])


# ---- run_gate_v2 end to end (stubbed backtest) ---------------------------------------------

GROUPS = {"RB": ("rush_yds", "y_rush_yds", 100.0), "QB": ("pass_yds", "y_pass_yds", 250.0),
          "WR": ("rec_yds", "y_rec_yds", 80.0), "TE": ("rec_yds", "y_rec_yds", 50.0)}
TOP = {"RB": -0.20, "QB": 0.10, "WR": 0.12, "TE": 0.08}
ACT = {g: [-v, -v / 2, 0.0, v / 2, v] for g, v in TOP.items()}
MX = [-2.0, -1.0, 0.0, 1.0, 2.0]
SEASONS = (2024, 2025)
WEEKS = 6


def _player_tbl():
    rows = []
    for i in range(5):
        for pos, (market, label, base) in GROUPS.items():
            pid = f"T{i}{pos}"
            for w in range(1, 9):   # 2023 history: baseline = base
                rows.append({"player_id": pid, "season": 2023, "week": w, "team": f"T{i}",
                             "opponent": "X", "position": pos, "is_stub": False, label: base,
                             "mx_pass_minus_rush": np.nan, "qb_changed": 0, "qb_ratio_ypa": 1.0})
            for s in SEASONS:
                for w in range(1, WEEKS + 1):
                    j = (i + w) % 5
                    rows.append({"player_id": pid, "season": s, "week": w, "team": f"T{i}",
                                 "opponent": f"D{j}", "position": pos, "is_stub": False,
                                 "mx_pass_minus_rush": MX[j],
                                 "qb_changed": int(i == 0 and w in (2, 3)),   # T0's backup starts
                                 "qb_ratio_ypa": 1.0})
    # 2026 spot check: PHI QB (Keenum) at CHI
    rows.append({"player_id": "KEENUM", "season": 2026, "week": 4, "team": "PHI",
                 "opponent": "CHI", "position": "QB", "is_stub": False, "mx_pass_minus_rush": 0.3,
                 "qb_changed": 1, "qb_ratio_ypa": 0.9})
    t = pd.DataFrame(rows)
    t["st_questionable"] = 0.0
    return t


def _served(version):
    rows = []
    for i in range(5):
        for pos, (market, label, base) in GROUPS.items():
            for s in SEASONS:
                for w in range(1, WEEKS + 1):
                    j = (i + w) % 5
                    act = base * (1 + ACT[pos][j])
                    dev = 0.0 if version == "v1" else 0.75 * ACT[pos][j]
                    rps1 = 5.0 + (w % 3) + i
                    qb_boost = 0.7 if (i == 0 and w in (2, 3) and market == "pass_yds") else 0.95
                    rows.append({"season": s, "week": w, "home": f"T{i}", "player_id": f"T{i}{pos}",
                                 "market": market, "mean": base * (1 + dev),
                                 "rps": rps1 if version == "v1" else rps1 * qb_boost,
                                 "pit": ((w * 7 + i * 3 + len(pos)) % 20) / 20 + 0.025,
                                 "actual": act})
    if version == "v2":
        rows.append({"season": 2026, "week": 4, "home": "CHI", "player_id": "KEENUM",
                     "market": "pass_yds", "mean": 231.0, "rps": 20.0, "pit": 0.4,
                     "actual": 260.0})
    else:
        rows.append({"season": 2026, "week": 4, "home": "CHI", "player_id": "KEENUM",
                     "market": "pass_yds", "mean": 262.0, "rps": 22.0, "pit": 0.6,
                     "actual": 260.0})
    return pd.DataFrame(rows)


def _row(season, week, home, away, hs, as_, sl=2.5, tl=44.5, gt="REG"):
    return {"season": season, "week": week, "game_type": gt, "home_team": home,
            "away_team": away, "home_score": hs, "away_score": as_, "spread_line": sl,
            "total_line": tl}


def _sims(home, away):
    return GameSims(home_score=np.asarray(home, dtype=int), away_score=np.asarray(away, dtype=int),
                    batter_stats={}, pitcher_stats={})


class _FakeBsn:
    SIM_SEED = 42

    def __init__(self):
        self.fetches, self.runs = [], []
        rows = [_row(s, w, "KC", "BUF", 24, 17 + w) for s in (2023,) + SEASONS
                for w in range(1, 5)]
        self.src = {"schedules": pd.DataFrame(rows),
                    "pbp": pd.DataFrame({"season": [2024], "week": [1], "epa": [0.1]})}

    def backtest_fetch_seasons(self, seasons):
        return [min(seasons) - 1] + list(seasons)

    def fetch_backtest_sources(self, fetch_seasons):
        self.fetches.append(list(fetch_seasons))
        return self.src

    def run_backtest(self, seasons, n_sims, *, seed, on_game, spec_hook, sources, **prod):
        self.runs.append({"seasons": list(seasons), "hook": spec_hook is not None})
        s = sources["schedules"]
        for r in s[s["season"].isin(seasons)].itertuples(index=False):
            spec_hook(int(r.season), int(r.week), r.home_team, r.away_team, object())
            on_game(int(r.season), int(r.week), r.home_team, r.away_team,
                    _sims([24, 21, 27], [20, 17 + int(r.week), 24]))


class _Models:
    share_fallbacks = 0


A_V1 = {"kept": ["context", "efficiency", "market", "volume"],
        "tuned": {str(s): [1.0, 150] for s in range(2021, 2026)}}
A_V2 = {"kept": ["context", "efficiency", "market", "matchup", "qb_profile", "volume"],
        "tuned": {str(s): [0.8, 300] for s in range(2021, 2026)}, "gate_name": "_v2"}
IDENTITY = {"git_head": "deadbeef", "player_features": {"size": 1, "sha256": "a" * 64},
            "team_features": {"size": 1, "sha256": "b" * 64}}


def _kw(tmp_path, **over):
    kw = dict(bsn=_FakeBsn(), player_tbl=_player_tbl(), team_tbl=pd.DataFrame({"season": [2024], "week": [1], "team": ["KC"]}),
              identity=dict(IDENTITY), a_gate_v1=A_V1, a_gate_v2=A_V2,
              served_v1=_served("v1"), served_v2=_served("v2"), data_dir=tmp_path / "data",
              gate_path=tmp_path / "v2_gate.json", report_dir=tmp_path / "reports",
              log=lambda m: None, run_date="2026-09-28")
    kw.update(over)
    return kw


def _stub_models(monkeypatch):
    fits = []
    monkeypatch.setattr(gv2.gml.learned, "fit_models",
                        lambda p, t, toggles, **k: fits.append((frozenset(toggles), k)) or _Models())
    monkeypatch.setattr(gv2.gml.learned, "apply_to_spec", lambda spec, *a: spec)
    return fits


def test_run_gate_v2_end_to_end(tmp_path, monkeypatch):
    fits = _stub_models(monkeypatch)
    kw = _kw(tmp_path)
    gate = gv2.run_gate_v2({"PROPS_ML_N_SIMS": "10"}, **kw)

    # two game-level backtests on the verdict seasons: v1's kept A, then v2's
    assert gate["seasons"] == [2024, 2025]
    assert kw["bsn"].fetches == [[2023, 2024, 2025]]
    assert [r["hook"] for r in kw["bsn"].runs] == [True, True]
    assert {t for t, _ in fits} == {frozenset(A_V1["kept"]), frozenset(A_V2["kept"])}
    assert {(k["decay"], k["max_iter"]) for t, k in fits if t == frozenset(A_V2["kept"])} == {
        (0.8, 300)}
    names = {p.name for p in (tmp_path / "data").glob("*.parquet")}
    assert names == {"game_records__s2024-2025__every4.parquet",
                     "game_records__s2024-2025__every4_v2.parquet"}

    # identical game sims -> game level equal -> within tolerance
    assert gate["game"]["pass"] is True
    # served composites: verdict seasons only; v2 5 % better, QB-change games 30 % better
    assert gate["n_paired"] == 2 * 5 * 4 * WEEKS
    assert gate["verdict"]["pooled"]["pass"] and gate["verdict"]["season_2025"]["pass"]
    qb = gate["qb_change"]
    assert qb["n"] == 2 * 2 and qb["pass"] is True     # T0's QB, weeks 2-3, two seasons
    assert gate["matchup"]["pass_top"] and gate["matchup"]["pass_bottom"]
    assert gate["matchup"]["pass"] is True
    assert gate["pass"] is True and gate["verdict"]["pass"] is True

    # outputs
    out = json.loads((tmp_path / "v2_gate.json").read_text())
    assert out["pass"] is True and len(out["matchup"]["table"]) == 20
    md = (tmp_path / "reports" / "2026-09-28-matchup-qb-gate.md").read_text()
    assert md.split("\n\n")[1].startswith("**Verdict: PASS.**")
    for needle in ("QB-change", "Matchup response", "| RB | 5 |", "Keenum", "2025 alone",
                   "win_brier", "margin_mae", "deadbeef"):
        assert needle in md, needle
    spot = gate["spot_checks"][0]
    assert spot["graded"] is True and spot["player_id"] == "KEENUM"
    assert spot["mean_v1"] == 262.0 and spot["mean_v2"] == 231.0 and spot["actual"] == 260.0


def test_run_gate_v2_fails_when_2025_alone_is_worse(tmp_path, monkeypatch):
    _stub_models(monkeypatch)
    v2 = _served("v2")
    v2.loc[v2.season == 2025, "rps"] = _served("v1").loc[v2.season == 2025, "rps"] * 1.03
    v2.loc[v2.season == 2024, "rps"] = _served("v1").loc[v2.season == 2024, "rps"] * 0.7
    gate = gv2.run_gate_v2({}, **_kw(tmp_path, served_v2=v2))
    assert gate["verdict"]["pooled"]["pass"] is True
    assert gate["verdict"]["season_2025"]["pass"] is False and gate["pass"] is False
    md = (tmp_path / "reports" / "2026-09-28-matchup-qb-gate.md").read_text()
    assert md.split("\n\n")[1].startswith("**Verdict: FAIL.**")


def test_run_gate_v2_guards_gate_names_and_served_keys(tmp_path, monkeypatch):
    _stub_models(monkeypatch)
    with pytest.raises(ValueError, match="gate_name"):
        gv2.run_gate_v2({}, **_kw(tmp_path, a_gate_v2=A_V1))
    with pytest.raises(ValueError, match="gate_name"):
        gv2.run_gate_v2({}, **_kw(tmp_path, a_gate_v1=A_V2))
    with pytest.raises(ValueError, match="key"):
        gv2.run_gate_v2({}, **_kw(tmp_path, served_v2=_served("v2").iloc[3:]))


def test_run_gate_v2_resume_reuses_game_checkpoints(tmp_path, monkeypatch):
    _stub_models(monkeypatch)
    first = gv2.run_gate_v2({}, **_kw(tmp_path))
    kw = _kw(tmp_path)
    again = gv2.run_gate_v2({"PROPS_ML_RESUME": "1"}, **kw)
    assert kw["bsn"].runs == []
    assert again["game"] == first["game"]


def test_paths_and_constants(tmp_path):
    assert gv2.VERDICT_SEASONS == (2024, 2025) and gv2.FINAL_SEASON == 2025
    assert gv2.SERVED_V1 == gv2.tpm.DATA_DIR / "served__s2021-2025__every4.parquet"
    assert gv2.SERVED_V2 == gv2.tpm.DATA_DIR / "served__s2021-2025__every4_v2.parquet"
    assert gv2.A_GATE_V2 == gv2.tpm.GATE_PATH.with_name("a_gate_v2.json")
    assert gv2.GATE_PATH.name == "v2_gate.json"
    assert gv2.output_paths("s2024-2025__every4", "2026-10-05", gate_path=tmp_path / "v2_gate.json",
                            report_dir=tmp_path) == (tmp_path / "v2_gate.json",
                                                     tmp_path / "2026-10-05-matchup-qb-gate.md")
    assert gv2.output_paths("s2025__every4", "2026-10-05", gate_path=tmp_path / "v2_gate.json",
                            report_dir=tmp_path) == (
        tmp_path / "v2_gate__s2025__every4.json",
        tmp_path / "2026-10-05-matchup-qb-gate__s2025__every4.md")


def test_spot_check_without_a_served_record_is_reported_not_graded():
    feats = _player_tbl()
    rows = gv2.spot_checks(_served("v1").query("season < 2026"),
                           _served("v2").query("season < 2026"), feats)
    assert rows == [{"label": "Keenum PHI @ CHI 2026-09-28", "graded": False,
                     "note": "no served record (not in the gate's backtest seasons)"}]
