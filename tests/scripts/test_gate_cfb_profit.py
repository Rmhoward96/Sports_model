"""Tests for scripts/gate_cfb_profit.py (CFB profit-model walk-forward gate).

The model walk-forward (`walk_forward_oof`) is stubbed in most tests; the
feature table comes from the real `profit_features.build_rows` on synthetic
walk-forward rows. No network, no DB, nothing written to assets/ or docs/.
"""
import importlib.util
import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from sportsmodel.cfb import profit_model
from sportsmodel.cfb.profit_features import build_rows

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gate_cfb_profit.py"
_spec = importlib.util.spec_from_file_location("gate_cfb_profit", _p)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

SIGMAS = {"sigma_margin": 16.0, "sigma_total": 17.0}


# ---------------------------------------------------------------- fixtures ---

def synth_raw(seasons=range(2015, 2026), n_games=60, seed=0, *, plant_verdict=False,
              thin_open=None, thin_ml=None):
    """Synthetic `raw_model_predictions` rows. Openers and moneylines only
    from 2021 (like the real data). `plant_verdict` makes the ratings model
    clairvoyant in verdict seasons (model numbers = actuals) WITHOUT touching
    any other season. `thin_open={season: frac}` / `thin_ml` drop that
    fraction of the season's openers / one moneyline side."""
    rng = np.random.default_rng(seed)
    raw = []
    pk = 0
    for s in seasons:
        for i in range(n_games):
            pk += 1
            week = 1 + i % 12
            mm = float(rng.normal(0, 14))
            mt = float(rng.normal(55, 8))
            spread_close = float(np.round(mm + rng.normal(0, 4)) + 0.5)
            total_close = float(np.round(mt + rng.normal(0, 4)) + 0.5)
            am = float(np.round(mm + rng.normal(0, 16)))
            at = float(max(3, np.round(mt + rng.normal(0, 16))))
            if am == 0:
                am = 1.0
            if plant_verdict and s in gate.VERDICT_SEASONS:
                mm = am + (am - spread_close) * 3.0
                mt = at + (at - total_close) * 3.0
            has_open = s >= 2021
            if thin_open and s in thin_open and (i / n_games) < thin_open[s]:
                has_open = False
            ml_home = -150.0 if spread_close > 0 else 130.0
            ml_away = 130.0 if spread_close > 0 else -150.0
            has_ml = s >= 2021
            ml_away_v = ml_away if not (thin_ml and s in thin_ml
                                        and (i / n_games) < thin_ml[s]) else None
            day = 3 + (i // 12) * 7 + (i % 3)
            raw.append({
                "season": s, "week": week, "home_team": f"H{i}", "away_team": f"A{i}",
                "game_pk": pk, "start_date": f"{s}-09-{1 + day % 28:02d}T{17 + i % 6}:00Z",
                "model_margin": mm, "model_total": mt,
                "market_spread": spread_close, "market_total": total_close,
                "spread_open": spread_close - 1.0 if has_open else None,
                "total_open": total_close + 1.0 if has_open else None,
                "ml_home": ml_home if has_ml else None,
                "ml_away": ml_away_v if has_ml else None,
                "actual_margin": am, "actual_total": at,
                "elo_home": 1500.0, "elo_away": 1500.0,
                "neutral_site": False, "conference_game": True,
            })
    return raw


def rows_and_games(raw):
    return build_rows(raw, pd.DataFrame(), set(), keep_pushes=True), pd.DataFrame(raw)


class StubOOF:
    """Stand-in for profit_model.walk_forward_oof. Deterministic p per row:
    tuning seasons get a weak hp-dependent signal; verdict seasons get either
    noise or (planted) a near-perfect signal for ONE hp combination."""

    def __init__(self, planted=False):
        self.planted = planted
        self.calls = []

    def __call__(self, df, market, seasons, min_train_season=2015, **hp):
        self.calls.append({"market": market, "seasons": sorted(seasons),
                           "max_season": int(df["season"].max()), "hp": hp})
        d = df[(df["market"] == market) & (df["season"].isin(seasons))].copy()
        idx = gate.HP_GRID.index(hp) if hp in gate.HP_GRID else 0
        noise = ((d["game_pk"].astype(int) * 7919) % 101) / 101.0 - 0.5
        y = d["y"].to_numpy()
        p = 0.5 + 0.3 * noise.to_numpy() + (0.02 + 0.01 * idx) * (y - 0.5)
        p = np.where(np.isnan(y), 0.8, p)          # unlabelled (push/void): strong home
        verdict = d["season"].isin(gate.VERDICT_SEASONS).to_numpy()
        if self.planted:
            p = np.where(verdict & (idx == 0), np.where(y == 1, 0.97, 0.03), p)
            p = np.where(verdict & (idx != 0), 0.5 + 0.45 * (y - 0.5), p)
        d["p_raw"] = p
        d["p"] = np.clip(p, 0.01, 0.99)
        return d.reset_index(drop=True)


def _tuning_view(result):
    """Everything the gate CHOSE (and the numbers it chose from)."""
    out = {}
    for m, r in result["markets"].items():
        out[m] = {k: r[k] for k in ("hyperparameters", "hp_grid", "policy", "policy_grid",
                                     "baseline_policy", "baseline_policy_grid",
                                     "hp_tuning_seasons", "tuning_streams")}
    return json.loads(json.dumps(out, default=str))


# ------------------------------------------------------ tuning isolation ---

def test_tuning_ignores_planted_verdict_only_signal():
    rows_a, games_a = rows_and_games(synth_raw(plant_verdict=False))
    rows_b, games_b = rows_and_games(synth_raw(plant_verdict=True))
    stub_a, stub_b = StubOOF(planted=False), StubOOF(planted=True)
    res_a = gate.run_gate(rows_a, games_a, SIGMAS, oof_fn=stub_a)
    res_b = gate.run_gate(rows_b, games_b, SIGMAS, oof_fn=stub_b)

    # every tuning choice (and every number behind it) is identical
    assert _tuning_view(res_a) == _tuning_view(res_b)

    # ... although the plant really changed the verdict seasons
    for m in ("spread", "total"):
        va = res_a["markets"][m]["verdict"]["open"]["combined"]
        vb = res_b["markets"][m]["verdict"]["open"]["combined"]
        assert vb["model"]["sim"]["log_growth_total"] > va["model"]["sim"]["log_growth_total"]
        assert vb["baseline"]["sim"]["log_growth_total"] > va["baseline"]["sim"]["log_growth_total"]

    # hyperparameter-grid calls only ever see tuning-era data
    grid_calls = [c for c in stub_b.calls if c["seasons"] and max(c["seasons"]) <= 2022
                  and c["max_season"] <= 2022]
    assert len(grid_calls) == len(gate.HP_GRID) * 3
    for c in grid_calls:
        assert set(c["seasons"]) <= set(gate.TUNING_SEASONS)
    # the only calls touching verdict seasons are the one full OOF per market
    full = [c for c in stub_b.calls if c["max_season"] > 2022]
    assert len(full) == 3


def test_policy_tuning_rejects_verdict_rows():
    rows, games = rows_and_games(synth_raw(seasons=range(2021, 2024), n_games=20))
    r = rows[rows["market"] == "spread"].copy()
    r["p"] = 0.55
    with pytest.raises(ValueError):
        gate.tune_policy([r], "spread", "p")


def test_moneyline_tunes_only_on_seasons_with_earlier_training_rows():
    rows, games = rows_and_games(synth_raw())
    res = gate.run_gate(rows, games, SIGMAS, oof_fn=StubOOF(), markets=("moneyline",))
    ml = res["markets"]["moneyline"]
    # ML prices start 2021 -> 2021 has no earlier training rows -> 2022 only
    assert ml["hp_tuning_seasons"] == [2022]
    assert ml["primary_price_point"] == "close"
    assert set(ml["verdict"]) == {"close"}


def test_ties_prefer_smaller_kelly_then_larger_min_edge():
    rows, _ = rows_and_games(synth_raw(seasons=range(2019, 2023), n_games=20))
    r = rows[rows["market"] == "spread"].copy()
    r["p"] = 0.5   # no bets anywhere -> every policy scores 0
    policy, grid = gate.tune_policy([r], "spread", "p")
    assert policy["kelly_frac"] == pytest.approx(0.25)
    assert policy["min_edge"] == pytest.approx(0.06)
    assert len(grid) == 21


def test_hp_ties_prefer_more_regularized():
    rows, _ = rows_and_games(synth_raw(seasons=range(2015, 2023), n_games=30))

    def flat(df, market, seasons, min_train_season=2015, **hp):
        d = df[(df["market"] == market) & df["season"].isin(seasons)].copy()
        d["p_raw"] = d["p"] = 0.5
        return d

    best, grid, seasons = gate.tune_hyperparameters(rows, "spread", flat)
    assert best == {"learning_rate": 0.03, "max_iter": 100, "min_samples_leaf": 400,
                    "l2_regularization": 10.0}
    assert len(grid) == 16
    assert seasons == [2019, 2020, 2021, 2022]


# ---------------------------------------------------------- ship decision ---

PASSING = {"log_growth_combined": 0.2,
           "log_growth_by_season": {2023: 0.1, 2024: 0.05, 2025: -0.01},
           "roi_ci_lower": 0.01, "ece": 0.02, "baseline_log_growth_combined": 0.1}


def test_ship_decision_passes_when_every_clause_holds():
    d = gate.ship_decision(PASSING)
    assert d["pass"] is True
    assert all(d["clauses"].values())
    assert d["reasons"] == []


@pytest.mark.parametrize("change,clause", [
    ({"log_growth_combined": -0.01, "baseline_log_growth_combined": -0.5},
     "log_growth_combined_positive"),
    ({"log_growth_combined": 0.0, "baseline_log_growth_combined": -0.5},
     "log_growth_combined_positive"),
    ({"log_growth_by_season": {2023: 0.1, 2024: -0.05, 2025: -0.01}},
     "log_growth_2_of_3_seasons"),
    ({"log_growth_by_season": {2023: 0.1, 2024: None, 2025: -0.01}},   # uncovered season
     "log_growth_2_of_3_seasons"),
    ({"roi_ci_lower": 0.0}, "roi_ci_lower_positive"),
    ({"roi_ci_lower": float("nan")}, "roi_ci_lower_positive"),
    ({"ece": 0.03}, "ece_below_0.03"),
    ({"ece": float("nan")}, "ece_below_0.03"),
    ({"baseline_log_growth_combined": 0.2}, "beats_baseline"),
])
def test_ship_decision_each_clause_fails_alone(change, clause):
    d = gate.ship_decision({**PASSING, **change})
    assert d["pass"] is False
    assert d["clauses"][clause] is False
    assert all(v for k, v in d["clauses"].items() if k != clause)
    assert len(d["reasons"]) == 1


# --------------------------------------------------------------- coverage ---

def test_coverage_table_and_thin_season_excluded():
    raw = synth_raw(thin_open={2024: 0.5}, thin_ml={2025: 0.4})
    rows, games = rows_and_games(raw)
    cov = gate.coverage_table(games)
    assert cov["spread"]["open"][2024] == pytest.approx(0.5)
    assert cov["spread"]["open"][2023] == pytest.approx(1.0)
    assert cov["spread"]["open"][2019] == 0.0
    assert cov["spread"]["close"][2019] == pytest.approx(1.0)
    assert cov["moneyline"]["close"][2025] == pytest.approx(0.6)   # both-ML rule

    res = gate.run_gate(rows, games, SIGMAS, oof_fn=StubOOF())
    sp = res["markets"]["spread"]["verdict"]["open"]
    assert sp["seasons"][2024]["counted"] is False
    assert sp["seasons"][2023]["counted"] is True
    assert sp["combined"]["seasons"] == [2023, 2025]
    n_counted = sum(sp["seasons"][s]["model"]["n_rows"] for s in (2023, 2025))
    assert sp["combined"]["model"]["n_rows"] == n_counted
    assert res["markets"]["spread"]["ship_inputs"]["log_growth_by_season"][2024] is None
    ml = res["markets"]["moneyline"]["verdict"]["close"]
    assert ml["seasons"][2025]["counted"] is False
    # tuning streams: close 2019-22, open only where openers exist (2021-22)
    streams = res["markets"]["spread"]["tuning_streams"]
    assert streams == {"close": [2019, 2020, 2021, 2022], "open": [2021, 2022]}


# ---------------------------------------------------- bets and baseline ---

def _row(**kw):
    base = {"season": 2023, "week": 1, "game_pk": 1, "market": "spread",
            "price_point": "open", "line": 3.5, "ml_home": np.nan, "ml_away": np.nan,
            "start_date": "2023-09-02T17:00Z", "y": 1.0, "p": 0.6}
    return {**base, **kw}


def test_make_bets_sides_stakes_results_and_et_day():
    rows = pd.DataFrame([
        _row(game_pk=1, p=0.60, y=1.0),                                    # home wins
        _row(game_pk=2, p=0.40, y=1.0, start_date="2023-09-03T02:00Z"),     # away loses
        _row(game_pk=3, p=0.51, y=0.0),                                    # edge < min
    ])
    bets = gate.make_bets(rows, "spread", "p", kelly_frac=0.5, min_edge=0.02)
    assert list(bets["side"]) == ["home", "away"]
    d = 1 + 100 / 110
    assert bets["dec"].tolist() == pytest.approx([d, d])
    assert list(bets["result"]) == ["win", "loss"]
    # 02:00Z on the 3rd is 22:00 ET on the 2nd -> same slate day
    assert bets["day"].nunique() == 1
    edge = 0.6 * d - 1
    assert bets["stake_frac"].iloc[0] == pytest.approx(min(0.03, 0.5 * edge / (d - 1)))
    assert {"season", "week"} <= set(bets.columns)


def test_make_bets_totals_moneyline_and_day_cap():
    t = pd.DataFrame([_row(market="total", p=0.3, y=0.0, line=55.5)])
    b = gate.make_bets(t, "total", "p", kelly_frac=0.25, min_edge=0.0)
    assert list(b["side"]) == ["under"] and list(b["result"]) == ["win"]

    m = pd.DataFrame([_row(market="moneyline", price_point="close", line=np.nan,
                           ml_home=150.0, ml_away=-170.0, p=0.5, y=1.0)])
    b = gate.make_bets(m, "moneyline", "p", kelly_frac=0.25, min_edge=0.0)
    assert list(b["side"]) == ["home"] and b["dec"].iloc[0] == pytest.approx(2.5)
    assert list(b["result"]) == ["win"]

    many = pd.DataFrame([_row(game_pk=i, p=0.9) for i in range(10)])
    b = gate.make_bets(many, "spread", "p", kelly_frac=0.5, min_edge=0.0)
    assert b["stake_frac"].sum() == pytest.approx(0.15)       # 10 x 3 % -> 15 % day cap


def test_baseline_probs_normal_push_excluded():
    rows = pd.DataFrame([
        {"market": "spread", "line": 3.0, "f_model_margin": 3.0, "f_model_total": 50.0},
        {"market": "spread", "line": 2.5, "f_model_margin": 2.5, "f_model_total": 50.0},
        {"market": "total", "line": 50.0, "f_model_margin": 0.0, "f_model_total": 55.0},
        {"market": "moneyline", "line": np.nan, "f_model_margin": 10.0, "f_model_total": 50.0},
    ])
    p = gate.baseline_probs(rows, SIGMAS)
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(0.5)
    s = SIGMAS["sigma_total"]
    over, under = norm.sf(50.5, 55, s), norm.cdf(49.5, 55, s)
    assert p[2] == pytest.approx(over / (over + under))
    sm = SIGMAS["sigma_margin"]
    win, lose = norm.sf(0.5, 10, sm), norm.cdf(-0.5, 10, sm)
    assert p[3] == pytest.approx(win / (win + lose))


# ------------------------------------------------------------- reporting ---

def test_report_renders_and_json_is_strict():
    rows, games = rows_and_games(synth_raw())
    res = gate.run_gate(rows, games, SIGMAS, oof_fn=StubOOF())
    md = gate.render_report(res)
    for m in ("spread", "total", "moneyline"):
        assert f"## {m}" in md
    assert "PASS" in md or "FAIL" in md
    assert "Kelly" in md and "baseline" in md.lower()
    assert "2023" in md and "open" in md and "close" in md
    txt = gate.to_json(res)
    json.loads(txt)                       # no NaN/inf tokens
    assert "NaN" not in txt and "Infinity" not in txt
    assert set(json.loads(txt)["markets"]) == {"spread", "total", "moneyline"}


def test_run_gate_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows, games = rows_and_games(synth_raw(n_games=30))
    gate.run_gate(rows, games, SIGMAS, oof_fn=StubOOF(), markets=("spread",))
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------- real model, tiny grid ---

def test_end_to_end_with_real_walk_forward_tiny_grid():
    rows, games = rows_and_games(synth_raw(n_games=60))
    grid = [{"learning_rate": 0.05, "max_iter": 10, "min_samples_leaf": 20,
             "l2_regularization": 1.0}]
    res = gate.run_gate(rows, games, SIGMAS, oof_fn=profit_model.walk_forward_oof,
                        markets=("total",), hp_grid=grid)
    t = res["markets"]["total"]
    assert t["hyperparameters"] == grid[0]
    assert t["ship"]["pass"] in (True, False)
    assert set(t["verdict"]) == {"open", "close"}
    assert math.isfinite(t["verdict"]["close"]["combined"]["model"]["ece"])


# -------------------------------------------------------------------- CLI ---

def test_main_writes_json_and_report_only_where_told(tmp_path, monkeypatch):
    rows, games = rows_and_games(synth_raw(n_games=30))
    seen = {}

    def fake_load(max_season):
        seen["max_season"] = max_season
        return rows, games, SIGMAS

    monkeypatch.setattr(gate, "load_inputs", fake_load)
    j, r = tmp_path / "gate.json", tmp_path / "gate.md"
    assert gate.main(["--markets", "spread", "--json-out", str(j), "--report-out", str(r)],
                     oof_fn=StubOOF()) == 0
    assert seen["max_season"] == 2025
    out = json.loads(j.read_text())
    assert out["markets"]["spread"]["policy"]["kelly_frac"] in (0.25, 1 / 3, 0.5)
    assert "passing_markets" in out
    assert "## spread" in r.read_text()


def test_main_tuning_only_cuts_data_and_writes_nothing(tmp_path, monkeypatch, capsys):
    rows, games = rows_and_games(synth_raw(seasons=range(2015, 2023), n_games=30))
    seen = {}

    def fake_load(max_season):
        seen["max_season"] = max_season
        return rows, games, SIGMAS

    monkeypatch.setattr(gate, "load_inputs", fake_load)
    monkeypatch.chdir(tmp_path)
    j = tmp_path / "gate.json"
    assert gate.main(["--tuning-only", "--markets", "spread", "--json-out", str(j)],
                     oof_fn=StubOOF()) == 0
    assert seen["max_season"] == 2022
    assert list(tmp_path.iterdir()) == []
    assert json.loads(capsys.readouterr().out)["markets"]["spread"]["policy"]


# ------------------------------------------------ pushes, voids, chronology ---

def test_push_bet_occupies_day_cap_and_settles_at_zero():
    wins = [_row(game_pk=i, p=0.9, y=1.0, result="win") for i in range(5)]
    push = _row(game_pk=99, p=0.9, y=np.nan, result="push")
    void = _row(game_pk=98, p=0.9, y=np.nan, result="void")
    with_push = gate.make_bets(pd.DataFrame(wins + [push]), "spread", "p", 0.5, 0.0)
    assert 99 in set(with_push["game_pk"])
    assert with_push.set_index("game_pk").loc[99, "result"] == "push"
    scale = 0.15 / 0.18                                   # 6 x 3 % -> 15 % day cap
    assert with_push["stake_frac"].tolist() == pytest.approx([0.03 * scale] * 6)
    d = 1 + 100 / 110
    sim = kelly_sim = gate.kelly.simulate(with_push, 100.0)
    assert sim["n_bets"] == 6
    assert sim["staked"] == pytest.approx(15.0)                       # push included
    assert kelly_sim["profit"] == pytest.approx(5 * 100 * 0.03 * scale * (d - 1))
    assert gate._stake_unit_roi(with_push) == pytest.approx(5 * (d - 1) / 6)
    # a void settles like a push
    v = gate.make_bets(pd.DataFrame(wins + [void]), "spread", "p", 0.5, 0.0)
    assert v.set_index("game_pk").loc[98, "result"] == "push"
    assert gate.kelly.simulate(v, 100.0)["profit"] == pytest.approx(kelly_sim["profit"])


def test_verdict_bet_set_keeps_pushes_but_ece_ignores_them():
    raw = synth_raw()
    pushed = [r for r in raw if r["season"] == 2023][:3]
    for r in pushed:                         # the close AND the opener push
        r["market_spread"] = r["spread_open"] = r["actual_margin"]
    rows, games = rows_and_games(raw)
    assert (rows["result"] == "push").sum() >= 6
    res = gate.run_gate(rows, games, SIGMAS, oof_fn=StubOOF(), markets=("spread",))
    s23 = res["markets"]["spread"]["verdict"]["open"]["seasons"][2023]["model"]
    assert s23["n_push"] == 3                 # stub prices pushes at p=.8 -> all bet
    assert s23["n_rows"] == 60 - 3            # ECE/log-loss rows exclude pushes


def test_nat_kickoff_takes_its_weeks_slate_day_or_is_dropped():
    rows = pd.DataFrame([
        _row(game_pk=1, week=1, start_date="2023-09-02T17:00Z"),
        _row(game_pk=2, week=2, start_date="2023-09-09T17:00Z"),
        _row(game_pk=3, week=1, start_date=None),                   # -> 2023-09-02
        _row(game_pk=4, week=5, start_date=None),                   # no week-5 date
    ])
    bets = gate.make_bets(rows, "spread", "p", 0.25, 0.0)
    assert list(bets["game_pk"]) == [1, 3, 2]                        # chronological
    assert bets.set_index("game_pk").loc[3, "day"] == "2023-09-02"
    assert bets.attrs["n_dropped_no_day"] == 1


def test_to_json_keeps_infinities_as_strings():
    out = json.loads(gate.to_json({"a": float("-inf"), "b": float("inf"),
                                   "c": float("nan"), "d": 1.5}))
    assert out == {"a": "-inf", "b": "inf", "c": None, "d": 1.5}


def test_baseline_verdict_uses_the_baseline_policy(monkeypatch):
    fixed = {"p": {"kelly_frac": 0.25, "min_edge": 0.0},
             "p_base": {"kelly_frac": 0.5, "min_edge": 0.03}}
    monkeypatch.setattr(gate, "tune_policy",
                        lambda streams, market, p_col: (dict(fixed[p_col]), []))
    seen = []
    real = gate.evaluate

    def spy(rows, market, p_col, policy):
        seen.append((p_col, dict(policy)))
        return real(rows, market, p_col, policy)

    monkeypatch.setattr(gate, "evaluate", spy)
    rows, games = rows_and_games(synth_raw(n_games=30))
    gate.run_gate(rows, games, SIGMAS, oof_fn=StubOOF(), markets=("spread",))
    assert {c for c, _ in seen} == {"p", "p_base"}
    for p_col, policy in seen:
        assert policy == fixed[p_col]


def test_load_inputs_keeps_pushes_for_betting(monkeypatch):
    from sportsmodel.cfb import walkforward
    raw = synth_raw(seasons=range(2021, 2023), n_games=10)
    seen = {}
    monkeypatch.setattr(walkforward, "raw_model_predictions",
                        lambda span, elo, blend, **kw: (seen.update(max=int(span.season.max()))
                                                        or raw))
    real = gate.build_rows

    def spy(r, priors, fbs, **kw):
        seen["kw"] = kw
        return real(r, priors, fbs, **kw)

    monkeypatch.setattr(gate, "build_rows", spy)
    rows, games, sigmas = gate.load_inputs(2022)
    assert seen["kw"] == {"keep_pushes": True}
    assert seen["max"] == 2022
    assert set(sigmas) == {"sigma_margin", "sigma_total"}
    assert len(games) == len(raw)
