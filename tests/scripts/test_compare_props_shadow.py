"""Tests for scripts/compare_props_shadow.py (props-ML Props-2 shadow
comparison: current sim `sim-nfl-v1` vs ML `nfl-sim-ml-v1`). Pure scorer and
report rendering only; the DB loader is stubbed -- no network, no DB."""
import importlib.util
import json
import pathlib

import pytest

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "compare_props_shadow.py"
_spec = importlib.util.spec_from_file_location("compare_props_shadow", _p)
cps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cps)


def _sim(game_pk, pid, market, pmf, as_json=False):
    dist = {"kind": "pmf", "pmf": list(pmf), "mean": 0.0}
    return {"game_pk": game_pk, "player_id": pid, "market": market,
            "dist": json.dumps(dist) if as_json else dist}


def _act(game_pk, pid, market, actual, season=2026, week=3):
    return {"game_pk": game_pk, "player_id": pid, "market": market,
            "actual": actual, "season": season, "week": week}


def _line(game_pk, pid, market, line):
    return {"game_pk": game_pk, "player_id": pid, "market": market, "line": line}


# ---------------------------------------------------------------- p_over_pmf

@pytest.mark.parametrize("line,expected", [
    (0.5, 0.8),    # 1 - F(0)
    (1.5, 0.5),    # 1 - F(1)
    (1.0, 0.5),    # integer line: P(X > 1)
    (-0.5, 1.0),   # below the support
    (5.5, 0.0),    # above the support
])
def test_p_over_pmf(line, expected):
    assert cps.p_over_pmf([0.2, 0.3, 0.5], line) == pytest.approx(expected)


# ---------------------------------------------------------------- score_shadow

def test_score_pairs_keys_and_computes_known_metrics():
    # One paired key with an actual; extra keys that must be dropped:
    # (2, "b") only in v1, (3, "c") only in ML, (4, "d") in both but no actual.
    v1 = [_sim(1, "a", "rush_att", [0.5, 0.5]), _sim(2, "b", "rush_att", [0.5, 0.5]),
          _sim(4, "d", "rush_att", [0.5, 0.5])]
    ml = [_sim(1, "a", "rush_att", [0.2, 0.8], as_json=True),
          _sim(3, "c", "rush_att", [0.2, 0.8]), _sim(4, "d", "rush_att", [0.2, 0.8])]
    actuals = [_act(1, "a", "rush_att", 1.0), _act(2, "b", "rush_att", 1.0),
               _act(3, "c", "rush_att", 1.0)]
    lines = [_line(1, "a", "rush_att", 0.5)]

    res = cps.score_shadow(v1, ml, actuals, lines)
    m = res["markets"]["rush_att"]
    assert m["n"] == 1
    # RPS = sum_k (F(k) - 1[k >= a])^2 with a = 1: v1 (0.5-0)^2 = 0.25; ML 0.2^2 = 0.04.
    assert m["rps_v1"] == pytest.approx(0.25)
    assert m["rps_ml"] == pytest.approx(0.04)
    # Over 0.5 hits (actual 1): Brier v1 (0.5-1)^2 = 0.25, ML (0.8-1)^2 = 0.04.
    assert m["n_line"] == 1
    assert m["brier_v1"] == pytest.approx(0.25)
    assert m["brier_ml"] == pytest.approx(0.04)
    # A single PIT lands in one decile: ECE = (0.9 + 9 * 0.1) / 10 = 0.18.
    assert m["ece_v1"] == pytest.approx(0.18)
    assert m["ece_ml"] == pytest.approx(0.18)
    assert m["verdict"] == "ML better"
    assert res["verdict"] == "inconclusive (n < 300 per market)"


def test_push_is_excluded_from_brier_but_kept_for_rps():
    v1 = [_sim(1, "a", "receptions", [0.2, 0.3, 0.5]),
          _sim(2, "a", "receptions", [0.2, 0.3, 0.5])]
    ml = [_sim(1, "a", "receptions", [0.1, 0.4, 0.5]),
          _sim(2, "a", "receptions", [0.1, 0.4, 0.5])]
    actuals = [_act(1, "a", "receptions", 1.0), _act(2, "a", "receptions", 2.0)]
    # Game 1: integer line 1.0 == actual -> push. Game 2: 2 > 1 -> over.
    lines = [_line(1, "a", "receptions", 1.0), _line(2, "a", "receptions", 1.0)]
    m = cps.score_shadow(v1, ml, actuals, lines)["markets"]["receptions"]
    assert m["n"] == 2
    assert m["n_line"] == 1
    # Game 2 only: P(X > 1) = 0.5 for both -> (0.5 - 1)^2.
    assert m["brier_v1"] == pytest.approx(0.25)
    assert m["brier_ml"] == pytest.approx(0.25)


def test_market_without_lines_uses_rps_only():
    v1 = [_sim(1, "a", "pass_tds", [0.5, 0.5])]
    ml = [_sim(1, "a", "pass_tds", [0.8, 0.2])]
    actuals = [_act(1, "a", "pass_tds", 1.0)]
    m = cps.score_shadow(v1, ml, actuals, [])["markets"]["pass_tds"]
    assert m["n_line"] == 0
    assert m["brier_v1"] is None and m["brier_ml"] is None
    # ML RPS 0.64 > v1 0.25 and no Brier to disagree -> worse.
    assert m["verdict"] == "ML worse"


def test_anytime_td_actual_is_clipped_to_binary_support():
    v1 = [_sim(1, "a", "anytime_td", [0.6, 0.4])]
    ml = [_sim(1, "a", "anytime_td", [0.3, 0.7])]
    actuals = [_act(1, "a", "anytime_td", 2.0)]  # total TDs
    m = cps.score_shadow(v1, ml, actuals, [])["markets"]["anytime_td"]
    assert m["rps_v1"] == pytest.approx(0.36)
    assert m["rps_ml"] == pytest.approx(0.09)


def test_rows_with_missing_pmf_or_actual_are_skipped():
    v1 = [_sim(1, "a", "rec_yds", [0.5, 0.5]),
          {"game_pk": 2, "player_id": "a", "market": "rec_yds", "dist": None}]
    ml = [_sim(1, "a", "rec_yds", [0.4, 0.6]), _sim(2, "a", "rec_yds", [0.5, 0.5])]
    actuals = [_act(1, "a", "rec_yds", 1.0), _act(2, "a", "rec_yds", 1.0)]
    res = cps.score_shadow(v1, ml, actuals, [])
    assert res["markets"]["rec_yds"]["n"] == 1
    assert res["skipped"] == 1


def test_identical_pairs_are_excluded_from_metrics_and_counted():
    # Game 1: identical pmfs (fallback copy) -> excluded. Game 2: differs -> scored.
    v1 = [_sim(1, "a", "rush_att", [0.5, 0.5]), _sim(2, "a", "rush_att", [0.5, 0.5])]
    ml = [_sim(1, "a", "rush_att", [0.5, 0.5], as_json=True),
          _sim(2, "a", "rush_att", [0.2, 0.8])]
    actuals = [_act(1, "a", "rush_att", 0.0), _act(2, "a", "rush_att", 1.0)]
    lines = [_line(1, "a", "rush_att", 0.5), _line(2, "a", "rush_att", 0.5)]
    res = cps.score_shadow(v1, ml, actuals, lines)
    m = res["markets"]["rush_att"]
    assert m["n"] == 1
    assert m["n_identical"] == 1
    assert res["n_identical"] == 1
    # Metrics are game 2 alone (game 1 would have pulled both RPS to 0.25).
    assert m["rps_v1"] == pytest.approx(0.25)
    assert m["rps_ml"] == pytest.approx(0.04)
    assert m["n_line"] == 1
    assert m["brier_ml"] == pytest.approx(0.04)


def test_pmfs_of_different_length_are_not_identical():
    v1 = [_sim(1, "a", "rush_att", [0.5, 0.5])]
    ml = [_sim(1, "a", "rush_att", [0.5, 0.5, 0.0])]
    m = cps.score_shadow(v1, ml, [_act(1, "a", "rush_att", 1.0)], [])["markets"]["rush_att"]
    assert m["n"] == 1
    assert m["n_identical"] == 0


def test_all_identical_market_is_reported_and_does_not_block_the_verdict():
    v1, ml, actuals, lines = [], [], [], []
    for g in range(300):
        v1.append(_sim(g, "a", "rush_att", [0.5, 0.5]))
        ml.append(_sim(g, "a", "rush_att", [0.2, 0.8]))
        actuals.append(_act(g, "a", "rush_att", 1.0))
        lines.append(_line(g, "a", "rush_att", 0.5))
    for g in range(3):  # baseline-sourced market: ML copies v1
        v1.append(_sim(g, "q", "pass_tds", [0.5, 0.5]))
        ml.append(_sim(g, "q", "pass_tds", [0.5, 0.5]))
        actuals.append(_act(g, "q", "pass_tds", 1.0))
    res = cps.score_shadow(v1, ml, actuals, lines)
    p = res["markets"]["pass_tds"]
    assert p["n"] == 0 and p["n_identical"] == 3
    assert p["rps_v1"] is None and p["rps_ml"] is None
    assert p["ece_v1"] is None and p["brier_v1"] is None
    assert p["verdict"] == "all identical (n_identical=3)"
    assert res["n_identical"] == 3
    assert res["verdict"] == "ML better"


def test_only_identical_pairs():
    v1 = [_sim(1, "a", "pass_tds", [0.5, 0.5])]
    ml = [_sim(1, "a", "pass_tds", [0.5, 0.5])]
    res = cps.score_shadow(v1, ml, [_act(1, "a", "pass_tds", 0.0)], [])
    assert res["markets"]["pass_tds"]["n"] == 0
    assert res["verdict"] == "no difference (all paired pmfs identical)"


# ---------------------------------------------------------------- verdicts

@pytest.mark.parametrize("rps_v1,rps_ml,b_v1,b_ml,expected", [
    (1.0, 0.9, 0.25, 0.25, "ML better"),    # Brier tie still counts as better
    (1.0, 0.9, 0.25, 0.20, "ML better"),
    (1.0, 1.1, 0.25, 0.30, "ML worse"),
    (1.0, 0.9, 0.25, 0.30, "mixed"),        # better RPS, worse Brier
    (1.0, 1.1, 0.25, 0.20, "mixed"),        # worse RPS, better Brier
    (1.0, 1.1, 0.25, 0.25, "ML worse"),     # worse RPS, Brier tie (mirror of better)
    (1.0, 1.0, 0.25, 0.20, "ML better"),    # RPS tie, Brier strictly better
    (1.0, 1.0, 0.25, 0.30, "ML worse"),     # RPS tie, Brier strictly worse
    (1.0, 0.9, None, None, "ML better"),    # no lines -> RPS decides
    (1.0, 1.1, None, None, "ML worse"),
    (1.0, 1.0, 0.25, 0.25, "no difference"),
    (1.0, 1.0, None, None, "no difference"),
])
def test_market_verdict(rps_v1, rps_ml, b_v1, b_ml, expected):
    assert cps.market_verdict(rps_v1, rps_ml, b_v1, b_ml) == expected


def _mk(n, verdict):
    return {"n": n, "verdict": verdict}


@pytest.mark.parametrize("markets,expected", [
    ({}, "inconclusive (no paired rows)"),
    ({"rec_yds": _mk(300, "ML better"), "pass_yds": _mk(299, "ML better")},
     "inconclusive (n < 300 per market)"),
    ({"rec_yds": _mk(300, "ML better"), "pass_yds": _mk(400, "ML better")}, "ML better"),
    ({"rec_yds": _mk(300, "ML better"), "pass_tds": _mk(400, "no difference")}, "ML better"),
    ({"rec_yds": _mk(300, "ML worse"), "pass_yds": _mk(400, "ML worse")}, "ML worse"),
    ({"rec_yds": _mk(300, "ML better"), "pass_yds": _mk(400, "ML worse")}, "mixed"),
    ({"rec_yds": _mk(300, "ML better"), "pass_yds": _mk(400, "mixed")}, "mixed"),
    ({"pass_tds": _mk(300, "no difference")}, "no difference"),
    # An all-identical market (n = 0) is exempt from the n < 300 gate.
    ({"rec_yds": _mk(300, "ML better"), "pass_tds": _mk(0, "all identical (n_identical=5)")},
     "ML better"),
    ({"pass_tds": _mk(0, "all identical (n_identical=5)")},
     "no difference (all paired pmfs identical)"),
])
def test_overall_verdict(markets, expected):
    assert cps.overall_verdict(markets) == expected


def test_enough_rows_per_market_gives_a_decided_verdict():
    v1, ml, actuals, lines = [], [], [], []
    for g in range(300):
        v1.append(_sim(g, "a", "rush_att", [0.5, 0.5]))
        ml.append(_sim(g, "a", "rush_att", [0.2, 0.8]))
        actuals.append(_act(g, "a", "rush_att", 1.0))
        lines.append(_line(g, "a", "rush_att", 0.5))
    res = cps.score_shadow(v1, ml, actuals, lines)
    assert res["markets"]["rush_att"]["n"] == 300
    assert res["verdict"] == "ML better"


# ---------------------------------------------------------------- report

def test_render_report_has_verdict_table_and_per_market_n():
    v1 = [_sim(1, "a", "rush_att", [0.5, 0.5]), _sim(1, "a", "pass_tds", [0.5, 0.5])]
    ml = [_sim(1, "a", "rush_att", [0.2, 0.8]), _sim(1, "a", "pass_tds", [0.5, 0.5])]
    actuals = [_act(1, "a", "rush_att", 1.0), _act(1, "a", "pass_tds", 0.0)]
    res = cps.score_shadow(v1, ml, actuals, [_line(1, "a", "rush_att", 0.5)])
    md = cps.render_report(res, since="2026-09-20T00:00:00+00:00", run_date="2026-09-27",
                           null_commence=4)
    assert md.startswith("# Props-ML shadow comparison — 2026-09-27")
    assert "**Verdict: inconclusive (n < 300 per market).**" in md
    assert "Per-market n: rush_att 1, pass_tds 0." in md
    assert "| rush_att | 1 | 0 | 0.2500 | 0.0400 | 0.1800 | 0.1800 | 1 | 0.2500 | 0.0400 | ML better |" in md
    assert "Since: 2026-09-20T00:00:00+00:00" in md
    # Markets are listed in the canonical order (rush_att before pass_tds).
    assert md.index("| rush_att |") < md.index("| pass_tds |")
    # pass_tds is all identical (no scored rows) and carries no book line: dashes.
    assert ("| pass_tds | 0 | 1 | – | – | – | – | 0 | – | – | all identical (n_identical=1) |"
            in md)
    assert "Identical-pmf pairs excluded: 1 (pass_tds 1)." in md
    assert "nfl_player_sim rows dropped for NULL commence_time: 4." in md
    assert "Brier is over the n line keys" in md


# ---------------------------------------------------------------- main (loader stubbed)

def test_main_exits_zero_without_ml_rows(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cps, "load_shadow_data", lambda since: None)
    assert cps.main(["--out-dir", str(tmp_path)]) == 0
    assert "no nfl-sim-ml-v1 rows" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_main_writes_report(monkeypatch, tmp_path, capsys):
    data = {"since": "2026-09-20T00:00:00+00:00",
            "v1": [_sim(1, "a", "rush_att", [0.5, 0.5])],
            "ml": [_sim(1, "a", "rush_att", [0.2, 0.8])],
            "actuals": [_act(1, "a", "rush_att", 1.0)],
            "lines": [_line(1, "a", "rush_att", 0.5)],
            "null_commence": 2}
    seen = {}

    def fake_load(since):
        seen["since"] = since
        return data

    monkeypatch.setattr(cps, "load_shadow_data", fake_load)
    assert cps.main(["--since", "2026-09-20", "--out-dir", str(tmp_path),
                     "--run-date", "2026-09-27"]) == 0
    assert seen["since"] == "2026-09-20"
    out = tmp_path / "2026-09-27-props-ml-shadow.md"
    assert out.exists()
    printed = capsys.readouterr().out
    assert "| rush_att | 1 |" in printed
    assert "inconclusive (n < 300 per market)" in printed
    text = out.read_text()
    assert "ML better" in text
    assert "nfl_player_sim rows dropped for NULL commence_time: 2." in text
