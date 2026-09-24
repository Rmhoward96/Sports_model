"""Tests for the pure seams of scripts/train_props_ml.py (the props-ML A-gate
walk-forward ablation ladder). main()'s nflverse IO, model fits and backtests
are not unit tested. Synthetic frames and stubs only -- no network."""
import importlib.util
import pathlib

import pandas as pd
import pytest

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "train_props_ml.py"
_spec = importlib.util.spec_from_file_location("train_props_ml", _p)
tpm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tpm)


# ---- constants ------------------------------------------------------------------

def test_ladder_constants():
    assert tpm.TEST_SEASONS == [2021, 2022, 2023, 2024, 2025]
    assert tpm.REFIT_WEEKS == (1, 5, 9, 13, 17)
    assert tpm.LADDER == ("volume", "efficiency", "context", "market")
    assert tpm.PROD == dict(season_decay=0.4, questionable_weight=0.75,
                            home_field=0.07, ratings_weight=0.5)
    # learned count models train on stubs as 0 and see st_questionable as a
    # feature, so learned shares get no extra questionable down-weight
    assert tpm.Q_WEIGHT == 1.0


# ---- refit_block ----------------------------------------------------------------

@pytest.mark.parametrize("week,block", [(1, 1), (4, 1), (5, 5), (8, 5), (9, 9),
                                        (16, 13), (17, 17), (18, 17)])
def test_refit_block_is_largest_refit_week_at_or_before(week, block):
    assert tpm.refit_block(week) == block


def test_refit_block_weekly_schedule():
    weekly = tuple(range(1, 19))
    assert [tpm.refit_block(w, weekly) for w in (1, 7, 18)] == [1, 7, 18]


def test_refit_block_before_first_refit_raises():
    with pytest.raises(ValueError):
        tpm.refit_block(0)


def test_refit_weeks_from_env():
    assert tpm.refit_weeks_from_env({}) == (1, 5, 9, 13, 17)
    assert tpm.refit_weeks_from_env({"PROPS_ML_WEEKLY_REFIT": "0"}) == (1, 5, 9, 13, 17)
    assert tpm.refit_weeks_from_env({"PROPS_ML_WEEKLY_REFIT": "1"}) == tuple(range(1, 19))


# ---- next_candidate -------------------------------------------------------------

def test_next_candidate_adds_rung_to_kept():
    assert tpm.next_candidate(frozenset(), "volume") == frozenset({"volume"})
    assert tpm.next_candidate(frozenset({"volume"}), "efficiency") == frozenset({"volume", "efficiency"})
    assert tpm.next_candidate(frozenset({"volume", "efficiency"}), "context") == \
        frozenset({"volume", "efficiency", "context"})


def test_next_candidate_later_rungs_imply_volume_even_if_volume_failed():
    for rung in ("efficiency", "context", "market"):
        assert tpm.next_candidate(frozenset(), rung) == frozenset({"volume", rung})
    assert tpm.next_candidate(frozenset({"volume", "context"}), "market") == \
        frozenset({"volume", "context", "market"})


def test_next_candidate_returns_frozenset():
    assert isinstance(tpm.next_candidate(frozenset(), "market"), frozenset)


# ---- questionable_index ---------------------------------------------------------

def _player_tbl():
    return pd.DataFrame({
        "player_id": ["a1", "a2", "b1", "b2", "c1", "a1", "a2"],
        "season":    [2024, 2024, 2024, 2024, 2024, 2024, 2023],
        "week":      [3, 3, 3, 3, 3, 4, 3],
        "team":      ["AAA", "AAA", "BBB", "BBB", "CCC", "AAA", "AAA"],
        "st_questionable": [1.0, 0.0, 1.0, float("nan"), 1.0, 1.0, 1.0],
        "p_x":       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    })


def _team_tbl():
    return pd.DataFrame({
        "team":   ["AAA", "BBB", "CCC", "AAA", "AAA"],
        "season": [2024, 2024, 2024, 2024, 2023],
        "week":   [3, 3, 3, 4, 3],
        "tm_x":   [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_questionable_index_by_season_week_team():
    q = tpm.questionable_index(_player_tbl())
    assert q[(2024, 3, "AAA")] == {"a1"}
    assert q[(2024, 3, "BBB")] == {"b1"}
    assert q[(2024, 4, "AAA")] == {"a1"}
    assert q[(2023, 3, "AAA")] == {"a2"}
    assert (2024, 3, "ZZZ") not in q


# ---- make_hook ------------------------------------------------------------------

class _Spec:
    pass


def _spy_apply(monkeypatch):
    calls = []

    def fake_apply(spec, models, player_rows, team_rows, questionable, q_weight):
        calls.append(dict(spec=spec, models=models, player_rows=player_rows.copy(),
                          team_rows=team_rows.copy(), questionable=set(questionable),
                          q_weight=q_weight))
        return "NEW-SPEC"

    monkeypatch.setattr(tpm.learned, "apply_to_spec", fake_apply)
    return calls


def test_make_hook_uses_block_model_and_only_that_week_and_teams(monkeypatch):
    calls = _spy_apply(monkeypatch)
    models = {(2024, 1): "M1", (2024, 5): "M5", (2023, 1): "OLD"}
    ptbl, ttbl = _player_tbl(), _team_tbl()
    hook = tpm.make_hook(models, ptbl, ttbl, tpm.questionable_index(ptbl))
    spec = _Spec()
    out = hook(2024, 3, "AAA", "BBB", spec)

    assert out == "NEW-SPEC"
    assert len(calls) == 1
    c = calls[0]
    assert c["spec"] is spec
    assert c["models"] == "M1"  # week 3 -> refit block 1 of 2024
    pr = c["player_rows"]
    assert set(pr["player_id"]) == {"a1", "a2", "b1", "b2"}
    assert set(zip(pr["season"], pr["week"])) == {(2024, 3)}
    assert set(pr["team"]) == {"AAA", "BBB"}
    tr = c["team_rows"]
    assert set(tr["team"]) == {"AAA", "BBB"}
    assert set(zip(tr["season"], tr["week"])) == {(2024, 3)}
    assert c["questionable"] == {"a1", "b1"}
    assert c["q_weight"] == 1.0  # learned shares: no questionable down-weight on top (I2)


def test_make_hook_picks_later_block(monkeypatch):
    calls = _spy_apply(monkeypatch)
    models = {(2024, 1): "M1", (2024, 5): "M5"}
    ptbl = _player_tbl().assign(week=lambda d: d["week"].replace({3: 6}))
    ttbl = _team_tbl().assign(week=lambda d: d["week"].replace({3: 6}))
    hook = tpm.make_hook(models, ptbl, ttbl, tpm.questionable_index(ptbl))
    hook(2024, 6, "BBB", "AAA", _Spec())
    assert calls[0]["models"] == "M5"


def test_make_hook_honours_refit_schedule(monkeypatch):
    calls = _spy_apply(monkeypatch)
    weekly = tuple(range(1, 19))
    models = {(2024, w): f"M{w}" for w in weekly}
    ptbl = _player_tbl()
    hook = tpm.make_hook(models, ptbl, _team_tbl(), tpm.questionable_index(ptbl),
                         refit_weeks=weekly)
    hook(2024, 3, "AAA", "BBB", _Spec())
    assert calls[0]["models"] == "M3"


def test_make_hook_missing_rows_passes_empty_frames(monkeypatch):
    calls = _spy_apply(monkeypatch)
    ptbl = _player_tbl()
    hook = tpm.make_hook({(2024, 9): "M9"}, ptbl, _team_tbl(), tpm.questionable_index(ptbl))
    hook(2024, 10, "AAA", "BBB", _Spec())
    c = calls[0]
    assert c["player_rows"].empty and c["team_rows"].empty
    assert list(c["player_rows"].columns) == list(ptbl.columns)
    assert c["questionable"] == set()


def test_make_hook_missing_model_block_raises(monkeypatch):
    _spy_apply(monkeypatch)
    ptbl = _player_tbl()
    hook = tpm.make_hook({}, ptbl, _team_tbl(), tpm.questionable_index(ptbl))
    with pytest.raises(KeyError):
        hook(2024, 3, "AAA", "BBB", _Spec())


# ---- hook guard / coverage ----------------------------------------------------------

def test_guard_hook_passes_through_and_captures_errors_with_context():
    def hook(season, week, home, away, spec):
        if week == 2:
            raise KeyError((season, 1))
        return ("ok", spec)

    guarded, errors = tpm.guard_hook(hook)
    assert guarded(2025, 1, "KC", "BAL", "S") == ("ok", "S")
    with pytest.raises(KeyError):  # still raised so run_backtest skips the game
        guarded(2025, 2, "DET", "LA", "S")
    assert errors == ["2025 wk2 LA@DET: KeyError: (2025, 1)"]
    tpm.raise_hook_errors([], "volume")  # no-op
    with pytest.raises(RuntimeError, match=r"(?s)volume.*1 game.*2025 wk2 LA@DET"):
        tpm.raise_hook_errors(errors, "volume")


def _game_recs(games):
    return [{"season": s, "week": w, "home": h, "player_id": "p", "market": "rec_yds",
             "mean": 1.0, "p50": 1.0, "p90": 2.0, "rps": 1.0, "pit": 0.5, "actual": 1.0}
            for s, w, h in games]


def test_game_set_and_coverage_check():
    base = tpm.game_set(_game_recs([(2025, 1, "KC"), (2025, 1, "DET"), (2025, 2, "KC")]))
    assert base == {(2025, 1, "KC"), (2025, 1, "DET"), (2025, 2, "KC")}
    same = _game_recs([(2025, 2, "KC"), (2025, 1, "DET"), (2025, 1, "KC"), (2025, 1, "KC")])
    assert tpm.check_coverage(base, same, "volume") == 3
    with pytest.raises(RuntimeError) as ei:
        tpm.check_coverage(base, _game_recs([(2025, 1, "KC"), (2025, 2, "KC"), (2025, 3, "NE")]),
                           "context")
    msg = str(ei.value)
    assert "context" in msg and "missing 1" in msg and "(2025, 1, 'DET')" in msg
    assert "extra 1" in msg and "(2025, 3, 'NE')" in msg


# ---- calibration vs baseline --------------------------------------------------------

def _pit_recs(pits, market="rec_yds"):
    return [{"season": 2025, "week": 1 + i % 17, "home": f"T{i % 16}", "player_id": f"p{i}",
             "market": market, "mean": 5.0, "p50": 5.0, "p90": 9.0, "rps": 1.0, "pit": p,
             "actual": 5.0} for i, p in enumerate(pits)]


def test_baseline_ece_check_flags_drift_beyond_tolerance():
    uniform = [(i + 0.5) / 100 for i in range(100)]          # ECE 0 vs deciles
    lumpy = [0.55] * 100                                      # all in one decile
    base, cand_bad, cand_ok = _pit_recs(uniform), _pit_recs(lumpy), _pit_recs(uniform)
    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}

    chk = tpm.baseline_ece_check(base, cand_bad, pop)
    assert chk["pass"] is False
    assert chk["per_market"]["rec_yds"]["n"] == 100
    assert chk["per_market"]["rec_yds"]["ece_base"] == pytest.approx(0.0)
    assert chk["per_market"]["rec_yds"]["ece_cand"] == pytest.approx(0.18)
    assert len(chk["reasons"]) == 1 and "rec_yds" in chk["reasons"][0]
    assert "baseline" in chk["reasons"][0]

    ok = tpm.baseline_ece_check(base, cand_ok, pop)
    assert ok["pass"] is True and ok["reasons"] == []


def test_rung_passes_needs_both_checks():
    assert tpm.rung_passes({"pass": True}, {"pass": True}) is True
    assert tpm.rung_passes({"pass": True}, {"pass": False}) is False
    assert tpm.rung_passes({"pass": False}, {"pass": True}) is False


# ---- identity / checkpoints ---------------------------------------------------------

def test_file_fingerprint_changes_with_content(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"abc")
    fp = tpm.file_fingerprint(f)
    assert fp["size"] == 3 and len(fp["sha256"]) == 64
    f.write_bytes(b"abd")
    assert tpm.file_fingerprint(f) != fp


def test_git_head_falls_back_to_unknown(monkeypatch):
    head = tpm.git_head()
    assert isinstance(head, str) and head

    def boom(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(tpm.subprocess, "run", boom)
    assert tpm.git_head() == "unknown"


def _recs():
    return [
        {"season": 2025, "week": 1, "home": "AAA", "player_id": "p1", "market": "rec_yds",
         "mean": 40.0, "p50": 38.0, "p90": 80.0, "rps": 3.2, "pit": 0.4, "actual": 51.0},
        {"season": 2025, "week": 2, "home": "BBB", "player_id": "p2", "market": "receptions",
         "mean": 4.0, "p50": 4.0, "p90": 7.0, "rps": 0.9, "pit": 0.7, "actual": 5.0},
    ]


def _meta():
    return {"n_sims": 200, "seasons": [2025], "refit_weeks": [1, 5, 9, 13, 17],
            "toggles": ["volume"],
            "identity": {"player_features": {"size": 10, "sha256": "a" * 64},
                         "team_features": {"size": 5, "sha256": "b" * 64},
                         "git_head": "deadbeef", "prod": dict(tpm.PROD), "seed": 42}}


def test_checkpoint_roundtrip_with_stats(tmp_path):
    path = tmp_path / "records_volume.parquet"
    tpm.save_records(path, _recs(), _meta(), {"share_fallbacks": 12, "seconds": 288.5})
    got = tpm.load_records(path, _meta())
    assert got is not None
    recs, stats = got
    assert recs == _recs()
    assert isinstance(recs[0]["season"], int)
    assert stats == {"share_fallbacks": 12, "seconds": 288.5}
    assert tpm.load_records(tmp_path / "missing.parquet", _meta()) is None


@pytest.mark.parametrize("change", [
    lambda m: m.update(n_sims=1000),
    lambda m: m.update(toggles=["context", "volume"]),
    lambda m: m["identity"]["player_features"].update(sha256="c" * 64),
    lambda m: m["identity"]["team_features"].update(size=6),
    lambda m: m["identity"].update(git_head="cafef00d"),
    lambda m: m["identity"]["prod"].update(home_field=0.0),
    lambda m: m["identity"].update(seed=7),
])
def test_checkpoint_rejected_when_meta_or_fingerprint_changes(tmp_path, change):
    path = tmp_path / "records_volume.parquet"
    tpm.save_records(path, _recs(), _meta(), {"share_fallbacks": 0, "seconds": 1.0})
    m = _meta()
    change(m)
    assert tpm.load_records(path, m) is None


# ---- report / json --------------------------------------------------------------------

def _decision(passed, skill=0.01, lo=0.004, hi=0.02, reasons=()):
    return {"skill": skill, "lo": lo, "hi": hi, "pass": passed, "reasons": list(reasons),
            "per_market": {"rec_yds": {"n": 100, "rps_b": 3.0, "rps_c": 2.9,
                                       "ece_b": 0.02, "ece_c": 0.021}}}


def _bcheck(passed, reasons=()):
    return {"pass": passed, "reasons": list(reasons),
            "per_market": {"rush_yds": {"n": 50, "ece_base": 0.0154, "ece_cand": 0.0228}}}


def _gate(final_pass, kept, seasons=(2025,)):
    ladder = [
        {"rung": "volume", "toggles": ["volume"], "compared_against": [],
         "decision": _decision(bool(kept)), "baseline_ece": _bcheck(True),
         "pass": bool(kept), "kept_after": kept, "note": None,
         "n_records": 10, "n_games": 272, "share_fallbacks": 0, "seconds": 1.0},
        {"rung": "efficiency", "toggles": ["efficiency", "volume"], "compared_against": kept,
         "decision": _decision(True),
         "baseline_ece": _bcheck(False, ["market rush_yds: ece_cand (0.0228) > baseline ece "
                                         "(0.0154) + 0.005"]),
         "pass": False, "kept_after": kept, "note": None,
         "n_records": 10, "n_games": 272, "share_fallbacks": 3, "seconds": 1.0},
    ]
    return {"run_date": "2026-09-24", "seasons": list(seasons), "n_sims": 200,
            "refit_schedule": "every 4 weeks", "refit_weeks": [1, 5, 9, 13, 17],
            "tuned": {"2025": [0.8, 150]}, "ladder": ladder, "kept": kept,
            "n_games_baseline": 272,
            "identity": {"git_head": "deadbeef", "seed": 42, "prod": dict(tpm.PROD),
                         "player_features": {"size": 10, "sha256": "a" * 64},
                         "team_features": {"size": 5, "sha256": "b" * 64}},
            "final": {"pass": final_pass,
                      "all": _decision(final_pass, reasons=() if final_pass else
                                       ["market rush_yds: ece_c (0.0228) > ece_b (0.0154) + 0.005"]),
                      "season_2025": _decision(final_pass)},
            "elapsed_s": 12.0}


def test_report_first_paragraph_states_verdict():
    md = tpm.render_report(_gate(False, []))
    first = md.split("\n\n")[1]  # after the title
    assert "FAIL" in first and "no rung" in first.lower()
    md = tpm.render_report(_gate(True, ["volume"], seasons=(2021, 2025)))
    first = md.split("\n\n")[1]
    assert "PASS" in first and "volume" in first
    assert "pooled seasons 2021, 2025" in first and "2025 alone" in first
    assert "every 4 weeks" in md and "0.8" in md
    assert "rec_yds" in md
    assert "per-game seeded random streams (game-level alignment across runs)" in md
    assert "common random numbers" not in md
    assert "deadbeef" in md and "272" in md


def test_report_single_season_wording():
    first = tpm.render_report(_gate(False, ["volume"])).split("\n\n")[1]
    assert "FAIL" in first and "volume" in first
    assert "pooled seasons 2025" not in first and "2025 alone" not in first
    assert "season 2025" in first
    assert "rush_yds" in first  # the failing reason is named


def test_report_shows_baseline_calibration_check_and_rung_result():
    md = tpm.render_report(_gate(False, ["volume"]))
    eff = md[md.index("### efficiency"):]
    assert "Calibration vs baseline" in eff
    assert "0.0154 → 0.0228" in eff
    assert "ece_cand (0.0228) > baseline ece" in eff
    assert "Rung result: **FAIL**" in eff


def test_gate_json_uses_real_booleans_and_native_numbers():
    import json

    import numpy as np
    gate = _gate(False, [])
    gate["ladder"][0]["decision"]["pass"] = np.bool_(False)
    gate["ladder"][0]["pass"] = np.bool_(False)
    gate["ladder"][0]["decision"]["skill"] = np.float64(-0.5)
    gate["ladder"][0]["decision"]["per_market"]["rec_yds"]["n"] = np.int64(100)
    gate["final"]["pass"] = np.bool_(False)
    text = tpm.gate_json(gate)
    assert '"pass": false' in text and '"pass": 0.0' not in text
    back = json.loads(text)
    assert back["ladder"][0]["pass"] is False
    assert back["ladder"][0]["decision"]["pass"] is False
    assert back["final"]["pass"] is False
    assert back["ladder"][0]["decision"]["skill"] == -0.5
    assert back["ladder"][0]["decision"]["per_market"]["rec_yds"]["n"] == 100
