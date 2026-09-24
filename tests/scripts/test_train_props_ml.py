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
    assert c["q_weight"] == 0.75


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


# ---- checkpoints ----------------------------------------------------------------

def _recs():
    return [
        {"season": 2025, "week": 1, "home": "AAA", "player_id": "p1", "market": "rec_yds",
         "mean": 40.0, "p50": 38.0, "p90": 80.0, "rps": 3.2, "pit": 0.4, "actual": 51.0},
        {"season": 2025, "week": 2, "home": "BBB", "player_id": "p2", "market": "receptions",
         "mean": 4.0, "p50": 4.0, "p90": 7.0, "rps": 0.9, "pit": 0.7, "actual": 5.0},
    ]


def test_checkpoint_roundtrip_and_meta_mismatch(tmp_path):
    meta = {"n_sims": 200, "seasons": [2025], "refit_weeks": [1, 5, 9, 13, 17],
            "toggles": ["volume"]}
    path = tmp_path / "records_volume.parquet"
    tpm.save_records(path, _recs(), meta)
    got = tpm.load_records(path, meta)
    assert got == _recs()
    assert isinstance(got[0]["season"], int)
    assert tpm.load_records(path, {**meta, "n_sims": 1000}) is None
    assert tpm.load_records(path, {**meta, "toggles": ["context", "volume"]}) is None
    assert tpm.load_records(tmp_path / "missing.parquet", meta) is None


# ---- report ---------------------------------------------------------------------

def _decision(passed, skill=0.01, lo=0.004, hi=0.02, reasons=()):
    return {"skill": skill, "lo": lo, "hi": hi, "pass": passed, "reasons": list(reasons),
            "per_market": {"rec_yds": {"n": 100, "rps_b": 3.0, "rps_c": 2.9,
                                       "ece_b": 0.02, "ece_c": 0.021}}}


def _gate(final_pass, kept):
    ladder = [
        {"rung": "volume", "toggles": ["volume"], "compared_against": [],
         "decision": _decision(bool(kept)), "kept_after": kept, "note": None,
         "n_records": 10, "share_fallbacks": 0, "seconds": 1.0},
        {"rung": "efficiency", "toggles": ["efficiency", "volume"], "compared_against": kept,
         "decision": _decision(False, lo=-0.01, reasons=["bootstrap 2.5% bound (-0.0100) <= 0"]),
         "kept_after": kept, "note": None, "n_records": 10, "share_fallbacks": 0, "seconds": 1.0},
    ]
    return {"run_date": "2026-09-24", "seasons": [2025], "n_sims": 200,
            "refit_schedule": "every 4 weeks", "refit_weeks": [1, 5, 9, 13, 17],
            "tuned": {"2025": [0.8, 150]}, "ladder": ladder, "kept": kept,
            "final": {"pass": final_pass, "all": _decision(final_pass),
                      "season_2025": _decision(final_pass)},
            "elapsed_s": 12.0}


def test_report_first_paragraph_states_verdict():
    md = tpm.render_report(_gate(False, []))
    first = md.split("\n\n")[1]  # after the title
    assert "FAIL" in first and "no rung" in first.lower()
    md = tpm.render_report(_gate(True, ["volume"]))
    first = md.split("\n\n")[1]
    assert "PASS" in first and "volume" in first
    assert "every 4 weeks" in md and "0.8" in md
    assert "bootstrap 2.5% bound" in md
    assert "rec_yds" in md
