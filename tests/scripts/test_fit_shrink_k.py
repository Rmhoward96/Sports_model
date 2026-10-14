"""scripts/fit_shrink_k.py: the fit and report from injected rows, the write-only-on-success rule, the guards. No database."""
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel import db
from sportsmodel.shrink import report
from sportsmodel.shrink.store import ShrinkTable, load_table

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fit_shrink_k", ROOT / "scripts" / "fit_shrink_k.py")
fs = importlib.util.module_from_spec(_spec)
sys.modules["fit_shrink_k"] = fs
_spec.loader.exec_module(fs)

AS_OF = datetime(2026, 10, 20, 15, 30, tzinfo=timezone.utc)


def grade(i, week, market="rush_yds", lean="over", result="hit", version="nfl-sim-ml-v2", p=0.6, over=-110, under=-110):
    return {"game_pk": 100 + i // 3, "player_id": f"00-{i}", "market": market, "line": 54.5, "over_price": over,
            "under_price": under, "p_over": p, "lean": lean, "result": result, "season": 2026, "week": week,
            "model_version": version}


def slate(weeks=(1, 2, 3, 4), per_week=120):
    """Alternating hit / miss leans: the model carries no information, so the fitted k is small but the machinery runs."""
    return [grade(w * 1000 + i, w, result="hit" if i % 2 else "miss") for w in weeks for i in range(per_week)]


def fake_q(rows, served="nfl-sim-ml-v2"):
    def q(sql, params=()):
        return rows if "nfl_prop_grades" in sql else ([{"model_version": served}] if served else [])
    return q


def run_main(monkeypatch, tmp_path, rows, *extra, served="nfl-sim-ml-v2", q=None, as_of=AS_OF):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setattr(fs, "_db_query", lambda: q or fake_q(rows, served))
    out, reports = tmp_path / "shrink_k.json", tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    code = fs.main(["--as-of", as_of.isoformat(), "--out", str(out), "--reports-dir", str(reports), *extra])
    return code, out, reports


# ---- the happy path -----------------------------------------------------------------------------------------------

def test_a_successful_run_writes_the_k_file_and_the_dated_report(monkeypatch, tmp_path, capsys):
    code, out, reports = run_main(monkeypatch, tmp_path, slate())
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["fitted_at"] == AS_OF.isoformat() and doc["current_version"] == "nfl-sim-ml-v2"
    assert doc["pooled"]["n"] == 480 and doc["data_window"] == {"first": [2026, 1], "last": [2026, 4], "n_legs": 480}
    assert doc["markets"]["nfl"]["rush_yds"]["n"] == 480 and all(e["k"] == 0.0 for e in doc["markets"]["mlb"].values())
    (rep,) = reports.glob("*-shrink-gate.md")
    assert rep.name == "2026-10-20-shrink-gate.md" and report.parse_report(rep.read_text())["decision"]["status"] == "collecting"
    assert "wrote shrink_k.json" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.tmp"))


def test_the_written_file_is_read_back_by_the_board_loader(monkeypatch, tmp_path):
    rows = [grade(i, 1 + i // 100, lean="over", result="hit" if i % 5 else "miss", p=0.62) for i in range(500)]
    code, out, _ = run_main(monkeypatch, tmp_path, rows)
    table = load_table(out)
    assert code == 0 and table.problem is None
    assert 0.0 <= table.k("nfl", "rush_yds") <= 1.0 and table.k("mlb", "total_bases") == 0.0
    assert table.k("nfl", "pass_tds") == 0.0 and table.warning() is None      # a market with no legs is an explicit k = 0 entry: no warning


def test_a_pooled_sample_under_300_legs_writes_an_all_zero_file(monkeypatch, tmp_path):
    code, out, _ = run_main(monkeypatch, tmp_path, slate(weeks=(1,), per_week=250))
    doc = json.loads(out.read_text())
    assert code == 0 and doc["pooled"]["n"] == 250 and doc["pooled"]["k"] == 0.0
    assert all(e["k"] == 0.0 for e in doc["markets"]["nfl"].values())


def test_the_current_version_comes_from_the_serving_table_and_falls_back_to_all_versions(monkeypatch, tmp_path):
    rows = [grade(i, 1 + i // 200, version="nfl-sim-ml-v1") for i in range(600)]
    code, out, _ = run_main(monkeypatch, tmp_path, rows, served="nfl-sim-ml-v2")
    assert json.loads(out.read_text())["model_version_used"] == "all"
    code, out2, _ = run_main(monkeypatch, tmp_path, rows, served="nfl-sim-ml-v1")
    assert json.loads(out2.read_text())["model_version_used"] == "nfl-sim-ml-v1"


def test_pushes_invalid_prices_and_unserved_markets_do_not_count(monkeypatch, tmp_path):
    rows = slate(weeks=(1, 2), per_week=100) + [grade(9001, 2, result="push"), grade(9002, 2, over=50),
                                                grade(9003, 2, under=None), grade(9004, 2, market="pass_yds"),
                                                grade(9005, 2, market="rush_att")]
    code, out, _ = run_main(monkeypatch, tmp_path, rows)
    assert json.loads(out.read_text())["data_window"]["n_legs"] == 200


def test_dry_run_prints_and_writes_nothing(monkeypatch, tmp_path, capsys):
    code, out, reports = run_main(monkeypatch, tmp_path, slate(), "--dry-run")
    text = capsys.readouterr().out
    assert code == 0 and not out.exists() and list(reports.iterdir()) == []
    assert '"pooled"' in text and "# Shrink prop probabilities: report of 2026-10-20" in text


# ---- never writes a k it did not compute ---------------------------------------------------------------------------

def test_a_fetch_failure_exits_nonzero_and_leaves_the_existing_file_and_reports_untouched(monkeypatch, tmp_path, capsys):
    out = tmp_path / "shrink_k.json"
    out.write_text('{"keep": "me"}\n')

    def boom(sql, params=()):
        raise ConnectionError("db down")
    code, out, reports = run_main(monkeypatch, tmp_path, [], q=boom)
    assert code == 1 and out.read_text() == '{"keep": "me"}\n' and list(reports.iterdir()) == []
    assert "::error::fit-shrink-k: ConnectionError" in capsys.readouterr().out


def test_a_failure_while_serving_lookup_also_keeps_the_file(monkeypatch, tmp_path):
    out = tmp_path / "shrink_k.json"
    out.write_text("old")

    def half(sql, params=()):
        if "nfl_sim_serving" in sql:
            raise RuntimeError('relation "nfl_sim_serving" does not exist')
        return slate()
    code, out, reports = run_main(monkeypatch, tmp_path, [], q=half)
    assert code == 1 and out.read_text() == "old" and list(reports.iterdir()) == []


def test_a_fit_failure_keeps_the_file(monkeypatch, tmp_path):
    out = tmp_path / "shrink_k.json"
    out.write_text("old")
    monkeypatch.setattr(fs.fit, "build_document", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad fit")))
    code, out, reports = run_main(monkeypatch, tmp_path, slate())
    assert code == 1 and out.read_text() == "old" and list(reports.iterdir()) == []


def test_a_read_with_fewer_legs_than_the_committed_file_is_refused(monkeypatch, tmp_path, capsys):
    code, out, reports = run_main(monkeypatch, tmp_path, slate())
    first = out.read_text()
    (reports / "2026-10-20-shrink-gate.md").unlink()
    code2, out2, _ = run_main(monkeypatch, tmp_path, slate(weeks=(1, 2), per_week=120))
    assert code == 0 and code2 == 1 and out2.read_text() == first and list(reports.iterdir()) == []
    assert "truncated" in capsys.readouterr().out


def test_a_read_with_more_legs_replaces_the_file(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, slate(weeks=(1, 2), per_week=120))
    code, out, _ = run_main(monkeypatch, tmp_path, slate())
    assert code == 0 and json.loads(out.read_text())["data_window"]["n_legs"] == 480


def test_write_atomic_replaces_without_leaving_a_temp_file(tmp_path):
    path = tmp_path / "sub" / "shrink_k.json"
    fs.write_atomic(path, "one")
    fs.write_atomic(path, "two")
    assert path.read_text() == "two" and [p.name for p in path.parent.iterdir()] == ["shrink_k.json"]


# ---- the frozen verdict -------------------------------------------------------------------------------------------

def test_a_previously_judged_verdict_is_carried_into_the_new_report(monkeypatch, tmp_path):
    rows = slate()
    code, out, reports = run_main(monkeypatch, tmp_path, rows)
    frozen = {"status": "judged", "judged_on": "2026-10-13", "scored": 2100, "need_scored": 2000, "running": None,
              "verdict": "PASS", "criteria": []}
    prior = report.render_report(as_of=AS_OF - timedelta(days=7), doc=json.loads(out.read_text()), decision=frozen)
    (reports / "2026-10-13-shrink-gate.md").write_text(prior)
    (reports / "2026-10-20-shrink-gate.md").unlink()
    code, _, reports = run_main(monkeypatch, tmp_path, rows)
    new = report.parse_report((reports / "2026-10-20-shrink-gate.md").read_text())["decision"]
    assert code == 0 and new == frozen


# ---- guards and read-only queries ------------------------------------------------------------------------------------

def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert fs.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_main_needs_the_database_url(monkeypatch, capsys):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert fs.main([]) == 1 and "DATABASE_URL is not set" in capsys.readouterr().out


def test_the_script_never_reaches_a_database_in_tests(monkeypatch):
    monkeypatch.setattr(db, "get_postgres", lambda *a, **k: pytest.fail("no connection expected"))
    assert fs.main([]) == 0                                               # SM_LEDGER_OFF is set by the root conftest


def test_the_queries_are_read_only_and_ask_for_the_legs_version_and_the_served_version():
    seen = []
    fs.collect(lambda sql, params=(): seen.append(sql) or [])
    assert "FROM nfl_prop_grades" in seen[0] and "nfl_player_sim" in seen[0] and "abs(s.mean - g.projection)" in seen[0]
    assert "result IN ('hit', 'miss')" in seen[0] and "FROM nfl_sim_serving" in seen[1]
    assert not any(w in s.upper().split() for s in seen for w in ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "DROP", "ALTER"))


def test_collect_returns_none_when_the_serving_table_has_no_row():
    assert fs.collect(fake_q([], served=None))[1] is None


# ---- the fetched rows carry what the pass test and the version rule need --------------------------------------------

def test_the_fetched_rows_carry_game_pk_and_a_per_leg_model_version():
    seen = []
    rows = slate(weeks=(1,), per_week=6)
    got, served = fs.collect(lambda sql, params=(): seen.append(sql) or (rows if "nfl_prop_grades" in sql else [{"model_version": "nfl-sim-ml-v2"}]))
    select_list = seen[0].split("FROM nfl_prop_grades")[0]
    assert "g.game_pk" in select_list and "v.model_version" in select_list         # nfl_prop_grades has no model_version column: joined
    assert "LEFT JOIN LATERAL" in seen[0] and "FROM nfl_player_sim s" in seen[0]
    assert all(r["game_pk"] is not None and r["model_version"] == "nfl-sim-ml-v2" for r in got) and served == "nfl-sim-ml-v2"
    legs = fs.fit.legs_from_grades(got)
    assert len(legs) == 6 and all(l["game_pk"] is not None and l["version"] == "nfl-sim-ml-v2" for l in legs)


def test_a_leg_with_no_game_pk_fails_the_job_and_keeps_the_file(monkeypatch, tmp_path, capsys):
    out = tmp_path / "shrink_k.json"
    out.write_text("old")
    rows = slate()
    rows[7]["game_pk"] = None
    code, out, reports = run_main(monkeypatch, tmp_path, rows)
    assert code == 1 and out.read_text() == "old" and list(reports.iterdir()) == []
    assert "game_pk" in capsys.readouterr().out


def test_legs_whose_sim_row_is_gone_have_no_version_and_only_count_in_the_all_versions_pool(monkeypatch, tmp_path):
    rows = [grade(i, 1 + i // 200, version=None) for i in range(600)]
    code, out, _ = run_main(monkeypatch, tmp_path, rows, served="nfl-sim-ml-v2")
    doc = json.loads(out.read_text())
    assert code == 0 and doc["model_version_used"] == "all" and doc["pooled"]["n"] == 600


# ---- judged once: a re-run on the day of the judgement cannot re-judge -----------------------------------------------

FROZEN = {"status": "judged", "judged_on": "2026-10-20", "scored": 2100, "need_scored": 2000, "running": None,
          "verdict": "PASS", "criteria": []}


def write_same_day_judged_report(reports):
    empty = fs.fit.build_document([], None, "t")
    (reports / "2026-10-20-shrink-gate.md").write_text(report.render_report(as_of=AS_OF, doc=empty, decision=FROZEN))


@pytest.mark.parametrize("later", [timedelta(0), timedelta(minutes=45), timedelta(hours=8)])
def test_a_same_day_rerun_keeps_the_verdict_judged_earlier_that_day(monkeypatch, tmp_path, later):
    reports = tmp_path / "reports"
    reports.mkdir()
    write_same_day_judged_report(reports)
    code, out, reports = run_main(monkeypatch, tmp_path, slate(), as_of=AS_OF + later)    # this data alone would say "collecting"
    new = report.parse_report((reports / "2026-10-20-shrink-gate.md").read_text())["decision"]
    assert code == 0 and new == FROZEN


def test_running_the_job_twice_in_a_day_judges_once(monkeypatch, tmp_path):
    first_rows = slate(weeks=range(1, 13), per_week=200)
    code, out, reports = run_main(monkeypatch, tmp_path, first_rows)
    first = report.parse_report((reports / "2026-10-20-shrink-gate.md").read_text())["decision"]
    assert code == 0 and first["status"] == "judged" and first["judged_on"] == "2026-10-20"
    more_rows = first_rows + slate(weeks=(13,), per_week=200)                    # a later read, more legs scored
    code, out, reports = run_main(monkeypatch, tmp_path, more_rows, as_of=AS_OF)
    second = report.parse_report((reports / "2026-10-20-shrink-gate.md").read_text())["decision"]
    assert code == 0 and second == first and second["scored"] == first["scored"]
