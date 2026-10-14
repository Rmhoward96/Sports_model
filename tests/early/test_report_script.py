"""scripts/report_early_tickets.py: decision carry-forward, the quiet-week guard, file output and the guards. No database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel import db
from sportsmodel.early import report

from .builders import res, spreads, totals

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("report_early_tickets", ROOT / "scripts" / "report_early_tickets.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["report_early_tickets"] = rs
_spec.loader.exec_module(rs)

AS_OF = datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc)


def ticket(i, days_ago=1):
    return {"id": i, "sport": "cfb", "market": "total", "side": "over", "observed_at": AS_OF - timedelta(days=days_ago),
            "commence_time": AS_OF - timedelta(hours=60)}


def graded(r, i, days_ago=1):
    return r | {"ticket_id": i, "graded_at": AS_OF - timedelta(days=days_ago)}


def test_a_collecting_report_is_built_from_the_numbers():
    results = [graded(r, i) for i, r in enumerate(totals(10), 1)]
    text, decision = rs.build_report(AS_OF, [ticket(i) for i in range(1, 11)], results, None)
    assert decision["status"] == "collecting" and "COLLECTING" in text and "10 of 150" in text


def test_the_first_report_that_meets_both_minimums_judges_and_later_ones_carry_it_forward():
    rows = [graded(r, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    text, decision = rs.build_report(AS_OF, [], rows, None)
    assert decision["status"] == "judged" and decision["judged_on"] == "2026-11-03"
    later = AS_OF + timedelta(days=14)
    worse = [graded(r | {"close_ev": -0.5, "beat_close": False}, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    text2, decision2 = rs.build_report(later, [], worse, report.parse_report(text)["decision"])
    assert decision2 == decision and "Later weeks below are informational only" in text2


def test_activity_means_a_ticket_logged_or_graded_in_the_last_seven_days():
    assert rs.has_activity([ticket(1, days_ago=2)], [], AS_OF) is True
    assert rs.has_activity([ticket(1, days_ago=9)], [], AS_OF) is False
    assert rs.has_activity([ticket(1, days_ago=9)], [graded(res(), 1, days_ago=1)], AS_OF) is True
    assert rs.has_activity([], [], AS_OF) is False
    assert rs.has_activity([ticket(1, days_ago=9)], [res() | {"ticket_id": 1, "graded_at": None}], AS_OF) is False


def run_main(monkeypatch, tmp_path, tickets, results, *extra, missing=()):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setattr(db, "missing_early_tables", lambda *a, **k: list(missing))
    monkeypatch.setattr(rs, "_db_query", lambda: lambda sql, params=(): tickets if "early_tickets" in sql else results)
    return rs.main(["--as-of", AS_OF.isoformat(), "--reports-dir", str(tmp_path), *extra])


def test_main_writes_a_dated_file_and_the_next_run_reads_its_decision_block(monkeypatch, tmp_path):
    rows = [graded(r, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    assert run_main(monkeypatch, tmp_path, [], rows) == 0
    (path,) = tmp_path.glob("*-early-monitor.md")
    assert path.name == "2026-11-03-early-monitor.md" and "JUDGED on 2026-11-03" in path.read_text()
    assert report.latest_decision([path.read_text()], AS_OF + timedelta(days=7))["status"] == "judged"


def test_a_quiet_week_writes_no_file(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, [ticket(1, days_ago=30)], []) == 0
    assert list(tmp_path.iterdir()) == [] and "no report written" in capsys.readouterr().out


def test_dry_run_prints_and_writes_nothing(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, [ticket(1)], [], "--dry-run") == 0
    assert list(tmp_path.iterdir()) == [] and "# Early-week monitor: report of 2026-11-03" in capsys.readouterr().out


def test_missing_tables_warn_and_exit_zero(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, [], [], missing=["early_tickets"]) == 0
    assert "tables missing" in capsys.readouterr().out and list(tmp_path.iterdir()) == []


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert rs.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_main_needs_the_database_url(monkeypatch, capsys):
    monkeypatch.setattr(rs, "writes_disabled", lambda: False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert rs.main([]) == 1 and "DATABASE_URL is not set" in capsys.readouterr().out


def test_report_queries_are_read_only(monkeypatch):
    seen = []
    monkeypatch.setattr(db, "get_postgres", lambda: pytest.fail("no connection expected"))
    rs.collect(lambda sql, params=(): seen.append(sql) or [])
    assert "FROM early_tickets" in seen[0] and "FROM early_ticket_results" in seen[1]
    assert not any(w in s.upper().split() for s in seen for w in ("INSERT", "UPDATE", "DELETE"))


def prior_report(tmp_path, as_of, results):
    text, decision = rs.build_report(as_of, [], results, None)
    (tmp_path / f"{as_of:%Y-%m-%d}-early-monitor.md").write_text(text)
    return decision


def test_a_prior_judged_fail_is_carried_even_when_new_data_would_pass(monkeypatch, tmp_path):
    bad = [graded(r | {"close_ev": -0.05, "beat_close": False}, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    prior_at = AS_OF - timedelta(days=14)
    frozen = prior_report(tmp_path, prior_at, bad)
    assert frozen["status"] == "judged" and frozen["markets"]["cfb_total"]["verdict"] == "FAIL"
    good = [graded(r, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    assert rs.build_report(AS_OF, [], good, None)[1]["markets"]["cfb_total"]["verdict"] == "PASS"       # sanity: it would pass
    assert run_main(monkeypatch, tmp_path, [], good) == 0
    new = report.parse_report((tmp_path / "2026-11-03-early-monitor.md").read_text())["decision"]
    assert new == frozen and new["judged_on"] == "2026-10-20" and new["markets"]["cfb_total"]["verdict"] == "FAIL"
    assert "JUDGED on 2026-10-20" in (tmp_path / "2026-11-03-early-monitor.md").read_text()


def test_a_corrupt_newer_report_does_not_unfreeze_an_older_judged_verdict(monkeypatch, tmp_path):
    bad = [graded(r | {"close_ev": -0.05, "beat_close": False}, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    frozen = prior_report(tmp_path, AS_OF - timedelta(days=14), bad)
    (tmp_path / "2026-10-27-early-monitor.md").write_text("<!-- early-monitor: {not json} -->")
    good = [graded(r, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    assert run_main(monkeypatch, tmp_path, [], good) == 0
    assert report.parse_report((tmp_path / "2026-11-03-early-monitor.md").read_text())["decision"] == frozen


def test_the_report_shows_stale_ungraded_tickets(monkeypatch, tmp_path):
    old = ticket(7, days_ago=5) | {"commence_time": AS_OF - timedelta(hours=100)}
    assert run_main(monkeypatch, tmp_path, [old, ticket(8) | {"commence_time": AS_OF - timedelta(hours=10)}], []) == 0
    text = (tmp_path / "2026-11-03-early-monitor.md").read_text()
    assert "2 tickets whose game started more than 4 hours ago are not graded yet" in text
    assert "1 of them started more than 48 hours ago and are still ungraded" in text
