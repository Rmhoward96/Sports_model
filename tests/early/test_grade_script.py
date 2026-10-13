"""scripts/grade_early_tickets.py on injected loaders and recorded SQL. No database."""
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from .builders import NOW, StubTable, mass_at, row

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("grade_early_tickets", ROOT / "scripts" / "grade_early_tickets.py")
ge = importlib.util.module_from_spec(_spec)
sys.modules["grade_early_tickets"] = ge
_spec.loader.exec_module(ge)

KICK = NOW + timedelta(hours=60)
TABLES = {("cfb", "total"): StubTable(mass_at((55, 56), 0.0318))}


def ticket(i=11, pk=1):
    return {"id": i, "sport": "cfb", "game_pk": pk, "market": "total", "side": "over", "book": "draftkings",
            "book_line": 54.5, "price": -110, "observed_at": NOW, "commence_time": KICK}


def close(pk=1):
    at = KICK - timedelta(minutes=15)
    return {pk: [row(pk, "total", "over", "pinnacle", 55.5, -110, KICK, captured_at=at),
                 row(pk, "total", "under", "pinnacle", 55.5, -110, KICK, captured_at=at)]}


def loaders(**over):
    calls = {}
    base = dict(
        missing_tables=lambda: [],
        pending_tickets=lambda days: [ticket()],
        finals=lambda pks: {("cfb", 1): {"actual_margin": 3.0, "actual_total": 60.0}},
        close_rows=lambda pks: close(), later_rows=lambda pks: {},
        load_tables=lambda: TABLES,
        store=lambda rows: calls.setdefault("stored", rows) and len(rows))
    return ge.Loaders(**{**base, **over}), calls


def test_grades_pending_tickets_and_stores_one_row_per_ticket():
    ld, calls = loaders()
    out = ge.grade_all(ld, 14)
    assert out == {"status": "ok", "pending": 1, "graded": 1, "close_kinds": {"pinnacle_novig": 1}}
    (row_,) = calls["stored"]
    assert row_["ticket_id"] == 11 and row_["result"] == "W" and row_["close_ev"] > 0


def test_nothing_pending_stores_nothing():
    ld, calls = loaders(pending_tickets=lambda days: [])
    assert ge.grade_all(ld, 14) == {"status": "ok", "pending": 0, "graded": 0} and "stored" not in calls


def test_a_game_without_a_final_stays_ungraded_and_is_retried_the_next_day():
    ld, calls = loaders(finals=lambda pks: {})
    out = ge.grade_all(ld, 14)
    assert out["pending"] == 1 and out["graded"] == 0 and "stored" not in calls


def test_a_missing_close_is_graded_but_marked_missing():
    ld, calls = loaders(close_rows=lambda pks: {})
    out = ge.grade_all(ld, 14)
    assert out["close_kinds"] == {"missing": 1} and calls["stored"][0]["close_ev"] is None


def test_missing_tables_warn_and_stop(capsys):
    ld, calls = loaders(missing_tables=lambda: ["early_ticket_results"], pending_tickets=lambda d: pytest.fail("queried"))
    assert ge.grade_all(ld, 14) == {"status": "no_table"}
    assert "::warning::grade-early-tickets: table early_ticket_results missing" in capsys.readouterr().out


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert ge.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_days_is_clamped_to_1_60(monkeypatch):
    seen = []
    monkeypatch.setattr(ge, "grade_all", lambda ld, days: seen.append(days) or {})
    monkeypatch.setattr(ge, "_db_loaders", lambda: None)
    monkeypatch.setattr(ge, "writes_disabled", lambda: False)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    for arg in ("0", "-5", "14", "9999"):
        assert ge.main(["--days", arg]) == 0
    assert seen == [1, 1, 14, 60]


def test_main_needs_the_database_url(monkeypatch, capsys):
    monkeypatch.setattr(ge, "writes_disabled", lambda: False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert ge.main([]) == 1 and "DATABASE_URL is not set" in capsys.readouterr().out


def test_default_queries(monkeypatch):
    from sportsmodel import db
    seen = []

    class Cur:
        description = [("a",)]

        def execute(self, sql, params):
            seen.append((" ".join(sql.split()), params))

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn(Cur):
        def cursor(self):
            return Cur()

    monkeypatch.setattr(db, "get_postgres", lambda: Conn())
    ld = ge._db_loaders()
    ld.pending_tickets(14)
    ld.finals([7])
    ld.close_rows([7])
    ld.later_rows([7])
    sql = [s for s, _ in seen]
    assert "FROM early_tickets t" in sql[0] and "interval '4 hours'" in sql[0] and seen[0][1] == (14,)
    assert "NOT EXISTS (SELECT 1 FROM early_ticket_results r WHERE r.ticket_id = t.id)" in sql[0]
    assert "FROM prediction_accuracy" in sql[1] and seen[1][1] == ([7],)
    assert "JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at" in sql[2]
    assert "c.kind = 'lines' AND c.status = 'ok'" in sql[2] and "s.player_name = ''" in sql[2]
    assert "FROM odds_snapshot s" in sql[3] and "interval '3 hours'" in sql[3] and "t.observed_at" in sql[3]
    assert not any(w in s.upper().split() for s in sql for w in ("INSERT", "UPDATE", "DELETE"))
    for col in ("t.id", "t.book_line", "t.price", "t.observed_at", "t.commence_time"):
        assert col in sql[0]
