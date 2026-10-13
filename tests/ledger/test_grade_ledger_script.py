"""scripts/grade_ledger.py on injected loaders and recorded SQL. No database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("grade_ledger", ROOT / "scripts" / "grade_ledger.py")
gl = importlib.util.module_from_spec(_spec)
sys.modules["grade_ledger"] = gl
_spec.loader.exec_module(gl)

KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)


def loaders(**over):
    calls = {}
    base = dict(
        missing_tables=lambda: [],
        pending_keys=lambda days: ["k1"],
        ledger_rows=lambda keys: [{"id": 1, "pick_key": "k1", "source": "ev_game", "sport": "nfl", "game_pk": 7,
                                   "market": "moneyline", "side": "home", "line": None, "player_id": None,
                                   "player_name": None, "best_price": 120, "observed_at": KICK - timedelta(hours=30),
                                   "commence_time": KICK, "minutes_to_kick": 1800.0}],
        finals=lambda pks: {("nfl", 7): {"actual_margin": 3.0, "actual_total": 40.0}},
        prop_actuals=lambda pks: {},
        close_rows=lambda pks: ([{"game_pk": 7, "market": "moneyline", "side": "home", "book": "pinnacle", "price": 110,
                                  "line": None, "player_name": "", "captured_at": KICK - timedelta(minutes=15)},
                                 {"game_pk": 7, "market": "moneyline", "side": "away", "book": "pinnacle", "price": -125,
                                  "line": None, "player_name": "", "captured_at": KICK - timedelta(minutes=15)}], []),
        store=lambda rows: calls.setdefault("stored", rows) and len(rows))   # remember the rows, return their count
    return gl.Loaders(**{**base, **over}), calls


def test_grades_pending_keys_on_both_bases_and_stores_them():
    ld, calls = loaders()
    out = gl.grade_all(ld, 14)
    assert out["status"] == "ok" and out["keys"] == 1 and out["graded"] == 2
    assert {r["basis"] for r in calls["stored"]} == {"first", "last"} and out["first_basis_close_kinds"] == {"pinnacle_novig": 1}


def test_nothing_pending_stores_nothing():
    ld, calls = loaders(pending_keys=lambda days: [])
    assert gl.grade_all(ld, 14) == {"status": "ok", "keys": 0, "graded": 0} and "stored" not in calls


def test_missing_tables_warn_and_stop(capsys):
    ld, calls = loaders(missing_tables=lambda: ["pick_ledger_results"], pending_keys=lambda d: pytest.fail("queried"))
    assert gl.grade_all(ld, 14) == {"status": "no_table"}
    assert "::warning::grade-ledger: table pick_ledger_results missing" in capsys.readouterr().out


def test_unfinished_games_stay_pending():
    ld, calls = loaders(finals=lambda pks: {})
    assert gl.grade_all(ld, 14)["graded"] == 0 and "stored" not in calls


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert gl.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


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
    ld = gl._db_loaders()
    ld.pending_keys(14)
    ld.ledger_rows(["k"])
    ld.finals([7])
    ld.prop_actuals([7])
    ld.close_rows([7])
    sql = [s for s, _ in seen]
    assert "interval '4 hours'" in sql[0] and "r.basis = 'last'" in sql[0] and seen[0][1] == (14,)
    assert "FROM pick_ledger WHERE pick_key = ANY(%s) ORDER BY pick_key, observed_at, id" in sql[1]
    assert "FROM prediction_accuracy" in sql[2] and "FROM nfl_player_actuals" in sql[3] and "FROM ev_prop_results" in sql[4]
    assert "JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at" in sql[5]
    assert "c.status = 'ok'" in sql[5] and "interval '180 minutes'" in sql[6]
    assert not any(w in s.upper().split() for s in sql for w in ("INSERT", "UPDATE", "DELETE"))


def test_days_is_clamped_to_1_60(monkeypatch):
    seen = []
    monkeypatch.setattr(gl, "grade_all", lambda ld, days: seen.append(days) or {})
    monkeypatch.setattr(gl, "_db_loaders", lambda: None)
    monkeypatch.setattr(gl, "writes_disabled", lambda: False)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    for arg in ("0", "-5", "14", "9999"):
        assert gl.main(["--days", arg]) == 0
    assert seen == [1, 1, 14, 60]
