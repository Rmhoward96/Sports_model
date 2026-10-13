"""scripts/capture_close.py gains early_tickets as a candidate source: a ticketed game with no prediction and no ledger
pick still gets its pre-kick line capture, and nothing changes before the migration has run. No network, no database."""
from datetime import timedelta

import pytest

from sportsmodel import db
from tests.ledger import test_capture_close_script as ccs

cc, NOW = ccs.cc, ccs.NOW


def cfb_fake(**kw):
    base = dict(schedule={"cfb": [ccs.game(9, 12)]}, events=[ccs.event("c1", 12)], line_events=[ccs.odds_event("c1", 12)])
    return ccs.Fake(**{**base, **kw})


def test_a_ticketed_game_alone_makes_the_run_non_idle_and_gets_its_lines_captured():
    fake = cfb_fake()                                              # no prediction, no ledger pick: load_candidates is empty
    deps = fake.deps()
    deps.load_early_candidates = lambda now, end: [ccs.cand("cfb", 9, 12)]
    out = cc.run_capture(NOW, deps)
    assert out["rows"] == [("cfb", 9, "lines", "ok")] and ("lines", "cfb", "us,eu") in fake.calls


def test_without_early_candidates_the_same_game_is_idle_as_before():
    fake = cfb_fake()
    assert cc.run_capture(NOW, fake.deps()) == {"status": "idle"} and fake.calls == []


def test_a_far_away_early_candidate_stays_idle():
    fake = cfb_fake()
    deps = fake.deps()
    deps.load_early_candidates = lambda now, end: [ccs.cand("cfb", 9, 600)]
    assert cc.run_capture(NOW, deps) == {"status": "idle"}


def test_both_sources_are_asked_with_the_same_one_hour_horizon():
    fake = cfb_fake()
    deps = fake.deps()
    seen = []
    deps.load_candidates = lambda now, end: seen.append(("base", now, end)) or []
    deps.load_early_candidates = lambda now, end: seen.append(("early", now, end)) or []
    cc.run_capture(NOW, deps)
    assert seen == [("base", NOW, NOW + timedelta(minutes=60)), ("early", NOW, NOW + timedelta(minutes=60))]


class Cur:
    def __init__(self, seen, rows):
        self.seen, self.rows = seen, rows

    def execute(self, sql, params=None):
        self.seen.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Conn(Cur):
    def cursor(self):
        return self


def test_production_query_reads_early_tickets_only_once_the_table_exists(monkeypatch):
    seen = []
    kick = NOW + timedelta(minutes=12)
    monkeypatch.setattr(db, "get_postgres", lambda: Conn(seen, [("cfb", 9, kick)]))
    monkeypatch.setattr(db, "missing_early_tables", lambda tables=None, **kw: [])
    rows = cc._db_deps().load_early_candidates(NOW, NOW + timedelta(hours=1))
    assert rows == [{"sport": "cfb", "game_pk": 9, "commence_time": kick}]
    sql, params = seen[0]
    assert sql.startswith("SELECT DISTINCT sport, game_pk, commence_time FROM early_tickets WHERE commence_time > %s")
    assert params == (NOW, NOW + timedelta(hours=1))


def test_before_the_migration_the_early_source_is_empty_and_runs_no_query(monkeypatch):
    monkeypatch.setattr(db, "get_postgres", lambda: pytest.fail("no query expected"))
    monkeypatch.setattr(db, "missing_early_tables", lambda tables=None, **kw: ["early_tickets"])
    assert cc._db_deps().load_early_candidates(NOW, NOW + timedelta(hours=1)) == []


def test_production_deps_wire_the_early_source():
    deps = cc._db_deps()
    assert deps.load_early_candidates is not None and deps.load_early_candidates.__name__ == "early_candidates"
