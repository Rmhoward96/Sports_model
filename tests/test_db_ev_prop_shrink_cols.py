"""ev_prop_picks' optional audit columns (raw_model_prob, shrink_k): written only when the table has them, and
db/migration_shrink_props.sql. No live DB (fake connections), same pattern as tests/test_db_prediction_accuracy.py."""
import re
from pathlib import Path

from sportsmodel import db
from sportsmodel import db as db_module

MIGRATION = (Path(__file__).resolve().parents[1] / "db" / "migration_shrink_props.sql").read_text()


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.setdefault("probes", []).append((sql, params))

    def fetchall(self):
        return [(c,) for c in self.sink.get("existing_cols", [])]

    def executemany(self, sql, rows):
        self.sink["sql"], self.sink["rows"] = sql, list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return FakeCursor(self.sink)

    def commit(self):
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def row(**o):
    base = {"sport": "nfl", "game_pk": 1, "player_id": "p", "market": "rush_yds", "line": 54.5, "model_version": "v",
            "model_prob": 0.55, "raw_model_prob": 0.66, "shrink_k": 0.25, "is_pick": True}
    return base | o


def test_the_optional_columns_are_the_two_audit_columns():
    assert db._EV_PROP_PICKS_OPTIONAL_COLS == ["raw_model_prob", "shrink_k"]
    assert not set(db._EV_PROP_PICKS_OPTIONAL_COLS) & set(db._EV_PROP_PICKS_COLS)


def test_both_columns_are_written_after_the_migration_and_updated_on_conflict(monkeypatch):
    sink = {"existing_cols": ["raw_model_prob", "shrink_k"]}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    assert db.upsert_ev_prop_picks([row()]) == 1
    cols = db._EV_PROP_PICKS_COLS + ["raw_model_prob", "shrink_k"]
    (tup,) = sink["rows"]
    assert len(tup) == len(cols) and tup[-2:] == (0.66, 0.25) and tup[cols.index("model_prob")] == 0.55
    assert "raw_model_prob = EXCLUDED.raw_model_prob" in sink["sql"] and "shrink_k = EXCLUDED.shrink_k" in sink["sql"]
    assert "open_pinnacle_price = EXCLUDED" not in sink["sql"]                       # the immutable column rule is unchanged
    assert len(sink["probes"]) == 1 and "information_schema.columns" in sink["probes"][0][0] and sink["committed"]


def test_before_the_migration_the_board_is_written_without_them_and_warns(monkeypatch, capsys):
    sink = {"existing_cols": []}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    assert db.upsert_ev_prop_picks([row(), row(player_id="q")]) == 2
    assert all(len(t) == len(db._EV_PROP_PICKS_COLS) for t in sink["rows"])
    assert "raw_model_prob" not in sink["sql"] and "shrink_k" not in sink["sql"]
    out = capsys.readouterr().out
    assert "WARN" in out and "migration_shrink_props.sql" in out and len(sink["probes"]) == 1


def test_only_the_existing_optional_column_is_written(monkeypatch, capsys):
    sink = {"existing_cols": ["shrink_k"]}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    db.upsert_ev_prop_picks([row()])
    assert sink["rows"][0][-1] == 0.25 and "raw_model_prob" not in sink["sql"] and "shrink_k = EXCLUDED.shrink_k" in sink["sql"]
    assert "raw_model_prob" in capsys.readouterr().out


def test_a_none_shrink_k_is_still_a_carried_key_and_written_as_null(monkeypatch):
    sink = {"existing_cols": ["raw_model_prob", "shrink_k"]}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    db.upsert_ev_prop_picks([row(shrink_k=None)])
    assert sink["rows"][0][-1] is None


def test_no_probe_when_records_carry_neither_key(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    legacy = {k: v for k, v in row().items() if k not in ("raw_model_prob", "shrink_k")}
    assert db.upsert_ev_prop_picks([legacy]) == 1
    assert "probes" not in sink and len(sink["rows"][0]) == len(db._EV_PROP_PICKS_COLS)


def test_an_empty_list_never_connects(monkeypatch):
    monkeypatch.setattr(db_module, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("no connection")))
    assert db.upsert_ev_prop_picks([]) == 0


def test_the_migration_is_the_specs_idempotent_additive_alter():
    stmts = [l for l in MIGRATION.splitlines() if l.strip() and not l.startswith("--")]
    assert stmts == ["ALTER TABLE ev_prop_picks ADD COLUMN IF NOT EXISTS raw_model_prob double precision, "
                     "ADD COLUMN IF NOT EXISTS shrink_k double precision;"]
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE|UPDATE|INSERT)\b", "\n".join(stmts), re.IGNORECASE)


class ProbeFailsCursor(FakeCursor):
    def execute(self, sql, params=None):
        if "information_schema" in sql:
            raise RuntimeError("canceling statement due to statement timeout")
        super().execute(sql, params)

    def executemany(self, sql, rows):
        self.sink.setdefault("events", []).append("upsert")
        super().executemany(sql, rows)


class ProbeFailsConn(FakeConn):
    def cursor(self):
        return ProbeFailsCursor(self.sink)

    def rollback(self):
        self.sink.setdefault("events", []).append("rollback")


def test_a_failed_column_probe_rolls_back_warns_and_writes_the_legacy_columns(monkeypatch, capsys):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: ProbeFailsConn(sink))
    assert db.upsert_ev_prop_picks([row(), row(player_id="q")]) == 2
    assert sink["events"] == ["rollback", "upsert"]                  # the aborted transaction is cleared before the write
    assert all(len(t) == len(db._EV_PROP_PICKS_COLS) for t in sink["rows"])
    assert "raw_model_prob" not in sink["sql"] and "shrink_k" not in sink["sql"] and sink["committed"]
    out = capsys.readouterr().out
    assert "WARN" in out and "RuntimeError" in out
