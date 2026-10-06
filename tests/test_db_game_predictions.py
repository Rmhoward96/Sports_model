"""upsert_game_predictions: a stored commence_time is never overwritten with NULL (no live DB)."""
from sportsmodel import db


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def executemany(self, sql, rows):
        self.sink["sql"] = sql
        self.sink["rows"] = list(rows)

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
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _upsert(monkeypatch, records):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    n = db.upsert_game_predictions(records)
    return n, sink


def test_conflict_update_keeps_stored_commence_time_when_new_is_null(monkeypatch):
    n, sink = _upsert(monkeypatch, [{"game_pk": 1, "model_version": "v", "commence_time": None}])
    assert n == 1
    sql = sink["sql"]
    assert "commence_time = COALESCE(EXCLUDED.commence_time, game_predictions.commence_time)" in sql
    assert "commence_time = EXCLUDED.commence_time" not in sql
    # other columns still take the new value; the key columns are not updated
    assert "home_win_prob = EXCLUDED.home_win_prob" in sql and "game_pk = EXCLUDED" not in sql
    assert "generated_at = now()" in sql


def test_commence_time_still_inserted_and_sport_defaults_to_mlb(monkeypatch):
    _, sink = _upsert(monkeypatch, [{"game_pk": 1, "model_version": "v", "commence_time": "2026-10-06T22:00:00Z"}])
    cols = sink["sql"].split("(")[1].split(")")[0].split(", ")
    row = dict(zip(cols, sink["rows"][0]))
    assert row["commence_time"] == "2026-10-06T22:00:00Z" and row["sport"] == "mlb"
