"""upsert_team_betting_records: column-order tuple building + SQL shape, no live
DB -- mirrors tests/test_db_nfl_player_actuals.py's FakeConn pattern."""
from sportsmodel import db
from sportsmodel import db as db_module


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink
        self.rowcount = 0

    def executemany(self, sql, rows):
        self.sink["sql"] = sql
        self.sink["rows"] = list(rows)
        self.rowcount = len(self.sink["rows"])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink):
        self.sink = sink
        self.committed = False

    def cursor(self):
        return FakeCursor(self.sink)

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_upsert_team_betting_records_helper_exists():
    assert hasattr(db, "upsert_team_betting_records")


def test_upsert_team_betting_records_empty_returns_zero(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_team_betting_records([]) == 0


def test_upsert_team_betting_records_tuple_and_sql(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    row = {"sport": "nfl", "season": 2026, "team_name": "Buffalo Bills",
           "an_team_name": "Buffalo Bills", "abbr": "BUF",
           "records": {"ats": {"w": 2, "l": 1, "p": 1, "o": None, "u": None}}}
    assert db.upsert_team_betting_records([row]) == 1
    (tup,) = sink["rows"]
    cols = db._TEAM_RECORD_COLS
    assert tup[cols.index("sport")] == "nfl"
    assert tup[cols.index("season")] == 2026
    assert tup[cols.index("team_name")] == "Buffalo Bills"
    assert tup[cols.index("records")] == '{"ats": {"w": 2, "l": 1, "p": 1, "o": null, "u": null}}'
    assert "INSERT INTO team_betting_records" in sink["sql"]
    assert "ON CONFLICT (sport, season, team_name) DO UPDATE" in sink["sql"]
    assert "captured_at = now()" in sink["sql"]


def test_upsert_team_betting_records_commit_called(monkeypatch):
    sink = {}
    holder = {}

    def fake():
        c = FakeConn(sink)
        holder["c"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake)
    rows = [
        {"sport": "nfl", "season": 2026, "team_name": "Buffalo Bills",
         "an_team_name": "Buffalo Bills", "abbr": "BUF", "records": {}},
        {"sport": "cfb", "season": 2026, "team_name": "Alabama Crimson Tide",
         "an_team_name": "Alabama Crimson Tide", "abbr": None, "records": {}},
    ]
    db.upsert_team_betting_records(rows)
    assert len(sink["rows"]) == 2
    assert holder["c"].committed is True


class FakeCursorForReplace:
    """FakeCursor for replace_nfl_game_trends with execute for DELETE."""
    def __init__(self, sink):
        self.sink = sink
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.sink["execute_sql"] = sql
        self.sink["execute_params"] = params
        self.rowcount = 0

    def executemany(self, sql, rows):
        self.sink["executemany_sql"] = sql
        self.sink["executemany_rows"] = list(rows)
        self.rowcount = len(self.sink["executemany_rows"])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConnForReplace:
    def __init__(self, sink):
        self.sink = sink
        self.committed = False

    def cursor(self):
        return FakeCursorForReplace(self.sink)

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_replace_nfl_game_trends_helper_exists():
    assert hasattr(db, "replace_nfl_game_trends")


def test_replace_nfl_game_trends_empty_returns_zero(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.replace_nfl_game_trends([], []) == 0


def test_replace_nfl_game_trends_delete_then_insert_one_transaction(monkeypatch):
    sink = {}
    holder = {}

    def fake():
        c = FakeConnForReplace(sink)
        holder["c"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake)
    game_pks = [123, 456]
    rows = [
        {"game_pk": 123, "team_name": "BUF", "situation": "home", "label": "Home",
         "ats_w": 10, "ats_l": 5, "ats_p": 0.67, "ou_o": 8, "ou_u": 7, "ou_p": 0.53, "n": 15, "since_season": 2022},
        {"game_pk": 123, "team_name": "KC", "situation": "favorite", "label": "Favorite",
         "ats_w": 5, "ats_l": 2, "ats_p": 0.71, "ou_o": 3, "ou_u": 4, "ou_p": 0.43, "n": 7, "since_season": 2023},
    ]
    result = db.replace_nfl_game_trends(game_pks, rows)
    assert result == 2

    # Check DELETE was called with correct SQL and params
    assert "DELETE FROM nfl_game_trends WHERE game_pk = ANY(%s)" in sink["execute_sql"]
    assert sink["execute_params"] == (game_pks,)
    # Check INSERT was called with correct SQL and rows
    assert "INSERT INTO nfl_game_trends" in sink["executemany_sql"]
    assert len(sink["executemany_rows"]) == 2
    # Check commit was called
    assert holder["c"].committed is True


def test_replace_nfl_game_trends_tuple_column_order(monkeypatch):
    sink = {}
    holder = {}

    def fake():
        c = FakeConnForReplace(sink)
        holder["c"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake)
    rows = [
        {"game_pk": 789, "team_name": "NYG", "situation": "off_road", "label": "Off a road game",
         "ats_w": 12, "ats_l": 8, "ats_p": 0.6, "ou_o": 10, "ou_u": 9, "ou_p": 0.53, "n": 19, "since_season": 2021},
    ]
    db.replace_nfl_game_trends([789], rows)

    (tup,) = sink["executemany_rows"]  # Get the row tuple from executemany
    cols = db._NFL_GAME_TRENDS_COLS
    assert tup[cols.index("game_pk")] == 789
    assert tup[cols.index("team_name")] == "NYG"
    assert tup[cols.index("situation")] == "off_road"
    assert tup[cols.index("label")] == "Off a road game"
    assert tup[cols.index("ats_w")] == 12
    assert tup[cols.index("since_season")] == 2021
