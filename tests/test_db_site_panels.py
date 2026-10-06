"""db.upsert_game_info / upsert_cfb_team_insights / upsert_cfb_quarter_shares and
db/migration_site_panels.sql: SQL shape, tuple building and column agreement. No live DB (FakeConn)."""
import json
import re
from pathlib import Path

import pytest

from sportsmodel import db

MIGRATION = (Path(__file__).parent.parent / "db" / "migration_site_panels.sql").read_text()


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

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


def _boom():
    raise AssertionError("get_postgres should not be called")


@pytest.mark.parametrize("fn", ["upsert_game_info", "upsert_cfb_team_insights", "upsert_cfb_quarter_shares"])
def test_empty_rows_never_connect(monkeypatch, fn):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert getattr(db, fn)([]) == 0


def test_game_info_sql_keeps_earlier_readings_and_nulls_indoor_weather(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {"sport": "cfb", "game_pk": 7, "venue_name": "V", "city": "C", "state": "AL", "indoor": False,
           "temp_f": 41.0, "wind_mph": 14.0, "precip_chance": None, "precip_in": 0.04, "conditions": None,
           "weather_kind": "forecast", "source": "cfbd", "captured_at": "2026-10-07T14:00:00Z",
           "line_score": {"home": [1, 2, 3, 4], "away": [4, 3, 2, 1]}}
    assert db.upsert_game_info([row]) == 1
    (tup,) = sink["rows"]
    cols = db.SITE_PANEL_COLUMNS["game_info"]
    assert len(tup) == len(cols) and tup[cols.index("sport")] == "cfb" and tup[cols.index("game_pk")] == 7
    assert json.loads(tup[cols.index("line_score")]) == row["line_score"]
    sql = sink["sql"]
    assert "INSERT INTO game_info" in sql and "ON CONFLICT (sport, game_pk) DO UPDATE" in sql
    assert ("temp_f = CASE WHEN EXCLUDED.indoor IS TRUE THEN NULL WHEN EXCLUDED.weather_kind IS NOT NULL "
            "THEN EXCLUDED.temp_f ELSE game_info.temp_f END") in sql
    assert "weather_kind = CASE WHEN EXCLUDED.indoor IS TRUE" in sql
    assert "venue_name = COALESCE(EXCLUDED.venue_name, game_info.venue_name)" in sql
    assert "line_score = COALESCE(EXCLUDED.line_score, game_info.line_score)" in sql
    assert "indoor = COALESCE(EXCLUDED.indoor, game_info.indoor)" in sql
    assert "source = EXCLUDED.source" in sql
    assert ("captured_at = CASE WHEN EXCLUDED.weather_kind IS NOT NULL OR EXCLUDED.indoor IS TRUE "
            "THEN EXCLUDED.captured_at ELSE game_info.captured_at END") in sql
    assert sql.rstrip().endswith("updated_at = now()") and sink["committed"] is True
    row2 = {**row, "line_score": None}
    db.upsert_game_info([row2])
    assert sink["rows"][0][cols.index("line_score")] is None          # NULL stays SQL NULL, not the string "null"


def _set_clause(sql, col):
    m = re.search(rf"\b{col} = (CASE .*?END|COALESCE\([^)]*\)|EXCLUDED\.\w+)", sql.split("DO UPDATE SET")[1])
    assert m, col
    return m.group(1)


def test_game_info_weather_columns_replace_as_a_group_and_every_one_is_covered():
    sql = db._site_panel_sql("game_info")
    weather = ("temp_f", "wind_mph", "precip_chance", "precip_in", "conditions", "weather_kind")
    for col in weather:
        assert _set_clause(sql, col) == (f"CASE WHEN EXCLUDED.indoor IS TRUE THEN NULL "
                                         f"WHEN EXCLUDED.weather_kind IS NOT NULL THEN EXCLUDED.{col} "
                                         f"ELSE game_info.{col} END")      # new kind -> replace (NULLs too); none -> keep old
    for col in ("venue_name", "city", "state", "indoor", "line_score"):
        assert _set_clause(sql, col) == f"COALESCE(EXCLUDED.{col}, game_info.{col})"


def test_captured_at_only_advances_with_a_reading_or_indoor_verdict():
    sql = db._site_panel_sql("game_info")
    assert "captured_at = CASE WHEN EXCLUDED.weather_kind IS NOT NULL OR EXCLUDED.indoor IS TRUE" in sql
    assert "ELSE game_info.captured_at END" in sql
    assert "captured_at = EXCLUDED.captured_at" not in sql


def test_cfb_tables_overwrite_every_column(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    ins = {c: 1 for c in db.SITE_PANEL_COLUMNS["cfb_team_insights"]} | {"team": "333"}
    assert db.upsert_cfb_team_insights([ins]) == 1
    assert "ON CONFLICT (season, team) DO UPDATE" in sink["sql"] and "COALESCE" not in sink["sql"]
    assert "def_havoc_rank = EXCLUDED.def_havoc_rank" in sink["sql"] and "season = EXCLUDED.season" not in sink["sql"]
    sh = {c: 0.25 for c in db.SITE_PANEL_COLUMNS["cfb_quarter_shares"]} | {"team": "333", "games_used": 30}
    assert db.upsert_cfb_quarter_shares([sh]) == 1
    assert "ON CONFLICT (team) DO UPDATE" in sink["sql"] and sink["rows"][0][0] == "333" and sink["rows"][0][-1] == 30


def _create_table(name):
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name} \((.+?)\n\);", MIGRATION, re.DOTALL)
    assert m, f"no CREATE TABLE for {name}"
    return m.group(1)


@pytest.mark.parametrize("table", ["game_info", "cfb_team_insights", "cfb_quarter_shares"])
def test_migration_table_has_every_written_column_pk_rls_policy_and_grant(table):
    body = _create_table(table)
    cols = re.findall(r"^\s{2}(\w+)\s", body, re.MULTILINE)
    assert set(db.SITE_PANEL_COLUMNS[table]) <= set(cols), set(db.SITE_PANEL_COLUMNS[table]) - set(cols)
    assert "updated_at" in cols
    key = ", ".join(db.SITE_PANEL_KEYS[table])
    assert f"PRIMARY KEY ({key})" in body or (len(db.SITE_PANEL_KEYS[table]) == 1 and "PRIMARY KEY" in body)
    assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;" in MIGRATION
    assert f'CREATE POLICY "public read {table}" ON {table} FOR SELECT USING (true);' in MIGRATION
    assert re.search(rf"GRANT SELECT ON [^;]*\b{table}\b[^;]* TO anon, authenticated;", MIGRATION)


def test_migration_revokes_writes_from_api_roles():
    m = re.search(r"REVOKE ([A-Z, ]+) ON ([^;]+) FROM anon, authenticated;", MIGRATION)
    assert m, "no REVOKE"
    assert set(re.split(r",\s*", m.group(1))) == {"INSERT", "UPDATE", "DELETE", "TRUNCATE"}
    for t in ("game_info", "cfb_team_insights", "cfb_quarter_shares"):
        assert f"public.{t}" in m.group(2)
    assert MIGRATION.index("REVOKE") < MIGRATION.index("GRANT SELECT")


def test_migration_is_idempotent_and_writes_nothing():
    assert MIGRATION.count("CREATE TABLE IF NOT EXISTS") == 3
    assert MIGRATION.count("DROP POLICY IF EXISTS") == 3
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE|DROP TABLE|TRUNCATE)\b", MIGRATION, re.MULTILINE | re.IGNORECASE)
