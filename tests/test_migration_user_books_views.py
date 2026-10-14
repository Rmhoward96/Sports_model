"""db/migration_user_books_views.sql: the SQL views shop exactly the user's 9 books. File-parsing only -- no DB."""
import re
from pathlib import Path

from sportsmodel.serving.board import USER_BOOKS

DB = Path(__file__).resolve().parents[1] / "db"
NEW = DB / "migration_user_books_views.sql"
VIEW = re.compile(r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(\w+)\s+AS\b(.*?);", re.S | re.I)
KEY = re.compile(r"'([a-z0-9_]+)'")
# A hard-coded book list: a `book IN (...)` / `book = ANY (ARRAY[...])` literal that names a sportsbook.
BOOK_LIST = re.compile(r"\bbook\s*(?:IN\s*\(|=\s*ANY\s*\(\s*ARRAY\s*\[)([^\])]*)[\])]", re.S | re.I)


def _strip_comments(sql):
    return re.sub(r"--[^\n]*", "", sql)


def _views(path):
    return {m.group(1): m.group(2) for m in VIEW.finditer(_strip_comments(path.read_text()))}


def _book_lists(view_sql):
    return [KEY.findall(m.group(1)) for m in BOOK_LIST.finditer(view_sql)]


def test_every_view_in_the_migration_lists_exactly_the_users_nine_books():
    views = _views(NEW)
    assert views, "the migration defines no views"
    for name, sql in views.items():
        lists = _book_lists(sql)
        assert lists, f"{name}: no book list found"
        for keys in lists:
            assert tuple(sorted(keys)) == tuple(sorted(USER_BOOKS)) and len(keys) == 9, (name, keys)
            assert not {"bet365", "thescore", "caesars", "bovada", "pinnacle"} & set(keys), (name, keys)


def test_migration_recreates_every_view_that_hard_codes_a_book_list():
    """Scan the other db files: any view with a hard-coded book list must be recreated here (this
    migration is the latest definition of each)."""
    hardcoded = set()
    for path in sorted(DB.glob("*.sql")):
        if path == NEW:
            continue
        for name, sql in _views(path).items():
            if _book_lists(sql):
                hardcoded.add(name)
    assert hardcoded >= {"ev_pick_prices_current", "game_moneylines_current"}   # the scan finds the known ones
    assert hardcoded <= set(_views(NEW)), sorted(hardcoded - set(_views(NEW)))


def test_migration_is_additive_idempotent_and_keeps_grants():
    sql = _strip_comments(NEW.read_text())
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE)\b", sql, re.I)
    assert len(re.findall(r"CREATE\s+VIEW", sql, re.I)) == 0                 # only CREATE OR REPLACE
    assert re.search(r"GRANT\s+SELECT\s+ON\s+ev_pick_prices_current,\s*game_moneylines_current\s+TO\s+anon,\s*authenticated", sql, re.I)


def test_game_moneylines_keeps_the_latest_columns_in_order():
    """CREATE OR REPLACE VIEW needs the same columns in the same order as the latest definition
    (migration_settings_prices.sql: the base five plus the per-book price maps)."""
    sql = _views(NEW)["game_moneylines_current"]
    final = sql[sql.rindex("SELECT b.game_pk"):]
    assert re.findall(r"\bAS\s+(\w+)", final.split("FROM best")[0]) == [
        "home_price", "home_book", "away_price", "away_book", "home_prices", "away_prices"]
