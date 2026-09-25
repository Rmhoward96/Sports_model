"""Test that migration_nfl_sim_serving.sql contains the serving table and
updated views with the model_version filter. Text/regex assertions only,
no database connection."""
import re
from pathlib import Path


MIGRATION_PATH = Path(__file__).parent.parent / "db" / "migration_nfl_sim_serving.sql"
DB_DIR = Path(__file__).parent.parent / "db"

# LIVE column lists verified against Supabase information_schema on 2026-09-25.
# These are the definitive column orders the views MUST have.
LIVE_NFL_SIM_CURRENT_COLS = [
    "game_pk", "model_version", "matchup", "commence_time",
    "sim_home_win_prob", "sim_margin", "sim_total", "disagreement",
    "margin_dist", "total_dist", "away_score_dist", "home_score_dist"
]
LIVE_NFL_PLAYER_SIM_CURRENT_COLS = [
    "game_pk", "player_id", "model_version", "name", "pos", "team",
    "market", "mean", "dist", "commence_time"
]


def read_migration(path):
    """Read migration file as a single string."""
    return path.read_text()


def extract_select_cols(view_name, migration_text):
    """Extract SELECT column list between SELECT and FROM for a view.

    Returns a list of column names in order.
    """
    # Match: CREATE OR REPLACE VIEW view_name AS
    #        SELECT [DISTINCT ON (...)]
    #          col1, col2, ... colN
    #        FROM ...
    # Extract the column list between SELECT and FROM
    pattern = rf"CREATE OR REPLACE VIEW {view_name}\s+AS\s+SELECT\s+(?:DISTINCT ON\s*\([^)]*\)\s+)?(.+?)\s+FROM"
    match = re.search(pattern, migration_text, re.DOTALL | re.IGNORECASE)
    assert match, f"Could not extract columns for {view_name}"
    cols_text = match.group(1)
    # Split on comma and strip whitespace, clean up newlines
    cols = [c.strip() for c in cols_text.split(",")]
    return cols


def find_all_view_definitions(view_name):
    """Find all CREATE OR REPLACE VIEW definitions for a view across all db/*.sql files.

    Excludes migration_nfl_sim_serving.sql. Returns dict of {filepath: column_list}.
    """
    definitions = {}
    for sql_file in sorted(DB_DIR.glob("*.sql")):
        if sql_file.name == "migration_nfl_sim_serving.sql":
            continue
        content = sql_file.read_text()
        pattern = rf"CREATE OR REPLACE VIEW {view_name}\s+AS\s+SELECT\s+(?:DISTINCT ON\s*\([^)]*\)\s+)?(.+?)\s+FROM"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            cols_text = match.group(1)
            cols = [c.strip() for c in cols_text.split(",")]
            definitions[sql_file.name] = cols
    return definitions


def get_view_block(view_name, migration_text):
    """Extract the text block for a view (from CREATE OR REPLACE VIEW to next statement).

    Returns the view's SQL block as a single string.
    """
    pattern = rf"CREATE OR REPLACE VIEW {view_name}\s+AS.*?(?=CREATE OR REPLACE VIEW|GRANT SELECT|$)"
    match = re.search(pattern, migration_text, re.DOTALL | re.IGNORECASE)
    assert match, f"Could not extract view block for {view_name}"
    return match.group(0)


class TestNflSimServingMigration:
    """Verify migration_nfl_sim_serving.sql contains table, seed, and views."""

    def test_migration_file_exists(self):
        """Migration file must exist."""
        assert MIGRATION_PATH.exists(), f"Missing {MIGRATION_PATH}"

    def test_migration_contains_nfl_sim_serving_table(self):
        """Migration must create nfl_sim_serving table."""
        migration = read_migration(MIGRATION_PATH)
        assert "CREATE TABLE IF NOT EXISTS nfl_sim_serving" in migration
        assert "id            INT PRIMARY KEY DEFAULT 1 CHECK (id = 1)" in migration
        assert "model_version TEXT NOT NULL" in migration
        assert "updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()" in migration

    def test_migration_contains_nfl_sim_serving_seed_row(self):
        """Migration must seed nfl_sim_serving with the default version."""
        migration = read_migration(MIGRATION_PATH)
        assert "INSERT INTO nfl_sim_serving (id, model_version) VALUES (1, 'sim-nfl-v1')" in migration
        assert "ON CONFLICT (id) DO NOTHING" in migration

    def test_migration_enables_rls_on_nfl_sim_serving(self):
        """Migration must enable RLS and grant access."""
        migration = read_migration(MIGRATION_PATH)
        assert "ALTER TABLE nfl_sim_serving ENABLE ROW LEVEL SECURITY" in migration
        assert "CREATE POLICY nfl_sim_serving_read ON nfl_sim_serving FOR SELECT USING (true)" in migration
        assert "GRANT SELECT ON nfl_sim_serving TO anon, authenticated" in migration

    def test_migration_contains_nfl_sim_current_view(self):
        """Migration must create nfl_sim_current view."""
        migration = read_migration(MIGRATION_PATH)
        assert "CREATE OR REPLACE VIEW nfl_sim_current AS" in migration

    def test_migration_contains_nfl_player_sim_current_view(self):
        """Migration must create nfl_player_sim_current view."""
        migration = read_migration(MIGRATION_PATH)
        assert "CREATE OR REPLACE VIEW nfl_player_sim_current AS" in migration

    def test_nfl_sim_current_has_serving_filter(self):
        """nfl_sim_current view must have model_version serving filter."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_sim_current", migration)
        assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in view_block

    def test_nfl_player_sim_current_has_serving_filter(self):
        """nfl_player_sim_current view must have model_version serving filter."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_player_sim_current", migration)
        assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in view_block

    def test_nfl_sim_current_column_list_matches_live_definition(self):
        """nfl_sim_current SELECT columns must match LIVE definition (12 cols from migration_nfl_sim_dist.sql)."""
        new = read_migration(MIGRATION_PATH)
        new_cols = extract_select_cols("nfl_sim_current", new)
        assert new_cols == LIVE_NFL_SIM_CURRENT_COLS, f"Column mismatch: {new_cols} != {LIVE_NFL_SIM_CURRENT_COLS}"

    def test_nfl_sim_current_contains_all_columns_from_all_earlier_migrations(self):
        """nfl_sim_current must contain all columns from every earlier definition across all db/*.sql."""
        new = read_migration(MIGRATION_PATH)
        new_cols = extract_select_cols("nfl_sim_current", new)

        # Find all earlier definitions
        earlier_defs = find_all_view_definitions("nfl_sim_current")
        assert len(earlier_defs) > 0, "Must find at least one earlier nfl_sim_current definition in db/*.sql"

        # Assert new version contains all columns from each earlier definition
        for filepath, earlier_cols in earlier_defs.items():
            for col in earlier_cols:
                assert col in new_cols, f"Column {col} from {filepath} missing in new nfl_sim_current"

    def test_nfl_sim_current_from_clause(self):
        """nfl_sim_current must SELECT FROM nfl_sim (not subquery)."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_sim_current", migration)
        match = re.search(
            r"FROM\s+nfl_sim\s+WHERE",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must SELECT FROM nfl_sim (with word boundary)"

    def test_nfl_sim_current_distinct_on(self):
        """nfl_sim_current must use DISTINCT ON (game_pk)."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_sim_current", migration)
        match = re.search(
            r"DISTINCT ON\s*\(\s*game_pk\s*\)",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must use DISTINCT ON (game_pk)"

    def test_nfl_sim_current_where_clause_has_commence_time_check(self):
        """nfl_sim_current WHERE clause must filter by commence_time > now()."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_sim_current", migration)
        match = re.search(
            r"WHERE\s+commence_time\s*>\s*now\(\)",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must have WHERE commence_time > now()"

    def test_nfl_sim_current_order_by(self):
        """nfl_sim_current must ORDER BY game_pk, created_at DESC."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_sim_current", migration)
        match = re.search(
            r"ORDER BY\s+game_pk\s*,\s*created_at\s+DESC",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must ORDER BY game_pk, created_at DESC"

    def test_nfl_player_sim_current_column_list_matches_live_definition(self):
        """nfl_player_sim_current SELECT columns must match LIVE definition."""
        new = read_migration(MIGRATION_PATH)
        new_cols = extract_select_cols("nfl_player_sim_current", new)
        assert new_cols == LIVE_NFL_PLAYER_SIM_CURRENT_COLS, f"Column mismatch: {new_cols} != {LIVE_NFL_PLAYER_SIM_CURRENT_COLS}"

    def test_nfl_player_sim_current_contains_all_columns_from_all_earlier_migrations(self):
        """nfl_player_sim_current must contain all columns from every earlier definition across all db/*.sql."""
        new = read_migration(MIGRATION_PATH)
        new_cols = extract_select_cols("nfl_player_sim_current", new)

        # Find all earlier definitions
        earlier_defs = find_all_view_definitions("nfl_player_sim_current")
        assert len(earlier_defs) > 0, "Must find at least one earlier nfl_player_sim_current definition in db/*.sql"

        # Assert new version contains all columns from each earlier definition
        for filepath, earlier_cols in earlier_defs.items():
            for col in earlier_cols:
                assert col in new_cols, f"Column {col} from {filepath} missing in new nfl_player_sim_current"

    def test_nfl_player_sim_current_from_clause(self):
        """nfl_player_sim_current must SELECT FROM nfl_player_sim (not subquery)."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_player_sim_current", migration)
        match = re.search(
            r"FROM\s+nfl_player_sim\s+WHERE",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_player_sim_current must SELECT FROM nfl_player_sim (with word boundary)"

    def test_nfl_player_sim_current_distinct_on(self):
        """nfl_player_sim_current must use DISTINCT ON (game_pk, player_id, market)."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_player_sim_current", migration)
        match = re.search(
            r"DISTINCT ON\s*\(\s*game_pk\s*,\s*player_id\s*,\s*market\s*\)",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_player_sim_current must use DISTINCT ON (game_pk, player_id, market)"

    def test_nfl_player_sim_current_where_clause_has_commence_time_check(self):
        """nfl_player_sim_current WHERE clause must filter by commence_time > now()."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_player_sim_current", migration)
        match = re.search(
            r"WHERE\s+commence_time\s*>\s*now\(\)",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_player_sim_current must have WHERE commence_time > now()"

    def test_nfl_player_sim_current_order_by(self):
        """nfl_player_sim_current must ORDER BY game_pk, player_id, market, created_at DESC."""
        migration = read_migration(MIGRATION_PATH)
        view_block = get_view_block("nfl_player_sim_current", migration)
        match = re.search(
            r"ORDER BY\s+game_pk\s*,\s*player_id\s*,\s*market\s*,\s*created_at\s+DESC",
            view_block,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_player_sim_current must ORDER BY game_pk, player_id, market, created_at DESC"

    def test_migration_grants_access_to_both_views(self):
        """Migration must grant SELECT to both views."""
        migration = read_migration(MIGRATION_PATH)
        assert "GRANT SELECT ON nfl_sim_current, nfl_player_sim_current TO anon, authenticated" in migration

    def test_migration_includes_verification_comment(self):
        """Migration must include verification instructions."""
        migration = read_migration(MIGRATION_PATH)
        assert "Verify: SELECT count(*) FROM nfl_sim_current" in migration

    def test_migration_includes_updated_at_in_go_live(self):
        """Migration must include updated_at = now() in go-live UPDATE."""
        migration = read_migration(MIGRATION_PATH)
        assert "UPDATE nfl_sim_serving SET model_version = 'nfl-sim-ml-v1', updated_at = now()" in migration

    def test_migration_includes_updated_at_in_rollback(self):
        """Migration must include updated_at = now() in rollback UPDATE."""
        migration = read_migration(MIGRATION_PATH)
        assert "UPDATE nfl_sim_serving SET model_version = 'sim-nfl-v1', updated_at = now()" in migration
