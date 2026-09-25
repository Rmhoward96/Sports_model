"""Test that migration_nfl_sim_serving.sql contains the serving table and
updated views with the model_version filter. Text/regex assertions only,
no database connection."""
import re
from pathlib import Path


MIGRATION_PATH = Path(__file__).parent.parent / "db" / "migration_nfl_sim_serving.sql"
ORIGINAL_MIGRATION_PATH = Path(__file__).parent.parent / "db" / "migration_nfl_sim.sql"

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
        # The view must have the filter in its WHERE clause
        assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in migration
        # Verify it's in the nfl_sim_current view
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_sim_current AS.*?"
            r"(CREATE OR REPLACE VIEW|GRANT SELECT|$)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        nfl_sim_current_block = match.group(0)
        assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in nfl_sim_current_block

    def test_nfl_player_sim_current_has_serving_filter(self):
        """nfl_player_sim_current view must have model_version serving filter."""
        migration = read_migration(MIGRATION_PATH)
        # The view must have the filter in its WHERE clause
        assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in migration
        # Verify it's in the nfl_player_sim_current view
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_player_sim_current AS.*?"
            r"(GRANT SELECT|$)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        nfl_player_sim_current_block = match.group(0)
        assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in nfl_player_sim_current_block

    def test_nfl_sim_current_column_list_matches_live_definition(self):
        """nfl_sim_current SELECT columns must match LIVE definition (12 cols from migration_nfl_sim_dist.sql)."""
        new = read_migration(MIGRATION_PATH)
        new_cols = extract_select_cols("nfl_sim_current", new)
        assert new_cols == LIVE_NFL_SIM_CURRENT_COLS, f"Column mismatch: {new_cols} != {LIVE_NFL_SIM_CURRENT_COLS}"

    def test_nfl_sim_current_contains_all_columns_from_original_migrations(self):
        """nfl_sim_current must contain all columns from migration_nfl_sim.sql."""
        new = read_migration(MIGRATION_PATH)
        original = read_migration(ORIGINAL_MIGRATION_PATH)

        new_cols = extract_select_cols("nfl_sim_current", new)
        original_cols = extract_select_cols("nfl_sim_current", original)

        for col in original_cols:
            assert col in new_cols, f"Column {col} from original migration missing in new"

    def test_nfl_sim_current_from_clause(self):
        """nfl_sim_current must SELECT FROM nfl_sim."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_sim_current AS.*?FROM\s+nfl_sim",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must SELECT FROM nfl_sim"

    def test_nfl_sim_current_distinct_on(self):
        """nfl_sim_current must use DISTINCT ON (game_pk)."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_sim_current AS.*?DISTINCT ON\s*\(\s*game_pk\s*\)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must use DISTINCT ON (game_pk)"

    def test_nfl_sim_current_where_clause_has_commence_time_check(self):
        """nfl_sim_current WHERE clause must filter by commence_time > now()."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_sim_current AS.*?WHERE\s+commence_time\s*>\s*now\(\)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must have WHERE commence_time > now()"

    def test_nfl_sim_current_order_by(self):
        """nfl_sim_current must ORDER BY game_pk, created_at DESC."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_sim_current AS.*?ORDER BY\s+game_pk\s*,\s*created_at\s+DESC",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_sim_current must ORDER BY game_pk, created_at DESC"

    def test_nfl_player_sim_current_column_list_matches_live_definition(self):
        """nfl_player_sim_current SELECT columns must match LIVE definition."""
        new = read_migration(MIGRATION_PATH)
        new_cols = extract_select_cols("nfl_player_sim_current", new)
        assert new_cols == LIVE_NFL_PLAYER_SIM_CURRENT_COLS, f"Column mismatch: {new_cols} != {LIVE_NFL_PLAYER_SIM_CURRENT_COLS}"

    def test_nfl_player_sim_current_contains_all_columns_from_original_migrations(self):
        """nfl_player_sim_current must contain all columns from migration_nfl_sim.sql."""
        new = read_migration(MIGRATION_PATH)
        original = read_migration(ORIGINAL_MIGRATION_PATH)

        new_cols = extract_select_cols("nfl_player_sim_current", new)
        original_cols = extract_select_cols("nfl_player_sim_current", original)

        for col in original_cols:
            assert col in new_cols, f"Column {col} from original migration missing in new"

    def test_nfl_player_sim_current_from_clause(self):
        """nfl_player_sim_current must SELECT FROM nfl_player_sim."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_player_sim_current AS.*?FROM\s+nfl_player_sim",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_player_sim_current must SELECT FROM nfl_player_sim"

    def test_nfl_player_sim_current_distinct_on(self):
        """nfl_player_sim_current must use DISTINCT ON (game_pk, player_id, market)."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_player_sim_current AS.*?DISTINCT ON\s*\(\s*game_pk\s*,\s*player_id\s*,\s*market\s*\)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_player_sim_current must use DISTINCT ON (game_pk, player_id, market)"

    def test_nfl_player_sim_current_where_clause_has_commence_time_check(self):
        """nfl_player_sim_current WHERE clause must filter by commence_time > now()."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_player_sim_current AS.*?WHERE\s+commence_time\s*>\s*now\(\)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        assert match, "nfl_player_sim_current must have WHERE commence_time > now()"

    def test_nfl_player_sim_current_order_by(self):
        """nfl_player_sim_current must ORDER BY game_pk, player_id, market, created_at DESC."""
        migration = read_migration(MIGRATION_PATH)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_player_sim_current AS.*?ORDER BY\s+game_pk\s*,\s*player_id\s*,\s*market\s*,\s*created_at\s+DESC",
            migration,
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
