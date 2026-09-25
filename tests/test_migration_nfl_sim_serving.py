"""Test that migration_nfl_sim_serving.sql contains the serving table and
updated views with the model_version filter. Text/regex assertions only,
no database connection."""
import re
from pathlib import Path


MIGRATION_PATH = Path(__file__).parent.parent / "db" / "migration_nfl_sim_serving.sql"
ORIGINAL_MIGRATION_PATH = Path(__file__).parent.parent / "db" / "migration_nfl_sim.sql"


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
    if not match:
        return None
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
        # Verify it's in the nfl_sim_current view (rough check: appears after nfl_sim_current creation)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_sim_current AS.*?"
            r"(CREATE OR REPLACE VIEW|$)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            nfl_sim_current_block = match.group(0)
            assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in nfl_sim_current_block

    def test_nfl_player_sim_current_has_serving_filter(self):
        """nfl_player_sim_current view must have model_version serving filter."""
        migration = read_migration(MIGRATION_PATH)
        # The view must have the filter in its WHERE clause
        assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in migration
        # Verify it's in the nfl_player_sim_current view (rough check)
        match = re.search(
            r"CREATE OR REPLACE VIEW nfl_player_sim_current AS.*?"
            r"(CREATE OR REPLACE VIEW|$)",
            migration,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            nfl_player_sim_current_block = match.group(0)
            assert "AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)" in nfl_player_sim_current_block

    def test_nfl_sim_current_column_list_matches_original(self):
        """nfl_sim_current SELECT columns must match original migration exactly."""
        original = read_migration(ORIGINAL_MIGRATION_PATH)
        new = read_migration(MIGRATION_PATH)

        original_cols = extract_select_cols("nfl_sim_current", original)
        new_cols = extract_select_cols("nfl_sim_current", new)

        assert original_cols is not None, "Could not extract columns from original nfl_sim_current"
        assert new_cols is not None, "Could not extract columns from new nfl_sim_current"
        assert original_cols == new_cols, f"Column mismatch: {original_cols} != {new_cols}"

    def test_nfl_player_sim_current_column_list_matches_original(self):
        """nfl_player_sim_current SELECT columns must match original migration exactly."""
        original = read_migration(ORIGINAL_MIGRATION_PATH)
        new = read_migration(MIGRATION_PATH)

        original_cols = extract_select_cols("nfl_player_sim_current", original)
        new_cols = extract_select_cols("nfl_player_sim_current", new)

        assert original_cols is not None, "Could not extract columns from original nfl_player_sim_current"
        assert new_cols is not None, "Could not extract columns from new nfl_player_sim_current"
        assert original_cols == new_cols, f"Column mismatch: {original_cols} != {new_cols}"

    def test_migration_grants_access_to_both_views(self):
        """Migration must grant SELECT to both views."""
        migration = read_migration(MIGRATION_PATH)
        assert "GRANT SELECT ON nfl_sim_current, nfl_player_sim_current TO anon, authenticated" in migration
