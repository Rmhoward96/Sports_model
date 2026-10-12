"""db/migration_accuracy_pick_probs.sql: idempotent ADD COLUMN IF NOT EXISTS for the two nullable double columns."""
import re
from pathlib import Path

SQL = (Path(__file__).resolve().parents[1] / "db" / "migration_accuracy_pick_probs.sql").read_text()


def test_adds_both_columns_idempotently_as_nullable_double():
    for col in ("spread_pick_prob", "total_pick_prob"):
        assert re.search(rf"ALTER TABLE prediction_accuracy ADD COLUMN IF NOT EXISTS {col}\s+DOUBLE PRECISION;", SQL)
    code = "\n".join(line for line in SQL.splitlines() if not line.lstrip().startswith("--"))
    assert "NOT NULL" not in code and "DROP" not in code.upper()


def test_matches_the_columns_the_writers_know_about():
    from sportsmodel import db
    assert db._PREDICTION_ACCURACY_OPTIONAL_COLS == ["spread_pick_prob", "total_pick_prob"]
    for col in db._PREDICTION_ACCURACY_OPTIONAL_COLS:
        assert col in SQL
