"""migration_site_redesign_a.sql: predictions_any keeps the LIVE column list in order and
appends the model's distributions at the end (CREATE OR REPLACE VIEW fails with 42P16
otherwise). Text/regex assertions only, no database connection."""
import re
from pathlib import Path

MIGRATION = (Path(__file__).parent.parent / "db" / "migration_site_redesign_a.sql").read_text()

# LIVE predictions_any column order (pg_get_viewdef, final whole-branch review, 2026-10-05).
LIVE_PREDICTIONS_ANY_COLS = [
    "sport", "game_pk", "game_date", "home_team_name", "away_team_name", "home_win_prob",
    "pred_home_score", "pred_away_score", "commence_time", "market_spread", "market_total", "model_version",
]


def _view(name):
    m = re.search(rf"CREATE OR REPLACE VIEW {name}\s+AS\s+SELECT\s+DISTINCT ON\s*\(([^)]*)\)\s+(.+?)\s+FROM\s+(\w+)\s+ORDER BY\s+([^;]+);",
                  MIGRATION, re.DOTALL | re.IGNORECASE)
    assert m, f"no {name} definition"
    return m


def test_predictions_any_keeps_live_columns_in_order_and_appends_the_distributions():
    m = _view("predictions_any")
    cols = [c.strip() for c in m.group(2).split(",")]
    assert cols[:len(LIVE_PREDICTIONS_ANY_COLS)] == LIVE_PREDICTIONS_ANY_COLS
    assert cols[len(LIVE_PREDICTIONS_ANY_COLS):] == ["margin_dist", "total_dist"]


def test_predictions_any_same_source_and_latest_row_per_game():
    m = _view("predictions_any")
    assert [c.strip() for c in m.group(1).split(",")] == ["sport", "game_pk"]
    assert m.group(3) == "game_predictions"
    assert " ".join(m.group(4).split()) == "sport, game_pk, generated_at DESC"


def test_grants_and_comments():
    assert "GRANT SELECT ON predictions_any TO anon, authenticated;" in MIGRATION
    assert "42P16" in MIGRATION, "the column-order rule is explained"
    assert re.search(r"CREATE INDEX.*blocks writes to odds_snapshot", MIGRATION, re.DOTALL), "quiet-time note for the index"
    assert MIGRATION.index("CREATE OR REPLACE VIEW line_moves_current") < MIGRATION.index("CREATE OR REPLACE VIEW predictions_any")
