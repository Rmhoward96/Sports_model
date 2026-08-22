from sportsmodel import db


def test_board_and_picks_helpers_exist():
    for fn in ("upsert_board_picks", "insert_new_picks", "update_graded_picks"):
        assert hasattr(db, fn), f"db.{fn} missing"
