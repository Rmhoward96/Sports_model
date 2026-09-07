"""Unit tests for scripts/write_desk_picks.py's pure validation + pre-kickoff
filter functions.

Loads the script module directly (it isn't a package) via importlib, the same
pattern tests/test_grade_predictions.py uses. No network, no DB -- both
validate_picks and writable_picks are pure functions of their arguments.
"""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "write_desk_picks.py"
_spec = importlib.util.spec_from_file_location("write_desk_picks", _SCRIPT_PATH)
write_desk_picks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(write_desk_picks)

validate_picks = write_desk_picks.validate_picks
writable_picks = write_desk_picks.writable_picks


def _pick(**overrides) -> dict:
    base = {
        "sport": "cfb",
        "game_pk": 401671789,
        "model_version": "cfb-v1",
        "game_date": "2026-09-12",
        "commence_time": "2026-09-13T17:00:00Z",
        "matchup": "Ravens @ Chiefs",
        "ml_pick": "home",
        "spread_side": "home",
        "spread_line": -3.5,
        "total_side": "over",
        "total_line": 47.5,
        "confidence": 0.62,
        "conviction_tier": "medium",
        "rationale": "Model favors the home side on both lines.",
        "agent_notes": {"statistics": "..."},
    }
    base.update(overrides)
    return base


# =============================================================================
# validate_picks
# =============================================================================

def test_valid_pick_list_has_no_problems():
    assert validate_picks([_pick()]) == []


def test_invalid_pick_flags_every_bad_field():
    bad = _pick(
        conviction_tier="huge",
        spread_side="left",
        total_side="maybe",
        confidence=1.7,
    )
    del bad["rationale"]

    problems = validate_picks([bad])

    assert problems  # non-empty
    joined = "\n".join(problems)
    assert "conviction_tier" in joined
    assert "spread_side" in joined
    assert "total_side" in joined
    assert "confidence" in joined
    assert "rationale" in joined
    # each problem should name the offending game_pk
    assert all(str(bad["game_pk"]) in p for p in problems)


def test_missing_required_field_is_flagged():
    bad = _pick()
    del bad["matchup"]
    problems = validate_picks([bad])
    assert any("matchup" in p for p in problems)


def test_bad_ml_pick_side_is_flagged():
    bad = _pick(ml_pick="tie")
    problems = validate_picks([bad])
    assert any("ml_pick" in p for p in problems)


def test_spread_side_without_spread_line_is_flagged():
    bad = _pick(spread_line=None)  # spread_side still set to "home"
    problems = validate_picks([bad])
    assert any("spread_line" in p or "spread_side" in p for p in problems)


def test_no_market_no_side_is_valid():
    # A game with no spread market at all: both side and line are None -- not
    # flagged, since the desk simply declined to pick that market.
    ok = _pick(spread_side=None, spread_line=None, total_side=None, total_line=None)
    assert validate_picks([ok]) == []


def test_non_numeric_line_is_flagged():
    bad = _pick(spread_line="pick'em")
    problems = validate_picks([bad])
    assert any("spread_line" in p for p in problems)


# =============================================================================
# writable_picks
# =============================================================================

def test_writable_picks_drops_started_games_keeps_future_ones():
    now = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)
    past = _pick(game_pk=1, commence_time=(now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"))
    future = _pick(game_pk=2, commence_time=(now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"))
    at_kickoff = _pick(game_pk=3, commence_time=now.isoformat().replace("+00:00", "Z"))

    result = writable_picks([past, future, at_kickoff], now)

    game_pks = {p["game_pk"] for p in result}
    assert game_pks == {2}


def test_writable_picks_empty_input():
    assert writable_picks([], datetime.now(timezone.utc)) == []
