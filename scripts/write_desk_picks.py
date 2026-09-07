"""Validate a decision-desk picks JSON and upsert it into `desk_picks`.

This is the AUTHORITATIVE contract for the picks JSON the (in-session) desk
agents produce: see `validate_picks` below for the exact fields, allowed
values, and null rules it enforces. Task 6's runbook documents this contract
for humans; this module is where it's actually enforced.

Pre-kickoff immutability: a pick for a game that has already started is never
(re)written. `main()` filters with `writable_picks` right before upserting,
so a late/re-run write can't clobber a pick after the market it was made
against has closed.

Both `validate_picks` and `writable_picks` are pure (no I/O) -- easy to unit
test in isolation. `main()` is the thin IO wrapper: load JSON, validate
(fail closed -- abort and write nothing if there are any problems), filter to
pre-kickoff, upsert.

Usage:
    DATABASE_URL=... PYTHONPATH=src uv run python scripts/write_desk_picks.py \\
        --in path/to/picks.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import upsert_desk_picks

# =============================================================================
# validate_picks -- the authoritative picks-JSON contract
# =============================================================================

# Required on every pick. `spread_side`/`spread_line`/`total_side`/
# `total_line` are "required" in the sense that the key's ABSENCE is not
# meaningfully different from an explicit null here -- but unlike the other
# fields, null is a legitimate value for them (see NULLABLE_FIELDS below),
# used when the game simply had no market on that side.
REQUIRED_FIELDS = [
    "sport", "game_pk", "model_version", "commence_time", "matchup",
    "ml_pick", "spread_side", "spread_line", "total_side", "total_line",
    "confidence", "conviction_tier", "rationale",
]

# These four may be None -- "the game had no line/side to pick on this
# market" -- without being flagged as a missing required field. Every other
# required field must be present and non-null.
NULLABLE_FIELDS = {"spread_side", "spread_line", "total_side", "total_line"}

CONVICTION_TIERS = {"high", "medium", "low"}
ML_SIDES = {"home", "away"}
SPREAD_SIDES = {"home", "away"}
TOTAL_SIDES = {"over", "under"}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_picks(picks: list[dict]) -> list[str]:
    """Validate a list of pick dicts against the desk_picks contract. PURE.

    Returns a list of human-readable problem strings, each naming the
    offending game_pk and field; an empty list means every pick is valid.

    Contract enforced (this IS the picks-JSON spec -- Task 6's runbook
    mirrors it):
      - Required fields present and non-null: sport, game_pk, model_version,
        commence_time, matchup, ml_pick, confidence, conviction_tier,
        rationale. (spread_side, spread_line, total_side, total_line are
        also required KEYS but may be null -- see below.)
      - conviction_tier in {"high", "medium", "low"}.
      - ml_pick in {"home", "away"}.
      - spread_side, when non-null, in {"home", "away"}.
      - total_side, when non-null, in {"over", "under"}.
      - confidence is a number in [0, 1].
      - spread_line/total_line, when non-null, are numeric.
      - Null-line rule: spread_line/total_line MAY be null (the game had no
        market on that side). When null, the matching side
        (spread_side/total_side) must ALSO be null -- a side pick with no
        line to grade it against is flagged. The reverse (a line present but
        no side picked) is NOT flagged: the desk is allowed to decline a
        market that does have a line.
    """
    problems: list[str] = []

    for pick in picks:
        gid = pick.get("game_pk", "<missing game_pk>")

        for field in REQUIRED_FIELDS:
            if pick.get(field) is None and field not in NULLABLE_FIELDS:
                problems.append(f"game {gid}: missing required field '{field}'")

        conviction_tier = pick.get("conviction_tier")
        if conviction_tier is not None and conviction_tier not in CONVICTION_TIERS:
            problems.append(
                f"game {gid}: invalid conviction_tier {conviction_tier!r} "
                f"(must be one of {sorted(CONVICTION_TIERS)})"
            )

        ml_pick = pick.get("ml_pick")
        if ml_pick is not None and ml_pick not in ML_SIDES:
            problems.append(f"game {gid}: invalid ml_pick {ml_pick!r} (must be home/away)")

        spread_side = pick.get("spread_side")
        if spread_side is not None and spread_side not in SPREAD_SIDES:
            problems.append(f"game {gid}: invalid spread_side {spread_side!r} (must be home/away)")

        total_side = pick.get("total_side")
        if total_side is not None and total_side not in TOTAL_SIDES:
            problems.append(f"game {gid}: invalid total_side {total_side!r} (must be over/under)")

        confidence = pick.get("confidence")
        if confidence is not None:
            if not _is_number(confidence) or not (0.0 <= confidence <= 1.0):
                problems.append(f"game {gid}: confidence {confidence!r} must be a number in [0, 1]")

        spread_line = pick.get("spread_line")
        if spread_line is not None and not _is_number(spread_line):
            problems.append(f"game {gid}: spread_line {spread_line!r} is not numeric")

        total_line = pick.get("total_line")
        if total_line is not None and not _is_number(total_line):
            problems.append(f"game {gid}: total_line {total_line!r} is not numeric")

        if spread_side is not None and spread_line is None:
            problems.append(f"game {gid}: spread_side is set but spread_line is null")

        if total_side is not None and total_line is None:
            problems.append(f"game {gid}: total_side is set but total_line is null")

    return problems


# =============================================================================
# writable_picks -- pre-kickoff immutability filter
# =============================================================================

def _parse_iso(ts: str) -> datetime:
    """ISO-8601 -> aware datetime, tolerating a trailing 'Z' (mirrors
    desk_inputs._parse_iso)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def writable_picks(picks: list[dict], now: datetime) -> list[dict]:
    """Only picks whose commence_time is STRICTLY after `now`. PURE.

    Enforces pre-kickoff immutability: a pick for a game that has already
    started (commence_time <= now) is dropped and never (re)written.
    """
    return [p for p in picks if _parse_iso(p["commence_time"]) > now]


# =============================================================================
# fill_missing_game_dates -- safety net for a missing game_date
# =============================================================================

def _game_date_from_commence(commence_iso: str) -> str:
    """US game date from a UTC `commence_time`.

    Same 8h-shift trick as `generate_cfb.py::_game_date_from_commence`: CFB
    kickoffs run from ~15:00 UTC through ~04:00-07:00 UTC the next day, and
    shifting back 8h before taking the date maps every real kickoff onto its
    true US game day without a timezone lookup.
    """
    dt = _parse_iso(commence_iso)
    return (dt - timedelta(hours=8)).date().isoformat()


def fill_missing_game_dates(picks: list[dict]) -> list[dict]:
    """Derive `game_date` from `commence_time` wherever it's missing. PURE.

    `validate_picks` does not require `game_date`, but
    `grade_desk_picks._pending_desk_picks` windows its query on
    `game_date >= start` -- a pick written with `game_date = NULL` would
    silently never be graded (SQL `NULL >= anything` is never true). Rather
    than force every caller of this script to supply `game_date` (fragile --
    an agent-produced picks JSON could easily omit it), derive it here from
    `commence_time`, which IS required. A pick that already has an explicit
    `game_date` is left untouched.
    """
    filled = []
    for p in picks:
        if p.get("game_date") is None:
            p = {**p, "game_date": _game_date_from_commence(p["commence_time"])}
        filled.append(p)
    return filled


# =============================================================================
# main()
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Validate + upsert decision-desk picks.")
    ap.add_argument("--in", dest="in_path", type=Path, required=True,
                     help="Path to the picks JSON (a list of pick dicts).")
    args = ap.parse_args()

    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (writing picks upserts to Supabase).")

    picks = json.loads(args.in_path.read_text())

    problems = validate_picks(picks)
    if problems:
        print(f"{len(problems)} problem(s) found in {args.in_path}; writing NOTHING:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    picks = fill_missing_game_dates(picks)

    now = datetime.now(timezone.utc)
    to_write = writable_picks(picks, now)
    skipped = len(picks) - len(to_write)

    rows = [
        {**p, "agent_notes": json.dumps(p.get("agent_notes"))}
        for p in to_write
    ]
    written = upsert_desk_picks(rows)

    print(f"Upserted {written} desk_picks row(s); skipped {skipped} already-started game(s).")


if __name__ == "__main__":
    main()
