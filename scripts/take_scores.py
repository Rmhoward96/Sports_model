"""Settle graded 'take these' picks against final scores (.github/workflows/take-these-scores.yml).

Once a day. For each league holding a graded pick that has no outcome yet and whose game started over 3 hours ago, in
this order:

  1. A free /sports call refreshes the credit header.
  2. Per league, below 300 credits remaining the paid call is skipped, and so is every league after it.
  3. ONE /scores call (daysFrom=3), the final scores are stored on take_events, and each unsettled pick whose event is
     FINAL is settled at its own line and price (win / loss / push, units at 1 unit staked). A pick whose game is not
     final yet is left for the next run.
  4. The call and its credits are logged to take_api_calls.

A failed league warns and moves on; nothing is retried. The run exits 1 only when every league it attempted failed.
Missing tables: warn and exit 0.

    uv run python scripts/take_scores.py [--dry-run]

--dry-run still makes the paid scores calls but writes nothing: it prints what it would settle. SM_LEDGER_OFF also
means a dry run.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from sportsmodel.early.polling import parse_credits  # noqa: E402
from sportsmodel.ingest import odds  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402
from sportsmodel.take import constants as C  # noqa: E402
from sportsmodel.take.grade import settle  # noqa: E402
from sportsmodel.take.parse import parse_scores  # noqa: E402

DAYS_FROM = 3
SCORES_COST = 2                       # /scores with daysFrom bills 2 credits; used only when the header cannot be read


@dataclass
class Deps:
    """Every side effect of a run, injectable so the whole flow is testable without a network or a database."""
    missing_tables: Callable[[], list[str]]
    prime: Callable[[], Any]                                           # the free /sports call: refreshes the credit header
    fetch_scores: Callable[[Any, int], list[dict]]                     # (cfg, days_from)
    remaining: Callable[[], Any]
    last_cost: Callable[[], Any]
    unsettled_leagues: Callable[[], list[str]]
    unsettled: Callable[[str], list[dict]]
    store_scores: Callable[[dict], int]
    set_outcomes: Callable[[list[dict]], int]
    log_call: Callable[..., None]


def _warn(msg: str) -> None:
    print(f"::warning::take-scores: {msg}", flush=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _scores_league(league: str, deps: Deps, run_id: str, dry_run: bool, out: dict) -> None:
    response = deps.fetch_scores(C.league(league), DAYS_FROM)
    parsed = parse_credits(deps.last_cost())
    cost = parsed if parsed is not None else SCORES_COST
    out["credits"] += cost                                             # count it the moment it is spent
    finals = parse_scores(response)
    rows = []
    for p in deps.unsettled(league):
        if p["event_id"] in finals:                                    # only a FINAL game settles a pick
            outcome, units = settle(p, *finals[p["event_id"]])
            rows.append({"pick_id": p["pick_id"], "outcome": outcome, "units": units})
    out["leagues"][league] = {"finals": len(finals), "settled": len(rows), "credits": cost}
    if dry_run:
        out["settled"] += len(rows)
        for r in rows:
            print(f"[take-scores] DRY RUN would settle: {league} pick {r['pick_id']} {r['outcome']} {r['units']:+.3f}u",
                  flush=True)
        return
    if finals:
        deps.store_scores(finals)
    if rows:
        deps.set_outcomes(rows)
    deps.log_call("scores", league, "scores", cost, parse_credits(deps.remaining()), [], run_id)
    out["settled"] += len(rows)                                        # counted only once stored


def run_scores(now: datetime, deps: Deps, *, run_id: str = "local", dry_run: bool = False) -> dict:
    """Fetch final scores for every league with unsettled graded picks and settle them."""
    out: dict = {"status": "ok", "settled": 0, "credits": 0, "leagues": {}, "skipped": []}
    gone = deps.missing_tables()
    if gone:
        _warn(f"table {gone[0]} missing — run db/migration_take_these.sql")
        out["status"] = "no_table"
        return out
    leagues = deps.unsettled_leagues()
    if not leagues:
        return out
    try:
        deps.prime()                                                   # free: reads x-requests-remaining first
    except Exception as exc:  # noqa: BLE001 -- without the header the first paid call simply runs (unknown = run)
        _warn(f"credit check call failed: {type(exc).__name__}: {exc}")
    for i, league in enumerate(leagues):
        ok, why = C.budget_ok(deps.remaining(), C.SCORES_FLOOR)
        if not ok:
            _warn(why)
            out["skipped"] = leagues[i:]
            break
        try:
            _scores_league(league, deps, run_id, dry_run, out)
        except Exception as exc:  # noqa: BLE001 -- a failed league is not retried; tomorrow's run picks it up again
            _warn(f"{league} scores failed: {type(exc).__name__}: {exc}")
            out["leagues"][league] = {"failed": f"{type(exc).__name__}: {exc}"[:200]}   # also after a paid call: a write failure is a failure
    out["remaining"] = deps.remaining()
    return out


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db
    from sportsmodel.take import store

    timeouts = {"connect_timeout": db.LEDGER_CONNECT_TIMEOUT, "statement_timeout_ms": db.LEDGER_STATEMENT_TIMEOUT_MS}
    return Deps(
        missing_tables=lambda: store.missing_tables(**timeouts),
        prime=odds.fetch_sports, fetch_scores=odds.fetch_scores,
        remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
        unsettled_leagues=lambda: store.unsettled_leagues(**timeouts),
        unsettled=lambda league: store.unsettled(league, **timeouts),
        store_scores=lambda finals: store.store_scores(finals, **timeouts),
        set_outcomes=lambda rows: store.set_outcomes(rows, **timeouts),
        log_call=lambda *a: store.log_call(*a, **timeouts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="make the paid calls but write nothing; print what it would do")
    args = ap.parse_args(argv)
    dry_run = args.dry_run or writes_disabled()
    for name in ("DATABASE_URL", "ODDS_API_KEY"):
        if not os.getenv(name):
            print(f"::error::take-scores: {name} is not set")
            return 1
    result = run_scores(_now(), _db_deps(), run_id=os.getenv("GITHUB_RUN_ID") or "local", dry_run=dry_run)
    print(f"[take-scores] {result}", flush=True)
    attempted = list(result["leagues"].values())
    if attempted and all("failed" in v for v in attempted):
        print("::error::take-scores: every attempted league's scores call failed (check ODDS_API_KEY)", flush=True)
        return 1                                       # a dead scores job must turn the run red
    return 0


if __name__ == "__main__":
    sys.exit(main())
