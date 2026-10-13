"""Hourly early-week poll: log paper tickets where a US book's number beats Pinnacle's fair (profit step 3).

Run hourly from Sunday 16:00 UTC to Wednesday 04:00 UTC in football season (.github/workflows/early-monitor.yml).
Each run, for CFB (spreads + totals, 2 credits) and then NFL (spreads, 1 credit):

  1. Credit floor: below 1,500 Odds API credits remaining the remaining calls are skipped with a warning. A free
     /events call first reads x-requests-remaining, so even the first paid call is guarded; if the value is unknown
     the call runs.
  2. ONE Odds API odds call per sport with `bookmakers` = the 9 US books + pinnacle (10 keys = one billed region).
  3. Events are matched to game_pk with the same matchers as scripts/ingest_odds.py; the rows go to odds_snapshot in
     the shape ingest_odds writes.
  4. The flagging engine (sportsmodel.early.flags) turns this poll's prices into new tickets: one per
     (sport, game_pk, market, side), best US book, EV >= +1.5% against Pinnacle's same-poll fair, kickoff >= 24 h away.
     Tickets are appended right after each sport, so a failure part-way keeps what was already found.

A failed poll is not retried (the next hourly run proceeds normally). Missing tables: warn and exit 0. Under the test
guard SM_LEDGER_OFF: warn and exit 0, --dry-run included. Writes only odds_snapshot and early_tickets.

    uv run python scripts/early_monitor.py [--dry-run]

--dry-run still makes the paid odds calls (about 3 credits) but writes nothing: it prints the tickets it would log.
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

import ingest_odds  # noqa: E402  (scripts/ingest_odds.py: schedule fetchers, event matchers, game lookup)
from sportsmodel import sports  # noqa: E402
from sportsmodel.early import flags, landing, polling  # noqa: E402
from sportsmodel.early.constants import BOOKMAKERS, MARKETS, SPORT_ORDER  # noqa: E402
from sportsmodel.ingest import odds  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402


@dataclass
class Deps:
    """Every side effect of a run, injectable so the whole flow is testable without a network or a database."""
    missing_tables: Callable[[], list[str]]
    load_existing: Callable[[str, datetime], set]                      # (sport, now) -> {(sport, game_pk, market, side)}
    load_tables: Callable[[], dict]                                    # landing tables
    fetch_schedule: Callable[[str], list[dict]]
    match: Callable[[str], Callable]                                   # sport -> matcher(event, games) -> game_pk | None
    prime: Callable[[Any], Any]                                        # the free /events call: refreshes the credit header
    fetch_odds: Callable[[Any, list[str], list[str]], list[dict]]      # (cfg, bookmakers, markets)
    remaining: Callable[[], Any]
    last_cost: Callable[[], Any]
    store_snapshots: Callable[[list[dict]], int]
    store_tickets: Callable[[list[dict]], int]


def _warn(msg: str) -> None:
    print(f"::warning::early-monitor: {msg}", flush=True)


def _poll_sport(sport: str, now: datetime, deps: Deps, tables: dict, run_id: str, dry_run: bool, out: dict) -> None:
    cfg = sports.get(sport)
    schedule = deps.fetch_schedule(sport)                              # free reads first: a failure here costs no credits
    existing = deps.load_existing(sport, now)
    events = deps.fetch_odds(cfg, list(BOOKMAKERS), polling.api_markets(sport))
    cost = polling.parse_credits(deps.last_cost())
    out["credits"] += cost if cost is not None else polling.poll_cost(sport)     # count it the moment it is spent
    lookup = {k: int(pk) for k, pk in ingest_odds.build_game_lookup(events, schedule, deps.match(sport)).items()}
    existing = {(s, int(pk), m, side) for s, pk, m, side in existing}  # same key types as the rows: int game_pk both sides
    captured_at = now.isoformat()
    # `rows` are THIS call's freshly parsed prices only (one poll): flag_poll ignores captured_at, so it must never be
    # handed rows read back from odds_snapshot.
    rows = [r for r in odds.parse_game_odds(events, lookup, captured_at)
            if r["book"] in BOOKMAKERS and r["market"] in MARKETS[sport]]
    tickets = flags.flag_poll(rows, sport=sport, observed_at=now, tables=tables, existing=existing, poll_run_id=run_id)
    out["sports"][sport] = {"events": len(events), "matched": len(lookup), "rows": len(rows), "tickets": len(tickets)}
    if dry_run:
        out["would_write"] += tickets
        return
    if rows:
        try:
            out["snapshots"] += deps.store_snapshots(rows)
        except Exception as exc:  # noqa: BLE001 -- tickets are the point: still log what this poll found
            _warn(f"{sport}: storing the odds snapshot failed: {type(exc).__name__}: {exc}")
    if tickets:
        out["tickets"] += deps.store_tickets(tickets)                  # right now: a later sport's failure loses nothing


def run_poll(now: datetime, deps: Deps, *, run_id: str = "local", dry_run: bool = False) -> dict:
    """One poll of every sport. Returns a summary dict; status 'no_table' means nothing was fetched."""
    gone = deps.missing_tables()
    if gone:
        _warn(f"table {gone[0]} missing — run db/migration_early_tickets.sql")
        return {"status": "no_table"}
    tables = deps.load_tables()
    out: dict = {"status": "ok", "credits": 0, "snapshots": 0, "tickets": 0, "sports": {}, "would_write": [], "skipped": []}
    try:
        deps.prime(sports.get(SPORT_ORDER[0]))                         # free: reads x-requests-remaining before any paid call
    except Exception as exc:  # noqa: BLE001 -- without the header the first paid call simply runs (unknown = run)
        _warn(f"credit check call failed: {type(exc).__name__}: {exc}")
    for sport in SPORT_ORDER:
        decision = polling.budget_decision(deps.remaining())
        if not decision.ok:
            _warn(decision.warning)
            out["skipped"] = [s for s in SPORT_ORDER if s not in out["sports"]]
            break
        try:
            _poll_sport(sport, now, deps, tables, run_id, dry_run, out)
        except Exception as exc:  # noqa: BLE001 -- a failed poll is not retried; the next sport and the next hour go on
            _warn(f"{sport} poll failed: {type(exc).__name__}: {exc}")
            out["sports"].setdefault(sport, {"failed": f"{type(exc).__name__}: {exc}"[:200]})
    out["remaining"] = deps.remaining()
    return out


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db

    timeouts = {"connect_timeout": db.LEDGER_CONNECT_TIMEOUT, "statement_timeout_ms": db.LEDGER_STATEMENT_TIMEOUT_MS}
    return Deps(
        missing_tables=lambda: db.missing_early_tables(**timeouts),
        load_existing=lambda sport, now: db.early_ticket_keys(sport, now, **timeouts),
        load_tables=landing.load_tables,
        fetch_schedule=lambda sport: ingest_odds._ESPN_FETCHERS[sport](),
        match=ingest_odds._matcher_for,
        prime=odds.fetch_events, fetch_odds=odds.fetch_bookmaker_odds,
        remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
        store_snapshots=db.upsert_odds_snapshot,
        store_tickets=lambda rows: db.insert_early_tickets(rows, **timeouts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="poll (about 3 credits) and print the tickets it would log; write nothing")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("early-monitor: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    for name in ("DATABASE_URL", "ODDS_API_KEY"):
        if not os.getenv(name):
            print(f"::error::early-monitor: {name} is not set")
            return 1
    result = run_poll(datetime.now(timezone.utc), _db_deps(), run_id=os.getenv("GITHUB_RUN_ID") or "local",
                      dry_run=args.dry_run)
    for t in result.get("would_write", []):
        print(f"[early-monitor] DRY RUN would log: {t['sport']} {t['game_pk']} {t['market']} {t['side']} "
              f"{t['book']} {t['book_line']:+g} {t['price']:+d} ev={t['ev']:+.2%} kick in {t['hours_to_kick']:.1f}h", flush=True)
    print(f"[early-monitor] { {k: v for k, v in result.items() if k != 'would_write'} }", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
