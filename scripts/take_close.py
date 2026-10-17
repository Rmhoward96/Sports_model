"""Grade and close capture for the 'take these' list (.github/workflows/take-these-close.yml).

Every 10 minutes (minutes 3, 13, ... 53), in this order:

  1. GRADE (free, always runs, even with no credits): every pick whose game has started and that has no result yet is
     graded against its stored close (CLV, direction, or no_close) and written to take_results.
  2. TARGETS: leagues and markets holding ungraded picks whose game starts in 5-20 minutes. None: done, nothing spent.
  3. A free /sports call refreshes the credit header; below 300 credits remaining the paid calls are skipped.
  4. Per target league, ONE odds call for bookmakers=pinnacle and only the markets with a target pick: one credit per
     market. Credit math: spec section 4.
  5. Each returned event's start time is refreshed from the RESPONSE (a moved game is not trusted at its stored time) and
     Pinnacle's prices become the close for each target market, stored only if the game starts 0-30 minutes from now.

A failed league warns and moves on; nothing is retried. The run exits 1 only when every league it attempted failed.
Missing tables: warn and exit 0.

    uv run python scripts/take_close.py [--dry-run]

--dry-run still makes the paid odds calls but writes nothing (not even grades): it prints what it would grade and store.
SM_LEDGER_OFF also means a dry run.
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
from sportsmodel.take.grade import grade_pick  # noqa: E402
from sportsmodel.take.parse import parse_events, parse_prices  # noqa: E402


@dataclass
class Deps:
    """Every side effect of a run, injectable so the whole flow is testable without a network or a database."""
    missing_tables: Callable[[], list[str]]
    prime: Callable[[], Any]                                           # the free /sports call: refreshes the credit header
    fetch_odds: Callable[[Any, list[str], list[str]], list[dict]]      # (cfg, bookmakers, markets)
    remaining: Callable[[], Any]
    last_cost: Callable[[], Any]
    ungraded_started: Callable[[datetime], list[dict]]
    insert_results: Callable[[list[dict]], int]
    close_targets: Callable[[datetime, float, float], dict[str, set[str]]]
    upsert_closes: Callable[[list[dict]], int]
    upsert_events: Callable[[list[dict]], int]
    log_call: Callable[..., None]


def _warn(msg: str) -> None:
    print(f"::warning::take-close: {msg}", flush=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _grade(now: datetime, deps: Deps, dry_run: bool, out: dict) -> None:
    rows = [grade_pick(p, p["close"], now) for p in deps.ungraded_started(now)]
    out["graded"] = len(rows)
    for r in rows:
        if dry_run:
            print(f"[take-close] DRY RUN would grade: pick {r['pick_id']} {r['graded_by']} clv={r['clv']}", flush=True)
    if rows and not dry_run:
        deps.insert_results(rows)


def _close_rows(response: list[dict], league: str, markets: set[str], now: datetime) -> tuple[list[dict], list[dict]]:
    """(event rows with the response's start time, close rows within the valid window) from one Pinnacle response."""
    events = parse_events(response, league)
    minutes = {e["event_id"]: (e["commence_time"] - now).total_seconds() / 60 for e in events}
    prices: dict[tuple[str, str], dict] = {}
    for r in parse_prices(response, league, now):
        if r["book"] == "pinnacle" and r["market"] in markets:
            prices.setdefault((r["event_id"], r["market"]), {})[r["side"]] = {"line": r["line"], "price": r["price"]}
    closes = [{"event_id": eid, "market": market, "prices": sides, "captured_at": now, "minutes_to_start": minutes[eid]}
              for (eid, market), sides in prices.items() if 0 <= minutes[eid] <= C.CLOSE_VALID_MAX_MIN]
    return events, closes


def _close_league(league: str, markets: set[str], now: datetime, deps: Deps, run_id: str, dry_run: bool,
                  out: dict) -> None:
    cfg = C.league(league)
    api_markets = [C.API_MARKET[m] for m in sorted(markets)]
    response = deps.fetch_odds(cfg, [C.REFERENCE], api_markets)
    parsed = parse_credits(deps.last_cost())
    cost = parsed if parsed is not None else len(api_markets)
    out["credits"] += cost                                             # count it the moment it is spent
    events, closes = _close_rows(response, league, markets, now)
    out["closes"] += len(closes)
    out["leagues"][league] = {"events": len(events), "closes": len(closes), "credits": cost}
    if dry_run:
        for c in closes:
            print(f"[take-close] DRY RUN would store close: {league} {c['event_id']} {c['market']} "
                  f"{c['minutes_to_start']:.1f} min to start", flush=True)
        return
    deps.upsert_events(events)
    if closes:
        deps.upsert_closes(closes)
    priced = any(b.get("key") == C.REFERENCE for e in response for b in e.get("bookmakers") or [])
    missing = [] if priced else [C.REFERENCE]
    deps.log_call("close", league, "odds", cost, parse_credits(deps.remaining()), missing, run_id)


def run_close(now: datetime, deps: Deps, *, run_id: str = "local", dry_run: bool = False) -> dict:
    """Grade started picks, then capture Pinnacle closes for the picks about to start."""
    out: dict = {"status": "ok", "graded": 0, "closes": 0, "credits": 0, "leagues": {}, "skipped": []}
    gone = deps.missing_tables()
    if gone:
        _warn(f"table {gone[0]} missing — run db/migration_take_these.sql")
        out["status"] = "no_table"
        return out
    try:
        _grade(now, deps, dry_run, out)                                # free: never gated on credits
    except Exception as exc:  # noqa: BLE001 -- grading is retried by the next run; the closes are time-critical
        _warn(f"grading failed: {type(exc).__name__}: {exc}")
    targets = deps.close_targets(now, C.CLOSE_LO_MIN, C.CLOSE_HI_MIN)
    if not targets:
        return out
    try:
        deps.prime()                                                   # free: reads x-requests-remaining first
    except Exception as exc:  # noqa: BLE001 -- without the header the first paid call simply runs (unknown = run)
        _warn(f"credit check call failed: {type(exc).__name__}: {exc}")
    order = [lg for lg in C.LEAGUES if lg in targets]
    for i, league in enumerate(order):
        ok, why = C.budget_ok(deps.remaining(), C.CLOSE_FLOOR)
        if not ok:
            _warn(why)
            out["skipped"] = order[i:]
            break
        try:
            _close_league(league, targets[league], now, deps, run_id, dry_run, out)
        except Exception as exc:  # noqa: BLE001 -- a failed league is not retried; the next run is 10 minutes away
            _warn(f"{league} close failed: {type(exc).__name__}: {exc}")
            out["leagues"].setdefault(league, {"failed": f"{type(exc).__name__}: {exc}"[:200]})
    out["remaining"] = deps.remaining()
    return out


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db
    from sportsmodel.take import store

    timeouts = {"connect_timeout": db.LEDGER_CONNECT_TIMEOUT, "statement_timeout_ms": db.LEDGER_STATEMENT_TIMEOUT_MS}
    return Deps(
        missing_tables=lambda: store.missing_tables(**timeouts),
        prime=odds.fetch_sports, fetch_odds=odds.fetch_bookmaker_odds,
        remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
        ungraded_started=lambda now: store.ungraded_started(now, **timeouts),
        insert_results=lambda rows: store.insert_results(rows, **timeouts),
        close_targets=lambda now, lo, hi: store.close_targets(now, lo, hi, **timeouts),
        upsert_closes=lambda rows: store.upsert_closes(rows, **timeouts),
        upsert_events=lambda rows: store.upsert_events(rows, **timeouts),
        log_call=lambda *a: store.log_call(*a, **timeouts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="make the paid calls but write nothing; print what it would do")
    args = ap.parse_args(argv)
    dry_run = args.dry_run or writes_disabled()
    for name in ("DATABASE_URL", "ODDS_API_KEY"):
        if not os.getenv(name):
            print(f"::error::take-close: {name} is not set")
            return 1
    result = run_close(_now(), _db_deps(), run_id=os.getenv("GITHUB_RUN_ID") or "local", dry_run=dry_run)
    print(f"[take-close] {result}", flush=True)
    attempted = list(result["leagues"].values())
    if attempted and all("failed" in v for v in attempted):
        print("::error::take-close: every attempted league's close call failed (check ODDS_API_KEY)", flush=True)
        return 1                                       # a dead close job must turn the run red
    return 0


if __name__ == "__main__":
    sys.exit(main())
