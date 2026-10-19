"""Watcher for the 'take these' list: one long-running job that runs the close and the pull on time
(.github/workflows/take-these-watch.yml).

GitHub cron is best-effort and delays or drops runs (on 2026-10-18 the pull ran 6 times, 15-55 minutes late, and the 10-minute
close never landed inside a game's pre-start window). So one scheduled job starts every 2 hours and stays alive for
C.WATCH_MINUTES (340) minutes; the next scheduled start takes over when it ends. Nothing here restarts or dispatches a workflow.

Ticks are anchored to the wall clock: tick k is due at start + k * tick (C.WATCH_TICK_SEC, 300). After the work the loop sleeps
until the next FUTURE slot; if the work overran, the missed slots are skipped, never bunched. The loop stops when the next slot
would be at or past the deadline, and exits 0. Each tick, in this order, each part in its own try/except (a failure warns and the
loop goes on):

  1. CLOSE: take_close.run_close (grade started picks, capture Pinnacle closes for picks starting in 5-12 minutes).
  2. PULL, when due: take_pull.run_pull. Due means the UTC hour is in C.PULL_HOURS_UTC (12-23 and 0-4) and the last logged pull
     (max take_api_calls.called_at where job = 'pull') is at least TAKE_PULL_EVERY_HOURS (default 2; a bad, empty or < 1 value
     counts as 2) minus 10 minutes ago, or there is none. A pull that spent no credits logs nothing, so the next tick re-checks
     (the free /events calls cost nothing). A pull that DID spend credits is also remembered in memory, so a dry run (which logs
     nothing) cannot repeat the paid pull every tick.

    uv run python scripts/take_watch.py [--dry-run] [--minutes N] [--tick-seconds S]

--dry-run is passed to both parts: paid calls are still made, nothing is written. SM_LEDGER_OFF also means a dry run.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import take_close  # noqa: E402
import take_pull  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402
from sportsmodel.take import constants as C  # noqa: E402

PULL_SLACK = timedelta(minutes=10)                  # a pull is due this much before a full cadence has passed


def _warn(msg: str) -> None:
    print(f"::warning::take-watch: {msg}", flush=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Deps:
    """Every side effect of the loop, injectable so tests need no clock, network or database."""
    close_deps: Any                                                    # take_close.Deps
    pull_deps: Any                                                     # take_pull.Deps
    last_pull_at: Callable[[], datetime | None]
    now: Callable[[], datetime] = _now
    sleep: Callable[[float], None] = time.sleep


def every_hours_from(raw) -> int:
    """The pull cadence from a repo variable; a bad, empty or < 1 value counts as 2 (the same rule as take_pull.due)."""
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 2
    return n if n >= 1 else 2


def pull_due(now: datetime, last_pull_at: datetime | None, every_hours: int) -> bool:
    """Is a pull due: a pull hour, and none logged yet or one old enough (a full cadence less 10 minutes)."""
    if now.hour not in C.PULL_HOURS_UTC:
        return False
    return last_pull_at is None or now - last_pull_at >= timedelta(hours=every_hours) - PULL_SLACK


def _read_last_pull(deps: Deps) -> datetime | None:
    try:
        return deps.last_pull_at()
    except Exception as exc:  # noqa: BLE001 -- unknown means "no pull yet"; the 1,500-credit pull floor still protects
        _warn(f"could not read the last pull time: {type(exc).__name__}: {exc}")
        return None


def _print_pull(result: dict) -> None:
    for p in result.get("would_write", []):
        line = "" if p["line"] is None else f" {p['line']:+g}"
        print(f"[take-pull] DRY RUN would store: {p['league']} {p['event_id']} {p['market']} {p['side']} "
              f"{p['book']}{line} {p['price']:+d} ev={p['ev']:+.2%} good until {p['good_until']:+d}", flush=True)
    print(f"[take-pull] { {k: v for k, v in result.items() if k != 'would_write'} }", flush=True)


def tick(now: datetime, deps: Deps, *, every_hours: int, run_id: str, dry_run: bool,
         paid_pull_at: datetime | None) -> datetime | None:
    """One tick: the close, then the pull when due. Returns the updated in-memory time of the last paid pull."""
    try:
        result = take_close.run_close(now, deps.close_deps, run_id=run_id, dry_run=dry_run)
        print(f"[take-close] {result}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- the pull and the next tick still go on
        _warn(f"close failed: {type(exc).__name__}: {exc}")
    if now.hour not in C.PULL_HOURS_UTC:
        return paid_pull_at
    try:
        last = _read_last_pull(deps)
        last = max((t for t in (last, paid_pull_at) if t is not None), default=None)
        if pull_due(now, last, every_hours):
            result = take_pull.run_pull(now, deps.pull_deps, run_id=run_id, dry_run=dry_run)
            _print_pull(result)
            if result.get("credits", 0) > 0:
                paid_pull_at = now
    except Exception as exc:  # noqa: BLE001 -- the loop goes on
        _warn(f"pull failed: {type(exc).__name__}: {exc}")
    return paid_pull_at


def run_watch(deps: Deps, *, minutes: float = C.WATCH_MINUTES, tick_seconds: float = C.WATCH_TICK_SEC,
              every_hours: int = 2, run_id: str = "local", dry_run: bool = False) -> int:
    """Tick until the deadline. Returns the number of ticks run."""
    start = deps.now()
    deadline = start + timedelta(minutes=minutes)
    step = timedelta(seconds=tick_seconds)
    print(f"[take-watch] start {start.isoformat()} deadline {deadline.isoformat()} tick {tick_seconds:g}s "
          f"pull every {every_hours}h dry_run={dry_run}", flush=True)
    paid_pull_at: datetime | None = None
    ticks, k = 0, 0
    while start + k * step < deadline:
        paid_pull_at = tick(deps.now(), deps, every_hours=every_hours, run_id=run_id, dry_run=dry_run,
                            paid_pull_at=paid_pull_at)
        ticks += 1
        after = deps.now()
        k = max(k + 1, math.floor((after - start) / step) + 1)         # the next FUTURE slot: overrun slots are skipped
        slot = start + k * step
        if slot >= deadline:
            break
        deps.sleep(max(0.0, (slot - after).total_seconds()))
    print(f"[take-watch] done after {ticks} ticks", flush=True)
    return ticks


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db
    from sportsmodel.take import store

    timeouts = {"connect_timeout": db.LEDGER_CONNECT_TIMEOUT, "statement_timeout_ms": db.LEDGER_STATEMENT_TIMEOUT_MS}
    return Deps(close_deps=take_close._db_deps(), pull_deps=take_pull._db_deps(),
                last_pull_at=lambda: store.last_pull_at(**timeouts))


def _positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="make the paid calls but write nothing; print what it would do")
    ap.add_argument("--minutes", type=_positive, default=C.WATCH_MINUTES, help="how long to keep watching")
    ap.add_argument("--tick-seconds", type=_positive, default=C.WATCH_TICK_SEC, help="seconds between ticks")
    args = ap.parse_args(argv)
    dry_run = args.dry_run or writes_disabled()
    for name in ("DATABASE_URL", "ODDS_API_KEY"):
        if not os.getenv(name):
            print(f"::error::take-watch: {name} is not set")
            return 1
    run_watch(_db_deps(), minutes=args.minutes, tick_seconds=args.tick_seconds,
              every_hours=every_hours_from(os.getenv("TAKE_PULL_EVERY_HOURS")),
              run_id=os.getenv("GITHUB_RUN_ID") or "local", dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
