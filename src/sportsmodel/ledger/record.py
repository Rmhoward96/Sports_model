"""Best-effort ledger recording. The pick writers call `safe_record`; whatever goes wrong here (missing
table, DB down, a bug in an adapter) logs a warning and NEVER fails the job that produced the picks."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Iterable

from .core import plan_appends


def writes_disabled() -> bool:
    """SM_LEDGER_OFF=1 turns every ledger DB write off. The test suite sets it (root conftest.py) because the
    local .env carries a real DATABASE_URL; the workflows never set it."""
    return os.getenv("SM_LEDGER_OFF", "").strip().lower() in ("1", "true", "yes", "on")


class DbStore:
    """The production store: the db.py ledger helpers, imported lazily."""

    def missing_tables(self) -> list[str]:
        from sportsmodel import db
        return db.missing_ledger_tables(["pick_ledger"])

    def latest(self, keys: list[str]) -> dict[str, dict]:
        from sportsmodel import db
        return db.latest_pick_ledger_rows(keys)

    def insert(self, rows: list[dict]) -> int:
        from sportsmodel import db
        return db.insert_pick_ledger(rows)


def record_observations(candidates: Iterable[dict], *, writer: str, now=None, store=None,
                        run_id: str | None = None) -> dict:
    """Append the candidates that the append rule accepts. Returns a summary dict; never raises."""
    try:
        candidates = [c for c in candidates if c]
        if not candidates:
            return {"status": "empty"}
        if store is None:
            if writes_disabled():
                return {"status": "disabled"}
            store = DbStore()
        gone = store.missing_tables()
        if gone:
            print(f"::warning::pick-ledger: table {gone[0]} missing — run db/migration_pick_ledger.sql "
                  f"({writer}: {len(candidates)} observations not recorded)", flush=True)
            return {"status": "no_table"}
        keys = sorted({c["pick_key"] for c in candidates})
        rows, counts = plan_appends(
            candidates, store.latest(keys), now=now or datetime.now(timezone.utc), writer=writer,
            writer_run=run_id or os.getenv("GITHUB_RUN_ID") or "local")
        if rows:
            store.insert(rows)
        print(f"[pick-ledger] {writer}: {len(candidates)} picks seen, {len(rows)} appended {counts}", flush=True)
        return {"status": "ok", "appended": len(rows), **counts}
    except Exception as exc:  # noqa: BLE001 -- the ledger must never fail the writer
        print(f"::warning::pick-ledger: {writer}: {type(exc).__name__}: {exc}", flush=True)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def safe_record(build: Callable[[], Iterable[dict]], *, writer: str, **kw) -> dict:
    """`record_observations(build(), ...)` with the candidate adapter inside the same safety net."""
    try:
        candidates = list(build())
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::pick-ledger: {writer}: building observations failed: {type(exc).__name__}: {exc}",
              flush=True)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return record_observations(candidates, writer=writer, **kw)
