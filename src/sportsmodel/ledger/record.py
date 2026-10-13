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


_WARNED: set[tuple[str, str]] = set()      # (writer, status) pairs already warned about in this process


def reset_warnings() -> None:
    """Forget which warnings were already printed (tests)."""
    _WARNED.clear()


def _warn(writer: str, status: str, message: str) -> None:
    """One GitHub-Actions warning per process per (writer, status) -- a down DB must not spam the log."""
    if (writer, status) in _WARNED:
        return
    _WARNED.add((writer, status))
    print(f"::warning::pick-ledger: {message}", flush=True)


def _exc_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:200]


def _require_enabled() -> None:
    if writes_disabled():
        raise RuntimeError("pick ledger writes are disabled (SM_LEDGER_OFF)")


class DbStore:
    """The production store: the db.py ledger helpers, imported lazily. Every method refuses to run under
    SM_LEDGER_OFF, so no caller (not even a direct one) can reach the database from the test suite."""

    def missing_tables(self) -> list[str]:
        _require_enabled()
        from sportsmodel import db
        return db.missing_ledger_tables(["pick_ledger"], connect_timeout=db.LEDGER_CONNECT_TIMEOUT,
                                         statement_timeout_ms=db.LEDGER_STATEMENT_TIMEOUT_MS)

    def latest(self, keys: list[str]) -> dict[str, dict]:
        _require_enabled()
        from sportsmodel import db
        return db.latest_pick_ledger_rows(keys, connect_timeout=db.LEDGER_CONNECT_TIMEOUT,
                                         statement_timeout_ms=db.LEDGER_STATEMENT_TIMEOUT_MS)

    def insert(self, rows: list[dict]) -> int:
        _require_enabled()
        from sportsmodel import db
        return db.insert_pick_ledger(rows, connect_timeout=db.LEDGER_CONNECT_TIMEOUT,
                                         statement_timeout_ms=db.LEDGER_STATEMENT_TIMEOUT_MS)


def record_observations(candidates: Iterable[dict], *, writer: str, now=None, store=None,
                        run_id: str | None = None) -> dict:
    """Append the candidates that the append rule accepts. Returns a summary dict; never raises."""
    try:
        candidates = [c for c in candidates if c]
        good = [c for c in candidates if isinstance(c, dict) and c.get("pick_key")]
        malformed = len(candidates) - len(good)
        candidates = good
        if not candidates:
            return {"status": "empty", **({"skipped": malformed} if malformed else {})}
        if store is None:
            if writes_disabled():
                return {"status": "disabled"}
            store = DbStore()
        gone = store.missing_tables()
        if gone:
            _warn(writer, "no_table", f"table {gone[0]} missing — run db/migration_pick_ledger.sql "
                  f"({writer}: {len(candidates)} observations not recorded)")
            return {"status": "no_table"}
        keys = sorted({c["pick_key"] for c in candidates})
        rows, counts = plan_appends(
            candidates, store.latest(keys), now=now or datetime.now(timezone.utc), writer=writer,
            writer_run=run_id or os.getenv("GITHUB_RUN_ID") or "local")
        if rows:
            store.insert(rows)
        print(f"[pick-ledger] {writer}: {len(candidates)} picks seen, {len(rows)} appended {counts}", flush=True)
        if malformed:
            counts["skipped"] = counts.get("skipped", 0) + malformed
        return {"status": "ok", "appended": len(rows), **counts}
    except Exception as exc:  # noqa: BLE001 -- the ledger must never fail the writer
        _warn(writer, "error", f"{writer}: {_exc_text(exc)}")
        return {"status": "error", "error": _exc_text(exc)}


def safe_record(build: Callable[[], Iterable], *, writer: str, **kw) -> dict:
    """`record_observations(build(), ...)` with the candidate adapter inside the same safety net.

    `build()` yields candidate dicts and/or zero-argument callables that produce one (so an adapter can defer
    `core.observation(...)`): a callable that raises skips only that candidate (a bad
    `core.observation(...)` argument, a missing field) and is counted under `skipped` in the returned dict."""
    try:
        items = list(build())
    except Exception as exc:  # noqa: BLE001
        _warn(writer, "build", f"{writer}: building observations failed: {_exc_text(exc)}")
        return {"status": "error", "error": _exc_text(exc)}
    candidates, skipped = [], 0
    for item in items:
        try:
            candidates.append(item() if callable(item) else item)
        except Exception:  # noqa: BLE001 -- one bad candidate (any error) is skipped; it must not escape the writer
            skipped += 1
    out = record_observations(candidates, writer=writer, **kw)
    if skipped:
        out["skipped"] = out.get("skipped", 0) + skipped
        _warn(writer, "skipped", f"{writer}: {skipped} bad candidate(s) skipped")
    return out
