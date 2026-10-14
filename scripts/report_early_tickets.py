"""Weekly early-week monitor report -> docs/superpowers/reports/YYYY-MM-DD-early-monitor.md (profit step 3).

Tickets by market, close coverage, running mean close EV / beat-close rate / availability share, W/L/P and units, and the
decision status of the pre-registered pass test (sportsmodel.early.passtest). The test is judged ONCE, at the first report
where there are >= 150 graded tickets with a Pinnacle close and >= 60 of them CFB totals; until then the status is
"collecting" and nothing can pass or fail. A judged verdict is carried forward unchanged by every later report (it is read
back from the previous report's machine-readable block). Read-only against the database; writes only the markdown file
(the workflow commits it). A week with no new tickets and no new grades writes nothing.

    uv run python scripts/report_early_tickets.py [--as-of 2026-10-20T15:52:00Z] [--reports-dir docs/superpowers/reports]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.early import passtest, report  # noqa: E402
from sportsmodel.ledger.core import as_utc  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "superpowers" / "reports"
TICKETS_SQL = "SELECT id, sport, market, side, observed_at, commence_time FROM early_tickets ORDER BY id"
RESULTS_SQL = ("SELECT ticket_id, sport, game_pk, market, side, result, units, close_kind, close_ev, beat_close, clv_pts, "
               "available_next_poll, graded_at FROM early_ticket_results ORDER BY ticket_id")


def _db_query():
    from sportsmodel import db

    def q(sql, params=()):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    return q


def collect(q) -> tuple[list[dict], list[dict]]:
    """(tickets, results) via read-only queries `q(sql, params) -> list of dict rows`."""
    return q(TICKETS_SQL, ()), q(RESULTS_SQL, ())


def has_activity(tickets: list[dict], results: list[dict], as_of: datetime) -> bool:
    """Anything new in the last 7 days: a ticket logged or a ticket graded."""
    since = as_utc(as_of) - timedelta(days=7)
    return (any(as_utc(t["observed_at"]) > since for t in tickets)
            or any(r.get("graded_at") is not None and as_utc(r["graded_at"]) > since for r in results))


def build_report(as_of: datetime, tickets: list[dict], results: list[dict], previous: dict | None) -> tuple[str, dict]:
    decision = passtest.decide(results, previous, f"{as_utc(as_of):%Y-%m-%d}")
    text = report.render_report(as_of=as_of, summary=report.summarize(tickets, results, as_of), decision=decision)
    return text, decision


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default=None, help="report time (ISO, default: now UTC)")
    ap.add_argument("--reports-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="print the report instead of writing the file")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("report-early-tickets: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::report-early-tickets: DATABASE_URL is not set")
        return 1
    from sportsmodel import db
    if db.missing_early_tables():
        print("::warning::report-early-tickets: early-monitor tables missing — run db/migration_early_tickets.sql", flush=True)
        return 0
    as_of = as_utc(args.as_of) if args.as_of else datetime.now(timezone.utc)
    tickets, results = collect(_db_query())
    if not has_activity(tickets, results, as_of):
        print("[early-monitor-report] no new tickets or grades in the last 7 days; no report written", flush=True)
        return 0
    reports_dir = Path(args.reports_dir)
    previous = report.latest_decision([p.read_text() for p in sorted(reports_dir.glob("*-early-monitor.md"))], as_of)
    text, decision = build_report(as_of, tickets, results, previous)
    if args.dry_run:
        print(text)
        return 0
    path = reports_dir / f"{as_of:%Y-%m-%d}-early-monitor.md"
    path.write_text(text)
    print(f"[early-monitor-report] wrote {path}: status={decision['status']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
