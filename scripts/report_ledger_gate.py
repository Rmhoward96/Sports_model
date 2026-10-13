"""Weekly step-1 gate report for the pick ledger -> docs/superpowers/reports/YYYY-MM-DD-ledger-gate.md.

Checks the football weekend that ended last Tuesday 12:00 UTC (see sportsmodel.ledger.gate): >= 95% of graded
games with a game-lines close captured within 30 minutes of kickoff, >= 90% of graded prop picks with a Pinnacle
or >= 3-book close at their exact line, and no ledger row updated or deleted (row count + content fingerprint of
everything older than the previous report); two consecutive passing weekends pass step 1. Read-only against the
database; writes only the markdown file (the workflow commits it).

    uv run python scripts/report_ledger_gate.py [--end 2026-10-20T12:00:00Z] [--reports-dir docs/superpowers/reports]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ledger import gate  # noqa: E402
from sportsmodel.ledger.core import as_utc  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "superpowers" / "reports"
AUDIT_SQL = ("SELECT count(*), md5(string_agg(md5(t::text), '' ORDER BY t.id)) FROM pick_ledger t "
             "WHERE t.observed_at <= %s")


def _db_query():
    from sportsmodel import db

    def q(sql, params=()):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    return q


def audit_state(q, as_of: datetime) -> dict:
    n, md5 = q(AUDIT_SQL, (as_of,))[0]
    return {"n": int(n), "md5": md5, "as_of": as_of.isoformat()}


def collect(q, start: datetime, end: datetime, previous: dict | None) -> dict:
    """All the numbers the report needs, via read-only queries `q(sql, params) -> rows`."""
    games = [(s, pk) for s, pk in q(
        "SELECT DISTINCT sport, game_pk FROM pick_ledger_results WHERE basis = 'first' "
        "AND sport IN ('nfl', 'cfb') AND commence_time >= %s AND commence_time < %s", (start, end))]
    lines_minutes: dict = {}
    if games:
        for s, pk, m in q("SELECT sport, game_pk, min(minutes_to_kick) FROM close_capture WHERE kind = 'lines' "
                          "AND status = 'ok' AND game_pk = ANY(%s) GROUP BY sport, game_pk", ([pk for _, pk in games],)):
            lines_minutes[(s, pk)] = m
    props = [{"close_kind": k, "close_same_line": same} for k, same in q(
        "SELECT close_kind, close_same_line FROM pick_ledger_results WHERE basis = 'first' AND source = 'ev_prop' "
        "AND commence_time >= %s AND commence_time < %s", (start, end))]
    summary = [{"source": s, "n": n, "mean_clv_prob": c, "beat_close_pct": b, "units": u} for s, n, c, b, u in q(
        "SELECT source, count(*), avg(clv_prob), avg(beat_close::int), sum(units_at_pick) FROM pick_ledger_results "
        "WHERE basis = 'first' AND commence_time >= %s AND commence_time < %s GROUP BY source ORDER BY source",
        (start, end))]
    prop_regions = {r: n for r, n in q(
        "SELECT regions, count(*) FROM close_capture WHERE kind = 'props' AND status IN ('ok', 'empty') "
        "AND captured_at >= %s AND captured_at < %s GROUP BY regions", (start, end))}
    recomputed = audit_state(q, as_utc(previous["audit"]["as_of"])) if previous and previous.get("audit") else None
    return {"games": games, "lines_minutes": lines_minutes, "props": props, "summary": summary,
            "prop_regions": prop_regions, "recomputed_audit": recomputed, "current_audit": audit_state(q, end)}


def build_report(end: datetime, data: dict, previous: dict | None) -> tuple[str, dict]:
    start = end - timedelta(days=7)
    lines = gate.lines_coverage(data["games"], data["lines_minutes"])
    props = gate.props_coverage(data["props"])
    prev_audit = previous["audit"] if previous and previous.get("audit") else None
    audit = gate.audit_check(prev_audit, data["recomputed_audit"])
    audit["state"] = data["current_audit"]
    evaluation = gate.evaluate_week(lines, props, audit)
    status = gate.step_status(evaluation["week_pass"], start, previous)
    text = gate.render_report(window_start=start, window_end=end, evaluation=evaluation, status=status,
                              audit=audit, lines=lines, props=props, summary=data["summary"],
                              prop_regions=data["prop_regions"])
    return text, {"evaluation": evaluation, "status": status}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--end", default=None, help="window end (ISO, default: the most recent Tuesday 12:00 UTC)")
    ap.add_argument("--reports-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="print the report instead of writing the file")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("report-ledger-gate: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::report-ledger-gate: DATABASE_URL is not set")
        return 1
    from sportsmodel import db
    if db.missing_ledger_tables():
        print("::warning::report-ledger-gate: ledger tables missing — run db/migration_pick_ledger.sql", flush=True)
        return 0
    end = as_utc(args.end) if args.end else gate.window_end_for(datetime.now(timezone.utc))
    reports_dir = Path(args.reports_dir)
    previous = gate.latest_previous([p.read_text() for p in sorted(reports_dir.glob("*-ledger-gate.md"))], end)
    text, result = build_report(end, collect(_db_query(), end - timedelta(days=7), end, previous), previous)
    if args.dry_run:
        print(text)
        return 0
    path = reports_dir / f"{end:%Y-%m-%d}-ledger-gate.md"
    path.write_text(text)
    print(f"[ledger-gate] wrote {path}: week_pass={result['evaluation']['week_pass']} "
          f"step1_pass={result['status']['step1_pass']}", flush=True)
    for c in result["evaluation"]["checks"]:
        if not c["passed"]:
            print(f"::warning::ledger-gate FAIL: {c['name']}: {c['detail']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
