"""Weekly refit of the prop shrink (profit step 2 Part A): read the graded NFL prop leans, fit k, write assets/shrink_k.json
and the dated gate report docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md.

Read-only against the database (`nfl_prop_grades`, the sim table for each leg's model version, `nfl_sim_serving` for the
current version). The k file is written ONLY after the fetch and the whole fit succeeded; on any failure this exits non-zero and
leaves the existing file untouched. It never writes a k it did not compute, and refuses to replace a file fitted on more legs
than it just read (a truncated read). The workflow (fit-shrink-k.yml) commits the two files.

    uv run python scripts/fit_shrink_k.py [--as-of 2026-10-20T15:30:00Z] [--out assets/shrink_k.json]
                                          [--reports-dir docs/superpowers/reports] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ledger.record import writes_disabled  # noqa: E402
from sportsmodel.shrink import fit, passtest, report  # noqa: E402
from sportsmodel.shrink.store import SHRINK_K_PATH  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "superpowers" / "reports"

# Every graded sim lean with a book line. The model version of a leg is the nfl_player_sim row for the same (game, player,
# market) whose mean equals the line row's projection (nfl_prop_lines does not store it); a leg whose sim row is gone gets NULL
# and counts only in the all-versions pool.
GRADES_SQL = """
SELECT g.game_pk, g.player_id, g.market, g.line, g.over_price, g.under_price, g.p_over, g.lean, g.result,
       g.season, g.week, v.model_version
FROM nfl_prop_grades g
LEFT JOIN LATERAL (
    SELECT s.model_version
    FROM nfl_player_sim s
    WHERE s.game_pk = g.game_pk AND s.player_id = g.player_id AND s.market = g.market
    ORDER BY abs(s.mean - g.projection) ASC, s.created_at DESC
    LIMIT 1
) v ON true
WHERE g.result IN ('hit', 'miss')
"""
SERVING_SQL = "SELECT model_version FROM nfl_sim_serving WHERE id = 1"


def _db_query():
    from sportsmodel import db

    def q(sql, params=()):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    return q


def collect(q) -> tuple[list[dict], str | None]:
    """(graded rows, the served props model version) via read-only queries `q(sql, params) -> list of dict rows`."""
    rows = q(GRADES_SQL, ())
    served = q(SERVING_SQL, ())
    return rows, (served[0]["model_version"] if served else None)


def read_existing(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to a temp file next to `path`, then replace: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def judged_cutoff(as_of: datetime) -> datetime:
    """The `as_of` handed to report.latest_decision: the start of the NEXT UTC day, so every report of the same day is a
    prior report. A re-run (or a manual dispatch) on the day the test was judged then finds that day's judged report and
    carries its verdict, whatever instant the re-run is stamped with; the test is judged once and never re-judged."""
    day = as_of.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return day + timedelta(days=1)


def build(rows: list[dict], current_version: str | None, as_of: datetime, previous: dict | None):
    """(document, report text, decision): everything computed before anything is written."""
    legs = fit.legs_from_grades(rows)
    doc = fit.build_document(legs, current_version, as_of.isoformat())
    scored = passtest.score_walk_forward(legs, current_version)
    decision = passtest.decide(scored, previous, f"{as_of:%Y-%m-%d}")
    return doc, report.render_report(as_of=as_of, doc=doc, decision=decision), decision


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default=None, help="fit time (ISO, default: now UTC)")
    ap.add_argument("--out", default=str(SHRINK_K_PATH))
    ap.add_argument("--reports-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="print the document and report instead of writing them")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("fit-shrink-k: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::fit-shrink-k: DATABASE_URL is not set")
        return 1
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")) if args.as_of else datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    out, reports_dir = Path(args.out), Path(args.reports_dir)
    try:
        rows, current_version = collect(_db_query())
        previous = report.latest_decision([p.read_text() for p in sorted(reports_dir.glob("*-shrink-gate.md"))],
                                          judged_cutoff(as_of))
        doc, text, decision = build(rows, current_version, as_of, previous)
    except Exception as exc:  # noqa: BLE001 -- any fetch or fit failure: keep the existing file, fail the job
        print(f"::error::fit-shrink-k: {type(exc).__name__}: {exc}; assets/shrink_k.json left untouched")
        return 1
    if fit.has_fewer_legs(read_existing(out), doc):
        print(f"::error::fit-shrink-k: this read has {doc['data_window']['n_legs']} eligible legs, fewer than the committed "
              "file was fitted on; the read looks truncated, file left untouched")
        return 1
    if args.dry_run:
        print(json.dumps(doc, indent=2, sort_keys=True))
        print(text)
        return 0
    write_atomic(out, json.dumps(doc, indent=2, sort_keys=True) + "\n")
    path = reports_dir / f"{as_of:%Y-%m-%d}-shrink-gate.md"
    path.write_text(text)
    print(f"[fit-shrink-k] wrote {out.name}: pooled k={doc['pooled']['k']:.3f} on n={doc['pooled']['n']} "
          f"(version {doc['model_version_used']}); {path.name}: status={decision['status']} scored={decision['scored']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
