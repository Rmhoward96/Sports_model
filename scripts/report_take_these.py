"""Weekly 'take these' report -> docs/superpowers/reports/YYYY-MM-DD-take-these.md.

Progress toward the pre-registered CLV test (sportsmodel.take.verdict): "N of 300 graded picks since Nov 1" with the three
criteria shown live, the verdict once judged, per-league and per-market beat-close rate and mean CLV, better-line picks
separately, the direction and no-close counts, W-L-P and units, the October shakedown (not judged), credits used per job
and books missing per league over the last 7 days (take_api_calls). The test is judged ONCE, on the first 300 graded picks;
the first report to see them inserts the (append-only) take_verdict row, and every later report carries it unchanged.
Dates shown are US Eastern. Writes only the markdown file and, once, the verdict row. No picks at all writes nothing.

    uv run python scripts/report_take_these.py [--as-of 2026-12-07T13:00:00Z] [--reports-dir docs/superpowers/reports] [--dry-run]

--dry-run (or SM_LEDGER_OFF) prints the report instead of writing the file and the verdict row.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ledger.core import as_utc  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402
from sportsmodel.take import constants as C  # noqa: E402
from sportsmodel.take import verdict as V  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "superpowers" / "reports"
EASTERN = ZoneInfo("America/New_York")
JOBS = ("pull", "close", "scores")
MARKET_ORDER = ("ml", "spread", "total")
CAVEAT = ("Your own Pikkit beat-close rate over the same dates is on the Scorecard page "
          "(Pikkit's close is not necessarily Pinnacle's).")


def _et(value: datetime) -> datetime:
    return as_utc(value).astimezone(EASTERN)


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def _clv(v: float | None) -> str:
    return "-" if v is None else f"{v:+.4f}"


def _table(head: tuple[str, ...], body: list[tuple]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|",
            *("| " + " | ".join(str(c) for c in r) + " |" for r in body)]


def _record(rows: list[dict]) -> tuple[int, int, int, float, int]:
    """(W, L, P, units, settled) over the rows that have an outcome."""
    done = [r for r in rows if r.get("outcome") is not None]
    n = Counter(r["outcome"] for r in done)
    return n["win"], n["loss"], n["push"], sum(float(r["units"] or 0) for r in done), len(done)


def _breakdown(rows: list[dict], key: str, order: tuple[str, ...]) -> list[tuple]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = []
    for k in order:
        if k in groups:
            s = V.stats(groups[k])
            out.append((k, s["n"], _pct(s["beat_rate"]), _clv(s["mean_clv"])))
    return out


def _progress(decision: dict, n_test: int, as_of: datetime) -> list[str]:
    """The progress block. `decision` is verdict.decide's output, or for a carried verdict the stored row's numbers."""
    crit = decision.get("criteria") or V.criteria(decision)
    lines = ["## Progress", "", f"**{n_test} of {C.N_REQUIRED} graded picks since {C.TEST_START:%b} {C.TEST_START.day}** "
             f"(a graded pick has a CLV or a line direction; the first {C.N_REQUIRED} to be graded are judged, once).", ""]
    if decision.get("carried") or decision.get("judged"):
        when = _et(decision["judged_at"] if decision.get("carried") else as_of)
        word = "PASSED" if decision["passed"] else "FAILED"
        lines += [f"**Verdict: {word}** (judged {when:%Y-%m-%d}, on the first {decision['n']} graded picks).", ""]
    else:
        lines += [f"Verdict: not judged yet. Running numbers over the {n_test} graded so far:", ""]
    lines += _table(("Criterion", "Value", "Met"), [
        (name, fmt(decision[key]), "yes" if crit[name] else "no")
        for name, key, fmt in zip(V.CRITERIA, ("beat_rate", "mean_clv", "lower"), (_pct, _clv, _clv))])
    lines += ["", "Beat-close covers every graded pick; mean CLV and its event-clustered 95% lower bound cover those "
              "with a CLV."]
    return lines


def _credits(calls: list[dict], since: datetime) -> list[str]:
    used = Counter()
    for c in calls:
        used[c["job"]] += int(c.get("credits_used") or 0)
    lines = [f"## Credits used (last 7 days, since {_et(since):%Y-%m-%d})", ""]
    lines += _table(("Job", "Credits"), [(j, used[j]) for j in JOBS] + [("total", sum(used[j] for j in JOBS))])
    return lines


def _books_missing(calls: list[dict], since: datetime) -> list[str]:
    seen: Counter = Counter()
    missing: dict[str, Counter] = defaultdict(Counter)
    for c in calls:
        if c.get("league") and c.get("books_missing") is not None:
            seen[c["league"]] += 1
            missing[c["league"]].update(set(c["books_missing"]))
    lines = [f"## Books missing per league (last 7 days, since {_et(since):%Y-%m-%d})", ""]
    out = [f"- {lg}: " + ", ".join(f"{b} ({k} of {seen[lg]} calls)"
                                   for b, k in sorted(missing[lg].items(), key=lambda kv: (-kv[1], kv[0])))
           for lg in C.LEAGUES if missing[lg]]
    return lines + (out or ["None in the last 7 days."])


def build_report(as_of: datetime, rows: list[dict], verdict_row: dict | None, calls: list[dict]) -> tuple[str, dict | None]:
    """(markdown, the take_verdict row to insert or None). `rows` are take_test_v rows (every pick, graded or not),
    `verdict_row` the stored verdict (None until judged), `calls` the take_api_calls of the last 7 days."""
    decision = V.decide(rows, verdict_row, as_of)
    new_row = V.verdict_row(decision, as_of) if decision.get("judged") and not decision.get("carried") else None
    since = as_utc(as_of) - timedelta(days=7)
    test = V.test_rows(rows)
    after = [r for r in rows if as_utc(r["first_seen"]) >= C.TEST_START]
    october = [r for r in rows if as_utc(r["first_seen"]) < C.TEST_START]
    out = [f"# Take these: weekly report {_et(as_of):%Y-%m-%d}", "",
           f"Generated {_et(as_of):%Y-%m-%d %H:%M} ET. Test window starts {C.TEST_START:%b} {C.TEST_START.day}; the "
           f"October picks are a shakedown and are not judged.", ""]
    out += _progress(decision, len(test), as_of)

    out += ["", "## Per league", ""]
    out += _table(("League", "Graded", "Beat-close", "Mean CLV"), _breakdown(test, "league", tuple(C.LEAGUES))) \
        if test else ["No graded picks yet."]
    out += ["", "## Per market", ""]
    out += _table(("Market", "Graded", "Beat-close", "Mean CLV"), _breakdown(test, "market", MARKET_ORDER)) \
        if test else ["No graded picks yet."]

    better = V.stats([r for r in test if r.get("better_line")])
    out += ["", "## Better-line picks", ""]
    out += ([f"{better['n']} better-line picks: beat-close {_pct(better['beat_rate'])}, mean CLV {_clv(better['mean_clv'])} "
             f"(also counted in the tables above)."] if better["n"] else ["No better-line picks graded yet."])

    out += ["", "## Grading", "",
            f"- Graded by line direction (no CLV): {sum(1 for r in test if r['graded_by'] == 'direction')}",
            f"- No valid close (not counted in the test): {sum(1 for r in after if r.get('graded_by') == 'no_close')}",
            f"- Awaiting grade: {sum(1 for r in after if r.get('graded_by') is None)}"]

    w, l, p, units, settled = _record(after)
    out += ["", "## Results", "", f"W-L-P {w}-{l}-{p}, {units:+.3f} units ({settled} settled; 1 unit staked per pick at "
            f"its own price)."]

    out += ["", "## October shakedown (not judged)", ""]
    if october:
        graded = [r for r in october if r.get("graded_by") is not None]
        s = V.stats([r for r in graded if r["graded_by"] in ("clv", "direction")])
        w, l, p, units, _ = _record(october)
        out += [f"{len(october)} picks first seen before {C.TEST_START:%b} {C.TEST_START.day}, {len(graded)} graded: "
                f"beat-close {_pct(s['beat_rate'])}, mean CLV {_clv(s['mean_clv'])}, W-L-P {w}-{l}-{p}, "
                f"{units:+.3f} units. Not judged."]
    else:
        out += [f"No picks were logged before {C.TEST_START:%b} {C.TEST_START.day}."]

    out += [""] + _credits(calls, since) + [""] + _books_missing(calls, since) + ["", CAVEAT, ""]
    return "\n".join(out), new_row


# ---- IO -----------------------------------------------------------------------------------------------

def report_path(reports_dir: Path, as_of: datetime) -> Path:
    return reports_dir / f"{as_utc(as_of):%Y-%m-%d}-take-these.md"


def _missing_tables() -> list[str]:
    from sportsmodel.take import store
    return store.missing_tables()


def _collect(since: datetime) -> tuple[list[dict], dict | None, list[dict]]:
    from sportsmodel.take import store
    return store.test_rows(), store.get_verdict(), store.recent_calls(since)


def _insert_verdict(row: dict) -> bool:
    from sportsmodel.take import store
    return store.insert_verdict(row)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default=None, help="report time (ISO, default: now UTC)")
    ap.add_argument("--reports-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="print the report instead of writing the file and the verdict")
    args = ap.parse_args(argv)
    dry_run = args.dry_run or writes_disabled()
    if not os.getenv("DATABASE_URL"):
        print("::error::report-take-these: DATABASE_URL is not set")
        return 1
    gone = _missing_tables()
    if gone:
        print(f"::warning::report-take-these: table {gone[0]} missing — run db/migration_take_these.sql", flush=True)
        return 0
    as_of = as_utc(args.as_of) if args.as_of else datetime.now(timezone.utc)
    rows, verdict_row, calls = _collect(as_of - timedelta(days=7))
    if not rows:
        print("[take-these-report] no picks yet; no report written", flush=True)
        return 0
    text, new_row = build_report(as_of, rows, verdict_row, calls)
    if dry_run:
        print(text)
        return 0
    if new_row is not None:
        _insert_verdict(new_row)                       # first, so a report never states a verdict that was not stored
        print(f"[take-these-report] verdict judged and stored: passed={new_row['passed']}", flush=True)
    path = report_path(Path(args.reports_dir), as_of)
    path.write_text(text)
    print(f"[take-these-report] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
