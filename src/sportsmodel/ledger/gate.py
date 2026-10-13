"""The weekly step-1 gate for the pick ledger. PURE -- no IO.

A football weekend passes when ALL hold:
  * >= 95% of graded NFL / CFB games have a game-lines close captured within 30 minutes of kickoff,
  * >= 90% of graded prop picks have a Pinnacle or >= 3-book US close at their exact line,
  * no ledger row was updated or deleted (the row count and a content fingerprint of every row older than the
    previous report are unchanged since that report),
and step 1 passes only after TWO CONSECUTIVE passing weekends. A miss blocks steps 2-3 of the profit roadmap.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from .core import as_utc

LINES_MIN_PCT = 95.0
LINES_WITHIN_MIN = 30.0
PROPS_MIN_PCT = 90.0
WEEKENDS_REQUIRED = 2
MARKER = "ledger-gate"
FOOTBALL = ("nfl", "cfb")


def window_end_for(now: datetime) -> datetime:
    """The most recent Tuesday 12:00 UTC at or before `now` (the weekend's window closes Tuesday noon, after
    Monday night football has finished and been graded)."""
    now = as_utc(now)
    days_back = (now.isoweekday() - 2) % 7              # Tuesday = 2
    end = (now - timedelta(days=days_back)).replace(hour=12, minute=0, second=0, microsecond=0)
    return end if end <= now else end - timedelta(days=7)


def lines_coverage(graded_games: list[tuple], lines_minutes: dict[tuple, float | None]) -> dict:
    """Share of graded football games whose game-lines close was captured 0..30 minutes before kickoff."""
    games = {(s, pk) for s, pk in graded_games if s in FOOTBALL}
    covered = sum(1 for g in games if (m := lines_minutes.get(g)) is not None and 0 <= m <= LINES_WITHIN_MIN)
    return {"covered": covered, "total": len(games), "pct": 100.0 * covered / len(games) if games else None}


def props_coverage(prop_results: list[dict]) -> dict:
    """Share of graded prop picks with a Pinnacle or >= 3-book close at their exact line."""
    covered = sum(1 for r in prop_results if r.get("close_kind") in ("pinnacle_novig", "us_consensus_novig")
                  and r.get("close_same_line") is True)
    n = len(prop_results)
    return {"covered": covered, "total": n, "pct": 100.0 * covered / n if n else None}


def audit_check(previous: dict | None, recomputed: dict | None) -> dict:
    """Compare the previous report's audit block ({'n', 'md5', 'as_of'}) with the same measurement recomputed
    NOW over rows observed up to that same instant. Any difference means a row changed or vanished."""
    if previous is None:
        return {"passed": True, "detail": "baseline established (no previous report to compare with)"}
    rec = recomputed or {}
    if rec.get("n") == previous.get("n") and rec.get("md5") == previous.get("md5"):
        return {"passed": True, "detail": f"{previous['n']} rows up to {previous['as_of']} unchanged"}
    return {"passed": False, "detail": f"rows up to {previous.get('as_of')}: was {previous.get('n')} / "
                                       f"{previous.get('md5')}, now {rec.get('n')} / {rec.get('md5')}"}


def evaluate_week(lines: dict, props: dict, audit: dict) -> dict:
    checks = []
    for name, cov, floor, what in (
            ("game-line closes within 30 min of kickoff", lines, LINES_MIN_PCT, "graded NFL/CFB games"),
            ("prop closes at the exact line (Pinnacle or >= 3 books)", props, PROPS_MIN_PCT, "graded prop picks")):
        if cov["total"] == 0:
            checks.append({"name": name, "passed": False, "detail": f"no {what} in the window (nothing measured)"})
        else:
            checks.append({"name": name, "passed": cov["pct"] >= floor,
                           "detail": f"{cov['covered']}/{cov['total']} = {cov['pct']:.1f}% (need {floor:.0f}%)"})
    checks.append({"name": "no ledger row updated or deleted", "passed": audit["passed"], "detail": audit["detail"]})
    return {"checks": checks, "week_pass": all(c["passed"] for c in checks)}


def step_status(week_pass: bool, window_start: datetime, previous: dict | None) -> dict:
    """Step 1 passes when this weekend AND the immediately preceding weekend both passed."""
    prev_ok = bool(previous and previous.get("week_pass")
                   and as_utc(previous.get("window_end")) == as_utc(window_start))
    streak = (1 + (int(previous.get("streak", 1)) if prev_ok else 0)) if week_pass else 0
    return {"streak": streak, "step1_pass": streak >= WEEKENDS_REQUIRED}


def regions_line(prop_regions: dict[str, int] | None) -> str:
    """One sentence on which Odds API regions the prop closes used (the report states whether Pinnacle was asked)."""
    if not prop_regions:
        return "Prop closes captured this weekend: none."
    parts = ", ".join(f"`{r}` x {n}" for r, n in sorted(prop_regions.items()))
    pin = any("eu" in (r or "").split(",") for r in prop_regions)
    return (f"Prop closes captured this weekend by regions: {parts}. "
            + ("Pinnacle (EU) prop prices were requested." if pin else
               "Pinnacle was NOT requested for props (US books only; the probe found no Pinnacle props)."))


def render_report(*, window_start: datetime, window_end: datetime, evaluation: dict, status: dict,
                  audit: dict, lines: dict, props: dict, summary: list[dict],
                  prop_regions: dict[str, int] | None = None) -> str:
    ws, we = as_utc(window_start), as_utc(window_end)
    lines_md = "\n".join(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |"
                         for c in evaluation["checks"])
    sums = "\n".join(
        f"| {r['source']} | {r['n']} | {_fmt(r.get('mean_clv_prob'), 100, '{:+.2f} pp')} | "
        f"{_fmt(r.get('beat_close_pct'), 100, '{:.1f}%')} | {_fmt(r.get('units'), 1, '{:+.2f}')} |" for r in summary
    ) or "| (none) | 0 | | | |"
    verdict = ("STEP 1 PASSED: two consecutive football weekends met every check. Steps 2 and 3 are unblocked."
               if status["step1_pass"] else
               "Step 1 is NOT yet passed. " + ("This weekend met every check; one more consecutive passing weekend is needed."
                                                if evaluation["week_pass"] else "A check failed; steps 2 and 3 stay blocked."))
    block = {"window_start": ws.isoformat(), "window_end": we.isoformat(), "week_pass": evaluation["week_pass"],
             "streak": status["streak"], "step1_pass": status["step1_pass"], "audit": audit["state"],
             "lines": lines, "props": props}
    return f"""# Pick ledger gate: weekend ending {we:%Y-%m-%d}

Window: {ws:%Y-%m-%d %H:%M} to {we:%Y-%m-%d %H:%M} UTC. Generated by `scripts/report_ledger_gate.py`.

**{verdict}**

| Check | Result | Detail |
|---|---|---|
{lines_md}

Consecutive passing weekends: {status['streak']} of {WEEKENDS_REQUIRED} required.

{regions_line(prop_regions)}

## First-observation results in the window (informational, not part of the gate)

| Source | Picks graded | Mean CLV vs close (same line) | Beat the close | Units at the price taken |
|---|---|---|---|---|
{sums}

<!-- {MARKER}: {json.dumps(block, sort_keys=True)} -->
"""


def _fmt(v, scale, fmt) -> str:
    return "n/a" if v is None else fmt.format(float(v) * scale)


_BLOCK = re.compile(rf"<!-- {MARKER}: (\{{.*\}}) -->")


def parse_report(text: str) -> dict | None:
    """The machine-readable block of a previous report, or None."""
    m = _BLOCK.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def latest_previous(reports: list[str], window_end: datetime) -> dict | None:
    """Of the parsed reports, the one with the greatest window_end strictly before `window_end`."""
    best = None
    for text in reports:
        p = parse_report(text)
        if not p or not p.get("window_end"):
            continue
        end = as_utc(p["window_end"])
        if end is not None and end < as_utc(window_end) and (best is None or end > as_utc(best["window_end"])):
            best = p
    return best
