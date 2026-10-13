"""Grade the pick ledger against results and the pre-kick close -> pick_ledger_results (nightly).

For every pick_key whose game started more than 4 hours ago and that has no result yet, grade the key's FIRST
ledger row (earliest bettable observation) and its LAST pre-kick row: result (W / L / P), units at the price taken,
and CLV against the close captured by scripts/capture_close.py (Pinnacle no-vig; else a >= 3 US-book consensus at
the exact line; else 'missing'). Results come from rows we already store (prediction_accuracy, ev_prop_results,
nfl_player_actuals): no network. Idempotent.

    uv run python scripts/grade_ledger.py [--days 14]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ledger.grading import grade_rows, merge_close  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

STALE_CLOSE_MIN = 180      # no capture for a game: use its last pre-kick snapshot if it is at most this old
SETTLE_HOURS = 4           # grade a key only this long after kickoff (results and actuals have landed)


@dataclass
class Loaders:
    missing_tables: Callable[[], list[str]]
    pending_keys: Callable[[int], list[str]]
    ledger_rows: Callable[[list[str]], list[dict]]
    finals: Callable[[list[int]], dict]
    prop_actuals: Callable[[list[int]], dict]
    close_rows: Callable[[list[int]], tuple[list[dict], list[dict]]]       # (captured, fallback)
    store: Callable[[list[dict]], int]


def grade_all(ld: Loaders, days: int) -> dict:
    gone = ld.missing_tables()
    if gone:
        print(f"::warning::grade-ledger: table {gone[0]} missing — run db/migration_pick_ledger.sql", flush=True)
        return {"status": "no_table"}
    keys = ld.pending_keys(days)
    if not keys:
        return {"status": "ok", "keys": 0, "graded": 0}
    rows = ld.ledger_rows(keys)
    pks = sorted({r["game_pk"] for r in rows})
    captured, fallback = ld.close_rows(pks)
    results = grade_rows(rows, ld.finals(pks), ld.prop_actuals(pks), merge_close(captured, fallback))
    n = ld.store(results) if results else 0
    kinds: dict[str, int] = {}
    for r in results:
        if r["basis"] == "first":
            kinds[r["close_kind"]] = kinds.get(r["close_kind"], 0) + 1
    return {"status": "ok", "keys": len(keys), "graded": n, "first_basis_close_kinds": kinds}


def _db_loaders() -> Loaders:
    from sportsmodel import db

    def q(sql, params):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def pending(days):
        return [r["pick_key"] for r in q(
            f"SELECT DISTINCT l.pick_key FROM pick_ledger l "
            f"WHERE l.commence_time <= now() - interval '{SETTLE_HOURS} hours' "
            f"AND l.commence_time >= now() - make_interval(days => %s) "
            f"AND NOT EXISTS (SELECT 1 FROM pick_ledger_results r WHERE r.pick_key = l.pick_key AND r.basis = 'last')",
            (days,))]

    def ledger_rows(keys):
        cols = ["id"] + db.LEDGER_COLUMNS["pick_ledger"]
        return q(f"SELECT {', '.join(cols)} FROM pick_ledger WHERE pick_key = ANY(%s) "
                 f"ORDER BY pick_key, observed_at, id", (keys,))

    def finals(pks):
        rows = q("SELECT game_pk, actual_margin, actual_total FROM prediction_accuracy WHERE game_pk = ANY(%s)", (pks,))
        return {r["game_pk"]: r for r in rows}

    def prop_actuals(pks):
        out = {}
        for sql in ("SELECT game_pk, player_id, market, actual FROM nfl_player_actuals "
                    "WHERE game_pk = ANY(%s) AND actual IS NOT NULL",
                    "SELECT game_pk, player_id, market, actual FROM ev_prop_results "
                    "WHERE game_pk = ANY(%s) AND actual IS NOT NULL"):
            for r in q(sql, (pks,)):
                out[(r["game_pk"], str(r["player_id"]), r["market"])] = float(r["actual"])
        return out

    def close_rows(pks):
        captured = q(
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM close_capture c JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at "
            "WHERE c.game_pk = ANY(%s) AND c.status = 'ok' "
            "AND ((c.kind = 'lines' AND s.player_name = '') OR (c.kind = 'props' AND s.player_name <> ''))", (pks,))
        fallback = q(
            "WITH latest AS (SELECT game_pk, (player_name <> '') AS is_prop, max(captured_at) AS ts "
            "  FROM odds_snapshot WHERE game_pk = ANY(%s) AND captured_at <= commence_time "
            f"  AND captured_at >= commence_time - interval '{STALE_CLOSE_MIN} minutes' "
            "  GROUP BY game_pk, (player_name <> '')) "
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM odds_snapshot s JOIN latest l ON l.game_pk = s.game_pk AND l.is_prop = (s.player_name <> '') "
            "AND l.ts = s.captured_at", (pks,))
        return captured, fallback

    return Loaders(
        missing_tables=lambda: db.missing_ledger_tables(), pending_keys=pending, ledger_rows=ledger_rows,
        finals=finals, prop_actuals=prop_actuals, close_rows=close_rows, store=db.upsert_pick_ledger_results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=14, help="grade games that started within this many days")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("grade-ledger: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::grade-ledger: DATABASE_URL is not set")
        return 1
    print(f"[grade-ledger] {grade_all(_db_loaders(), args.days)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
