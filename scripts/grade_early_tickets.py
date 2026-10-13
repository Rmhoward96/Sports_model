"""Grade the early-week paper tickets against the result and Pinnacle's pre-kick close -> early_ticket_results (daily).

For every ticket whose game started more than 4 hours ago and that has no result row yet: result (W / L / P) and units at
the ticket's price, `close_ev` (the ticket's line and price valued at Pinnacle's closing no-vig fair through the same
landing-table adjustment as the flag), `beat_close`, `clv_pts`, and `available_next_poll` (same book, same line, same or
better price one poll later). The close is Pinnacle's two-way quote in the `close_capture` lines snapshot written by
scripts/capture_close.py; no Pinnacle quote there means close_kind 'missing' (the ticket is excluded from the test). Results
come from rows we already store (prediction_accuracy, odds_snapshot, close_capture): no network, no Odds API credits.
A game with no final result stays ungraded. Idempotent.

    uv run python scripts/grade_early_tickets.py [--days 14]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.early import landing  # noqa: E402
from sportsmodel.early.constants import NEXT_POLL_MAX_MINUTES, SETTLE_HOURS  # noqa: E402
from sportsmodel.early.grading import grade_tickets  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

MIN_DAYS, MAX_DAYS = 1, 60     # --days is clamped to this range
LATER_HOURS = int(NEXT_POLL_MAX_MINUTES // 60)


@dataclass
class Loaders:
    missing_tables: Callable[[], list[str]]
    pending_tickets: Callable[[int], list[dict]]
    finals: Callable[[list[int]], dict]
    close_rows: Callable[[list[int]], dict]            # {game_pk: close-capture snapshot rows}
    later_rows: Callable[[list[int]], dict]            # {game_pk: odds_snapshot rows within 3 h after any of its tickets}
    load_tables: Callable[[], dict]
    store: Callable[[list[dict]], int]


def grade_all(ld: Loaders, days: int) -> dict:
    gone = ld.missing_tables()
    if gone:
        print(f"::warning::grade-early-tickets: table {gone[0]} missing — run db/migration_early_tickets.sql", flush=True)
        return {"status": "no_table"}
    tickets = ld.pending_tickets(days)
    if not tickets:
        return {"status": "ok", "pending": 0, "graded": 0}
    pks = sorted({t["game_pk"] for t in tickets})
    results = grade_tickets(tickets, ld.finals(pks), ld.close_rows(pks), ld.later_rows(pks), ld.load_tables())
    n = ld.store(results) if results else 0
    kinds: dict[str, int] = {}
    for r in results:
        kinds[r["close_kind"]] = kinds.get(r["close_kind"], 0) + 1
    return {"status": "ok", "pending": len(tickets), "graded": n, "close_kinds": kinds}


def _db_loaders() -> Loaders:
    from sportsmodel import db

    def q(sql, params):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def by_game(rows):
        out = defaultdict(list)
        for r in rows:
            out[r["game_pk"]].append(r)
        return dict(out)

    def pending(days):
        cols = ", ".join(f"t.{c}" for c in ["id"] + db.EARLY_COLUMNS["early_tickets"])
        return q(f"SELECT {cols} FROM early_tickets t "
                 f"WHERE t.commence_time <= now() - interval '{SETTLE_HOURS} hours' "
                 f"AND t.commence_time >= now() - make_interval(days => %s) "
                 f"AND NOT EXISTS (SELECT 1 FROM early_ticket_results r WHERE r.ticket_id = t.id) ORDER BY t.id", (days,))

    def finals(pks):
        rows = q("SELECT sport, game_pk, actual_margin, actual_total FROM prediction_accuracy WHERE game_pk = ANY(%s)", (pks,))
        return {(r["sport"], r["game_pk"]): r for r in rows}

    def close_rows(pks):
        return by_game(q(
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM close_capture c JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at "
            "WHERE c.game_pk = ANY(%s) AND c.kind = 'lines' AND c.status = 'ok' AND s.player_name = '' "
            "AND s.market IN ('spread', 'total')", (pks,)))

    def later_rows(pks):
        return by_game(q(
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM odds_snapshot s WHERE s.game_pk = ANY(%s) AND s.player_name = '' AND s.market IN ('spread', 'total') "
            "AND EXISTS (SELECT 1 FROM early_tickets t WHERE t.game_pk = s.game_pk AND s.captured_at > t.observed_at "
            f"AND s.captured_at <= t.observed_at + interval '{LATER_HOURS} hours')", (pks,)))

    return Loaders(
        missing_tables=lambda: db.missing_early_tables(), pending_tickets=pending, finals=finals,
        close_rows=close_rows, later_rows=later_rows, load_tables=landing.load_tables,
        store=db.upsert_early_ticket_results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=14, help="grade games that started within this many days (1-60)")
    args = ap.parse_args(argv)
    args.days = max(MIN_DAYS, min(MAX_DAYS, args.days))
    if writes_disabled():
        print("grade-early-tickets: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::grade-early-tickets: DATABASE_URL is not set")
        return 1
    print(f"[grade-early-tickets] {grade_all(_db_loaders(), args.days)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
