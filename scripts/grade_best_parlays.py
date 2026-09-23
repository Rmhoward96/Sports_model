"""Grade the +EV Parlays: evaluate locked tickets against game finals and prop
actuals, upsert results to ev_best_parlay_results. See serving/best_parlays.py.

Usage: DATABASE_URL=... uv run python scripts/grade_best_parlays.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.db import (
    get_postgres, ungraded_locked_parlays, upsert_best_parlay_results)
from sportsmodel.serving.best_parlays import (
    game_leg_result, prop_leg_result, settle_parlay)


def _q(sql: str, params: list | None = None) -> list[dict]:
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def load_finals(game_pks: list[int]) -> dict[int, dict]:
    """Load graded game finals from prediction_accuracy."""
    if not game_pks:
        return {}
    rows = _q("""
        SELECT game_pk, actual_margin, actual_total
        FROM prediction_accuracy
        WHERE game_pk = ANY(%s)
    """, [game_pks])
    return {r["game_pk"]: r for r in rows}


def load_actuals(game_pks: list[int]) -> dict[tuple, float]:
    """Load player prop actuals from nfl_player_actuals as (game_pk, player_id, market) -> actual."""
    if not game_pks:
        return {}
    rows = _q("""
        SELECT game_pk, player_id, market, actual
        FROM nfl_player_actuals
        WHERE game_pk = ANY(%s)
    """, [game_pks])
    return {(r["game_pk"], r["player_id"], r["market"]): r["actual"] for r in rows}


def load_captured_games(game_pks: list[int]) -> set[int]:
    """Load game_pks that have captured actuals (any row in nfl_player_actuals)."""
    if not game_pks:
        return set()
    rows = _q("""
        SELECT DISTINCT game_pk FROM nfl_player_actuals WHERE game_pk = ANY(%s)
    """, [game_pks])
    return {r["game_pk"] for r in rows}


def _legs(ticket: dict) -> list:
    """Extract and decode ticket's legs (handles JSON string from DB driver)."""
    legs = ticket["legs"]
    return json.loads(legs) if isinstance(legs, str) else legs


def grade_ticket(ticket: dict, finals: dict[int, dict], actuals: dict[tuple, float],
                 captured_games: set[int]) -> dict | None:
    """Grade a parlay ticket. Per-leg results from game finals or prop actuals;
    settle_parlay handles any loss -> immediate loss, any pending -> None, else payout.
    Returns a results row {parlay_id, sport, book, parlay_price, first_commence,
    result, pnl, payout_dec, legs: [leg + {result}]} or None if pending."""
    legs = _legs(ticket)

    results = []
    for leg in legs:
        if leg["kind"] == "game":
            r = game_leg_result(leg, finals.get(leg["game_pk"]))
        else:  # prop
            r = prop_leg_result(leg, actuals.get((leg["game_pk"], leg["player_id"], leg["market"])),
                               leg["game_pk"] in captured_games)
        results.append(r)

    settlement = settle_parlay(legs, results)
    if settlement is None:
        return None

    # Construct the results row, including legs with their per-leg results
    legs_with_results = []
    for leg, result in zip(legs, results):
        leg_copy = {**leg, "result": result}
        legs_with_results.append(leg_copy)

    return {
        "parlay_id": ticket["parlay_id"],
        "sport": ticket["sport"],
        "book": ticket["book"],
        "parlay_price": ticket["parlay_price"],
        "first_commence": ticket["first_commence"],
        "result": settlement["result"],
        "pnl": settlement["pnl"],
        "payout_dec": settlement["payout_dec"],
        "legs": legs_with_results,
    }


def main() -> None:
    tickets = ungraded_locked_parlays()
    if not tickets:
        print("[grade_best_parlays] no ungraded locked parlays")
        return

    game_pks = sorted({leg["game_pk"] for t in tickets for leg in _legs(t)})
    finals = load_finals(game_pks)
    actuals = load_actuals(game_pks)
    captured_games = load_captured_games(game_pks)

    graded = []
    for ticket in tickets:
        result = grade_ticket(ticket, finals, actuals, captured_games)
        if result is not None:
            graded.append(result)

    n = upsert_best_parlay_results(graded)
    print(f"[grade_best_parlays] tickets={len(tickets)} graded={len(graded)} upserted={n}")


if __name__ == "__main__":
    main()
