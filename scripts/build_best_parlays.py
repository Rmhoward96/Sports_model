"""Build the +EV Parlays: up to 3 three-leg tickets at one book each, from the
current +EV game-line picks (ev_current) and NFL prop picks
(ev_prop_picks_current). Replaces the unlocked tickets; locked tickets (first
leg started) are left for grading. See serving/best_parlays.py.

Usage: DATABASE_URL=... uv run python scripts/build_best_parlays.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.db import get_postgres, locked_leg_keys, replace_unlocked_best_parlays
from sportsmodel.serving.best_parlays import (
    assemble_game_legs, assemble_prop_legs, build_best_parlays)
from sportsmodel.serving.props_ev import SIM_TO_ODDS_MARKET, latest_capture_only

MODEL_VERSION = "ev-parlays-v1"


def _q(sql: str, params: list | None = None) -> list[dict]:
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def load_game_picks() -> list[dict]:
    return _q("""
        SELECT e.sport, e.game_pk, e.market, e.side, e.matchup, e.commence_time, e.true_prob, b.line
        FROM ev_current e
        LEFT JOIN ev_best_lines b USING (sport, game_pk, market, side)
        WHERE e.is_pick
    """)


def load_game_odds(game_pks: list[int]) -> list[dict]:
    if not game_pks:
        return []
    return _q("""
        SELECT DISTINCT ON (game_pk, market, side, book) game_pk, market, side, book, line, price
        FROM odds_snapshot
        WHERE game_pk = ANY(%s) AND market IN ('moneyline','spread','total')
          AND COALESCE(player_name, '') = '' AND captured_at <= commence_time
          AND captured_at > now() - interval '48 hours'
        ORDER BY game_pk, market, side, book, captured_at DESC
    """, [game_pks])


def load_prop_picks() -> list[dict]:
    return _q("""
        SELECT game_pk, player_id, player_name, market, side, line, model_prob, matchup, commence_time
        FROM ev_prop_picks_current WHERE is_pick
    """)


def load_prop_odds(game_pks: list[int]) -> list[dict]:
    if not game_pks:
        return []
    rows = _q("""
        SELECT game_pk, market, side, player_name, book, line, price, captured_at
        FROM odds_snapshot
        WHERE game_pk = ANY(%s) AND market = ANY(%s) AND captured_at <= commence_time
          AND captured_at > now() - interval '48 hours'
    """, [game_pks, sorted(set(SIM_TO_ODDS_MARKET.values()))])
    return latest_capture_only(rows)


def ticket_summary(t: dict) -> str:
    return f"{t['book']} {t['parlay_price']:+d} ev={t['ev']:.3f} :: " + " / ".join(l["label"] for l in t["legs"])


def main() -> None:
    game_picks, prop_picks = load_game_picks(), load_prop_picks()
    legs = (assemble_game_legs(game_picks, load_game_odds(sorted({p["game_pk"] for p in game_picks})))
            + assemble_prop_legs(prop_picks, load_prop_odds(sorted({p["game_pk"] for p in prop_picks}))))
    tickets = build_best_parlays(legs, excluded_keys=locked_leg_keys())
    # Drop tickets whose first leg has already commenced (compare as aware datetimes)
    now = datetime.now(timezone.utc)
    tickets = [t for t in tickets if not _is_ticket_locked(t, now)]
    for t in tickets:
        t["model_version"] = MODEL_VERSION
    n = replace_unlocked_best_parlays(tickets)
    priced = sum(1 for l in legs if l["book_prices"])
    print(f"[build_best_parlays] legs={len(legs)} priced={priced} tickets={n}")
    for t in tickets:
        print("  " + ticket_summary(t))


def _is_ticket_locked(ticket: dict, now: datetime) -> bool:
    """Check if a ticket's first leg has already commenced."""
    first_commence = ticket.get("first_commence")
    if first_commence is None:
        return False
    # Normalize to aware datetime
    if isinstance(first_commence, datetime):
        commence_dt = first_commence
    else:
        commence_dt = datetime.fromisoformat(str(first_commence).replace("Z", "+00:00"))
    # Ensure UTC aware
    if commence_dt.tzinfo is None:
        commence_dt = commence_dt.replace(tzinfo=timezone.utc)
    return commence_dt <= now


if __name__ == "__main__":
    main()
