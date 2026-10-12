"""Fill prediction_accuracy.spread_pick_prob / total_pick_prob for rows that were graded before those columns existed.

Run AFTER db/migration_accuracy_pick_probs.sql. For each graded row with a NULL probability it reads the
prediction's stored distribution -- game_predictions.margin_dist / total_dist (latest generated_at per game; the
CFB and MLB source), backed up for NFL by the served-version nfl_sim row (else the newest) -- and computes the model's probability that the
pick wins at the stored closing line (market_spread / market_total), exactly as grade_predictions.py does.
Rows with no distribution, no line, or a model exactly on the line stay NULL; stored values are never overwritten.
Idempotent: a re-run only touches rows that are still NULL.

Usage:
    uv run python scripts/backfill_accuracy_pick_probs.py --dry-run     # read-only: count what would be filled
    uv run python scripts/backfill_accuracy_pick_probs.py               # write
    uv run python scripts/backfill_accuracy_pick_probs.py --sport cfb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sportsmodel import config
from sportsmodel.db import _existing_columns, get_postgres
from sportsmodel.model.distributions import pick_dist, spread_pick_prob, total_pick_prob

from grade_predictions import _safe_sim_dists, _served_version  # noqa: E402  (scripts/ is on sys.path above)

SPORTS = ("nfl", "cfb", "mlb")
CHUNK = 500

# COALESCE keeps a value that is already stored; graded_at is left alone (this is not a re-grade).
UPDATE_SQL = ("UPDATE prediction_accuracy SET spread_pick_prob = COALESCE(%s, spread_pick_prob), "
              "total_pick_prob = COALESCE(%s, total_pick_prob) WHERE sport = %s AND game_pk = %s")


def plan_updates(acc_rows: list[dict], pred_dists: dict, sim_dists: dict) -> list[dict]:
    """Pure: graded rows + {game_pk: {"margin_dist","total_dist"}} (game_predictions) + the same from nfl_sim ->
    [{"sport","game_pk","spread_pick_prob","total_pick_prob"}] for rows that gain at least one new value.
    A column that is already stored is neither recomputed nor overwritten."""
    out = []
    for r in acc_rows:
        pk = r["game_pk"]
        pred, sim = pred_dists.get(pk), sim_dists.get(pk)
        spread = None if r.get("spread_pick_prob") is not None else spread_pick_prob(
            pick_dist("margin_dist", pred, sim), r.get("pred_margin"), r.get("market_spread"))
        total = None if r.get("total_pick_prob") is not None else total_pick_prob(
            pick_dist("total_dist", pred, sim), r.get("pred_total"), r.get("market_total"))
        if spread is None and total is None:
            continue
        out.append({"sport": r["sport"], "game_pk": pk, "spread_pick_prob": spread, "total_pick_prob": total})
    return out


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _load_acc_rows(cur, sport: str, have_cols: bool) -> list[dict]:
    todo = "AND (spread_pick_prob IS NULL OR total_pick_prob IS NULL)" if have_cols else ""
    sel = "spread_pick_prob, total_pick_prob" if have_cols else "NULL::float8, NULL::float8"
    cur.execute(f"""
        SELECT sport, game_pk, pred_margin, market_spread, pred_total, market_total, {sel}
        FROM prediction_accuracy
        WHERE sport = %s AND actual_winner IS NOT NULL {todo}
        ORDER BY game_pk
    """, (sport,))
    cols = ["sport", "game_pk", "pred_margin", "market_spread", "pred_total", "market_total",
            "spread_pick_prob", "total_pick_prob"]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _load_pred_dists(cur, sport: str, game_pks: list) -> dict:
    out: dict = {}
    for chunk in _chunks(game_pks, CHUNK):
        cur.execute("""
            SELECT DISTINCT ON (game_pk) game_pk, margin_dist, total_dist
            FROM game_predictions WHERE sport = %s AND game_pk = ANY(%s)
            ORDER BY game_pk, generated_at DESC
        """, (sport, chunk))
        out.update({pk: {"margin_dist": md, "total_dist": td} for pk, md, td in cur.fetchall()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="read-only: report what would be filled, write nothing")
    ap.add_argument("--sport", choices=SPORTS, help="only this sport (default: all)")
    args = ap.parse_args()
    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (the backfill reads/writes Supabase).")

    plans: dict[str, list[dict]] = {}
    served = _served_version()   # prefer the served nfl_sim version's distribution, as the game page does
    with get_postgres() as conn, conn.cursor() as cur:
        have = _existing_columns(cur, "prediction_accuracy", ["spread_pick_prob", "total_pick_prob"])
        have_cols = len(have) == 2
        if not have_cols and not args.dry_run:
            raise SystemExit("prediction_accuracy has no spread_pick_prob / total_pick_prob yet -- run "
                             "db/migration_accuracy_pick_probs.sql first.")
        if not have_cols:
            print("NOTE migration not run yet: counting every graded row as unfilled (dry-run only).")
        for sport in ([args.sport] if args.sport else SPORTS):
            acc = _load_acc_rows(cur, sport, have_cols)
            pks = [r["game_pk"] for r in acc]
            preds = _load_pred_dists(cur, sport, pks)
            sims = _safe_sim_dists(conn, cur, sport, pks, served)
            plans[sport] = plan_updates(acc, preds, sims)
            n_s = sum(1 for u in plans[sport] if u["spread_pick_prob"] is not None)
            n_t = sum(1 for u in plans[sport] if u["total_pick_prob"] is not None)
            print(f"{sport}: {len(acc)} graded rows lacking a probability -> {len(plans[sport])} updatable "
                  f"(spread {n_s}, total {n_t}); {len(acc) - len(plans[sport])} stay NULL (no distribution / line)")
        total_rows = sum(len(p) for p in plans.values())
        if args.dry_run:
            print(f"dry-run: would update {total_rows} row(s); nothing written.")
            return
        for sport, plan in plans.items():
            for chunk in _chunks(plan, CHUNK):
                cur.executemany(UPDATE_SQL, [(u["spread_pick_prob"], u["total_pick_prob"], u["sport"], u["game_pk"])
                                             for u in chunk])
        conn.commit()
        print(f"Updated {total_rows} prediction_accuracy row(s).")


if __name__ == "__main__":
    main()
