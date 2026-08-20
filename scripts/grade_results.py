"""Grade predictions against closing lines + actual results -> prediction_results.

For each recently-finished game with predictions: pull the actual outcome (MLB
StatsAPI boxscore), the closing line (last odds snapshot before first pitch), decide
the side the model favored vs that closing line, and record win/loss/push, the edge,
and profit (1u staked at the closing price). This is the model's real track record.

Runs daily (grade-results.yml) over a rolling window; idempotent.

Usage:
    uv run python scripts/grade_results.py [--days 5]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import get_postgres, upsert_prediction_results
from sportsmodel.ingest import mlb_results
from sportsmodel.model.odds import american_to_prob

BATTER_MARKETS = {"hits", "total_bases", "home_run", "hrr"}
PITCHER_MARKETS = {"pitcher_ks", "hits_allowed", "outs_recorded"}


def _decimal(price) -> float:
    price = float(price)
    return 1 + (price / 100 if price > 0 else 100 / -price)


def _grade_ou(model_num, line, actual, price_over, price_under):
    """Over/under grade for totals + props. Returns (lean, price, result, profit, edge)."""
    lean = "over" if model_num > line else "under"
    price = price_over if lean == "over" else price_under
    edge = (model_num - line) if lean == "over" else (line - model_num)
    if actual == line:
        return lean, price, "push", 0.0, edge
    won = (actual > line) if lean == "over" else (actual < line)
    return lean, price, ("win" if won else "loss"), (_decimal(price) - 1 if won else -1.0), edge


def closing_lines(cur, game_pk) -> dict:
    """{(market, side, player_lower): (line, price)} — consensus closing across books."""
    cur.execute("""
        SELECT market, side, lower(coalesce(player_name, '')) pn, avg(line), avg(price)
        FROM (
            SELECT DISTINCT ON (market, side, player_name, book)
                   market, side, player_name, book, line, price
            FROM odds_snapshot WHERE game_pk = %s AND captured_at <= commence_time
            ORDER BY market, side, player_name, book, captured_at DESC
        ) t GROUP BY market, side, lower(coalesce(player_name, ''))
    """, [game_pk])
    return {(m, s, pn): (line, price) for m, s, pn, line, price in cur.fetchall()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    args = ap.parse_args()
    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (grading reads/writes Supabase).")

    start = (date.today() - timedelta(days=args.days)).isoformat()
    rows: list[dict] = []
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT game_pk, game_date, model_version, home_team_name, away_team_name
            FROM game_predictions WHERE game_date >= %s
        """, [start])
        games = cur.fetchall()
        print(f"{len(games)} predicted games in window since {start}")

        graded_games = 0
        for game_pk, gdate, mv, home_name, away_name in games:
            res = mlb_results.fetch_results(game_pk)
            if res is None:
                continue  # not final yet
            close = closing_lines(cur, game_pk)
            graded_games += 1

            # game predictions for this game/model
            cur.execute("""SELECT pred_total, home_win_prob FROM game_predictions
                           WHERE game_pk = %s AND model_version = %s""", [game_pk, mv])
            gp = cur.fetchone()
            if gp:
                pred_total, home_wp = gp
                rows += _grade_game(game_pk, gdate, mv, res, close, pred_total, home_wp)

            # prop predictions for this game/model
            cur.execute("""SELECT player_id, player_name, market, projected_mean
                           FROM prop_predictions WHERE game_pk = %s AND model_version = %s""",
                        [game_pk, mv])
            for pid, pname, market, proj in cur.fetchall():
                r = _grade_prop(game_pk, gdate, mv, res, close, pid, pname, market, proj)
                if r:
                    rows.append(r)

    print(f"graded {graded_games} final games -> {len(rows)} results")
    if rows:
        n = upsert_prediction_results(rows)
        print(f"Upserted {n} rows into prediction_results.")


def _row(game_pk, market, pid, pname, mv, gdate, model_num, line, price, lean, actual, result, profit, edge):
    return {"game_pk": game_pk, "market": market, "player_id": pid, "player_name": pname,
            "model_version": mv, "game_date": gdate, "model_number": model_num,
            "closing_line": line, "closing_price": int(price) if price is not None else None,
            "lean": lean, "actual": actual, "result": result, "profit": profit, "edge": edge}


def _grade_game(game_pk, gdate, mv, res, close, pred_total, home_wp) -> list[dict]:
    out = []
    hr_, ar_ = res["home_runs"], res["away_runs"]
    # Moneyline
    mh, ma = close.get(("moneyline", "home", "")), close.get(("moneyline", "away", ""))
    if mh and ma:
        ph, pa = american_to_prob(mh[1]), american_to_prob(ma[1])
        novig_home = ph / (ph + pa)
        lean = "home" if home_wp > novig_home else "away"
        price = mh[1] if lean == "home" else ma[1]
        won = (hr_ > ar_) if lean == "home" else (ar_ > hr_)
        model_p = home_wp if lean == "home" else 1 - home_wp
        market_p = novig_home if lean == "home" else 1 - novig_home
        out.append(_row(game_pk, "moneyline", 0, "", mv, gdate, home_wp, None, price, lean,
                        1.0 if hr_ > ar_ else 0.0, "win" if won else "loss",
                        _decimal(price) - 1 if won else -1.0, model_p - market_p))
    # Total
    over = close.get(("total", "over", ""))
    if over:
        under = close.get(("total", "under", ""))
        po, pu = over[1], (under[1] if under else over[1])
        lean, price, result, profit, edge = _grade_ou(pred_total, over[0], hr_ + ar_, po, pu)
        out.append(_row(game_pk, "total", 0, "", mv, gdate, pred_total, over[0], price, lean,
                        hr_ + ar_, result, profit, edge))
    return out


def _grade_prop(game_pk, gdate, mv, res, close, pid, pname, market, proj) -> dict | None:
    src = res["batters"] if market in BATTER_MARKETS else res["pitchers"]
    stat = src.get(pid, {}).get(market) if pid in src else None
    if stat is None:
        return None
    pn = str(pname).lower().strip()
    over = close.get((market, "over", pn))
    if not over:
        return None
    under = close.get((market, "under", pn))
    po, pu = over[1], (under[1] if under else over[1])
    lean, price, result, profit, edge = _grade_ou(proj, over[0], stat, po, pu)
    return _row(game_pk, market, pid, pname, mv, gdate, proj, over[0], price, lean,
                float(stat), result, profit, edge)


if __name__ == "__main__":
    main()
