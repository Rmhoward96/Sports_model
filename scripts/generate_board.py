"""Precompute the serving board for the Blue Edge front-end.

Reads the latest game/prop predictions + latest per-book odds from Supabase, builds
best-book board rows via sportsmodel.serving.board, upserts the live `board_picks`
table, and locks any newly-+EV pick into the `picks` bet log (first-+EV price). Run
after each odds capture. Sport is selected via --sport (default: mlb).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel import config, sports
from sportsmodel.db import clear_board_picks, get_postgres, insert_new_picks, upsert_board_picks
from sportsmodel.model import calibration
from sportsmodel.serving import board

MARKET_LABEL = {
    "moneyline": "Moneyline", "spread": "Spread", "total": "Total",
    "hits": "Hits", "total_bases": "Total Bases", "home_run": "Home Runs",
    "hrr": "Hits + Runs + RBIs", "pitcher_ks": "Strikeouts",
    "hits_allowed": "Hits Allowed", "outs_recorded": "Outs Recorded",
}
# home_run is deliberately excluded: an over-only longshot that manufactures inflated
# EV and skews the board/record. Add it back here if that ever changes.
PROP_MARKETS = ("hits", "total_bases", "hrr",
                "pitcher_ks", "hits_allowed", "outs_recorded")
# NFL prop-market names are placeholders wired up in P4, once NFL prop predictions
# exist to board; keyed off the sport's odds prop_market_map for now.
PROP_MARKETS_BY_SPORT = {
    "mlb": PROP_MARKETS,
    "nfl": tuple(sports.get("nfl").prop_market_map.keys()),
}


def _load(dist):
    return json.loads(dist) if isinstance(dist, str) else dist


def _flatten(by_line):
    return [e for entries in by_line.values() for e in entries]


def _main_line(by_line):
    """Line with the most books (tie -> lowest); None if empty."""
    if not by_line:
        return None
    return sorted(by_line.items(), key=lambda kv: (-len(kv[1]), kv[0]))[0][0]


def _enrich(row, game, market, player_id=0, player_name=None, team=None, sport="mlb"):
    if row is None:
        return None
    row = dict(row)
    row.update({
        "sport": sport, "game_pk": game["game_pk"], "game_date": game["game_date"],
        "commence_time": game.get("commence_time"), "matchup": game["matchup"],
        "market_label": MARKET_LABEL[market], "player_id": player_id,
        "player_name": player_name, "team": team,
    })
    return row


def build_rows(game, prop_preds, odds, cals, sport="mlb"):
    """Assemble board_picks rows for one game across all markets. `odds` is
    {(market, side, player_lower): {line: [(book, american), ...]}}. `cals` is
    (total_cal, margin_cal). Returns a list of enriched row dicts (drops Nones)."""
    total_cal, margin_cal = cals
    rows = []

    r = board.moneyline_row(
        game["home_win_prob"],
        _flatten(odds.get(("moneyline", "home", ""), {})),
        _flatten(odds.get(("moneyline", "away", ""), {})),
        game["home_name"], game["away_name"])
    rows.append(_enrich(r, game, "moneyline", sport=sport))

    over, under = odds.get(("total", "over", ""), {}), odds.get(("total", "under", ""), {})
    tl = _main_line(over)
    if tl is not None:
        r = board.total_row(_load(game["total_dist"]), total_cal, tl, over.get(tl, []), under.get(tl, []))
        rows.append(_enrich(r, game, "total", sport=sport))

    sh, sa = odds.get(("spread", "home", ""), {}), odds.get(("spread", "away", ""), {})
    hl = _main_line(sh)
    if hl is not None:
        r = board.spread_row(_load(game["margin_dist"]), margin_cal, hl,
                             sh.get(hl, []), sa.get(-hl, []), game["home_name"], game["away_name"])
        rows.append(_enrich(r, game, "spread", sport=sport))

    for p in prop_preds:
        m, pl = p["market"], str(p["player_name"]).lower().strip()
        over = odds.get((m, "over", pl), {})
        under = odds.get((m, "under", pl), {})
        ln = _main_line(over)
        if ln is None:
            continue
        r = board.prop_row(m, _load(p["dist"]), m, ln, over.get(ln, []), under.get(ln, []))
        rows.append(_enrich(r, game, m, p["player_id"], p["player_name"], p.get("team"), sport=sport))

    return [r for r in rows if r]


def _to_pick(row):
    """Map a +EV board row to a `picks` insert row (locks the bet at this price)."""
    return {
        "game_pk": row["game_pk"], "market": row["market"], "player_id": row["player_id"],
        "sport": row["sport"], "game_date": row["game_date"],
        "commence_time": row["commence_time"], "matchup": row["matchup"],
        "market_label": row["market_label"], "player_name": row["player_name"],
        "team": row["team"], "pick_label": row["pick_label"], "side": row["side"],
        "line": row["line"], "bet_odds": row["odds"], "bet_book": row["book"],
        "model_prob": row["model_prob"], "novig_bet": row["implied_prob"],
        "ev_bet": row["ev"],
    }


def _load_odds(cur, game_pk):
    """{(market, side, player_lower): {line: [(book, american), ...]}} from the latest
    per-book capture (pre-commence) for a game, plus the game's commence_time."""
    cur.execute("""
        SELECT market, side, lower(coalesce(player_name, '')) pn, line, book, price, commence_time
        FROM (
            SELECT DISTINCT ON (market, side, player_name, line, book)
                   market, side, player_name, line, book, price, commence_time
            FROM odds_snapshot
            WHERE game_pk = %s AND captured_at <= commence_time
            ORDER BY market, side, player_name, line, book, captured_at DESC
        ) t
    """, [game_pk])
    odds, commence = {}, None
    for market, side, pn, line, book, price, ct in cur.fetchall():
        commence = commence or ct
        odds.setdefault((market, side, pn), {}).setdefault(line, []).append((book, int(price)))
    return odds, commence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="mlb", help="Sport key (default: mlb)")
    args = parser.parse_args()
    sport = args.sport
    prop_markets = PROP_MARKETS_BY_SPORT[sport]

    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (generate_board reads/writes Supabase).")
    cal = calibration.load()
    total_cal = (cal.get("total_dist", {}).get("loc", 0.0), cal.get("total_dist", {}).get("scale", 1.0))
    margin_cal = (cal.get("margin_dist", {}).get("loc", 0.0), cal.get("margin_dist", {}).get("scale", 1.0))
    today = date.today().isoformat()
    all_rows = []
    with get_postgres() as conn, conn.cursor() as cur:
        # latest game-prediction version per game for today+
        cur.execute("""
            SELECT DISTINCT ON (game_pk) game_pk, game_date, home_team_name, away_team_name,
                   total_dist, pred_margin, margin_dist, home_win_prob, pred_total
            FROM game_predictions WHERE game_date >= %s
            ORDER BY game_pk, generated_at DESC
        """, [today])
        games = cur.fetchall()
        for gp, gdate, home, away, tdist, pmargin, mdist, hwp, ptotal in games:
            odds, commence = _load_odds(cur, gp)
            if not odds:
                continue
            # latest prop-prediction version's slate for this game
            cur.execute("""
                SELECT player_id, player_name, team_name, market, dist FROM prop_predictions
                WHERE game_pk = %s AND model_version = (
                    SELECT model_version FROM prop_predictions WHERE game_pk = %s
                    ORDER BY generated_at DESC LIMIT 1)
            """, [gp, gp])
            props = [{"player_id": pid, "player_name": pn, "team": tm, "market": mk, "dist": d}
                     for pid, pn, tm, mk, d in cur.fetchall() if mk in prop_markets]
            game = {"game_pk": gp, "game_date": gdate, "commence_time": commence,
                    "home_name": home, "away_name": away, "matchup": f"{away} @ {home}",
                    "pred_total": ptotal, "total_dist": tdist, "pred_margin": pmargin,
                    "margin_dist": mdist, "home_win_prob": hwp}
            all_rows += build_rows(game, props, odds, (total_cal, margin_cal), sport=sport)

    if all_rows:
        clear_board_picks()  # full refresh: drop orphaned rows before writing the current slate
    n_board = upsert_board_picks(all_rows)
    picks = [_to_pick(r) for r in all_rows if r["is_pick"]]
    n_picks = insert_new_picks(picks)
    print(f"board_picks upserted: {n_board} | +EV picks logged (new): {n_picks} | "
          f"at {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}")


if __name__ == "__main__":
    main()
