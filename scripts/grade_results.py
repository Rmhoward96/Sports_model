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
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import get_postgres, update_graded_picks, upsert_prediction_results
from sportsmodel.ingest import mlb_results
from sportsmodel.model import calibration
from sportsmodel.model.distributions import apply_affine, prob_cover, prob_over_dist
from sportsmodel.model.odds import american_to_prob

# Results-provider seam: sport key -> module exposing final_game_pks(start,end) and
# fetch_results(game_pk). NFL provider is registered in a later plan (P1); MLB is the
# only entry here, so the default --sport (mlb) path is byte-for-byte unchanged.
RESULTS_PROVIDERS = {"mlb": mlb_results}

# Totals/margin distribution calibration (loc, scale), fit by fit_calibration_sim.py.
# loc re-centers the sim mean the scoring channels didn't fully close; scale finishes
# the width. Identity (0, 1) if absent. Applied before prob_over_dist / prob_cover.
_cal = calibration.load()
_TOTAL_CAL = (_cal.get("total_dist", {}).get("loc", 0.0), _cal.get("total_dist", {}).get("scale", 1.0))
_MARGIN_CAL = (_cal.get("margin_dist", {}).get("loc", 0.0), _cal.get("margin_dist", {}).get("scale", 1.0))


def _calibrated_total(d):
    return apply_affine(d, _TOTAL_CAL[0], _TOTAL_CAL[1]) if d else d


def _calibrated_margin(d):
    return apply_affine(d, _MARGIN_CAL[0], _MARGIN_CAL[1]) if d else d

BATTER_MARKETS = {"hits", "total_bases", "home_run", "hrr"}
PITCHER_MARKETS = {"pitcher_ks", "hits_allowed", "outs_recorded"}

# Hard CLV fresh-start floor: games with game_date BEFORE this are NEVER graded, so the
# track record starts clean here (excludes pre-fix HR/prop noise). ISO date strings
# compare chronologically. Set to "" to remove the floor and grade the full rolling window.
FRESH_START = "2026-08-21"


def _window_start(days: int, today: date | None = None) -> str:
    """Rolling grade-window start (today - days), floored at FRESH_START."""
    d = today or date.today()
    rolling = (d - timedelta(days=days)).isoformat()
    return max(rolling, FRESH_START) if FRESH_START else rolling


def _decimal(price) -> float:
    price = float(price)
    if price == 0:
        return float("nan")  # 0 is not a valid American price
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


def grade_pick(pick, actual, novig_close, home_score=None, away_score=None):
    """Grade one logged `picks` row at its locked bet price and compute CLV.

    `actual` is the market-appropriate outcome value: margin (home-away) for moneyline
    and spread, total runs for total, the stat for props. `home_score`/`away_score` are
    the final game runs, stored so the track record can show the score / margin / total.
    `novig_close` is the consensus no-vig closing prob of the picked side. Profit is flat
    1u at the bet price.
    """
    market, side, line = pick["market"], pick["side"], pick.get("line")
    if market == "moneyline":
        won = (actual > 0) if side == "home" else (actual < 0)
        result = "win" if won else "loss"   # baseball games are decided; no push
    elif market == "spread":
        signed = actual if side == "home" else -actual   # margin from the picked side
        m = signed + line
        result = "push" if m == 0 else ("win" if m > 0 else "loss")
    else:  # total + props: over/under at the line
        if actual == line:
            result = "push"
        else:
            result = "win" if ((actual > line) == (side == "over")) else "loss"
    profit = 0.0 if result == "push" else (_decimal(pick["bet_odds"]) - 1 if result == "win" else -1.0)
    return {"game_pk": pick["game_pk"], "market": market, "player_id": pick["player_id"],
            "actual": float(actual), "result": result, "profit": profit,
            "novig_close": novig_close, "clv": novig_close - pick["novig_bet"],
            "home_score": home_score, "away_score": away_score}


def closing_lines(cur, game_pk, game_date) -> dict:
    """{(market, side, player_lower): [(line, price, books), ...]} closing consensus.

    One entry per distinct LINE (books post alternate lines; averaging across them —
    or averaging American prices — is meaningless). `price` is the median posted price
    at that line; `books` is how many books offered it (used to pick the main line).

    Guard: only trust rows whose commence_time, shifted -10h (the same rule the odds
    matcher uses to assign game_pk), lands on this game's official date. A legacy
    date-boundary mismatch once stapled an adjacent series game's odds onto a game_pk;
    this ensures such a stray can never contribute to a closing line.
    """
    cur.execute("""
        SELECT market, side, lower(coalesce(player_name, '')) pn, line,
               count(*) books,
               percentile_disc(0.5) WITHIN GROUP (ORDER BY price) price
        FROM (
            SELECT DISTINCT ON (market, side, player_name, line, book)
                   market, side, player_name, line, book, price
            FROM odds_snapshot
            WHERE game_pk = %s AND captured_at <= commence_time
              AND ((commence_time AT TIME ZONE 'UTC') - interval '10 hours')::date = %s
            ORDER BY market, side, player_name, line, book, captured_at DESC
        ) t GROUP BY market, side, lower(coalesce(player_name, '')), line
    """, [game_pk, game_date])
    out: dict = {}
    for m, s, pn, line, books, price in cur.fetchall():
        out.setdefault((m, s, pn), []).append(
            (float(line) if line is not None else None,
             float(price) if price is not None else None, int(books)))
    return out


def _primary(entries):
    """Main line among consensus entries: most books, tie -> lowest line. -> (line, price)."""
    if not entries:
        return None
    best = sorted(entries, key=lambda r: (-r[2], r[0] if r[0] is not None else -1))[0]
    return best[0], best[1]


def _price_at(entries, line):
    """Price at a specific line among entries, or None."""
    for l, p, _ in (entries or []):
        if l == line:
            return p
    return None


def _latest_per_game(rows):
    """From (game_pk, game_date, model_version, home, away, generated_at) rows, keep the
    latest-generated model_version per game_pk. A game predicted under multiple versions
    would otherwise be graded once per version -> its game-line pick double-counted."""
    best: dict = {}
    for r in rows:
        gp = r[0]
        if gp not in best or r[5] > best[gp][5]:
            best[gp] = r
    return list(best.values())


def _latest_version_props(rows):
    """From one game's prop rows (pid, pname, market, proj, dist, model_version,
    generated_at), keep only those under the latest-generated model_version, so a stale
    prop slate under an old version isn't graded alongside the current one."""
    if not rows:
        return []
    latest_mv = max(rows, key=lambda r: r[6])[5]
    return [r for r in rows if r[5] == latest_mv]


def _actual_for(market, side, res, pid):
    """Market-appropriate outcome value from a game result: margin (home-away) for
    moneyline/spread, total runs for total, the stat for props. None if a prop stat
    is missing (player DNP)."""
    hr_, ar_ = res["home_runs"], res["away_runs"]
    if market in ("moneyline", "spread"):
        return float(hr_ - ar_)
    if market == "total":
        return float(hr_ + ar_)
    src = res["batters"] if market in BATTER_MARKETS else res["pitchers"]
    stat = src.get(pid, {}).get(market) if pid in src else None
    return None if stat is None else float(stat)


def _other(market, side, line):
    """The opposite side + its line, for computing the consensus no-vig close."""
    if market == "moneyline":
        return ("away" if side == "home" else "home"), None
    if market == "spread":
        return ("away" if side == "home" else "home"), -line
    return ("under" if side == "over" else "over"), line  # total + props


def _novig_close(close, market, side, line, player_lower):
    """Consensus no-vig closing prob of the picked side; None if no close captured.
    Over-only markets (e.g. home_run, no under posted) fall back to the raw implied."""
    picked = _price_at(close.get((market, side, player_lower)) or [], line)
    if picked is None:
        return None
    oside, oline = _other(market, side, line)
    other = _price_at(close.get((market, oside, player_lower)) or [], oline)
    if other is None:
        return american_to_prob(picked)
    io, iu = american_to_prob(picked), american_to_prob(other)
    return io / (io + iu)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--sport", default="mlb")
    args = ap.parse_args()
    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (grading reads/writes Supabase).")

    prov = RESULTS_PROVIDERS[args.sport]
    start = _window_start(args.days)
    end = date.today().isoformat()
    if start > end:  # fresh-start floor is still in the future — nothing to grade yet
        print(f"No games to grade: window start {start} is after today {end} (fresh-start floor).")
        return
    finals = prov.final_game_pks(start, end)
    print(f"{len(finals)} final games in window {start}..{end}")

    graded_rows: list[dict] = []
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("""SELECT game_pk, market, player_id, game_date, side, line, bet_odds,
                              novig_bet, player_name
                       FROM picks WHERE status = 'pending' AND game_date >= %s AND sport = %s""",
                    [start, args.sport])
        pending = cur.fetchall()
        res_cache: dict = {}
        close_cache: dict = {}
        for game_pk, market, player_id, gdate, side, line, bet_odds, novig_bet, pname in pending:
            if game_pk not in finals:
                continue  # only grade truly-final games
            if game_pk not in res_cache:
                res_cache[game_pk] = prov.fetch_results(game_pk)
                close_cache[game_pk] = closing_lines(cur, game_pk, gdate)
            res = res_cache[game_pk]
            if res is None:
                continue
            actual = _actual_for(market, side, res, player_id)
            if actual is None:
                continue  # prop stat missing (player didn't play)
            pl = (pname or "").lower().strip()
            novig_close = _novig_close(close_cache[game_pk], market, side, line, pl)
            if novig_close is None:
                novig_close = novig_bet  # no closing price captured -> CLV 0
            pick = {"game_pk": game_pk, "market": market, "player_id": player_id,
                    "side": side, "line": line, "bet_odds": bet_odds, "novig_bet": novig_bet}
            graded_rows.append(grade_pick(pick, actual, novig_close,
                                          res["home_runs"], res["away_runs"]))

    print(f"grading {len(graded_rows)} pending picks")
    if graded_rows:
        n = update_graded_picks(graded_rows)
        print(f"Updated {n} graded picks with result + CLV.")


def _row(game_pk, market, pid, pname, mv, gdate, model_num, line, price, lean, actual,
         result, profit, edge, model_prob=None, market_prob=None, ev=None):
    return {"game_pk": game_pk, "market": market, "player_id": pid, "player_name": pname,
            "model_version": mv, "game_date": gdate, "model_number": model_num,
            "closing_line": line, "closing_price": int(price) if price is not None else None,
            "lean": lean, "actual": actual, "result": result, "profit": profit, "edge": edge,
            "model_prob": model_prob, "market_prob": market_prob, "ev": ev}


def _grade_game(game_pk, gdate, mv, res, close, pred_total, home_wp, pred_margin,
                 total_dist=None, margin_dist=None, home_name="", away_name="") -> list[dict]:
    out = []
    hr_, ar_ = res["home_runs"], res["away_runs"]
    # Moneyline
    mh, ma = _primary(close.get(("moneyline", "home", ""))), _primary(close.get(("moneyline", "away", "")))
    if mh and ma and mh[1] and ma[1]:
        ph, pa = american_to_prob(mh[1]), american_to_prob(ma[1])
        novig_home = ph / (ph + pa)
        lean = "home" if home_wp > novig_home else "away"
        price = mh[1] if lean == "home" else ma[1]
        won = (hr_ > ar_) if lean == "home" else (ar_ > hr_)
        model_p = home_wp if lean == "home" else 1 - home_wp
        market_p = novig_home if lean == "home" else 1 - novig_home
        pname = home_name if lean == "home" else away_name
        out.append(_row(game_pk, "moneyline", 0, pname, mv, gdate, home_wp, None, price, lean,
                        1.0 if hr_ > ar_ else 0.0, "win" if won else "loss",
                        _decimal(price) - 1 if won else -1.0, model_p - market_p,
                        model_prob=model_p, market_prob=market_p,
                        ev=model_p * _decimal(price) - 1))
    # Total — grade at the main (most-booked) closing line.
    over = _primary(close.get(("total", "over", "")))
    if pred_total is not None and over and over[0] is not None and over[1]:
        line = over[0]
        pu = _price_at(close.get(("total", "under", "")), line) or over[1]
        cal_total = pred_total + _TOTAL_CAL[0]
        actual_total = hr_ + ar_
        model_p = market_p = ev = None
        # Pick the side by EV from the calibrated total distribution's P(over), NOT by
        # mean-vs-line. MLB totals are right-skewed (mean ~8.9 >> median ~8.0), so the
        # market line sits near the median; leaning on the calibrated MEAN would take
        # OVER on nearly every game. P(over) from the distribution prices the skew.
        p_over_line = float("nan")
        if total_dist:
            td = json.loads(total_dist) if isinstance(total_dist, str) else total_dist
            p_over_line = prob_over_dist(_calibrated_total(td), line)
        if p_over_line == p_over_line:  # distribution available
            io = american_to_prob(over[1])
            iu = american_to_prob(pu) if pu else 1 - io
            novig_over = io / (io + iu)
            ev_over = p_over_line * _decimal(over[1]) - 1
            ev_under = (1 - p_over_line) * _decimal(pu) - 1
            if ev_over >= ev_under:
                lean, price, model_p, market_p, ev = "over", over[1], p_over_line, novig_over, ev_over
            else:
                lean, price, model_p, market_p, ev = "under", pu, 1 - p_over_line, 1 - novig_over, ev_under
            if actual_total == line:
                result, profit = "push", 0.0
            else:
                won = (actual_total > line) if lean == "over" else (actual_total < line)
                result, profit = ("win", _decimal(price) - 1) if won else ("loss", -1.0)
            edge = (model_p - market_p)
        else:  # legacy fallback: no stored distribution -> mean-vs-line
            lean, price, result, profit, edge = _grade_ou(cal_total, line, actual_total, over[1], pu)
        # Only track a +EV pick in the CLV. A "pass" (computed EV <= 0) isn't recorded.
        # A legacy row without a stored dist (ev is None) still records (mean-based).
        if ev is None or ev > 0:
            pname = f"{lean.title()} {line:g}"
            out.append(_row(game_pk, "total", 0, pname, mv, gdate, cal_total, line, price, lean,
                            actual_total, result, profit, edge,
                            model_prob=model_p, market_prob=market_p, ev=ev))
    # Spread (run line) — home_line is the home team's spread point (e.g. -1.5).
    # "Home covers" iff actual_margin + home_line > 0 (equivalently: away_line is the
    # negation of home_line, and away covers iff actual_margin + away_line < 0, i.e.
    # actual_margin - home_line < 0, i.e. actual_margin < home_line -> same boundary).
    sp = _primary(close.get(("spread", "home", "")))
    if pred_margin is not None and sp and sp[0] is not None and sp[1]:
        sl = sp[0]
        away_price = _price_at(close.get(("spread", "away", "")), -sl) or sp[1]
        actual_margin = hr_ - ar_
        home_covers = actual_margin + sl > 0
        push = actual_margin + sl == 0
        # Pick the run line by EV, not by mean margin vs the number. A game's expected
        # margin is almost always inside 1.5 runs, so a mean-vs-line rule would take the
        # +1.5 side nearly every time even though it's juiced (~-175). Use P(cover) from
        # the margin distribution and the real price on each side; the +EV side wins.
        model_p = market_p = ev = None
        lean_home = pred_margin + sl > 0  # fallback when no distribution is stored
        if margin_dist:
            md = json.loads(margin_dist) if isinstance(margin_dist, str) else margin_dist
            p_home_cover = prob_cover(_calibrated_margin(md), sl)
            if p_home_cover == p_home_cover:  # not NaN
                io = american_to_prob(sp[1])
                iu = american_to_prob(away_price) if away_price else 1 - io
                novig_home = io / (io + iu)
                ev_home = p_home_cover * _decimal(sp[1]) - 1
                ev_away = (1 - p_home_cover) * _decimal(away_price) - 1
                lean_home = ev_home >= ev_away
                if lean_home:
                    model_p, market_p, ev = p_home_cover, novig_home, ev_home
                else:
                    model_p, market_p, ev = 1 - p_home_cover, 1 - novig_home, ev_away
        lean = "home" if lean_home else "away"
        price = sp[1] if lean_home else away_price
        if push:
            result, profit = "push", 0.0
        else:
            won = home_covers if lean_home else (not home_covers)
            result, profit = ("win", _decimal(price) - 1) if won else ("loss", -1.0)
        edge = (pred_margin + sl) if lean_home else -(pred_margin + sl)
        # Only track a +EV run-line pick in the CLV. A "pass" (EV <= 0) isn't recorded;
        # a legacy row without a margin dist (ev is None) still records (mean-based).
        if ev is None or ev > 0:
            pname = f"{home_name} {sl:+g}" if lean_home else f"{away_name} {-sl:+g}"
            out.append(_row(game_pk, "spread", 0, pname, mv, gdate, pred_margin, sl, price, lean,
                            actual_margin, result, profit, edge,
                            model_prob=model_p, market_prob=market_p, ev=ev))
    return out


def _model_p_over(market, dist, line):
    """Calibrated model P(stat > line) from the stored distribution, or None."""
    if not dist:
        return None
    if isinstance(dist, str):
        import json
        dist = json.loads(dist)
    p = prob_over_dist(dist, line)
    if p != p:  # NaN (empty/malformed distribution)
        return None
    return calibration.calibrate(market, p)


def _grade_prop(game_pk, gdate, mv, res, close, pid, pname, market, proj, dist) -> dict | None:
    """Grade one prop on EXPECTED VALUE at the book's line, not mean-vs-line.

    The edge that matters for a threshold bet is P(clear the line) vs the price's
    implied probability. Compute the model's P(over) at the *book's* line, pick the
    side with the better EV, and record model/market probabilities + EV.
    """
    src = res["batters"] if market in BATTER_MARKETS else res["pitchers"]
    stat = src.get(pid, {}).get(market) if pid in src else None
    if stat is None:
        return None
    pn = str(pname).lower().strip()
    over = _primary(close.get((market, "over", pn)))  # main (most-booked) closing line
    if not over or over[0] is None or not over[1]:  # `not over[1]` also drops price 0
        return None
    line, po = over[0], over[1]

    p_over = _model_p_over(market, dist, line)
    if p_over is None:
        return None  # legacy row without a stored distribution -> no EV, skip

    pu = _price_at(close.get((market, "under", pn)), line)  # under price AT the same line
    pu = pu if pu else None  # falsy (0 / None) -> treat as missing

    ev_over = p_over * _decimal(po) - 1
    ev_under = (1 - p_over) * _decimal(pu) - 1 if pu is not None else float("-inf")

    # No-vig market probability of clearing the line.
    io = american_to_prob(po)
    iu = american_to_prob(pu) if pu is not None else (1 - io)
    novig_over = io / (io + iu)

    if ev_over >= ev_under:
        lean, price, model_p, market_p, ev = "over", po, p_over, novig_over, ev_over
    else:
        lean, price, model_p, market_p, ev = "under", pu, 1 - p_over, 1 - novig_over, ev_under

    if ev <= 0:
        return None  # "pass" -- not a +EV bet, so it isn't tracked in the CLV record

    if stat == line:
        result, profit = "push", 0.0
    else:
        won = (stat > line) if lean == "over" else (stat < line)
        result, profit = ("win", _decimal(price) - 1) if won else ("loss", -1.0)

    return _row(game_pk, market, pid, pname, mv, gdate, proj, line, price, lean,
                float(stat), result, profit, model_p - market_p,
                model_prob=model_p, market_prob=market_p, ev=ev)


if __name__ == "__main__":
    main()
