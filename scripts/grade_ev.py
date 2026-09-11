"""Grade `ev_picks` FORWARD against the final result and the Pinnacle
CLOSING price, tracking closing-line value (CLV) -> `ev_results`.

Sibling of grade_desk_picks.py (model it closely follows), but for the +EV
desk-driven pilot board: not "did the desk's number beat the closing
number" (that's grade_desk_picks's job) but "did the +EV bet actually
placed -- one row per (game, market, side) in ev_picks -- win, and did its
price/number beat what the market closed at."

Two conventions carried over unchanged from grade_desk_picks:
  - Both the pick-time and closing spread/total NUMBERS are the SAME
    standard-sportsbook convention: home team's spread, negative = home
    favored. No conversion between them.
  - `won` for market="spread"/"total" is graded with the exact same
    cover/over math as grade_desk_picks.grade_pick -- just against the
    PICK-TIME number instead of the closing number, because a placed bet's
    outcome is fixed by the number it was placed at, not by where the
    market later closed.

One convention that's new here: `ev_picks` stores only the pick-time
Pinnacle PRICE (pinnacle_price), not the pick-time line -- so main()
reconstructs the pick-time line for spread/total from `odds_snapshot`
(the last Pinnacle snapshot at/before the ev_picks row's created_at), and
looks up the CLOSING Pinnacle price from the same table (the last Pinnacle
snapshot at/before commence_time -- by grading time this is genuinely the
close, since the game has already been played). See docstrings below for
the exact CLV convention.

Runs on a rolling window (like grade_desk_picks); idempotent -- re-running
just re-upserts the same rows via (sport, game_pk, market, side)
(db.upsert_ev_results).

Usage:
    uv run python scripts/grade_ev.py [--days 7]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.cfb import espn as cfb_espn
from sportsmodel.db import get_postgres, upsert_ev_results
from sportsmodel.nfl import espn as nfl_espn

# Results-provider seam: sport key -> module exposing fetch_final(game_pk) ->
# dict|None. Same seam as grade_desk_picks.FINAL_PROVIDERS -- both NFL and
# CFB use ESPN event ids as game_pk.
FINAL_PROVIDERS = {"nfl": nfl_espn, "cfb": cfb_espn}


def _window_start(days: int, today: date | None = None) -> str:
    d = today or date.today()
    return (d - timedelta(days=days)).isoformat()


def _implied_prob(american_price: int) -> float:
    """Single-sided implied probability from an American price. Not de-vigged
    -- grade_ev_pick compares this same conversion at two points in time for
    the SAME side, so the vig (whatever it is at each snapshot) is along for
    the ride on both sides of the comparison; de-vigging would need the
    opposing side's price at both timestamps too, which isn't always
    available (that's exactly why the fallback path below exists)."""
    if american_price > 0:
        return 100.0 / (american_price + 100.0)
    return -american_price / (-american_price + 100.0)


def grade_ev_pick(pick: dict, final: dict, pinnacle_close: dict) -> dict:
    """Pure grading: one `ev_picks` row (as a dict) + its final-score dict +
    the closing Pinnacle price for that same (game, market, side) -> an
    `ev_results` row (dict, columns matching db.upsert_ev_results).

    No network, no DB -- easy to unit test in isolation.

    `pick` = {"sport", "game_pk", "market", "side", "line", "pinnacle_price"}:
      - market: "moneyline" | "spread" | "total".
      - side: "home"/"away" for moneyline/spread, "over"/"under" for total.
      - line: the PICK-TIME Pinnacle number for spread/total (home-referenced
        spread, negative = home favored; or the total number). None for
        moneyline (no line to grade a moneyline pick against).
      - pinnacle_price: the PICK-TIME Pinnacle American price for `side`
        (== ev_picks.pinnacle_price, straight passthrough -- that column IS
        the pick-time price already, no reconstruction needed).

    `final` is the dict returned by cfb.espn.fetch_final / nfl.espn.fetch_final
    (same shape grade_desk_picks.grade_pick consumes):
    {"home_score", "away_score", "market_spread", "market_total"}, where
    market_spread/market_total are the ESPN pickcenter CLOSING NUMBER --
    used here only as the fallback CLV source (see below), never for `won`.

    `pinnacle_close` = {"price": american_int_or_None} -- the CLOSING
    Pinnacle American price for pick["side"] on pick["market"] (the last
    Pinnacle odds_snapshot row at/before commence_time). None if no such
    snapshot exists (e.g. odds ingestion didn't cover that side/book).

    won: did the picked side actually win/cover/hit, graded against the
    PICK-TIME number (NOT the close -- a placed bet's outcome is fixed by
    the number taken, unlike grade_desk_picks's forward re-grade of the
    desk's call). market="moneyline": pick["side"] == actual winner; None on
    a tie. market="spread": SAME cover math as grade_desk_picks.grade_pick,
    with pick["line"] standing in for its market_spread -- cover_margin =
    actual_margin + pick["line"]; push (cover_margin == 0) -> None; None if
    pick["line"] is missing. market="total": SAME over/under math with
    pick["line"] standing in for market_total; push (actual_total ==
    pick["line"]) -> None; None if pick["line"] is missing.

    clv: signed closing-line value from the side the pick took. PRIMARY path
    is price-based: convert both pick["pinnacle_price"] and
    pinnacle_close["price"] to single-sided implied probability via
    _implied_prob, and take
        clv = implied_prob(closing_price) - implied_prob(pick_time_price).
    A HIGHER closing implied probability than the pick-time implied
    probability means the market moved TOWARD this side after the pick was
    placed -- i.e. the side got shorter/more expensive over time -- which
    means the pick-time price was the cheaper, better one -> positive clv.
    Example: took -110 (implied .5238), closed -150 (implied .6) ->
    clv = .6 - .5238 = +.0762 (positive: beat the close). Computed
    regardless of `won` (still meaningful on a push), same as
    grade_desk_picks's clv_spread/clv_total.

    FALLBACK: if pinnacle_close["price"] (or pick["pinnacle_price"]) is
    missing, and the market is spread/total, fall back to grade_desk_picks's
    number-based CLV convention -- pick["line"] (pick-time) vs
    final["market_spread"/"market_total"] (closing NUMBER, from ESPN
    pickcenter): clv_spread = pick_line - closing_line for a home pick,
    negated for away; clv_total = closing_total - pick_line for an over
    pick, negated for under. Same sign meaning: positive = pick-time number
    beat the close. No fallback exists for moneyline (no number to fall back
    to) -- clv is None there if the price is unavailable.
    """
    market = pick.get("market")
    side = pick.get("side")
    home_score = final.get("home_score")
    away_score = final.get("away_score")
    actual_margin = home_score - away_score
    actual_total = home_score + away_score

    # ---- won ----
    if market == "moneyline":
        if home_score == away_score:
            won = None  # tie
        else:
            actual_winner = "home" if home_score > away_score else "away"
            won = (side == actual_winner)
    elif market == "spread":
        pick_line = pick.get("line")
        if pick_line is None:
            won = None
        else:
            cover_margin = actual_margin + pick_line
            if cover_margin == 0:
                won = None  # push
            else:
                covering_side = "home" if cover_margin > 0 else "away"
                won = (side == covering_side)
    elif market == "total":
        pick_line = pick.get("line")
        if pick_line is None:
            won = None
        else:
            if actual_total == pick_line:
                won = None  # push
            else:
                winning_side = "over" if actual_total > pick_line else "under"
                won = (side == winning_side)
    else:
        raise ValueError(f"unknown market: {market!r}")

    # ---- clv ----
    pick_price = pick.get("pinnacle_price")
    close_price = pinnacle_close.get("price")
    if pick_price is not None and close_price is not None:
        clv = _implied_prob(close_price) - _implied_prob(pick_price)
    elif market == "spread" and pick.get("line") is not None and final.get("market_spread") is not None:
        raw = pick["line"] - final["market_spread"]
        clv = raw if side == "home" else -raw
    elif market == "total" and pick.get("line") is not None and final.get("market_total") is not None:
        raw = final["market_total"] - pick["line"]
        clv = raw if side == "over" else -raw
    else:
        clv = None

    return {
        "sport": pick.get("sport"),
        "game_pk": pick.get("game_pk"),
        "market": market,
        "side": side,
        "won": won,
        "clv": clv,
    }


def _pending_ev_picks(cur, sport: str, start: str) -> list[dict]:
    """`ev_picks` rows for `sport` -- one per (game_pk, market, side) --
    whose game has already started (commence_time <= now(), so a final
    score should exist) and that haven't been graded yet.

    DISTINCT ON (game_pk, market, side) keeps only the most-recently-created
    model_version per row (mirrors grade_desk_picks._pending_desk_picks). A
    NOT EXISTS against ev_results skips rows already graded, making the
    script idempotent and cheap to run on a schedule.

    Filtered to is_pick = true: ev_picks also carries "pass" rows (markets
    the desk didn't take, or that didn't clear MIN_EDGE) for every game the
    desk touched -- those aren't real bets, so there's nothing to grade a
    forward CLV against. If a future consumer wants pass-row CLV too (e.g.
    for backtesting "would a pass have won"), drop this filter.
    """
    cur.execute("""
        SELECT DISTINCT ON (ep.game_pk, ep.market, ep.side)
               ep.game_pk, ep.market, ep.side,
               -- pick-time price: the FROZEN opening price when available,
               -- else the (mutable) pinnacle_price for rows written before
               -- open_pinnacle_price existed. This is what CLV is measured from.
               COALESCE(ep.open_pinnacle_price, ep.pinnacle_price) AS pinnacle_price,
               ep.commence_time, ep.created_at
        FROM ev_picks ep
        WHERE ep.sport = %(sport)s AND ep.commence_time >= %(start)s
          AND ep.commence_time <= now()
          AND ep.is_pick = true
          AND NOT EXISTS (
              SELECT 1 FROM ev_results er
              WHERE er.sport = %(sport)s AND er.game_pk = ep.game_pk
                AND er.market = ep.market AND er.side = ep.side
          )
        ORDER BY ep.game_pk, ep.market, ep.side, ep.created_at DESC
    """, {"sport": sport, "start": start})
    cols = ["game_pk", "market", "side", "pinnacle_price", "commence_time", "created_at"]
    return [dict(zip(cols, row), sport=sport) for row in cur.fetchall()]


def _pinnacle_line_at(cur, game_pk: int, market: str, side: str, cutoff) -> float | None:
    """The Pinnacle `line` for (game_pk, market, side) as of the last
    odds_snapshot at/before `cutoff` -- used to reconstruct the PICK-TIME
    line (cutoff = the ev_picks row's created_at) since ev_picks itself only
    stores the pick-time price, not the line."""
    cur.execute("""
        SELECT line FROM odds_snapshot
        WHERE game_pk = %s AND market = %s AND side = %s AND book = 'pinnacle'
          AND player_name = '' AND captured_at <= %s
        ORDER BY captured_at DESC LIMIT 1
    """, (game_pk, market, side, cutoff))
    row = cur.fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _pinnacle_close_price(cur, game_pk: int, market: str, side: str, commence_time) -> dict:
    """The CLOSING Pinnacle price for (game_pk, market, side): the last
    odds_snapshot at/before commence_time. By grading time (the game has
    already been played), "last snapshot before commence_time" genuinely is
    the close -- same captured_at <= commence_time convention
    build_ev_board.load_latest_odds uses for "latest" pre-kickoff odds."""
    cur.execute("""
        SELECT price FROM odds_snapshot
        WHERE game_pk = %s AND market = %s AND side = %s AND book = 'pinnacle'
          AND player_name = '' AND captured_at <= %s
        ORDER BY captured_at DESC LIMIT 1
    """, (game_pk, market, side, commence_time))
    row = cur.fetchone()
    return {"price": int(row[0])} if row and row[0] is not None else {"price": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (grading reads/writes Supabase).")

    start = _window_start(args.days)
    graded_rows: list[dict] = []
    counts: dict[str, int] = {}

    with get_postgres() as conn, conn.cursor() as cur:
        for sport, provider in FINAL_PROVIDERS.items():
            pending = _pending_ev_picks(cur, sport, start)
            n = 0
            for row in pending:
                try:
                    final = provider.fetch_final(row["game_pk"])
                    if final is None:
                        continue  # not final yet -- skip until a later run

                    pick_line = None
                    if row["market"] in ("spread", "total"):
                        # grade_ev_pick expects the HOME-referenced spread line
                        # (cover math: actual_margin + line, negative = home
                        # favored). odds_snapshot stores each spread outcome with
                        # its OWN handicap, so the away row carries +3 where home
                        # carries -3 -- always pull the home line for spreads so
                        # away spread picks aren't graded against a sign-flipped
                        # number. Totals carry the same line on both sides.
                        line_side = "home" if row["market"] == "spread" else row["side"]
                        pick_line = _pinnacle_line_at(
                            cur, row["game_pk"], row["market"], line_side, row["created_at"])
                    pinnacle_close = _pinnacle_close_price(
                        cur, row["game_pk"], row["market"], row["side"], row["commence_time"])

                    pick = {
                        "sport": sport,
                        "game_pk": row["game_pk"],
                        "market": row["market"],
                        "side": row["side"],
                        "line": pick_line,
                        "pinnacle_price": row["pinnacle_price"],
                    }
                    graded_rows.append(grade_ev_pick(pick, final, pinnacle_close))
                    n += 1
                except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the batch
                    print(f"  {sport} {row['game_pk']} {row['market']}/{row['side']}: "
                          f"grading failed ({exc}); skipping")
                    continue
            counts[sport] = n
            print(f"{sport}: {len(pending)} pending, {n} final and graded")

    if graded_rows:
        written = upsert_ev_results(graded_rows)
        print(f"Upserted {written} ev_results rows.")
    for sport, n in counts.items():
        print(f"graded {n} {sport} ev picks")


if __name__ == "__main__":
    main()
