"""Grade `desk_picks` FORWARD against the final score and the closing line,
tracking closing-line value (CLV) -> `desk_pick_results`.

Sibling of grade_predictions.py, but this is the decision desk's real bet
record: not "did the model call the right winner" but "did the picked SIDE
win/cover/hit, and did the desk's number beat the market's closing number."
No CFBD, no lines.parquet -- both the pick-time line (desk_picks.spread_line/
total_line) and the closing line (cfb.espn.fetch_final's market_spread/
market_total, from ESPN's pickcenter) are the SAME convention already:
standard sportsbook, home-team spread, negative = home favored. CLV is a
straight subtraction between those two numbers -- no conversion.

Runs on a rolling window (like grade_predictions); idempotent -- re-running
just re-upserts the same rows via (sport, game_pk) (db.upsert_desk_pick_results).

Usage:
    uv run python scripts/grade_desk_picks.py [--days 7]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.cfb import espn as cfb_espn
from sportsmodel.db import get_postgres, upsert_desk_pick_results
from sportsmodel.nfl import espn as nfl_espn

# Results-provider seam: sport key -> module exposing fetch_final(game_pk) ->
# dict|None. Both NFL and CFB use ESPN event ids as game_pk (mirrors
# grade_predictions.FINAL_PROVIDERS).
FINAL_PROVIDERS = {"nfl": nfl_espn, "cfb": cfb_espn}


def _window_start(days: int, today: date | None = None) -> str:
    d = today or date.today()
    return (d - timedelta(days=days)).isoformat()


def grade_pick(pick: dict, final: dict) -> dict:
    """Pure grading: one desk_picks row + its final/closing-line dict ->
    a desk_pick_results row (dict, columns matching db.upsert_desk_pick_results).

    No network, no DB -- easy to unit test in isolation.

    `final` is the dict returned by cfb.espn.fetch_final / nfl.espn.fetch_final:
    {"home_score", "away_score", "market_spread", "market_total"}, where
    market_spread/market_total are the ESPN pickcenter CLOSING line -- SAME
    sportsbook convention as `pick["spread_line"]`/`pick["total_line"]` (the
    PICK-TIME line the desk took): home team's spread, negative = home
    favored. No convention conversion happens anywhere in this function.

    ml_correct: did pick["ml_pick"] (home/away) match the actual winner?
    None on a tie (no winner to have picked correctly).

    spread_cover: graded vs the CLOSING line only (not the pick-time line --
    that's what makes this a "forward" grade of the pick, not a re-grade of
    the market at pick time). Home covers iff
    actual_margin + closing_home_spread > 0 (cover_margin). A push
    (cover_margin == 0) -> None, matching backtest_cfb_priors.ats_result's
    push handling. correct iff the picked side is the covering side. None if
    spread_side/spread_line (no pick made on this market) or market_spread
    (no closing line) is missing.

    total_result: graded vs the CLOSING total only. Over wins iff
    actual_total > closing_total; a push (equal) -> None. correct iff
    total_side matches the winning side. None if total_side/total_line or
    market_total is missing.

    clv_spread / clv_total: signed closing-line value FROM THE SIDE THE DESK
    TOOK -- purely a comparison of two lines, independent of the game's
    outcome (so still computed on a push, unlike spread_cover/total_result).
    Positive means the desk's pick-time number was better than what the
    line closed at, for the side taken.

      clv_spread: both spread_line and market_spread are HOME-team numbers,
      negative = home favored. For a HOME pick, a *higher* pick-time number
      relative to the close means the desk laid fewer points home (or got
      more points if home was an underdog) -- a better number -- so
      clv_spread = spread_line - market_spread. Example: took home -3,
      closed -5 -> -3 - (-5) = +2 (positive: the number improved from the
      desk's perspective). For an AWAY pick the same home-line delta reads
      backwards from the away side's perspective (a home line that fell
      further, e.g. -3 -> -5, moved additional points AWAY's way, i.e. worse
      for a bettor who already locked in away's smaller number) so it is
      inverted: clv_spread = -(spread_line - market_spread) =
      market_spread - spread_line.

      clv_total: for an OVER pick, a *lower* pick-time total than the close
      is better (less scoring needed to clear it), so
      clv_total = market_total - total_line. Example: took Over 47, closed
      50 -> 50 - 47 = +3 (positive). For an UNDER pick it's inverted (a
      *higher* pick-time total is better for Under), so
      clv_total = -(market_total - total_line) = total_line - market_total.

    Missing pick-time or closing line for a market -> that market's
    spread_cover/clv_spread (or total_result/clv_total) are both None.
    """
    home_score = final["home_score"]
    away_score = final["away_score"]
    actual_margin = home_score - away_score
    actual_total = home_score + away_score

    # ---- ml_correct ----
    if home_score > away_score:
        actual_winner = "home"
    elif away_score > home_score:
        actual_winner = "away"
    else:
        actual_winner = None  # tie
    ml_correct = None if actual_winner is None else (pick.get("ml_pick") == actual_winner)

    # ---- spread: cover + CLV ----
    spread_side = pick.get("spread_side")
    spread_line = pick.get("spread_line")
    market_spread = final.get("market_spread")
    have_spread = spread_side is not None and spread_line is not None and market_spread is not None

    if not have_spread:
        spread_cover = None
        clv_spread = None
    else:
        cover_margin = actual_margin + market_spread
        if cover_margin == 0:
            spread_cover = None  # push
        else:
            covering_side = "home" if cover_margin > 0 else "away"
            spread_cover = (spread_side == covering_side)
        raw_clv = spread_line - market_spread
        clv_spread = raw_clv if spread_side == "home" else -raw_clv

    # ---- total: result + CLV ----
    total_side = pick.get("total_side")
    total_line = pick.get("total_line")
    market_total = final.get("market_total")
    have_total = total_side is not None and total_line is not None and market_total is not None

    if not have_total:
        total_result = None
        clv_total = None
    else:
        if actual_total == market_total:
            total_result = None  # push
        else:
            winning_side = "over" if actual_total > market_total else "under"
            total_result = (total_side == winning_side)
        raw_clv_total = market_total - total_line
        clv_total = raw_clv_total if total_side == "over" else -raw_clv_total

    return {
        "sport": pick.get("sport"),
        "game_pk": pick.get("game_pk"),
        "ml_correct": ml_correct,
        "spread_cover": spread_cover,
        "total_result": total_result,
        "clv_spread": clv_spread,
        "clv_total": clv_total,
    }


def _pending_desk_picks(cur, sport: str, start: str) -> list[dict]:
    """desk_picks rows for `sport` with game_date >= `start` to grade.

    DISTINCT ON (game_pk) keeps only the most-recently-created model_version
    per game (mirrors grade_predictions._pending_predictions). A NOT EXISTS
    against desk_pick_results skips games already graded, making the script
    idempotent and cheap to run on a schedule.
    """
    cur.execute("""
        SELECT DISTINCT ON (dp.game_pk)
               dp.game_pk, dp.game_date, dp.ml_pick,
               dp.spread_side, dp.spread_line, dp.total_side, dp.total_line
        FROM desk_picks dp
        WHERE dp.sport = %(sport)s AND dp.game_date >= %(start)s
          AND NOT EXISTS (
              SELECT 1 FROM desk_pick_results dr
              WHERE dr.sport = %(sport)s AND dr.game_pk = dp.game_pk
          )
        ORDER BY dp.game_pk, dp.created_at DESC
    """, {"sport": sport, "start": start})
    cols = ["game_pk", "game_date", "ml_pick", "spread_side", "spread_line",
            "total_side", "total_line"]
    return [dict(zip(cols, row), sport=sport) for row in cur.fetchall()]


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
            pending = _pending_desk_picks(cur, sport, start)
            n = 0
            for pick in pending:
                try:
                    final = provider.fetch_final(pick["game_pk"])
                except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the batch
                    # fetch_final already retries transient blips; if it still
                    # fails, skip just this game and let a later run pick it up
                    # (idempotent), rather than failing every other game's grade.
                    print(f"  {sport} {pick['game_pk']}: fetch_final failed ({exc}); skipping")
                    continue
                if final is None:
                    continue  # not final yet -- skip until a later run
                graded_rows.append(grade_pick(pick, final))
                n += 1
            counts[sport] = n
            print(f"{sport}: {len(pending)} pending, {n} final and graded")

    if graded_rows:
        written = upsert_desk_pick_results(graded_rows)
        print(f"Upserted {written} desk_pick_results rows.")
    for sport, n in counts.items():
        print(f"graded {n} {sport} desk picks")


if __name__ == "__main__":
    main()
