"""Grade game_predictions against actual final scores -> prediction_accuracy.

Sibling of grade_results.py, but simpler: no odds, no ROI, no CLV. For each
recently-finished NFL/CFB game with a game_predictions row not yet graded,
fetch the final score (ESPN) and record whether the model picked the right
winner, plus margin/total error. This is the model's accuracy track record.

Runs on a rolling window (like grade_results); idempotent -- re-running just
re-upserts the same rows via (sport, game_pk).

Usage:
    uv run python scripts/grade_predictions.py [--days 7]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.cfb import espn as cfb_espn
from sportsmodel.db import get_postgres, upsert_prediction_accuracy
from sportsmodel.nfl import espn as nfl_espn

# Results-provider seam: sport key -> module exposing fetch_final(game_pk) -> dict|None.
# Both NFL and CFB use ESPN event ids as game_pk, so fetch_final takes the same
# argument shape for either sport.
FINAL_PROVIDERS = {"nfl": nfl_espn, "cfb": cfb_espn}


def _window_start(days: int, today: date | None = None) -> str:
    d = today or date.today()
    return (d - timedelta(days=days)).isoformat()


def _accuracy_row(prediction: dict, final: dict) -> dict:
    """Pure accuracy computation: one game_predictions row + its final score ->
    a prediction_accuracy row (dict, columns matching db.upsert_prediction_accuracy).

    No network, no DB -- easy to unit test in isolation.
    """
    home_wp = prediction.get("home_win_prob")
    home_name = prediction.get("home_team_name")
    away_name = prediction.get("away_team_name")
    home_score = final["home_score"]
    away_score = final["away_score"]

    predicted_winner = home_name if (home_wp is not None and home_wp >= 0.5) else away_name
    if home_score > away_score:
        actual_winner = home_name
    elif away_score > home_score:
        actual_winner = away_name
    else:
        actual_winner = None  # tie -- neither side "won"
    winner_correct = actual_winner is not None and predicted_winner == actual_winner

    pred_home = prediction.get("pred_home_score")
    pred_away = prediction.get("pred_away_score")
    has_pred_scores = pred_home is not None and pred_away is not None
    pred_margin = (pred_home - pred_away) if has_pred_scores else None
    actual_margin = home_score - away_score
    margin_error = abs(pred_margin - actual_margin) if pred_margin is not None else None
    pred_total = (pred_home + pred_away) if has_pred_scores else None
    actual_total = home_score + away_score
    total_error = abs(pred_total - actual_total) if pred_total is not None else None

    return {
        "sport": prediction.get("sport"),
        "game_pk": prediction.get("game_pk"),
        "game_date": prediction.get("game_date"),
        "home_team_name": home_name,
        "away_team_name": away_name,
        "win_prob": home_wp,
        "predicted_winner": predicted_winner,
        "actual_winner": actual_winner,
        "winner_correct": winner_correct,
        "pred_margin": pred_margin,
        "actual_margin": actual_margin,
        "margin_error": margin_error,
        "pred_total": pred_total,
        "actual_total": actual_total,
        "total_error": total_error,
    }


def _pending_predictions(cur, sport: str, start: str) -> list[dict]:
    """Ungraded game_predictions rows for `sport` with game_date >= `start`.

    DISTINCT ON (game_pk) keeps only the latest-generated model_version per
    game (a game re-predicted under a newer version shouldn't be graded once
    per version). NOT EXISTS against prediction_accuracy skips games already
    graded -- makes the whole script idempotent.
    """
    cur.execute("""
        SELECT DISTINCT ON (gp.game_pk)
               gp.game_pk, gp.game_date, gp.home_team_name, gp.away_team_name,
               gp.home_win_prob, gp.pred_home_score, gp.pred_away_score
        FROM game_predictions gp
        WHERE gp.sport = %s AND gp.game_date >= %s
          AND NOT EXISTS (
              SELECT 1 FROM prediction_accuracy pa
              WHERE pa.sport = %s AND pa.game_pk = gp.game_pk
          )
        ORDER BY gp.game_pk, gp.generated_at DESC
    """, [sport, start, sport])
    cols = ["game_pk", "game_date", "home_team_name", "away_team_name",
            "home_win_prob", "pred_home_score", "pred_away_score"]
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
            pending = _pending_predictions(cur, sport, start)
            n = 0
            for pred in pending:
                final = provider.fetch_final(pred["game_pk"])
                if final is None:
                    continue  # not final yet -- skip until a later run
                graded_rows.append(_accuracy_row(pred, final))
                n += 1
            counts[sport] = n
            print(f"{sport}: {len(pending)} pending, {n} final and graded")

    if graded_rows:
        written = upsert_prediction_accuracy(graded_rows)
        print(f"Upserted {written} prediction_accuracy rows.")
    for sport, n in counts.items():
        print(f"graded {n} {sport} predictions")


if __name__ == "__main__":
    main()
