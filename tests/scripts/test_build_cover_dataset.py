"""Tests for the pure assembly seam in build_cover_dataset.py (`assemble_rows`).

main()'s nflverse/IO wiring is not unit-tested (see the module docstring) --
this is the primary deliverable and review gate for Task 4.

Label-sign note (controller ruling, overrides the Task 4 brief's formula):
`assets/nfl/schedules.parquet` uses the nflverse sign convention where
POSITIVE `spread_line` means the HOME team is favored (verified empirically:
corr(result, spread_line) = +0.43 on the real asset; the brief's
`result + spread_line > 0` formula yields an implausible ~58.9% home-cover
rate, while `result - spread_line > 0` yields the plausible ~47.7% seen in
the real data). So:
    home_cover = int(result - spread_line > 0)
and a cover PUSH (result == spread_line) or total PUSH
(home_score + away_score == total_line) drops the game from the training
frame entirely -- a push is neither a cover nor a non-cover, and
`assemble_rows` returns one row per completed, NON-PUSH game.
"""
import importlib.util
import pathlib

import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cover_dataset.py"
_spec = importlib.util.spec_from_file_location("build_cover_dataset", _p)
bcd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcd)


def _schedule(rows):
    return pd.DataFrame(rows)


def _game_epa():
    # Week-1 (season 2023) pbp-derived aggregates for the six teams involved
    # in the week-2 synthetic schedule below, so adjusted_efficiency has
    # real prior-week data to adjust from (leakage-free: only week < 2).
    return {
        (2023, 1, "KC"): {"off": 0.30, "def": -0.10, "n": 10, "opp": "DET"},
        (2023, 1, "DET"): {"off": -0.10, "def": 0.30, "n": 10, "opp": "KC"},
        (2023, 1, "BUF"): {"off": 0.10, "def": 0.05, "n": 10, "opp": "NYJ"},
        (2023, 1, "NYJ"): {"off": 0.05, "def": 0.10, "n": 10, "opp": "BUF"},
        (2023, 1, "SF"): {"off": 0.20, "def": -0.05, "n": 10, "opp": "LA"},
        (2023, 1, "LA"): {"off": -0.05, "def": 0.20, "n": 10, "opp": "SF"},
    }


def _stub_ratings_fn():
    table = {
        ("KC", "DET"): (5.0, 44.0),
        ("BUF", "NYJ"): (-3.0, 41.0),
        ("SF", "LA"): (-1.0, 44.0),
    }

    def ratings_fn(home, away, season, week):
        return table[(home, away)]

    return ratings_fn


def _synthetic_schedule():
    return _schedule([
        # KC covers (-)3 fav... spread_line=3.0 => KC favored by 3; result=7
        # home_cover = int(7 - 3 > 0) = 1; total 47 > 45 -> over = 1
        {"season": 2023, "week": 2, "home_team": "KC", "away_team": "DET",
         "home_score": 27, "away_score": 20, "result": 7.0,
         "spread_line": 3.0, "total_line": 45.0},
        # BUF: result == spread_line (-3.0 == -3.0) -> COVER PUSH, dropped.
        {"season": 2023, "week": 2, "home_team": "BUF", "away_team": "NYJ",
         "home_score": 17, "away_score": 20, "result": -3.0,
         "spread_line": -3.0, "total_line": 41.0},
        # SF: home_cover = int(0 - 2 > 0) = 0; total 48 == total_line 48 ->
        # TOTAL PUSH, dropped (whole row, per Task 4 controller ruling).
        {"season": 2023, "week": 2, "home_team": "SF", "away_team": "LA",
         "home_score": 24, "away_score": 24, "result": 0.0,
         "spread_line": 2.0, "total_line": 48.0},
    ])


def test_build_cover_dataset_labels_features_and_ratings_prob():
    schedule_df = _synthetic_schedule()
    game_epa = _game_epa()
    ratings_fn = _stub_ratings_fn()

    rows = bcd.assemble_rows(schedule_df, game_epa, ratings_fn)

    # Both push games (cover push BUF/NYJ, total push SF/LA) are dropped --
    # only the clean KC/DET game survives.
    assert len(rows) == 1
    row = rows[0]

    result, spread_line = 7.0, 3.0
    home_score, away_score, total_line = 27, 20, 45.0
    assert row["home_cover"] == int(result - spread_line > 0)
    assert row["home_cover"] == 1
    assert row["over"] == int(home_score + away_score > total_line)
    assert row["over"] == 1
    assert isinstance(row["home_cover"], int)
    assert isinstance(row["over"], int)

    # Efficiency features present (from efficiency_features()).
    for key in ("eff_diff", "home_off_adj", "home_def_adj",
                "away_off_adj", "away_def_adj", "total_off"):
        assert key in row

    # Context.
    assert row["week"] == 2
    assert row["home_field"] == 1
    assert row["rest_diff"] == 0
    assert row["spread_line"] == 3.0
    assert row["total_line"] == 45.0

    # Ratings base probs (via gbm_prob on the stub ratings_fn's margin/total).
    assert 0.0 < row["ratings_cover_p"] < 1.0
    assert 0.0 < row["ratings_over_p"] < 1.0


def test_build_cover_dataset_excludes_push_rows():
    schedule_df = _synthetic_schedule()
    game_epa = _game_epa()
    ratings_fn = _stub_ratings_fn()

    rows = bcd.assemble_rows(schedule_df, game_epa, ratings_fn)

    kept = {(r["home_team"], r["away_team"]) for r in rows}
    assert ("BUF", "NYJ") not in kept   # cover push (result == spread_line)
    assert ("SF", "LA") not in kept     # total push (total == total_line)
    assert ("KC", "DET") in kept        # clean game retained
