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
import math
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


def test_build_cover_dataset_falls_back_when_team_absent_from_game_epa():
    # DET has NO entries anywhere in game_epa (e.g. a data gap) -- so
    # adjusted_efficiency's returned dict won't even have a "DET" key.
    # assemble_rows must fall back to {"off_adj": 0.0, "def_adj": 0.0}
    # rather than raising KeyError out of efficiency_features.
    schedule_df = _schedule([
        {"season": 2023, "week": 2, "home_team": "KC", "away_team": "DET",
         "home_score": 27, "away_score": 20, "result": 7.0,
         "spread_line": 3.0, "total_line": 45.0},
    ])
    game_epa = {
        (2023, 1, "KC"): {"off": 0.30, "def": -0.10, "n": 10, "opp": "DET"},
        # (2023, 1, "DET") intentionally omitted.
    }
    ratings_fn = _stub_ratings_fn()

    rows = bcd.assemble_rows(schedule_df, game_epa, ratings_fn)

    assert len(rows) == 1
    row = rows[0]
    # Away team (DET) fell back to the 0.0 stub rather than blowing up.
    assert row["away_off_adj"] == 0.0
    assert row["away_def_adj"] == 0.0
    # Every efficiency feature is a finite float, not NaN/inf.
    for key in ("eff_diff", "home_off_adj", "home_def_adj",
                "away_off_adj", "away_def_adj", "total_off"):
        assert math.isfinite(row[key])


def _point_margin_dist(margin: float, half_range: int = 25) -> dict:
    """A margin_dist ({"kind": "margin", ...}, see sim.engine.margin_pmf) with
    ALL its mass on one integer margin value -- lets a test assert an exact
    sim_cover_p (1.0 or 0.0) instead of reasoning about a real distribution's
    shape."""
    pmf = [0.0] * (2 * half_range + 1)
    pmf[int(margin) + half_range] = 1.0
    return {"kind": "margin", "offset": half_range, "pmf": pmf}


def _point_total_dist(total: float, max_total: int = 90) -> dict:
    """A total_dist ({"kind": "pmf", ...}, see sim.engine.total_pmf) with all
    its mass on one integer total."""
    pmf = [0.0] * (max_total + 1)
    pmf[int(total)] = 1.0
    return {"kind": "pmf", "pmf": pmf}


def test_sim_probs_from_dists_matches_home_cover_label_sign_convention():
    # assemble_rows' home_cover label is `int(result - spread_line > 0)` --
    # nflverse sign, home covers iff the ACTUAL margin exceeds spread_line.
    # sim_probs_from_dists must apply the same convention to the SIM's
    # margin_dist (via prob_cover's ESPN-signed home_line = -spread_line,
    # exactly like ratings_cover_p's `-float(spread_line)` conversion) so a
    # sim that's certain the home team wins by more than the spread reports
    # sim_cover_p == 1.0, not 0.0.
    spread_line = 3.0  # nflverse sign: home favored by 3

    covers = _point_margin_dist(7.0)  # home wins by 7 > 3 -> covers
    sim_cover_p, _ = bcd.sim_probs_from_dists(covers, _point_total_dist(44.0), spread_line, 44.0)
    assert sim_cover_p == 1.0

    doesnt_cover = _point_margin_dist(1.0)  # home wins by 1 < 3 -> doesn't cover
    sim_cover_p2, _ = bcd.sim_probs_from_dists(doesnt_cover, _point_total_dist(44.0), spread_line, 44.0)
    assert sim_cover_p2 == 0.0


def test_sim_probs_from_dists_over_under():
    high = _point_total_dist(50.0)
    _, sim_over_p = bcd.sim_probs_from_dists(_point_margin_dist(0.0), high, 0.0, 44.0)
    assert sim_over_p == 1.0

    low = _point_total_dist(30.0)
    _, sim_over_p2 = bcd.sim_probs_from_dists(_point_margin_dist(0.0), low, 0.0, 44.0)
    assert sim_over_p2 == 0.0


def test_sim_probs_from_dists_nan_when_dist_missing_never_fabricates():
    # Where a game's sim can't be reconstructed, the seam must return NaN --
    # never a fabricated/default probability (controller ruling).
    sim_cover_p, sim_over_p = bcd.sim_probs_from_dists(None, None, 3.0, 44.0)
    assert math.isnan(sim_cover_p)
    assert math.isnan(sim_over_p)

    sim_cover_p2, sim_over_p2 = bcd.sim_probs_from_dists({}, {}, 3.0, 44.0)
    assert math.isnan(sim_cover_p2)
    assert math.isnan(sim_over_p2)

    # One dist present, the other missing -> still NaN for both (a lone
    # margin_dist with no total_dist can't safely be treated as "sim data
    # available" -- assemble_rows' lookup always provides both together, but
    # the seam itself shouldn't assume that).
    sim_cover_p3, sim_over_p3 = bcd.sim_probs_from_dists(_point_margin_dist(0.0), None, 3.0, 44.0)
    assert math.isnan(sim_cover_p3)
    assert math.isnan(sim_over_p3)


def test_assemble_rows_adds_sim_probs_from_sim_lookup():
    schedule_df = _schedule([_synthetic_schedule().iloc[0].to_dict()])  # KC/DET
    game_epa = _game_epa()
    ratings_fn = _stub_ratings_fn()
    sim_lookup = {
        (2023, 2, "KC", "DET"): (_point_margin_dist(7.0), _point_total_dist(47.0)),
    }

    rows = bcd.assemble_rows(schedule_df, game_epa, ratings_fn, sim_lookup=sim_lookup)

    assert len(rows) == 1
    assert rows[0]["sim_cover_p"] == 1.0  # margin 7 > spread_line 3.0
    assert rows[0]["sim_over_p"] == 1.0   # total 47 > total_line 45.0


def test_assemble_rows_sim_probs_nan_without_sim_lookup():
    # No sim_lookup passed at all (the default) -> NaN, not 0/fabricated.
    schedule_df = _schedule([_synthetic_schedule().iloc[0].to_dict()])
    game_epa = _game_epa()
    ratings_fn = _stub_ratings_fn()

    rows = bcd.assemble_rows(schedule_df, game_epa, ratings_fn)

    assert math.isnan(rows[0]["sim_cover_p"])
    assert math.isnan(rows[0]["sim_over_p"])


def test_assemble_rows_sim_probs_nan_when_game_absent_from_lookup():
    # sim_lookup provided but doesn't cover THIS game (e.g. bounded 2024-only
    # backfill and this row is an earlier season) -> NaN for that row only.
    schedule_df = _schedule([_synthetic_schedule().iloc[0].to_dict()])
    game_epa = _game_epa()
    ratings_fn = _stub_ratings_fn()
    sim_lookup = {(2099, 1, "XX", "YY"): (_point_margin_dist(0.0), _point_total_dist(44.0))}

    rows = bcd.assemble_rows(schedule_df, game_epa, ratings_fn, sim_lookup=sim_lookup)

    assert math.isnan(rows[0]["sim_cover_p"])
    assert math.isnan(rows[0]["sim_over_p"])


def test_build_cover_dataset_uses_passed_in_sigmas_for_ratings_prob():
    # ratings_cover_p/ratings_over_p must actually respond to the
    # sigma_margin/sigma_total args (main() passes the FITTED
    # gameline.json sigmas here rather than the illustrative defaults) --
    # a tighter sigma should push a probability further from 0.5 for the
    # same off-center point prediction.
    schedule_df = _schedule([_synthetic_schedule().iloc[0].to_dict()])
    game_epa = _game_epa()
    ratings_fn = _stub_ratings_fn()  # KC/DET -> margin=5.0, total=44.0

    tight = bcd.assemble_rows(schedule_df, game_epa, ratings_fn,
                              sigma_margin=1.0, sigma_total=1.0)[0]
    wide = bcd.assemble_rows(schedule_df, game_epa, ratings_fn,
                             sigma_margin=50.0, sigma_total=50.0)[0]

    assert abs(tight["ratings_cover_p"] - 0.5) > abs(wide["ratings_cover_p"] - 0.5)
    assert abs(tight["ratings_over_p"] - 0.5) > abs(wide["ratings_over_p"] - 0.5)
