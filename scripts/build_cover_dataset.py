"""Training-frame builder for the NFL cover/total stacked ensemble (Task 4 of
docs/superpowers/specs/2026-09-21-nfl-cover-ensemble-design.md).

One row per completed, non-push game with the cover/over labels, the
opponent-adjusted efficiency features (Task 2: `sportsmodel.nfl.efficiency`),
minimal context, the closing line, and a "ratings" base cover/over
probability derived from the point-in-time gameline model (Elo+SRS+points,
shrunk toward that game's own closing line -- see `_build_ratings_lookup`,
which mirrors `scripts/backtest_nfl_gameline.py`'s walk-forward exactly).

Label sign (controller ruling -- overrides an earlier ESPN-signed draft of
this formula): `assets/nfl/schedules.parquet` uses the nflverse convention
where POSITIVE `spread_line` means the HOME team is favored (verified
empirically: corr(result, spread_line) = +0.43 on the real asset). So:

    home_cover = int(result - spread_line > 0)
    over       = int(home_score + away_score > total_line)

and a PUSH on either line (`result == spread_line` or
`home_score + away_score == total_line`) drops the game from the training
frame entirely -- a push is neither a cover nor a non-cover, so it cannot
supply a valid int label.

`assemble_rows` is pure (frames/dict/callable in, list[dict] out) and is the
primary deliverable + review gate; unit tested directly in
tests/scripts/test_build_cover_dataset.py. `main()`'s nflverse pbp fetch +
point-in-time rating walk-forward + parquet write is IO and not unit tested.

Usage:
    uv run python scripts/build_cover_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel.nfl.efficiency import (
    adjusted_efficiency,
    efficiency_features,
    team_game_epa,
)
from sportsmodel.serving.ensemble import gbm_prob

# Illustrative fallback dispersion for the "ratings base prob" conversion,
# used only if a caller doesn't pass the real fitted sigmas. `main()` always
# passes the FITTED `gl_cfg.sigma_margin`/`gl_cfg.sigma_total` (read from the
# committed, walk-forward-fit `assets/nfl/gameline.json` -- see
# `_build_ratings_lookup`) through to `assemble_rows` explicitly, so these
# constants only matter for a caller (e.g. a quick script or REPL check)
# that doesn't have a fitted GameLineConfig handy. They match
# `sportsmodel.nfl.gameline.GameLineConfig`'s own illustrative defaults.
_RATINGS_SIGMA_MARGIN = 13.2
_RATINGS_SIGMA_TOTAL = 10.0

_SEASONS = list(range(2015, 2026))


def assemble_rows(schedule_df: pd.DataFrame, game_epa: dict, ratings_fn,
                   sigma_margin: float = _RATINGS_SIGMA_MARGIN,
                   sigma_total: float = _RATINGS_SIGMA_TOTAL) -> list[dict]:
    """Pure: build one training row per completed, non-push game.

    Args:
        schedule_df: rows with (at least) `season`, `week`, `home_team`,
            `away_team`, `home_score`, `away_score`, `result` (home - away),
            `spread_line` (nflverse-signed: positive = home favored),
            `total_line`.
        game_epa: `sportsmodel.nfl.efficiency.team_game_epa` output --
            per-(season, week, team) offensive/defensive EPA.
        ratings_fn: `(home, away, season, week) -> (margin, total)`, a
            POINT-IN-TIME model prediction (must not see this game's own
            result) used to derive the ratings base cover/over prob via
            `gbm_prob`.
        sigma_margin, sigma_total: dispersion used to convert `ratings_fn`'s
            point margin/total into `ratings_cover_p`/`ratings_over_p` via
            `gbm_prob`. Defaults are the illustrative `GameLineConfig`
            values; `main()` passes the real FITTED sigmas from
            `assets/nfl/gameline.json` instead so the ratings base prob
            reflects the model's actual calibrated dispersion, not a
            placeholder.

    Returns:
        list of dicts, one per completed non-push game:
          - `home_cover`, `over` (int labels)
          - efficiency features from `efficiency_features` (`eff_diff`,
            `home_off_adj`, `home_def_adj`, `away_off_adj`, `away_def_adj`,
            `total_off`)
          - context: `season`, `week`, `home_team`, `away_team`,
            `home_field` (always 1 -- `home_team` is always the host in this
            schema), `rest_diff` (always 0 -- rest-day data isn't wired in
            yet)
          - the line: `spread_line`, `total_line`
          - ratings base probs: `ratings_cover_p`, `ratings_over_p`

    A game is skipped (not just label-dropped) when it isn't finished yet
    (`home_score`/`away_score`/`result` missing) or has no closing line
    (`spread_line`/`total_line` missing) -- neither label could be computed.
    A game with a valid line that is a PUSH on either market is also
    skipped entirely, per the controller ruling above.

    Leakage: `adjusted_efficiency` only uses games strictly before `week`
    within `season` (Task 2's own leak-free guarantee); `ratings_fn` must
    independently be point-in-time -- `main()`'s wiring guarantees this by
    construction (a walk-forward that only ever sees prior games).
    """
    adj_cache: dict[tuple[int, int], dict] = {}
    rows: list[dict] = []

    for _, g in schedule_df.iterrows():
        home_score, away_score = g.get("home_score"), g.get("away_score")
        result, spread_line, total_line = g.get("result"), g.get("spread_line"), g.get("total_line")
        if any(pd.isna(v) for v in (home_score, away_score, result, spread_line, total_line)):
            continue  # unfinished game or no closing line -> unlabelable

        season, week = int(g["season"]), int(g["week"])
        home, away = g["home_team"], g["away_team"]
        total_points = home_score + away_score

        if result == spread_line or total_points == total_line:
            continue  # push on either market -> drop the whole row

        home_cover = int(result - spread_line > 0)
        over = int(total_points > total_line)

        cache_key = (season, week)
        if cache_key not in adj_cache:
            adj_cache[cache_key] = adjusted_efficiency(game_epa, season, week)
        adj = adj_cache[cache_key]
        # Teams with no prior games this season (e.g. week 1, or a team
        # entirely absent from game_epa) fall back to the same 0.0
        # league-average adjusted_efficiency itself uses internally when it
        # has nothing to go on -- efficiency_features requires both keys.
        for team in (home, away):
            if team not in adj:
                adj[team] = {"off_adj": 0.0, "def_adj": 0.0}
        feats = efficiency_features(adj, home, away)

        margin, total = ratings_fn(home, away, season, week)
        # gbm_prob's `line` for dist_kind="margin" is ESPN-signed (home
        # covers iff margin + line > 0, i.e. margin > -line); spread_line
        # here is nflverse-signed (positive = home favored, home covers iff
        # margin > spread_line) -- negate to convert conventions.
        ratings_cover_p = gbm_prob(margin, sigma_margin, "margin", -float(spread_line))
        ratings_over_p = gbm_prob(total, sigma_total, "total", float(total_line))

        rows.append({
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
            "home_cover": home_cover,
            "over": over,
            **feats,
            "home_field": 1,
            "rest_diff": 0,
            "spread_line": float(spread_line),
            "total_line": float(total_line),
            "ratings_cover_p": ratings_cover_p,
            "ratings_over_p": ratings_over_p,
        })

    return rows


def _clean_market(value):
    """nflverse spread_line/total_line -> float, or None if missing/NaN.

    Identical to backtest_nfl_gameline._clean_market -- build_gameline/
    shrink() only recognize Python `None` as "no market line"; NaN would
    poison the (1-w)*model + w*market blend.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _build_ratings_lookup(schedule_df: pd.DataFrame) -> dict[tuple[int, int, str, str], tuple[float, float]]:
    """Point-in-time gameline (Elo+SRS+points model, shrunk toward that
    game's own closing market line) margin/total, keyed by
    `(season, week, home_team, away_team)`.

    This is NOT a new rating -- it is the exact same walk-forward
    `scripts/backtest_nfl_gameline.py`'s `_raw_model_predictions` +
    `_apply_gl` run (same Elo/SRS/points/build_gameline call sequence, same
    fitted configs from `assets/nfl/{rating,gameline}.json`), reimplemented
    locally only so team identity survives into a lookup keyed the way
    `assemble_rows`' `ratings_fn` needs (the original functions return a
    positional list with no team columns). Leak-free: each game's
    model_margin/model_total is computed from Elo/SRS/points state built up
    only from games strictly earlier in the walk (identical invariant to
    `_raw_model_predictions`).
    """
    # SYNC NOTE: this intentionally mirrors backtest_nfl_gameline.py's
    # _raw_model_predictions + _apply_gl step-for-step (same Elo/SRS/points/
    # build_gameline call sequence) -- if that walk-forward's logic changes,
    # this copy must be updated to match (de-duplication deferred by
    # controller ruling; see the module docstring above for why a local
    # reimplementation exists instead of importing the original).
    from sportsmodel.nfl import config as nfl_config
    from sportsmodel.nfl.elo import run_elo
    from sportsmodel.nfl.gameline import build_gameline
    from sportsmodel.nfl.points import compute_points_ratings, expected_total
    from sportsmodel.nfl.ratings import expected_margin
    from sportsmodel.nfl.srs import compute_srs

    elo_cfg, blend_cfg = nfl_config.load_rating()
    gl_cfg = nfl_config.load_gameline()

    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    games = run_elo(df, elo_cfg).games

    lookup: dict[tuple[int, int, str, str], tuple[float, float]] = {}
    for season, sdf in games.groupby("season"):
        srs_hist = sdf.iloc[0:0]
        pts_hist = sdf.iloc[0:0]
        counts: dict[str, int] = {}
        srs_cache: dict = {}
        pts_cache: dict = {}
        lg_cache = 0.0
        for _, g in sdf.iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            h, a = g["home_team"], g["away_team"]
            gh, ga = counts.get(h, 0), counts.get(a, 0)
            model_margin = expected_margin(g["elo_home"], g["elo_away"],
                                           srs_cache.get(h), srs_cache.get(a),
                                           gh, ga, elo_cfg, blend_cfg)
            model_total = (expected_total(pts_cache, lg_cache, h, a)
                           if pts_cache else 2 * lg_cache) if lg_cache else 44.0
            market = {"spread_line": _clean_market(g.get("spread_line")),
                      "total_line": _clean_market(g.get("total_line"))}
            week = int(g["week"])
            row = build_gameline(model_margin, model_total, market, week, gl_cfg)
            lookup[(int(season), week, h, a)] = (row["pred_margin"], row["pred_total"])

            counts[h] = gh + 1
            counts[a] = ga + 1
            srs_hist = pd.concat([srs_hist, pd.DataFrame([g])], ignore_index=True)
            pts_hist = pd.concat([pts_hist, pd.DataFrame([g])], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
            pts_cache, lg_cache = compute_points_ratings(pts_hist, k_points=4.0)
    return lookup


def main() -> None:
    import nfl_data_py as nfl_data_py_import

    from sportsmodel.nfl import config as nfl_config
    from sportsmodel.nfl.nflverse import import_by_season

    assets = Path(__file__).resolve().parents[1] / "assets" / "nfl"
    sched = pd.read_parquet(assets / "schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched.columns else sched
    reg = reg[reg["season"].isin(_SEASONS)].copy()

    pbp = import_by_season(
        lambda yrs: nfl_data_py_import.import_pbp_data(yrs)[
            ["season", "week", "posteam", "defteam", "epa"]
        ],
        _SEASONS,
        "pbp",
    )
    game_epa = team_game_epa(pbp)

    ratings_lookup = _build_ratings_lookup(reg)

    def ratings_fn(home: str, away: str, season: int, week: int) -> tuple[float, float]:
        return ratings_lookup.get((season, week, home, away), (0.0, 44.0))

    # Use the real FITTED sigmas (assets/nfl/gameline.json), not the
    # illustrative assemble_rows() defaults -- otherwise ratings_cover_p/
    # ratings_over_p would be silently miscalibrated vs. the model's actual
    # walk-forward fit.
    gl_cfg = nfl_config.load_gameline()
    rows = assemble_rows(reg, game_epa, ratings_fn,
                         sigma_margin=gl_cfg.sigma_margin, sigma_total=gl_cfg.sigma_total)
    out = pd.DataFrame(rows)
    out_path = assets / "cover_dataset.parquet"
    out.to_parquet(out_path, index=False)

    print(f"wrote {out_path}: {len(out)} rows")
    if len(out):
        print("home_cover rate:", round(out["home_cover"].mean(), 4))
        print("over rate:", round(out["over"].mean(), 4))
        print("seasons:", sorted(out["season"].unique().tolist()))
        print("columns:", out.columns.tolist())


if __name__ == "__main__":
    main()
