"""Walk-forward, point-in-time backtest of the NFL drive-based Monte Carlo sim
(the ship gate for sub-project B).

For every completed regular-season game in the validation span, builds a
leakage-free `NflGameSpec` (rates/player shares computed strictly before that
game's season/week, per `sportsmodel.sim.nfl.rates`'s no-leakage contract),
simulates it, and compares the sim's outputs against what actually happened:

  GAME:   sim home_win_prob vs actual home win (Brier score), sim margin/total
          vs actual margin/total (MAE), and the margin<->total correlation in
          the sim's own predictions vs. the same correlation in the actual
          results (a sanity check that the sim reproduces a realistic
          margin/total relationship, not just correct marginals).
  PLAYER: for pass_yds/rush_yds/rec_yds/receptions, the sim's per-player mean
          vs. that player's ACTUAL stat that week (MAE), plus a calibration
          check -- the share of actuals falling at/under the sim's p50 and
          p90 quantiles should be ~=0.50 and ~=0.90 if the sim's spread is
          well-calibrated. Each market's pairs are filtered to players who'd
          realistically have a prop line that week, under TWO independent
          gates: `is_propable` (ACTUAL usage that week -- realized,
          selection-biased, kept for reference/visibility into that bias) and
          `is_propable_projected` (the sim's own PROJECTED/pre-game usage --
          the deployment population, since books set lines off expectation,
          not off what happens to occur; this is the headline shippability
          metric). Without either gate the metric would be swamped by
          non-participants (e.g. a WR's pass_yds is 0-vs-0 every week and
          would otherwise dilute the pass_yds numbers).

PURE / IO split
----------------
`brier`, `mae`, `pearson_corr`, `quantile_from_pmf`, and `share_below` are
pure metric helpers -- no IO, no sim-specific types -- and are unit-tested in
tests/sim/nfl/test_backtest_sim_nfl.py.

`run_backtest`/`report`/`main` are the walk-forward harness: heavy IO
(nflverse fetch + thousands of simulate_game calls) and NOT unit-tested here,
guarded under `__main__` per this project's backtest-script convention (see
scripts/backtest_sim.py).

CONCERN -- actual-stat join: `_actual_player_stats` keys nflverse `weekly`
rows by `player_id` alone within a (season, week) slice. This matches
`active_usage`'s own keying (both come from the same nflverse `player_id` /
`gsis_id` column), so the common case is a clean join. Two edge cases are NOT
specifically guarded against: (1) a mid-season trade producing two rows for
the same player_id in the same (season, week) slice under different
`recent_team` values (e.g., a Tuesday waiver claim ahead of that week's game)
-- `_actual_player_stats` keeps whichever row is seen last, which could
silently attribute the wrong team's box score to that player-week; (2) a
player who has enough PRIOR history to receive a share-based `PlayerInput`
(and therefore gets a simulated distribution) but did not play at all in the
target week (bye, healthy scratch -- in-season injury IS now modeled, see
below) has no weekly row and is silently dropped from that market's pairs,
which slightly biases the player-level coverage numbers toward players who
play every week. Neither is believed to be a large effect at v1 but both are
worth re-checking if the ship-gate numbers look off for a specific market.

Per-week active-roster + injury filter (mirrors production): each historical
game's spec is now built the same way `scripts/generate_sim_nfl.py` builds a
live slate -- `usage.active_usage` per (team, season, week), using that
week's depth chart and `nfl_data_py.import_injuries` OUT/Doubtful
designations for that exact (season, week) (see `out_names_by_team_week`),
not `injuries={}`. `pass_yds` is paired against the actual stats keyed by the
sim's starting-QB gsis_id (`active_usage`'s second return value), which is
the same gsis `build_spec_from_usage` asserts the spec's lone QB carries --
see `actual_qb_pass_yds` -- fixing a prior mispairing artifact (MAE-178)
where pass_yds could land on the wrong player.

Usage:
    PYTHONPATH=src uv run python scripts/backtest_sim_nfl.py

Requires nflverse network access; NOT run as part of this task (slow: fetches
multiple seasons of pbp/weekly data and runs one Monte Carlo sim per historical
game). Tune with the DESK_SIM_N env var (sims per game; default 2000).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sportsmodel.nfl.data import load_schedules
from sportsmodel.nfl.teams import TEAMS, normalize_team
from sportsmodel.serving.props_ev import PROJECTED_USAGE_GATE, is_propable_projected
from sportsmodel.sim.engine import GameSims, pred_scores
from sportsmodel.sim.nfl.aggregate import nfl_player_prop_dists
from sportsmodel.sim.nfl.inputs import build_spec_from_usage
from sportsmodel.sim.nfl.kernel import simulate_game
from sportsmodel.sim.nfl.rates import (fetch_nflverse, team_rates_from_pbp,
                                       team_defense_rates_from_pbp)
from sportsmodel.sim.nfl.usage import (
    abbrev_alignment,
    active_usage,
    build_pfr_to_gsis,
    fetch_usage_sources,
    normalize_depth_charts,
)

DEFAULT_N_SIMS = 2000
SIM_SEED = 42

# Ship-gate validation span: completed seasons walked forward game-by-game.
VALIDATION_SEASONS = [2023, 2024, 2025]

# Extra prior seasons fetched (but never validated on) purely to warm up
# team_rates_from_pbp and active_usage's recency-weighted usage window for
# early weeks of the first validation season -- without this, week 1 of
# VALIDATION_SEASONS[0] would have zero history before its cutoff and
# team_rates_from_pbp would hand back all-zero drive_outcomes (a
# divide-by-zero landmine in kernel.sample_drive's renormalization).
# 1 = only the previous season, matching generate_sim_nfl's FETCH_SEASONS_BACK
# (prev + current window) so the backtest validates the shipped config.
WARMUP_SEASONS_BACK = 1

MARKET_MAX = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 200, "receptions": 15, "pass_tds": 6}
PLAYER_MARKETS: tuple[str, ...] = ("pass_yds", "rush_yds", "rec_yds", "receptions")
_ACTUAL_COL = {
    "pass_yds": "passing_yards",
    "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "receptions": "receptions",
}

# ACTUAL-usage columns pulled alongside the stat columns above, purely to gate
# per-market relevance (see `is_propable`) -- nflverse weekly column names.
_USAGE_COLS: tuple[str, ...] = ("attempts", "carries", "targets")

# Per-market (usage column, minimum ACTUAL usage) below which a player-week
# wouldn't realistically have had a prop line that market, so it's excluded
# from that market's evaluation pairs.
_MARKET_USAGE_GATE: dict[str, tuple[str, float]] = {
    "pass_yds": ("attempts", 10.0),
    "rush_yds": ("carries", 5.0),
    "rec_yds": ("targets", 3.0),
    "receptions": ("targets", 3.0),
}

# PROJECTED_USAGE_GATE and is_propable_projected are relocated to
# sportsmodel.serving.props_ev (Ruling C1 / Task 2) and imported above, so
# this module and the upcoming prop board producer share one definition.


# =============================================================================
# PURE metric helpers (unit-tested)
# =============================================================================

def brier(probs: list[float], outcomes: list[float]) -> float:
    """Mean squared error between predicted probabilities and 0/1 (or 0.5-tie)
    outcomes. Lower is better; a well-calibrated binary forecaster scores
    around 0.25 for coin-flip games and lower as skill improves."""
    if not probs:
        return float("nan")
    diffs = [(p - o) ** 2 for p, o in zip(probs, outcomes)]
    return float(np.mean(diffs))


def mae(preds: list[float], actuals: list[float]) -> float:
    """Mean absolute error between two equal-length sequences."""
    if not preds:
        return float("nan")
    diffs = [abs(p - a) for p, a in zip(preds, actuals)]
    return float(np.mean(diffs))


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient. NaN if fewer than 2 points or either
    series has zero variance (correlation is undefined, not zero, there)."""
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if xs_arr.std() == 0.0 or ys_arr.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(xs_arr, ys_arr)[0, 1])


def quantile_from_pmf(dist: dict, q: float) -> float:
    """Extract the q-th percentile VALUE from a `{"kind", "pmf", "mean"}` pmf
    dict (see `sportsmodel.sim.engine._pmf_mean` / `aggregate.nfl_player_prop_dists`).

    `pmf[k]` is P(X == k) for integer k = 0..len(pmf)-1. Returns the smallest
    k such that the cumulative sum P(X <= k) >= q (the standard discrete
    quantile / inverse-CDF definition). Falls back to the top bin if q is not
    reached (e.g. q >= 1.0, or a pmf that doesn't sum to 1 due to clipping).

    Compares with a small epsilon tolerance so float64 summation error (e.g. a
    uniform 10-bin pmf's running total landing on 0.8999999999999999 instead
    of exactly 0.9) can't push the crossing point out by one bin.
    """
    _eps = 1e-9
    cum = 0.0
    pmf = dist["pmf"]
    for k, p in enumerate(pmf):
        cum += p
        if cum >= q - _eps:
            return float(k)
    return float(len(pmf) - 1)


def share_below(quantile_values: list[float], actuals: list[float]) -> float:
    """Coverage / calibration check for a fixed quantile q.

    Given parallel lists where `quantile_values[i]` is the sim's q-th
    percentile value for observation i and `actuals[i]` is what actually
    happened, returns the fraction of observations where
    `actuals[i] <= quantile_values[i]`. For a well-calibrated q-quantile this
    fraction should be approximately q (e.g. ~0.50 for p50, ~0.90 for p90).
    """
    if not quantile_values:
        return float("nan")
    hits = [1.0 if a <= qv else 0.0 for qv, a in zip(quantile_values, actuals)]
    return float(np.mean(hits))


def is_propable(market: str, actual_usage: dict) -> bool:
    """True if a player's ACTUAL usage that week clears the threshold at
    which that market would realistically have had a prop line offered.

    `actual_usage` is a mapping that includes (at least) the usage column
    `_MARKET_USAGE_GATE[market]` needs (`attempts`/`carries`/`targets`);
    missing keys are treated as 0 usage. Gating on ACTUAL (not projected)
    usage keeps a market's evaluation sample to players who genuinely had a
    role in that stat that week, e.g. a WR run 0 pass routes doesn't belong
    in the pass_yds sample even though their sim'd pass_yds distribution and
    actual pass_yds (0) both exist. Unknown markets are never propable.
    """
    gate = _MARKET_USAGE_GATE.get(market)
    if gate is None:
        return False
    usage_col, threshold = gate
    return float(actual_usage.get(usage_col, 0.0) or 0.0) >= threshold


# Historical injury designations (nfl_data_py `import_injuries` `report_status`
# values, case-insensitive) treated as ruled out for a given player-week --
# matches `scripts/generate_sim_nfl.py`'s `_OUT_STATUSES` live-slate gate.
_OUT_STATUSES = frozenset({"out", "doubtful"})


def out_names_by_team_week(
    injuries_df: pd.DataFrame, season: int, week: int
) -> dict[str, set[str]]:
    """team -> {lowercased full_name} of players ruled OUT/Doubtful for one
    (season, week), from nflverse `import_injuries()` (columns include
    `season`, `week`, `team`, `full_name`, `report_status`). PURE.

    `report_status` is matched case-insensitively against `_OUT_STATUSES`
    ("out"/"doubtful"); any other value (e.g. "Questionable") or a NaN status
    excludes the row. A NaN/blank `full_name` is likewise skipped -- there's
    nothing to match against a depth chart's name column. This is the
    per-week active/inactive filter `active_usage`'s `injuries_out_names`
    expects, sourced from real historical designations instead of the
    `injuries={}` placeholder the backtest used before this.
    """
    out: dict[str, set[str]] = {}
    rows = injuries_df[
        (injuries_df["season"] == season) & (injuries_df["week"] == week)
    ]
    for row in rows.itertuples(index=False):
        status = getattr(row, "report_status", None)
        if pd.isna(status) or str(status).strip().lower() not in _OUT_STATUSES:
            continue
        name = getattr(row, "full_name", None)
        if pd.isna(name) or not str(name).strip():
            continue
        team = str(getattr(row, "team", "")).strip()
        out.setdefault(team, set()).add(str(name).strip().lower())
    return out


def questionable_names_by_team_week(
    injuries_df: pd.DataFrame, season: int, week: int
) -> dict[str, set[str]]:
    """team -> {lowercased full_name} of players tagged QUESTIONABLE for one
    (season, week), for active_usage's `questionable_names` (down-weighted, not
    dropped). Same source/shape as out_names_by_team_week but matches only
    "questionable". PURE."""
    out: dict[str, set[str]] = {}
    rows = injuries_df[
        (injuries_df["season"] == season) & (injuries_df["week"] == week)
    ]
    for row in rows.itertuples(index=False):
        status = getattr(row, "report_status", None)
        if pd.isna(status) or str(status).strip().lower() != "questionable":
            continue
        name = getattr(row, "full_name", None)
        if pd.isna(name) or not str(name).strip():
            continue
        team = str(getattr(row, "team", "")).strip()
        out.setdefault(team, set()).add(str(name).strip().lower())
    return out


def actual_qb_pass_yds(
    qb_gsis: str | None, actual_stats: dict[str, dict[str, float]]
) -> float | None:
    """Not called from the main dists loop (that loop already pairs pass_yds
    correctly via the generic gsis-keyed path below); this function documents
    and unit-tests the gsis-pairing CONTRACT that loop relies on, so it isn't
    mistaken for dead code -- see module docstring's Ruling C2.

    Ruling C2: the actual value the sim's starting-QB pass_yds
    distribution should be compared against -- `actual_stats[qb_gsis]
    ["pass_yds"]` if `qb_gsis` is known and present in `actual_stats`, else
    None. PURE.

    This makes explicit (and unit-testable) the pairing contract the main
    dists loop already relies on implicitly: `nfl_player_prop_dists` and
    `_actual_player_stats` are both keyed by gsis_id, so once `active_usage`
    hands back the true starting QB's gsis, pass_yds pairs against THAT
    player's actual stats -- not a mispaired player_id (the MAE-178 root
    cause).
    """
    if qb_gsis is None:
        return None
    stats = actual_stats.get(qb_gsis)
    if stats is None:
        return None
    return stats.get("pass_yds")


# NOTE: `abbrev_alignment` (the depth/injuries/schedule abbrev-mismatch
# diagnostic) now lives in `sportsmodel.sim.nfl.usage` -- imported above --
# since `scripts/generate_sim_nfl.py` needs the same pure check for the live
# slate and scripts aren't an importable package. See usage.py's docstring.


# =============================================================================
# Walk-forward harness (heavy IO -- not unit-tested; see module docstring)
# =============================================================================

def _actual_player_stats(weekly_df, season: int, week: int) -> dict[str, dict[str, float]]:
    """player_id -> {"pass_yds","rush_yds","rec_yds","receptions", plus the
    ACTUAL-usage columns in `_USAGE_COLS` ("attempts","carries","targets")}
    for one (season, week) slice of nflverse `weekly` data. A usage column
    absent from this pull (or NaN for a given row) is treated as 0, which
    only ever makes `is_propable` gate that player-week out -- never in. See
    module docstring's "actual-stat join" concern for this keying's edge
    cases."""
    rows = weekly_df[(weekly_df["season"] == season) & (weekly_df["week"] == week)]
    out: dict[str, dict[str, float]] = {}
    for row in rows.itertuples(index=False):
        pid = str(getattr(row, "player_id"))
        stats = {
            market: float(getattr(row, col, 0.0) or 0.0)
            for market, col in _ACTUAL_COL.items()
        }
        stats.update(
            {col: float(getattr(row, col, 0.0) or 0.0) for col in _USAGE_COLS}
        )
        out[pid] = stats
    return out


def run_backtest(
    seasons: list[int],
    n_sims: int,
    seed: int = SIM_SEED,
    on_game: Callable[[int, int, str, str, GameSims], None] | None = None,
    season_decay: float = 1.0,
    questionable_weight: float = 1.0,
    home_field: float = 0.0,
    use_defense: bool = True,
) -> dict:
    """Walk forward over every completed REG-season game in `seasons`.

    on_game: optional callback invoked once per successfully-simulated game
        as `on_game(season, week, home, away, sims)` (`home`/`away` already
        `normalize_team`-normalized, `sims` the raw `GameSims` from
        `simulate_game`) -- lets a caller (e.g.
        `scripts/build_cover_dataset.py`'s `_build_sim_lookup`) capture the
        same leakage-free walk-forward's per-game margin/total distributions
        (via `sim.engine.margin_pmf`/`total_pmf`) without duplicating this
        function's rates/usage/spec-building wiring. Not called for a game
        the per-game try/except below skips.

    Returns a dict of raw sample lists for `report()` to summarize:
      game_probs/game_outcomes, margin_preds/margin_actuals,
      total_preds/total_actuals, per-market player_mean_pairs/
      player_p50_pairs/player_p90_pairs (each a list of (sim_value, actual)
      tuples, restricted to player-weeks that clear that market's
      `is_propable` ACTUAL-usage gate -- selection-biased, reference only),
      the parallel player_mean_pairs_proj/player_p50_pairs_proj/
      player_p90_pairs_proj (same shape, but restricted to player-weeks that
      clear that market's `is_propable_projected` PROJECTED-usage gate --
      the deployment/shippability population; a player-week can land in
      either gate's pairs, both, or neither, since the two gates are
      evaluated independently), and `n_empty_active` (count of games where
      `active_usage` handed back an empty roster for either side -- see
      `abbrev_alignment`'s docstring for why this can happen silently).
    """
    fetch_seasons = list(range(min(seasons) - WARMUP_SEASONS_BACK, max(seasons) + 1))
    print(f"fetching nflverse seasons {fetch_seasons}")
    nflverse = fetch_nflverse(fetch_seasons)
    pbp, weekly, snaps = nflverse["pbp"], nflverse["weekly"], nflverse["snaps"]

    print(f"fetching usage sources for seasons {fetch_seasons}")
    usage_src = fetch_usage_sources(fetch_seasons)
    pfr2gsis = build_pfr_to_gsis(usage_src["ids"])
    # Normalize nflverse's current (snapshot-based) depth schema onto the old
    # columns active_usage expects. Stamped at (earliest fetch season, week 1)
    # so active_usage's _latest_depth_week fallback resolves it for EVERY
    # walk-forward (season, week). LIMITATION: the new feed carries only recent
    # snapshots, so every historical week gets the CURRENT chart (upstream has
    # no depth history) -- acceptable for this rough/bounded backfill; a
    # no-op passthrough when the old (frozen) schema is what's installed.
    depth_df = normalize_depth_charts(usage_src["depth"], min(fetch_seasons), 1)

    print(f"fetching historical injuries for seasons {fetch_seasons}")
    import nfl_data_py as nfl

    injuries_df = nfl.import_injuries(fetch_seasons)

    schedules = load_schedules(fetch_seasons)
    games = schedules[
        schedules["season"].isin(seasons)
        & (schedules["game_type"] == "REG")
        & schedules["home_score"].notna()
        & schedules["away_score"].notna()
    ].sort_values(["season", "week"])

    game_teams: set[str] = set()
    for row in games.itertuples(index=False):
        for raw in (row.home_team, row.away_team):
            try:
                game_teams.add(normalize_team(raw))
            except ValueError:
                continue  # surfaces via the per-game try/except in the loop below

    alignment = abbrev_alignment(depth_df, injuries_df, game_teams, TEAMS)
    if alignment["depth_unknown"] or alignment["injuries_unknown"] or alignment["games_unknown"]:
        print(
            f"WARN abbrev mismatch: depth_unknown={alignment['depth_unknown']} "
            f"injuries_unknown={alignment['injuries_unknown']} "
            f"games_unknown={alignment['games_unknown']}"
        )
    else:
        print("abbrev alignment: OK (depth/injuries/schedule codes all known)")

    rng = np.random.default_rng(seed)

    game_probs: list[float] = []
    game_outcomes: list[float] = []
    margin_preds: list[float] = []
    margin_actuals: list[float] = []
    total_preds: list[float] = []
    total_actuals: list[float] = []
    player_mean_pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in PLAYER_MARKETS}
    player_p50_pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in PLAYER_MARKETS}
    player_p90_pairs: dict[str, list[tuple[float, float]]] = {m: [] for m in PLAYER_MARKETS}
    # Parallel projected-usage-gated pairs (see is_propable_projected) -- the
    # deployment/shippability population. Independent of the actual-usage
    # gate above: a player-week can be in one, both, or neither.
    player_mean_pairs_proj: dict[str, list[tuple[float, float]]] = {m: [] for m in PLAYER_MARKETS}
    player_p50_pairs_proj: dict[str, list[tuple[float, float]]] = {m: [] for m in PLAYER_MARKETS}
    player_p90_pairs_proj: dict[str, list[tuple[float, float]]] = {m: [] for m in PLAYER_MARKETS}

    n_ok = 0
    n_skipped = 0
    n_empty_active = 0
    cutoff_key = None
    rates: dict = {}
    actual_stats: dict = {}
    out_by_team: dict[str, set[str]] = {}
    q_by_team: dict[str, set[str]] = {}
    # (team, season, week) -> active_usage's (players, qb_gsis) result.
    # Memoized so each team's active roster for a given week is computed
    # ONCE (active_usage does a league-wide weekly groupby internally) even
    # though many games in a cutoff block reuse the same team-week.
    active_cache: dict[tuple[str, int, int], tuple[list, str | None]] = {}

    def _active_for(team: str, season: int, week: int) -> tuple[list, str | None]:
        key = (team, season, week)
        if key not in active_cache:
            active_cache[key] = active_usage(
                team,
                season,
                week,
                depth_df,
                weekly,
                usage_src["snaps"],
                pfr2gsis,
                out_by_team.get(team, set()),
                questionable_names=q_by_team.get(team, set()),
                questionable_weight=questionable_weight,
            )
        return active_cache[key]

    for row in games.itertuples(index=False):
        season, week = int(row.season), int(row.week)
        key = (season, week)
        if key != cutoff_key:
            rates = team_rates_from_pbp(pbp, season, week, season_decay=season_decay)
            def_rates = (team_defense_rates_from_pbp(pbp, season, week, season_decay=season_decay)
                         if use_defense else {})
            actual_stats = _actual_player_stats(weekly, season, week)
            out_by_team = out_names_by_team_week(injuries_df, season, week)
            q_by_team = questionable_names_by_team_week(injuries_df, season, week)
            cutoff_key = key

        try:
            home = normalize_team(row.home_team)
            away = normalize_team(row.away_team)
            home_players, home_qb = _active_for(home, season, week)
            away_players, away_qb = _active_for(away, season, week)
            if not home_players or home_qb is None or not away_players or away_qb is None:
                # A team-abbrev convention mismatch (depth_df/injuries_df vs.
                # normalize_team) makes active_usage silently hand back an
                # empty roster instead of raising -- see abbrev_alignment's
                # docstring. Count it loudly rather than let it silently
                # shrink/pollute the player-market sample while n_ok climbs.
                n_empty_active += 1
            spec = build_spec_from_usage(
                home, away, rates, home_players, away_players, home_qb, away_qb,
                def_rates=def_rates,
            )
            sims = simulate_game(spec, n_sims, rng, home_field=home_field)
        except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the walk
            print(f"skipping {season} wk{week} {row.away_team}@{row.home_team}: {exc}")
            n_skipped += 1
            continue

        if on_game is not None:
            on_game(season, week, home, away, sims)

        scores = pred_scores(sims)
        home_score, away_score = float(row.home_score), float(row.away_score)
        actual_home_win = (
            0.5 if home_score == away_score else (1.0 if home_score > away_score else 0.0)
        )
        game_probs.append(scores["home_win_prob"])
        game_outcomes.append(actual_home_win)
        margin_preds.append(scores["pred_margin"])
        margin_actuals.append(home_score - away_score)
        total_preds.append(scores["pred_total"])
        total_actuals.append(home_score + away_score)

        dists = nfl_player_prop_dists(sims, MARKET_MAX)
        for player_id, markets in dists.items():
            actual = actual_stats.get(player_id)
            if actual is None:
                continue
            for market in PLAYER_MARKETS:
                if market not in markets:
                    continue
                dist = markets[market]
                a = actual[market]
                if is_propable(market, actual):
                    player_mean_pairs[market].append((dist["mean"], a))
                    player_p50_pairs[market].append((quantile_from_pmf(dist, 0.50), a))
                    player_p90_pairs[market].append((quantile_from_pmf(dist, 0.90), a))
                if is_propable_projected(market, dist["mean"]):
                    player_mean_pairs_proj[market].append((dist["mean"], a))
                    player_p50_pairs_proj[market].append((quantile_from_pmf(dist, 0.50), a))
                    player_p90_pairs_proj[market].append((quantile_from_pmf(dist, 0.90), a))

        n_ok += 1

    print(
        f"walk-forward: {n_ok} games scored, {n_skipped} skipped, "
        f"{n_empty_active} games had an empty active roster"
    )

    return {
        "game_probs": game_probs,
        "game_outcomes": game_outcomes,
        "margin_preds": margin_preds,
        "margin_actuals": margin_actuals,
        "total_preds": total_preds,
        "total_actuals": total_actuals,
        "player_mean_pairs": player_mean_pairs,
        "player_p50_pairs": player_p50_pairs,
        "player_p90_pairs": player_p90_pairs,
        "player_mean_pairs_proj": player_mean_pairs_proj,
        "player_p50_pairs_proj": player_p50_pairs_proj,
        "player_p90_pairs_proj": player_p90_pairs_proj,
        "n_empty_active": n_empty_active,
    }


def report(results: dict) -> None:
    """Print the ship-gate report: GAME metrics, then PLAYER metrics per market."""
    print("\n=== GAME ===")
    print(f"Brier (home_win_prob):    {brier(results['game_probs'], results['game_outcomes']):.4f}")
    print(f"Margin MAE:               {mae(results['margin_preds'], results['margin_actuals']):.3f}")
    print(f"Total MAE:                {mae(results['total_preds'], results['total_actuals']):.3f}")
    sim_corr = pearson_corr(results["margin_preds"], results["total_preds"])
    actual_corr = pearson_corr(results["margin_actuals"], results["total_actuals"])
    print(f"Margin<->Total corr:      sim={sim_corr:.3f}  actual={actual_corr:.3f}")

    print("\n=== PLAYER (actual-usage gate -- realized, selection-biased; reference only) ===")
    for market in PLAYER_MARKETS:
        means = results["player_mean_pairs"][market]
        p50s = results["player_p50_pairs"][market]
        p90s = results["player_p90_pairs"][market]
        if not means:
            print(f"{market:12s}  no data")
            continue
        mean_mae = mae([m for m, _ in means], [a for _, a in means])
        cov50 = share_below([q for q, _ in p50s], [a for _, a in p50s])
        cov90 = share_below([q for q, _ in p90s], [a for _, a in p90s])
        print(
            f"{market:12s}  n={len(means):6d}  mean_MAE={mean_mae:7.2f}  "
            f"coverage_p50={cov50:.3f} (target ~0.50)  coverage_p90={cov90:.3f} (target ~0.90)"
        )

    print("\n=== PLAYER (projected-usage gate -- deployment/shippability) ===")
    for market in PLAYER_MARKETS:
        means = results["player_mean_pairs_proj"][market]
        p50s = results["player_p50_pairs_proj"][market]
        p90s = results["player_p90_pairs_proj"][market]
        if not means:
            print(f"{market:12s}  no data")
            continue
        mean_mae = mae([m for m, _ in means], [a for _, a in means])
        cov50 = share_below([q for q, _ in p50s], [a for _, a in p50s])
        cov90 = share_below([q for q, _ in p90s], [a for _, a in p90s])
        print(
            f"{market:12s}  n={len(means):6d}  mean_MAE={mean_mae:7.2f}  "
            f"coverage_p50={cov50:.3f} (target ~0.50)  coverage_p90={cov90:.3f} (target ~0.90)"
        )


def main() -> None:
    n_sims = int(os.environ.get("DESK_SIM_N", str(DEFAULT_N_SIMS)))
    # SIM_SEASON_DECAY mirrors generate_sim_nfl so a bare backtest validates the
    # shipped config; default 0.4 matches production. Set SIM_SEASON_DECAY=1.0 to
    # compare against equal weighting.
    season_decay = float(os.environ.get("SIM_SEASON_DECAY", "0.4"))
    q_weight = float(os.environ.get("SIM_QUESTIONABLE_WEIGHT", "0.75"))
    home_field = float(os.environ.get("SIM_HOME_FIELD", "0.07"))
    print(f"season_decay={season_decay} questionable_weight={q_weight} home_field={home_field}")
    results = run_backtest(VALIDATION_SEASONS, n_sims, season_decay=season_decay,
                           questionable_weight=q_weight, home_field=home_field)
    report(results)


if __name__ == "__main__":
    main()
