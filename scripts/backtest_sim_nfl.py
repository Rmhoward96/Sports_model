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
          realistically have a prop line that week (see `is_propable`) so the
          metric isn't swamped by non-participants (e.g. a WR's pass_yds is
          0-vs-0 every week and would otherwise dilute the pass_yds numbers).

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
`player_inputs_from_weekly`'s own keying (both come from the same nflverse
`player_id` column), so the common case is a clean join. Two edge cases are
NOT specifically guarded against: (1) a mid-season trade producing two rows
for the same player_id in the same (season, week) slice under different
`recent_team` values (e.g., a Tuesday waiver claim ahead of that week's game)
-- `_actual_player_stats` keeps whichever row is seen last, which could
silently attribute the wrong team's box score to that player-week; (2) a
player who has enough PRIOR history to receive a share-based `PlayerInput`
(and therefore gets a simulated distribution) but did not play at all in the
target week (bye, healthy scratch, in-season injury not modeled here since
`injuries={}` is passed to `build_spec` -- see below) has no weekly row and is
silently dropped from that market's pairs, which slightly biases the
player-level coverage numbers toward players who play every week. Neither is
believed to be a large effect at v1 but both are worth re-checking if the
ship-gate numbers look off for a specific market.

CONCERN -- no injury zeroing: unlike `scripts/generate_sim_nfl.py` (which
zeros out OUT/Doubtful players via `injuries_nflverse.current_injuries`),
this backtest passes `injuries={}` to `build_spec` for every historical game,
i.e. every player with enough share to appear in the roster is simulated
regardless of whether they actually played that week. Historical week-by-week
injury *designations* (not just game logs) would need to be pinned per
season/week to fix this properly; left as a known limitation of the v1 ship
gate rather than blocking on it.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sportsmodel.nfl.data import load_schedules
from sportsmodel.nfl.teams import normalize_team
from sportsmodel.sim.engine import pred_scores
from sportsmodel.sim.nfl.aggregate import nfl_player_prop_dists
from sportsmodel.sim.nfl.inputs import build_spec
from sportsmodel.sim.nfl.kernel import simulate_game
from sportsmodel.sim.nfl.rates import (
    fetch_nflverse,
    player_inputs_from_weekly,
    team_rates_from_pbp,
)

DEFAULT_N_SIMS = 2000
SIM_SEED = 42

# Ship-gate validation span: completed seasons walked forward game-by-game.
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]

# Extra prior seasons fetched (but never validated on) purely to warm up
# team_rates_from_pbp/player_inputs_from_weekly for early weeks of the first
# validation season -- without this, week 1 of VALIDATION_SEASONS[0] would
# have zero history before its cutoff and team_rates_from_pbp would hand back
# all-zero drive_outcomes (a divide-by-zero landmine in kernel.sample_drive's
# renormalization).
WARMUP_SEASONS_BACK = 2

MARKET_MAX = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 200, "receptions": 15}
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


def run_backtest(seasons: list[int], n_sims: int, seed: int = SIM_SEED) -> dict:
    """Walk forward over every completed REG-season game in `seasons`.

    Returns a dict of raw sample lists for `report()` to summarize:
      game_probs/game_outcomes, margin_preds/margin_actuals,
      total_preds/total_actuals, and per-market player_mean_pairs/
      player_p50_pairs/player_p90_pairs (each a list of (sim_value, actual)
      tuples, restricted to player-weeks that clear that market's
      `is_propable` ACTUAL-usage gate).
    """
    fetch_seasons = list(range(min(seasons) - WARMUP_SEASONS_BACK, max(seasons) + 1))
    print(f"fetching nflverse seasons {fetch_seasons}")
    nflverse = fetch_nflverse(fetch_seasons)
    pbp, weekly, snaps = nflverse["pbp"], nflverse["weekly"], nflverse["snaps"]

    schedules = load_schedules(fetch_seasons)
    games = schedules[
        schedules["season"].isin(seasons)
        & (schedules["game_type"] == "REG")
        & schedules["home_score"].notna()
        & schedules["away_score"].notna()
    ].sort_values(["season", "week"])

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

    n_ok = 0
    n_skipped = 0
    cutoff_key = None
    rates: dict = {}
    players: dict = {}
    actual_stats: dict = {}

    for row in games.itertuples(index=False):
        season, week = int(row.season), int(row.week)
        key = (season, week)
        if key != cutoff_key:
            rates = team_rates_from_pbp(pbp, season, week)
            players = player_inputs_from_weekly(weekly, snaps, season, week)
            actual_stats = _actual_player_stats(weekly, season, week)
            cutoff_key = key

        try:
            home = normalize_team(row.home_team)
            away = normalize_team(row.away_team)
            spec = build_spec(home, away, rates, players, injuries={})
            sims = simulate_game(spec, n_sims, rng)
        except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the walk
            print(f"skipping {season} wk{week} {row.away_team}@{row.home_team}: {exc}")
            n_skipped += 1
            continue

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
                if not is_propable(market, actual):
                    continue
                dist = markets[market]
                a = actual[market]
                player_mean_pairs[market].append((dist["mean"], a))
                player_p50_pairs[market].append((quantile_from_pmf(dist, 0.50), a))
                player_p90_pairs[market].append((quantile_from_pmf(dist, 0.90), a))

        n_ok += 1

    print(f"walk-forward: {n_ok} games scored, {n_skipped} skipped")

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

    print("\n=== PLAYER ===")
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


def main() -> None:
    n_sims = int(os.environ.get("DESK_SIM_N", str(DEFAULT_N_SIMS)))
    results = run_backtest(VALIDATION_SEASONS, n_sims)
    report(results)


if __name__ == "__main__":
    main()
