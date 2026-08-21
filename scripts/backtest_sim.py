"""Walk-forward, point-in-time backtest of the Monte Carlo GAME simulator.

Mirrors scripts/backtest_game.py's harness exactly (same month cutoffs, same
profile builders, same actual-starter extraction) but swaps the closed-form
`predict()` for a full play-by-play simulation of every plate appearance,
using each team's ACTUAL 9-batter lineup (ordered by first appearance) vs the
ACTUAL opposing starter. Emits the same (p_home_win, home_won, pred_total,
actual_total) tuple shape backtest_game.report() consumes, so the two reports
are directly comparable.

Usage:
    uv run python scripts/backtest_sim.py --season 2025 --n-sims 4000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb
import numpy as np

from sportsmodel import transforms
from sportsmodel.sim.engine import pred_scores
from sportsmodel.sim.mlb import kernel
from sportsmodel.sim.mlb.advancement import AdvancementTable
from sportsmodel.sim.mlb.build_advancement import build_advancement_table
from sportsmodel.sim.mlb import config_dispersion
from sportsmodel.sim.mlb.inputs import build_game_spec

from backtest_game import (
    GLOB,
    _MONTHS,
    _load_league,
    _load_single,
    _load_vecs,
    month_games,
    report,
)

_LINEUP_SIZE = 9


def build_sim_profiles(con):
    """Build every profile the sim path needs into `con` (respects the active cutoff)."""
    transforms.build_batter_profiles(con)
    transforms.build_pitcher_profiles(con)
    transforms.build_team_bullpen_profiles(con)
    transforms.build_team_defense(con)
    transforms.build_park_factors(con)
    transforms.build_pitcher_workload(con)
    transforms.build_league_rates(con)
    transforms.shrink_toward_league(con, "feat_batter_profile")
    transforms.shrink_toward_league(con, "feat_pitcher_profile")


def month_lineups(con, season, month):
    """(game_pk, side) -> ordered list of up to 9 batter ids, ordered by first
    appearance (arg_min(at_bat_number) per batter within the game). side is the
    Statcast inning_topbot value: 'Top' = away batting, 'Bot' = home batting."""
    start = f"{season}-{month:02d}-01"
    end = f"{season}-{month + 1:02d}-01" if month < 12 else f"{season + 1}-01-01"
    rows = con.execute(f"""
        WITH pas AS (
            SELECT game_pk, inning_topbot AS side, at_bat_number,
                   CAST(batter AS BIGINT) AS batter
            FROM read_parquet('{GLOB}')
            WHERE CAST(game_date AS DATE) >= DATE '{start}'
              AND CAST(game_date AS DATE) < DATE '{end}'
              AND events IS NOT NULL AND batter IS NOT NULL AND inning_topbot IS NOT NULL
        ),
        bg AS (
            SELECT game_pk, side, batter, min(at_bat_number) AS first_ab
            FROM pas GROUP BY game_pk, side, batter
        ),
        ranked AS (
            SELECT game_pk, side, batter,
                   row_number() OVER (PARTITION BY game_pk, side ORDER BY first_ab) AS slot
            FROM bg
            QUALIFY slot <= {_LINEUP_SIZE}
        )
        SELECT game_pk, side, batter FROM ranked ORDER BY game_pk, side, slot
    """).fetchall()
    lineup: dict = {}
    for gp, side, batter in rows:
        lineup.setdefault((gp, side), []).append(batter)
    return lineup


def _load_workload(con):
    """{player_id: (avg_outs, sd_outs)} -- feeds kernel.Pitcher's outs-recorded hook."""
    return {int(r[0]): (r[1], r[2]) for r in
            con.execute("SELECT player_id, avg_outs, sd_outs FROM feat_pitcher_workload").fetchall()}


def run_sim_backtest(season: int, n_sims: int, seed: int = 42, dispersion=...) -> list:
    """Walk-forward; returns (p_home_win, home_won, pred_total, actual_total) per game.

    dispersion: sentinel `...` -> use the tuned production DISPERSION; pass an explicit
    Dispersion (or None) to override, e.g. from the tuning search."""
    if dispersion is ...:
        dispersion = config_dispersion.DISPERSION
    _rates = config_dispersion.load_rates()
    rng = np.random.default_rng(seed)
    samples = []
    n_skipped_lineup = 0
    for month in _MONTHS:
        cutoff = f"{season}-{month:02d}-01"
        transforms.set_cutoff(cutoff)
        con = duckdb.connect(":memory:")
        con.execute("INSTALL json; LOAD json;")
        build_sim_profiles(con)
        bat = _load_vecs(con, "feat_batter_profile", "player_id", min_season_pa=150, min_recent_pa=40)
        pit = _load_vecs(con, "feat_pitcher_profile", "player_id", min_season_pa=150, min_recent_pa=40)
        bp = _load_vecs(con, "feat_team_bullpen", "team", min_season_pa=1000, min_recent_pa=300)
        dfn = _load_single(con, "feat_team_defense", "def_factor")
        park = _load_single(con, "park_factors", "pf_runs")
        league = _load_league(con)
        wl = _load_workload(con)
        adv_rows = build_advancement_table(con)
        adv = AdvancementTable.from_rows(adv_rows)
        transforms.set_cutoff(None)

        games = month_games(con, season, month)
        lineups = month_lineups(con, season, month)
        con.close()

        used = 0
        for gp, home, away, hr, ar, hsp, asp in games:
            if hr is None or ar is None or hsp is None or asp is None:
                continue
            h_sp_vec, a_sp_vec = pit.get(hsp), pit.get(asp)
            wl_h, wl_a = wl.get(hsp), wl.get(asp)
            if None in (h_sp_vec, a_sp_vec, wl_h, wl_a):
                continue

            home_ids = lineups.get((gp, "Bot"))
            away_ids = lineups.get((gp, "Top"))
            if not home_ids or not away_ids or len(home_ids) != _LINEUP_SIZE or len(away_ids) != _LINEUP_SIZE:
                n_skipped_lineup += 1
                continue
            home_order = [(pid, bat[pid]) for pid in home_ids if pid in bat]
            away_order = [(pid, bat[pid]) for pid in away_ids if pid in bat]
            if len(home_order) != _LINEUP_SIZE or len(away_order) != _LINEUP_SIZE:
                n_skipped_lineup += 1
                continue

            pf = park.get(home, 1.0)
            context = {
                "home_pf": pf, "hr_mult": 1.0,
                "home_def": dfn.get(home, 1.0), "away_def": dfn.get(away, 1.0),
            }
            spec = build_game_spec(
                home_order, away_order, h_sp_vec, a_sp_vec,
                bp.get(home), bp.get(away), workload=wl, context=context,
                league=league, adv=adv, home_starter_id=hsp, away_starter_id=asp,
                roe_p=_rates["p_roe"], wp_p=_rates["p_wp"], dispersion=dispersion,
            )
            sims = kernel.simulate(spec, n_sims, rng)
            ps = pred_scores(sims)
            samples.append((ps["home_win_prob"], 1 if hr > ar else 0,
                            ps["pred_total"], hr + ar))
            used += 1
        print(f"  {season}-{month:02d}: {used} games scored")
    if n_skipped_lineup:
        print(f"  (skipped {n_skipped_lineup} games with incomplete lineups)")
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--n-sims", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    report(run_sim_backtest(args.season, args.n_sims, args.seed))


if __name__ == "__main__":
    main()
