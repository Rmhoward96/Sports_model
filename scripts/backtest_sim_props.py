"""Walk-forward, point-in-time backtest of the Monte Carlo sim's PLAYER PROPS.

Mirrors scripts/backtest_sim.py's harness exactly (same month cutoffs, same
profile builders, same actual-lineup/starter extraction) but instead of scoring
game-level win prob / total, it turns each game's simulated play-by-play into
per-player prop distributions (engine.player_prop_dists) and scores them
against ACTUAL per-player outcomes (backtest_props.month_actuals), using the
same Score class backtest_props.py uses. This makes the two prop backtests
(analytic closed-form vs. Monte Carlo sim) directly comparable line-for-line.

H+R+RBI (hrr) is computed by the sim (player_prop_dists always includes it)
but NOT scored here: backtest_props.month_actuals has no per-batter runs/RBI
recoverable from Statcast at the PA level, so there's no actual to compare
against. The analytic prop backtest (backtest_props.py) omits it for the same
reason.

Usage:
    uv run python scripts/backtest_sim_props.py --season 2025 --n-sims 2000
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
from sportsmodel.model.distributions import prob_over_dist
from sportsmodel.model.props import DEFAULT_LINE
from sportsmodel.sim.engine import player_prop_dists
from sportsmodel.sim.mlb import config_dispersion, kernel
from sportsmodel.sim.mlb.advancement import AdvancementTable
from sportsmodel.sim.mlb.build_advancement import build_advancement_table
from sportsmodel.sim.mlb.inputs import build_game_spec

from backtest_sim import (
    GLOB,
    _LINEUP_SIZE,
    _MONTHS,
    _load_league,
    _load_single,
    _load_vecs,
    _load_workload,
    build_sim_profiles,
    month_lineups,
)
from backtest_props import BATTER_MARKETS, PITCHER_MARKETS, Score, month_actuals

MARKET_MAX = {
    "hits": 6, "total_bases": 10, "hrr": 15,
    "pitcher_ks": 15, "hits_allowed": 15, "outs_recorded": 30,
}


def run_sim_props_backtest(season: int, n_sims: int, seed: int = 42, dispersion=...) -> dict:
    """Walk-forward; returns {market: Score} of sim-derived P(over) vs actual outcomes.
    Uses the same production scoring channels + dispersion as generate_sim, so the prop
    calibration is fit against the model that actually serves props. `dispersion` sentinel
    `...` -> production config; pass a Dispersion (or None) to override."""
    if dispersion is ...:
        dispersion = config_dispersion.DISPERSION
    _rates = config_dispersion.load_rates()
    rng = np.random.default_rng(seed)
    scores = {m: Score() for m in BATTER_MARKETS + PITCHER_MARKETS}
    n_skipped_lineup = 0
    n_skipped_profile = 0

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

        lineups = month_lineups(con, season, month)
        games, batters, pitchers = month_actuals(con, season, month)
        con.close()

        bat_actual: dict = {}   # (game_pk, side) -> [(batter, slot, pa, hits, tb, hr), ...]
        for gp, b, side, slot, pa, h, tb, hr in batters:
            bat_actual.setdefault((gp, side), []).append((b, slot, pa, h, tb, hr))
        pit_actual = {(gp, pid): (bf, ks, ha, outs) for gp, pid, bf, ks, ha, outs in pitchers}

        used = 0
        for g in games:
            gp, home, away, hsp, asp = g
            if hsp is None or asp is None:
                continue
            h_sp_vec, a_sp_vec = pit.get(hsp), pit.get(asp)
            wl_h, wl_a = wl.get(hsp), wl.get(asp)
            if None in (h_sp_vec, a_sp_vec, wl_h, wl_a):
                n_skipped_profile += 1
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
            dists = player_prop_dists(sims, MARKET_MAX)

            # Batter props: score every ACTUAL batter (both sides) matched by player_id
            # to the sim's per-player dists (only players in the simulated 9-lineup have one).
            for side in ("Bot", "Top"):
                for b, slot, pa, h, tb, hr in bat_actual.get((gp, side), []):
                    d = dists.get(b)
                    if d is None:
                        continue
                    scores["hits"].add(
                        prob_over_dist(d["hits"], DEFAULT_LINE["hits"]),
                        h, d["hits"]["mean"], DEFAULT_LINE["hits"])
                    scores["total_bases"].add(
                        prob_over_dist(d["total_bases"], DEFAULT_LINE["total_bases"]),
                        tb, d["total_bases"]["mean"], DEFAULT_LINE["total_bases"])
                    scores["home_run"].add(
                        prob_over_dist(d["home_run"], DEFAULT_LINE["home_run"]),
                        hr, d["home_run"]["mean"], DEFAULT_LINE["home_run"])
                    # hrr computed in `d["hrr"]` but not scored -- no per-batter runs/RBI
                    # actual is recoverable from Statcast (see module docstring).

            # Pitcher props: score both ACTUAL starters, matched by player_id to the
            # sim's per-pitcher dists (kernel only tracks starters, not relievers).
            for sp_id in (hsp, asp):
                d = dists.get(sp_id)
                act = pit_actual.get((gp, sp_id))
                if d is None or act is None:
                    continue
                _bf, ks, ha, outs = act
                scores["pitcher_ks"].add(
                    prob_over_dist(d["pitcher_ks"], DEFAULT_LINE["pitcher_ks"]),
                    ks, d["pitcher_ks"]["mean"], DEFAULT_LINE["pitcher_ks"])
                scores["hits_allowed"].add(
                    prob_over_dist(d["hits_allowed"], DEFAULT_LINE["hits_allowed"]),
                    ha, d["hits_allowed"]["mean"], DEFAULT_LINE["hits_allowed"])
                scores["outs_recorded"].add(
                    prob_over_dist(d["outs_recorded"], DEFAULT_LINE["outs_recorded"]),
                    outs, d["outs_recorded"]["mean"], DEFAULT_LINE["outs_recorded"])

            used += 1
        print(f"  {season}-{month:02d}: {used} games scored")
    if n_skipped_lineup:
        print(f"  (skipped {n_skipped_lineup} games with incomplete lineups)")
    if n_skipped_profile:
        print(f"  (skipped {n_skipped_profile} games with missing starter profile/workload)")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    scores = run_sim_props_backtest(args.season, args.n_sims, args.seed)
    print("\n" + "=" * 78)
    print(f"SIM PROPS BACKTEST — {args.season}   (well-calibrated: pred P(over) ~ actual over-rate)")
    print("=" * 78)
    print("Batter props:")
    for m in BATTER_MARKETS:
        scores[m].report(m)
    print("Pitcher props:")
    for m in PITCHER_MARKETS:
        scores[m].report(m)
    print("=" * 78)


if __name__ == "__main__":
    main()
