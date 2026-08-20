"""Walk-forward, point-in-time backtest of the game model (no odds needed).

For each month of the test season: rebuild profiles using ONLY games before that
month (no leakage), predict every game that month from its ACTUAL starters, and
compare to the ACTUAL result. Scores win-probability calibration (Brier / log-loss /
reliability buckets) and total-runs error (MAE / RMSE), vs. naive baselines.

Usage:
    uv run python scripts/backtest_game.py --season 2025
"""
from __future__ import annotations

import argparse
import sys
from math import log
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb

from sportsmodel import transforms
from sportsmodel.model import game, rates

OUTCOMES = ["p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out"]
GLOB = transforms._PARQUET_GLOB
STARTER_SHARE = 0.62
_MONTHS = [4, 5, 6, 7, 8, 9]  # regular-season months to test


def _load_vecs(con, table, id_col, min_pa=150):
    rows = con.execute(
        f"SELECT {id_col}, window_name, pa, {', '.join(OUTCOMES)} "
        f"FROM {table} WHERE vs_hand='ALL'"
    ).fetchall()
    best = {}
    for r in rows:
        key, win, pa = r[0], r[1], r[2]
        vec = {o: r[3 + i] for i, o in enumerate(OUTCOMES)}
        rank = 0 if (win == "season" and pa >= min_pa) else (1 if win == "career" else 2)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, vec)
    return {k: v for k, (_, v) in best.items()}


def _load_single(con, table, val_col):
    return {r[0]: r[1] for r in con.execute(f"SELECT team, {val_col} FROM {table}").fetchall()}


def _load_league(con):
    r = con.execute(
        f"SELECT {', '.join(OUTCOMES)} FROM ref_league_rates "
        f"WHERE vs_hand='ALL' AND window_name='career' LIMIT 1"
    ).fetchone()
    return {o: r[i] for i, o in enumerate(OUTCOMES)}


def build_pit_profiles(con):
    """Build only the game-model profiles into `con` (respecting the active cutoff)."""
    transforms.build_pitcher_profiles(con)
    transforms.build_team_offense_profiles(con)
    transforms.build_team_bullpen_profiles(con)
    transforms.build_team_defense(con)
    transforms.build_park_factors(con)
    transforms.build_league_rates(con)
    transforms.shrink_toward_league(con, "feat_pitcher_profile")


def month_games(con, season, month):
    start = f"{season}-{month:02d}-01"
    end = f"{season}-{month + 1:02d}-01" if month < 12 else f"{season + 1}-01-01"
    return con.execute(f"""
        WITH pas AS (
            SELECT game_pk, home_team, away_team, inning, inning_topbot, at_bat_number,
                   CAST(pitcher AS BIGINT) AS pitcher, post_home_score, post_away_score
            FROM read_parquet('{GLOB}')
            WHERE CAST(game_date AS DATE) >= DATE '{start}'
              AND CAST(game_date AS DATE) < DATE '{end}'
        ),
        g AS (
            SELECT game_pk, any_value(home_team) home_team, any_value(away_team) away_team,
                   max(post_home_score) home_runs, max(post_away_score) away_runs
            FROM pas GROUP BY game_pk
        ),
        hs AS (SELECT game_pk, arg_min(pitcher, at_bat_number) pid FROM pas
               WHERE inning=1 AND inning_topbot='Top' GROUP BY game_pk),
        aw AS (SELECT game_pk, arg_min(pitcher, at_bat_number) pid FROM pas
               WHERE inning=1 AND inning_topbot='Bot' GROUP BY game_pk)
        SELECT g.game_pk, g.home_team, g.away_team, g.home_runs, g.away_runs,
               hs.pid AS home_sp, aw.pid AS away_sp
        FROM g LEFT JOIN hs USING(game_pk) LEFT JOIN aw USING(game_pk)
    """).fetchall()


def predict(off, opp_sp, opp_bp, opp_def, pf, league):
    vec_sp = rates.matchup_vector(off, opp_sp, league)
    vec_bp = rates.matchup_vector(off, opp_bp, league) if opp_bp else vec_sp
    vec = {o: STARTER_SHARE * vec_sp[o] + (1 - STARTER_SHARE) * vec_bp[o] for o in OUTCOMES}
    if opp_def != 1.0:
        vec = game.apply_bip_defense(vec, opp_def)
    return game.expected_runs(vec, park_factor=pf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()

    samples = []  # (p_home_win, home_won, pred_total, actual_total)
    for month in _MONTHS:
        cutoff = f"{args.season}-{month:02d}-01"
        transforms.set_cutoff(cutoff)
        con = duckdb.connect(":memory:")
        con.execute("INSTALL json; LOAD json;")
        build_pit_profiles(con)
        pit = _load_vecs(con, "feat_pitcher_profile", "player_id")
        off = _load_vecs(con, "feat_team_offense", "team", min_pa=1000)
        bp = _load_vecs(con, "feat_team_bullpen", "team", min_pa=1000)
        dfn = _load_single(con, "feat_team_defense", "def_factor")
        park = _load_single(con, "park_factors", "pf_runs")
        league = _load_league(con)
        transforms.set_cutoff(None)

        games = month_games(con, args.season, month)
        con.close()
        used = 0
        for gp, home, away, hr, ar, hsp, asp in games:
            if hr is None or ar is None or hsp is None or asp is None:
                continue
            h_sp, a_sp = pit.get(hsp), pit.get(asp)
            h_off, a_off = off.get(home), off.get(away)
            if None in (h_sp, a_sp, h_off, a_off):
                continue
            pf = park.get(home, 1.0)
            h_runs = predict(h_off, a_sp, bp.get(away), dfn.get(away, 1.0), pf, league)
            a_runs = predict(a_off, h_sp, bp.get(home), dfn.get(home, 1.0), pf, league)
            res = game.win_total_probabilities(h_runs, a_runs)
            samples.append((res["home_win_prob"], 1 if hr > ar else 0,
                            h_runs + a_runs, hr + ar))
            used += 1
        print(f"  {args.season}-{month:02d}: {used} games scored")

    report(samples)


def report(samples):
    n = len(samples)
    if not n:
        print("No games scored.")
        return
    ps = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    eps = 1e-9
    logloss = -sum(y * log(max(p, eps)) + (1 - y) * log(max(1 - p, eps))
                   for p, y in zip(ps, ys)) / n
    base_rate = sum(ys) / n
    brier_base = sum((base_rate - y) ** 2 for y in ys) / n
    tot_mae = sum(abs(s[2] - s[3]) for s in samples) / n
    tot_rmse = (sum((s[2] - s[3]) ** 2 for s in samples) / n) ** 0.5
    lg_total = sum(s[3] for s in samples) / n
    naive_mae = sum(abs(lg_total - s[3]) for s in samples) / n

    print("\n" + "=" * 58)
    print(f"BACKTEST — {n} games")
    print("=" * 58)
    print("Win probability")
    print(f"  Brier score : {brier:.4f}   (baseline always-{base_rate:.0%}: {brier_base:.4f})")
    print(f"  Log-loss    : {logloss:.4f}")
    print(f"  Home win rate actual: {base_rate:.1%}")
    print("  Calibration (predicted -> actual):")
    for lo in [i / 10 for i in range(10)]:
        hi = lo + 0.1
        bucket = [(p, y) for p, y in zip(ps, ys) if lo <= p < hi]
        if bucket:
            pm = sum(p for p, _ in bucket) / len(bucket)
            am = sum(y for _, y in bucket) / len(bucket)
            print(f"    {lo:.1f}-{hi:.1f}: pred {pm:.2f}  actual {am:.2f}  (n={len(bucket)})")
    print("Totals")
    print(f"  MAE  : {tot_mae:.2f} runs   (naive league-avg {lg_total:.1f}: {naive_mae:.2f})")
    print(f"  RMSE : {tot_rmse:.2f} runs")
    print("=" * 58)


if __name__ == "__main__":
    main()
