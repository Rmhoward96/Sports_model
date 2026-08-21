"""Generate game + prop predictions for the current slate with the hybrid
Monte Carlo simulator (LIVE producer). REPLACES generate_predictions.py +
generate_props.py as the single producer of game_predictions/prop_predictions.

For every game: simulate the whole play-by-play ONCE (kernel.simulate) and derive
both the game prediction and most prop markets from that single set of sims.
The two pitcher markets that need book-realistic tail shape more than sample-noise
robustness (pitcher_ks, hits_allowed) still come from the closed-form ANALYTIC
model (model.props.pitcher_props), exactly as generate_props.py computed them.

    SIM (kernel.simulate + engine):     game score/win-prob, hits, total_bases,
                                         home_run, hrr, outs_recorded
    ANALYTIC (model.props.pitcher_props): pitcher_ks, hits_allowed

Data sources (all committed snapshots so this runs in CI without the ~1GB raw backfill):
  - profiles: assets/profiles/*.parquet, assets/advancement/mlb_advancement.parquet
  - lineups: MLB StatsAPI (confirmed when posted, else last-game projection)
  - schedule: Supabase daily_schedule if DATABASE_URL is set, else local DuckDB
  - output: Supabase game_predictions/prop_predictions if DATABASE_URL is set,
            else local DuckDB

Usage:
    uv run python scripts/generate_sim.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from sportsmodel import config, profiles, teams, venues, weather
from sportsmodel.db import get_duckdb, upsert_game_predictions, upsert_prop_predictions
from sportsmodel.ingest import mlb_lineups
from sportsmodel.model import calibration, distributions, game, props, rates
from sportsmodel.sim import engine
from sportsmodel.sim.mlb import kernel
from sportsmodel.sim.mlb.advancement import AdvancementTable
from sportsmodel.sim.mlb.inputs import build_game_spec

GAME_MODEL_VERSION = "mlb-hybrid-v1"
PROP_MODEL_VERSION = "mlb-hybrid-props-v1"
N_SIMS = 20_000
SIM_SEED = 42
_LINEUP_SIZE = 9

SIM_BATTER_MARKETS = ["hits", "total_bases", "home_run", "hrr"]
ANALYTIC_PITCHER_MARKETS = ["pitcher_ks", "hits_allowed"]

# player_prop_dists (src/sportsmodel/sim/engine.py) computes EVERY pitcher market
# it knows about (pitcher_ks, hits_allowed, outs_recorded) for each tracked starter,
# so market_max needs bounds for all three even though only outs_recorded is
# persisted from the sim path here -- pitcher_ks/hits_allowed are written from the
# ANALYTIC path below instead (same split backtest_sim_props.py's MARKET_MAX uses).
MARKET_MAX = {
    "hits": 6, "total_bases": 10, "hrr": 15, "outs_recorded": 30,
    "pitcher_ks": 15, "hits_allowed": 15,
}

SCHED_COLS = [
    "game_pk", "game_date",
    "home_team_id", "home_team_name", "away_team_id", "away_team_name",
    "home_probable_pitcher_id", "home_probable_pitcher_name",
    "away_probable_pitcher_id", "away_probable_pitcher_name",
]


def load_schedule() -> list[dict]:
    today = date.today().isoformat()
    if config.DATABASE_URL:
        from sportsmodel.db import get_postgres
        with get_postgres() as pg, pg.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(SCHED_COLS)} FROM daily_schedule WHERE game_date >= %s",
                [today],
            )
            rows = cur.fetchall()
    else:
        con = get_duckdb(read_only=True)
        rows = con.execute(
            f"SELECT {', '.join(SCHED_COLS)} FROM stg_schedule_raw WHERE game_date >= ?",
            [today],
        ).fetchall()
        con.close()
    return [dict(zip(SCHED_COLS, r)) for r in rows]


def sim_batter_rows(order, dists, lineup, lineup_source, team_name, g) -> list[dict]:
    """SIM prop rows (hits/total_bases/home_run/hrr) for one team's lineup.

    `order` is the (player_id, vec) list actually fed to the kernel (== the
    lineup, filtered to the 9 batters we had profiles for); `lineup` is the
    original (player_id, name) batting order used to recover names.
    """
    names = dict(lineup)
    out = []
    for slot, (pid, _vec) in enumerate(order, start=1):
        d = dists.get(pid)
        if d is None:
            continue
        for market in SIM_BATTER_MARKETS:
            m = d[market]
            line = props.DEFAULT_LINE[market]
            out.append({
                "game_pk": g["game_pk"], "player_id": int(pid), "market": market,
                "model_version": PROP_MODEL_VERSION, "game_date": g["game_date"],
                "player_name": names.get(pid, ""), "team_name": team_name,
                "batting_slot": slot, "projected_pa": props.SLOT_PA.get(slot, 4.1),
                "lineup_source": lineup_source,
                "projected_mean": m["mean"], "line": line,
                "prob_over": calibration.calibrate(market, distributions.prob_over_dist(m, line)),
                "dist": json.dumps(m),
            })
    return out


def sim_pitcher_outs_row(pid, pname, team_name, dists, workload_full, g) -> dict | None:
    """SIM prop row for outs_recorded (the one pitcher market the sim -- not the
    analytic path -- produces, since it needs the full-game hook simulation)."""
    d = dists.get(pid)
    if d is None:
        return None
    m = d["outs_recorded"]
    line = props.DEFAULT_LINE["outs_recorded"]
    avg_bf = workload_full[pid][0]
    return {
        "game_pk": g["game_pk"], "player_id": int(pid), "market": "outs_recorded",
        "model_version": PROP_MODEL_VERSION, "game_date": g["game_date"],
        "player_name": pname, "team_name": team_name, "batting_slot": None,
        "projected_pa": avg_bf, "lineup_source": None,
        "projected_mean": m["mean"], "line": line,
        "prob_over": calibration.calibrate("outs_recorded", distributions.prob_over_dist(m, line)),
        "dist": json.dumps(m),
    }


def analytic_pitcher_rows(pid, pvec, pname, team_name, opp_lineup, workload_full, league, g) -> list[dict]:
    """ANALYTIC prop rows (pitcher_ks, hits_allowed) for one starter vs the
    OPPOSING lineup -- exactly generate_props.py's pitcher_rows(), but limited
    to the two markets that stay analytic (outs_recorded comes from the sim)."""
    if pvec is None or pid is None or pid not in workload_full or not opp_lineup["batting_order"]:
        return []
    opp_ids = [b for b, _ in opp_lineup["batting_order"]]
    bvecs = profiles.load_batter_vectors(opp_ids)
    opp_vecs = [rates.matchup_vector(bv, pvec, league)
                for b in opp_ids if (bv := bvecs.get(b)) is not None]
    if not opp_vecs:
        return []
    avg_bf, sd_bf, avg_outs, sd_outs = workload_full[pid]
    pp = props.pitcher_props(opp_vecs, avg_bf, sd_bf ** 2, avg_outs, sd_outs)
    out = []
    for market in ANALYTIC_PITCHER_MARKETS:
        m = pp[market]
        out.append({
            "game_pk": g["game_pk"], "player_id": int(pid), "market": market,
            "model_version": PROP_MODEL_VERSION, "game_date": g["game_date"],
            "player_name": pname, "team_name": team_name, "batting_slot": None,
            "projected_pa": avg_bf, "lineup_source": opp_lineup["source"],
            "projected_mean": m["mean"], "line": m["line"],
            "prob_over": calibration.calibrate(market, m["prob_over"]),
            "dist": json.dumps(m["dist"]),
        })
    return out


def write_game_predictions(preds: list[dict]) -> None:
    if config.DATABASE_URL:
        n = upsert_game_predictions(preds)
        print(f"Upserted {n} rows into Supabase game_predictions.")
        return
    con = get_duckdb()
    con.execute("""
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_pk BIGINT, model_version TEXT, game_date DATE,
            home_team_name TEXT, away_team_name TEXT,
            home_probable_pitcher_name TEXT, away_probable_pitcher_name TEXT,
            pred_home_score REAL, pred_away_score REAL, pred_total REAL,
            pred_margin REAL, home_win_prob REAL,
            PRIMARY KEY (game_pk, model_version))
    """)
    for p in preds:
        con.execute(
            "INSERT OR REPLACE INTO game_predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [p["game_pk"], p["model_version"], p["game_date"],
             p["home_team_name"], p["away_team_name"],
             p["home_probable_pitcher_name"], p["away_probable_pitcher_name"],
             p["pred_home_score"], p["pred_away_score"], p["pred_total"],
             p["pred_margin"], p["home_win_prob"]],
        )
    con.close()
    print(f"Wrote {len(preds)} rows into local DuckDB game_predictions.")


def write_prop_predictions(rows: list[dict]) -> None:
    if config.DATABASE_URL:
        n = upsert_prop_predictions(rows)
        print(f"Upserted {n} rows into Supabase prop_predictions.")
        return
    con = get_duckdb()
    con.execute("""
        CREATE TABLE IF NOT EXISTS prop_predictions (
            game_pk BIGINT, player_id BIGINT, market TEXT, model_version TEXT,
            game_date DATE, player_name TEXT, team_name TEXT, batting_slot INT,
            projected_pa REAL, lineup_source TEXT, projected_mean REAL, line REAL,
            prob_over REAL, dist JSON,
            PRIMARY KEY (game_pk, player_id, market, model_version))
    """)
    cols = ["game_pk", "player_id", "market", "model_version", "game_date",
            "player_name", "team_name", "batting_slot", "projected_pa",
            "lineup_source", "projected_mean", "line", "prob_over", "dist"]
    for r in rows:
        con.execute(
            f"INSERT OR REPLACE INTO prop_predictions ({','.join(cols)}) "
            f"VALUES ({','.join(['?'] * len(cols))})", [r[c] for c in cols])
    con.close()
    print(f"Wrote {len(rows)} rows into local DuckDB prop_predictions.")


def main() -> None:
    games = load_schedule()
    print(f"{len(games)} games on today's slate")

    ids = [g["home_probable_pitcher_id"] for g in games] + \
          [g["away_probable_pitcher_id"] for g in games]
    pitchers = profiles.load_pitcher_vectors(ids)
    workload_full = profiles.load_pitcher_workload(ids)  # {pid: (avg_bf, sd_bf, avg_outs, sd_outs)}
    bullpen = profiles.load_team_bullpen_vectors()
    defense = profiles.load_team_defense()
    park = profiles.load_park_factors()
    league = profiles.load_league_vector()
    adv = AdvancementTable.from_rows(profiles.load_advancement_rows())
    rng = np.random.default_rng(SIM_SEED)

    game_rows, prop_rows = [], []
    skipped_starter, skipped_lineup = 0, 0

    for g in games:
        home_ab = teams.statcast_abbrev(g["home_team_id"])
        away_ab = teams.statcast_abbrev(g["away_team_id"])
        home_pid, away_pid = g["home_probable_pitcher_id"], g["away_probable_pitcher_id"]
        home_sp, away_sp = pitchers.get(home_pid), pitchers.get(away_pid)
        wl_home, wl_away = workload_full.get(home_pid), workload_full.get(away_pid)
        if None in (home_sp, away_sp, wl_home, wl_away):
            skipped_starter += 1
            continue

        lu = mlb_lineups.lineups_for_game(
            g["game_pk"], g["home_team_id"], g["away_team_id"], str(g["game_date"]))
        home_ids = [pid for pid, _ in lu["home"]["batting_order"]]
        away_ids = [pid for pid, _ in lu["away"]["batting_order"]]
        bvecs = profiles.load_batter_vectors(home_ids + away_ids)
        home_order = [(pid, bvecs[pid]) for pid in home_ids if pid in bvecs]
        away_order = [(pid, bvecs[pid]) for pid in away_ids if pid in bvecs]
        if len(home_order) != _LINEUP_SIZE or len(away_order) != _LINEUP_SIZE:
            skipped_lineup += 1
            continue

        pf = park.get(home_ab, 1.0)
        hr_mult = 1.0
        p = venues.park(home_ab)
        if p and p[2]:
            temp = weather.fetch_game_temp(p[0], p[1], g["game_date"])
            if temp is not None:
                hr_mult = game.weather_hr_multiplier(temp)

        context = {
            "home_pf": pf, "hr_mult": hr_mult,
            "home_def": defense.get(home_ab, 1.0), "away_def": defense.get(away_ab, 1.0),
        }
        workload_spec = {
            home_pid: (wl_home[2], wl_home[3]),
            away_pid: (wl_away[2], wl_away[3]),
        }
        spec = build_game_spec(
            home_order, away_order, home_sp, away_sp,
            bullpen.get(home_ab), bullpen.get(away_ab),
            workload=workload_spec, context=context, league=league, adv=adv,
            home_starter_id=home_pid, away_starter_id=away_pid,
        )
        sims = kernel.simulate(spec, N_SIMS, rng)

        # --- GAME prediction (sim) ---
        res = engine.pred_scores(sims)
        res["home_win_prob"] = calibration.calibrate("win_prob", res["home_win_prob"])
        game_rows.append({
            "game_pk": g["game_pk"], "model_version": GAME_MODEL_VERSION,
            "game_date": g["game_date"],
            "home_team_name": g["home_team_name"], "away_team_name": g["away_team_name"],
            "home_probable_pitcher_name": g["home_probable_pitcher_name"],
            "away_probable_pitcher_name": g["away_probable_pitcher_name"],
            **res,
        })

        # --- SIM props: batter markets + starter outs_recorded ---
        dists = engine.player_prop_dists(sims, MARKET_MAX)
        prop_rows += sim_batter_rows(
            home_order, dists, lu["home"]["batting_order"], lu["home"]["source"],
            g["home_team_name"], g)
        prop_rows += sim_batter_rows(
            away_order, dists, lu["away"]["batting_order"], lu["away"]["source"],
            g["away_team_name"], g)
        r = sim_pitcher_outs_row(home_pid, g["home_probable_pitcher_name"],
                                 g["home_team_name"], dists, workload_full, g)
        if r:
            prop_rows.append(r)
        r = sim_pitcher_outs_row(away_pid, g["away_probable_pitcher_name"],
                                 g["away_team_name"], dists, workload_full, g)
        if r:
            prop_rows.append(r)

        # --- ANALYTIC props: pitcher_ks, hits_allowed (each starter vs OPPOSING lineup) ---
        prop_rows += analytic_pitcher_rows(
            home_pid, home_sp, g["home_probable_pitcher_name"], g["home_team_name"],
            lu["away"], workload_full, league, g)
        prop_rows += analytic_pitcher_rows(
            away_pid, away_sp, g["away_probable_pitcher_name"], g["away_team_name"],
            lu["home"], workload_full, league, g)

    print(f"predicted {len(game_rows)} games, skipped {skipped_starter} (missing starter/"
          f"workload) + {skipped_lineup} (missing/incomplete lineup)")
    print(f"{len(prop_rows)} prop rows")
    if game_rows:
        write_game_predictions(game_rows)
    if prop_rows:
        write_prop_predictions(prop_rows)


if __name__ == "__main__":
    main()
