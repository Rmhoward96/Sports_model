"""Generate player-prop projections (Hits, Total Bases, Home Run) for today's slate.

For each batter in each lineup: build the in-game per-PA vector = odds-ratio(batter,
opposing starter, league), then opposing defense, weather, and park; project PA from
lineup slot; compute prop mean + P(over standard line). Lineups are confirmed when
posted, else projected from the team's last game (project-early / confirm-late).

Data: committed profile snapshots + lineups/schedule from MLB StatsAPI.
Output: Supabase prop_predictions if DATABASE_URL is set, else local DuckDB.

Usage:
    uv run python scripts/generate_props.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config, profiles, teams, venues, weather
from sportsmodel.db import get_duckdb, upsert_prop_predictions
from sportsmodel.ingest import mlb_lineups
from sportsmodel.model import game, props, rates

MODEL_VERSION = "mlb-props-v1"
HITTER_MARKETS = ["hits", "total_bases", "home_run", "hrr"]
PITCHER_MARKETS = ["pitcher_ks", "hits_allowed", "outs_recorded"]
SCHED_COLS = [
    "game_pk", "game_date", "home_team_id", "home_team_name", "away_team_id",
    "away_team_name", "home_probable_pitcher_id", "home_probable_pitcher_name",
    "away_probable_pitcher_id", "away_probable_pitcher_name",
]


def load_schedule() -> list[dict]:
    today = date.today().isoformat()
    if config.DATABASE_URL:
        from sportsmodel.db import get_postgres
        with get_postgres() as pg, pg.cursor() as cur:
            cur.execute(f"SELECT {', '.join(SCHED_COLS)} FROM daily_schedule WHERE game_date = %s", [today])
            rows = cur.fetchall()
    else:
        con = get_duckdb(read_only=True)
        rows = con.execute(f"SELECT {', '.join(SCHED_COLS)} FROM stg_schedule_raw WHERE game_date = ?", [today]).fetchall()
        con.close()
    return [dict(zip(SCHED_COLS, r)) for r in rows]


def side_rows(lineup, opp_sp, opp_def, pf, hr_mult, team_name, g) -> list[dict]:
    """Prop rows for one team's batters vs the opposing starter + context."""
    if opp_sp is None or not lineup["batting_order"]:
        return []
    ids = [pid for pid, _ in lineup["batting_order"]]
    bvecs = profiles.load_batter_vectors(ids)
    league = _LEAGUE
    out = []
    for slot, (pid, name) in enumerate(lineup["batting_order"], start=1):
        bvec = bvecs.get(pid)
        if bvec is None:
            continue
        vec = rates.matchup_vector(bvec, opp_sp, league)
        vec = game.apply_bip_defense(vec, opp_def)
        if hr_mult != 1.0:
            vec = game.apply_hr_multiplier(vec, hr_mult)
        if pf != 1.0:
            vec = game.apply_park_to_vector(vec, pf)
        proj = props.batter_props(vec, slot)
        for market in HITTER_MARKETS:
            m = proj[market]
            out.append({
                "game_pk": g["game_pk"], "player_id": pid, "market": market,
                "model_version": MODEL_VERSION, "game_date": g["game_date"],
                "player_name": name, "team_name": team_name, "batting_slot": slot,
                "projected_pa": proj["projected_pa"], "lineup_source": lineup["source"],
                "projected_mean": m["mean"], "line": m["line"], "prob_over": m["prob_over"],
            })
    return out


def pitcher_rows(pid, pvec, pname, team_name, opp_lineup, workload, g) -> list[dict]:
    """Pitcher-prop rows for one starter vs the opposing lineup."""
    if pvec is None or pid is None or pid not in workload or not opp_lineup["batting_order"]:
        return []
    opp_ids = [b for b, _ in opp_lineup["batting_order"]]
    bvecs = profiles.load_batter_vectors(opp_ids)
    opp_vecs = [rates.matchup_vector(bv, pvec, _LEAGUE)
                for b in opp_ids if (bv := bvecs.get(b)) is not None]
    if not opp_vecs:
        return []
    avg_bf, sd_bf, avg_outs, sd_outs = workload[pid]
    pp = props.pitcher_props(opp_vecs, avg_bf, sd_bf ** 2, avg_outs, sd_outs)
    out = []
    for market in PITCHER_MARKETS:
        m = pp[market]
        out.append({
            "game_pk": g["game_pk"], "player_id": int(pid), "market": market,
            "model_version": MODEL_VERSION, "game_date": g["game_date"],
            "player_name": pname, "team_name": team_name, "batting_slot": None,
            "projected_pa": avg_bf, "lineup_source": opp_lineup["source"],
            "projected_mean": m["mean"], "line": m["line"], "prob_over": m["prob_over"],
        })
    return out


_LEAGUE: dict = {}


def main() -> None:
    global _LEAGUE
    games = load_schedule()
    print(f"{len(games)} games on today's slate")

    pids = [g["home_probable_pitcher_id"] for g in games] + [g["away_probable_pitcher_id"] for g in games]
    pitchers = profiles.load_pitcher_vectors(pids)
    workload = profiles.load_pitcher_workload(pids)
    defense = profiles.load_team_defense()
    park = profiles.load_park_factors()
    _LEAGUE = profiles.load_league_vector()

    rows, confirmed, projected = [], 0, 0
    for g in games:
        home_ab = teams.statcast_abbrev(g["home_team_id"])
        away_ab = teams.statcast_abbrev(g["away_team_id"])
        home_sp = pitchers.get(g["home_probable_pitcher_id"])
        away_sp = pitchers.get(g["away_probable_pitcher_id"])
        pf = park.get(home_ab, 1.0)
        hr_mult = 1.0
        p = venues.park(home_ab)
        if p and p[2]:
            temp = weather.fetch_game_temp(p[0], p[1], g["game_date"])
            if temp is not None:
                hr_mult = game.weather_hr_multiplier(temp)
        lu = mlb_lineups.lineups_for_game(
            g["game_pk"], g["home_team_id"], g["away_team_id"], str(g["game_date"]))
        for side, opp_sp, opp_ab, tname in (
            ("home", away_sp, away_ab, g["home_team_name"]),
            ("away", home_sp, home_ab, g["away_team_name"]),
        ):
            r = side_rows(lu[side], opp_sp, defense.get(opp_ab, 1.0), pf, hr_mult, tname, g)
            rows += r
            if r:
                if lu[side]["source"] == "confirmed":
                    confirmed += 1
                else:
                    projected += 1
        # Pitcher props: each starter vs the OPPOSING lineup.
        rows += pitcher_rows(g["home_probable_pitcher_id"], home_sp,
                             g["home_probable_pitcher_name"], g["home_team_name"],
                             lu["away"], workload, g)
        rows += pitcher_rows(g["away_probable_pitcher_id"], away_sp,
                             g["away_probable_pitcher_name"], g["away_team_name"],
                             lu["home"], workload, g)

    print(f"{len(rows)} prop rows; lineups: {confirmed} confirmed / {projected} projected sides")
    if rows:
        write_rows(rows)


def write_rows(rows: list[dict]) -> None:
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
            prob_over REAL, PRIMARY KEY (game_pk, player_id, market, model_version))
    """)
    cols = ["game_pk", "player_id", "market", "model_version", "game_date",
            "player_name", "team_name", "batting_slot", "projected_pa",
            "lineup_source", "projected_mean", "line", "prob_over"]
    for r in rows:
        con.execute(
            f"INSERT OR REPLACE INTO prop_predictions ({','.join(cols)}) "
            f"VALUES ({','.join(['?']*len(cols))})", [r[c] for c in cols])
    con.close()
    print(f"Wrote {len(rows)} rows into local DuckDB prop_predictions.")


if __name__ == "__main__":
    main()
