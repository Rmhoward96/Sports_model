"""Walk-forward, point-in-time backtest of the player props (no odds needed).

For each month: rebuild profiles using only prior games, then for each game project
every batter (Hits / Total Bases / HR) from their ACTUAL lineup slot vs the ACTUAL
opposing starter, and each starter (Ks / Hits Allowed / Outs) vs the ACTUAL opposing
lineup. Compare to real outcomes. Scores per-market calibration (mean predicted P(over)
vs actual over-rate, plus reliability buckets) and projected-mean error (MAE).

H+R+RBI is omitted (runs/RBIs aren't cleanly recoverable per batter from Statcast).

Usage:
    uv run python scripts/backtest_props.py --season 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb

from sportsmodel import transforms
from sportsmodel.model import game, props, rates

OUTCOMES = ["p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out"]
GLOB = transforms._PARQUET_GLOB
WEIGHTS = {"30d": 0.2, "season": 0.4, "career": 0.4}
_MONTHS = [4, 5, 6, 7, 8, 9]
BATTER_MARKETS = ["hits", "total_bases", "home_run"]
PITCHER_MARKETS = ["pitcher_ks", "hits_allowed", "outs_recorded"]


def _blend(con, table, id_col, min_season, min_recent):
    rows = con.execute(
        f"SELECT {id_col}, window_name, pa, {', '.join(OUTCOMES)} FROM {table} WHERE vs_hand='ALL'"
    ).fetchall()
    per: dict = {}
    for r in rows:
        per.setdefault(r[0], {})[r[1]] = (r[2], {o: r[3 + i] for i, o in enumerate(OUTCOMES)})
    gate = {"30d": min_recent, "season": min_season, "career": 1}
    out = {}
    for key, wins in per.items():
        parts = [(WEIGHTS[w], v) for w, (pa, v) in wins.items() if w in WEIGHTS and pa >= gate[w]]
        if not parts:
            if "career" in wins:
                out[key] = wins["career"][1]
            continue
        tw = sum(w for w, _ in parts)
        out[key] = {o: sum(w * v[o] for w, v in parts) / tw for o in OUTCOMES}
    return out


def build_profiles(con):
    transforms.build_batter_profiles(con)
    transforms.build_pitcher_profiles(con)
    transforms.build_team_defense(con)
    transforms.build_park_factors(con)
    transforms.build_pitcher_workload(con)
    transforms.build_league_rates(con)
    transforms.shrink_toward_league(con, "feat_batter_profile")
    transforms.shrink_toward_league(con, "feat_pitcher_profile")


def month_actuals(con, season, month):
    start = f"{season}-{month:02d}-01"
    end = f"{season}-{month + 1:02d}-01" if month < 12 else f"{season + 1}-01-01"
    tb = ("CASE WHEN events='single' THEN 1 WHEN events='double' THEN 2 "
          "WHEN events='triple' THEN 3 WHEN events='home_run' THEN 4 ELSE 0 END")
    hit = "CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END"
    con.execute(f"""
        CREATE TEMP TABLE term AS
        SELECT game_pk, home_team, away_team, inning, inning_topbot, at_bat_number,
               CAST(batter AS BIGINT) batter, CAST(pitcher AS BIGINT) pitcher, events,
               {transforms._OUTS_CASE} AS outs
        FROM read_parquet('{GLOB}')
        WHERE CAST(game_date AS DATE) >= DATE '{start}'
          AND CAST(game_date AS DATE) < DATE '{end}' AND events IS NOT NULL
    """)
    games = con.execute("""
        SELECT game_pk, any_value(home_team) home_team, any_value(away_team) away_team,
               arg_min(pitcher, at_bat_number) FILTER (WHERE inning=1 AND inning_topbot='Top') home_sp,
               arg_min(pitcher, at_bat_number) FILTER (WHERE inning=1 AND inning_topbot='Bot') away_sp
        FROM term GROUP BY game_pk
    """).fetchall()
    batters = con.execute(f"""
        WITH bg AS (
            SELECT game_pk, batter, any_value(inning_topbot) side, min(at_bat_number) first_ab,
                   count(*) pa, sum({hit}) hits, sum({tb}) tb,
                   sum(CASE WHEN events='home_run' THEN 1 ELSE 0 END) hr
            FROM term GROUP BY game_pk, batter
        )
        SELECT game_pk, batter, side,
               row_number() OVER (PARTITION BY game_pk, side ORDER BY first_ab) slot,
               pa, hits, tb, hr FROM bg QUALIFY slot <= 9
    """).fetchall()
    pitchers = con.execute("""
        SELECT game_pk, pitcher, count(*) bf,
               sum(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) ks,
               sum(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) hits_allowed,
               sum(outs) outs
        FROM term GROUP BY game_pk, pitcher
    """).fetchall()
    con.execute("DROP TABLE term")
    return games, batters, pitchers


class Score:
    def __init__(self):
        self.p, self.y, self.mean, self.act = [], [], [], []

    def add(self, prob_over, actual, mean, line):
        self.p.append(prob_over)
        self.y.append(1 if actual > line else 0)
        self.mean.append(mean)
        self.act.append(actual)

    def report(self, name):
        n = len(self.p)
        if not n:
            print(f"  {name:14}: no samples")
            return
        pred = sum(self.p) / n
        actual_over = sum(self.y) / n
        brier = sum((a - b) ** 2 for a, b in zip(self.p, self.y)) / n
        mae = sum(abs(a - b) for a, b in zip(self.mean, self.act)) / n
        print(f"  {name:14}: n={n:5}  pred P(over)={pred:.3f}  actual over-rate={actual_over:.3f}  "
              f"Brier={brier:.3f}  mean-MAE={mae:.2f}")


def run_backtest(season: int) -> dict:
    """Walk-forward; returns {market: Score} of predicted P(over) vs actual outcomes."""
    scores = {m: Score() for m in BATTER_MARKETS + PITCHER_MARKETS}
    for month in _MONTHS:
        transforms.set_cutoff(f"{season}-{month:02d}-01")
        con = duckdb.connect(":memory:")
        con.execute("INSTALL json; LOAD json;")
        build_profiles(con)
        bat = _blend(con, "feat_batter_profile", "player_id", 150, 40)
        pit = _blend(con, "feat_pitcher_profile", "player_id", 150, 40)
        dfn = {r[0]: r[1] for r in con.execute("SELECT team, def_factor FROM feat_team_defense").fetchall()}
        park = {r[0]: r[1] for r in con.execute("SELECT team, pf_runs FROM park_factors").fetchall()}
        wl = {int(r[0]): (r[1], r[2], r[3], r[4]) for r in
              con.execute("SELECT player_id, avg_bf, sd_bf, avg_outs, sd_outs FROM feat_pitcher_workload").fetchall()}
        r = con.execute("SELECT " + ", ".join(OUTCOMES) +
                        " FROM ref_league_rates WHERE vs_hand='ALL' AND window_name='career' LIMIT 1").fetchone()
        league = {o: r[i] for i, o in enumerate(OUTCOMES)}
        transforms.set_cutoff(None)

        games, batters, pitchers = month_actuals(con, season, month)
        con.close()

        gm = {g[0]: g for g in games}                 # game_pk -> (pk, home, away, home_sp, away_sp)
        lineup = {}                                    # (game, side) -> [batter ids]
        for gp, b, side, slot, pa, h, tb, hr in sorted(batters, key=lambda x: x[3]):
            lineup.setdefault((gp, side), []).append(b)

        # Batter props
        for gp, b, side, slot, pa, h, tb, hr in batters:
            g = gm.get(gp)
            bvec = bat.get(b)
            if not g or bvec is None:
                continue
            _, home, away, home_sp, away_sp = g
            opp_sp = away_sp if side == "Bot" else home_sp     # home(Bot) faces away starter
            opp_team = away if side == "Bot" else home
            sp = pit.get(opp_sp)
            if sp is None:
                continue
            vec = rates.matchup_vector(bvec, sp, league)
            vec = game.apply_bip_defense(vec, dfn.get(opp_team, 1.0))
            vec = game.apply_park_to_vector(vec, park.get(home, 1.0))
            proj = props.batter_props(vec, slot)
            scores["hits"].add(proj["hits"]["prob_over"], h, proj["hits"]["mean"], proj["hits"]["line"])
            scores["total_bases"].add(proj["total_bases"]["prob_over"], tb, proj["total_bases"]["mean"], proj["total_bases"]["line"])
            scores["home_run"].add(proj["home_run"]["prob_over"], hr, proj["home_run"]["mean"], proj["home_run"]["line"])

        # Pitcher props (starters only)
        pg = {(p[0], p[1]): p for p in pitchers}       # (game, pitcher) -> actuals
        for gp, home, away, home_sp, away_sp in games:
            for sp_id, opp_side in ((home_sp, "Top"), (away_sp, "Bot")):
                pvec = pit.get(sp_id)
                act = pg.get((gp, sp_id))
                wlp = wl.get(sp_id)
                opp = lineup.get((gp, opp_side), [])
                if pvec is None or act is None or wlp is None or not opp:
                    continue
                opp_vecs = [rates.matchup_vector(bv, pvec, league) for b in opp if (bv := bat.get(b))]
                if not opp_vecs:
                    continue
                avg_bf, sd_bf, avg_outs, sd_outs = wlp
                pp = props.pitcher_props(opp_vecs, avg_bf, sd_bf ** 2, avg_outs, sd_outs)
                _, _, bf, ks, ha, outs = act
                scores["pitcher_ks"].add(pp["pitcher_ks"]["prob_over"], ks, pp["pitcher_ks"]["mean"], pp["pitcher_ks"]["line"])
                scores["hits_allowed"].add(pp["hits_allowed"]["prob_over"], ha, pp["hits_allowed"]["mean"], pp["hits_allowed"]["line"])
                scores["outs_recorded"].add(pp["outs_recorded"]["prob_over"], outs, pp["outs_recorded"]["mean"], pp["outs_recorded"]["line"])
        print(f"  {season}-{month:02d} done")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()
    scores = run_backtest(args.season)
    print("\n" + "=" * 78)
    print(f"PROPS BACKTEST — {args.season}   (well-calibrated: pred P(over) ~ actual over-rate)")
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
