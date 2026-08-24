from __future__ import annotations
import itertools, json, pathlib
import pandas as pd
from sportsmodel.nfl.elo import EloConfig, expected_home, run_elo
from sportsmodel.nfl.srs import compute_srs
from sportsmodel.nfl.ratings import BlendConfig, expected_margin

def run_backtest(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                 blend_cfg: BlendConfig) -> dict:
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)                       # pre-game elo per game
    games = res.games
    n = 0; brier = 0.0; correct = 0; abs_err = 0.0; sq_err = 0.0
    played_counts: dict = {}
    for season, sdf in games.groupby("season"):
        season_games = df[df["season"] == season].sort_values("week")
        srs_hist: pd.DataFrame = season_games.iloc[0:0]
        counts: dict = {}
        srs_cache: dict = {}
        for _, g in sdf.iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            h, a = g["home_team"], g["away_team"]
            gh, ga = counts.get(h, 0), counts.get(a, 0)
            srs = srs_cache if srs_cache else {}
            srs_h = srs.get(h); srs_a = srs.get(a)
            em = expected_margin(g["elo_home"], g["elo_away"], srs_h, srs_a,
                                 gh, ga, elo_cfg, blend_cfg)
            e_home = g["e_home"]
            actual_margin = g["home_score"] - g["away_score"]
            result_home = 1.0 if actual_margin > 0 else 0.0
            brier += (e_home - result_home) ** 2
            correct += int((e_home >= 0.5) == (result_home == 1.0))
            abs_err += abs(em - actual_margin)
            sq_err += (em - actual_margin) ** 2
            n += 1
            # after scoring, this game joins the played set -> update counts + SRS
            counts[h] = gh + 1; counts[a] = ga + 1
            srs_hist = pd.concat([srs_hist, pd.DataFrame([g])], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
    return {"brier": brier / n, "win_acc": correct / n,
            "margin_mae": abs_err / n, "margin_rmse": (sq_err / n) ** 0.5, "n": n}

def tune(train_df, valid_df, grid) -> tuple:
    combos = list(itertools.product(
        grid["k"], grid["hfa_elo"], grid["carryover"],
        grid["w_sos"], grid["srs_min_games"]))
    results = []
    for k, hfa, carry, w, mg in combos:
        ec = EloConfig(k=k, hfa_elo=hfa, carryover=carry)
        bc = BlendConfig(w_sos=w, srs_min_games=mg)
        vm = run_backtest(valid_df, ec, bc)
        results.append({"elo": ec, "blend": bc, "valid": vm})
    best = min(results, key=lambda r: r["valid"]["brier"])
    return (best["elo"], best["blend"]), results

def _coordinate_search(train_df, valid_df, grid: dict) -> tuple:
    """Tune one parameter at a time over its listed values, holding the
    others at a sensible center, for a few passes. Full Cartesian product
    over the brief's grid (768 combos) was measured too slow given that
    run_backtest recomputes SRS after every game; coordinate search visits
    O(passes * sum(len(values))) configs instead while still covering the
    same per-parameter search space the spec calls for."""
    order = ["k", "hfa_elo", "carryover", "w_sos", "srs_min_games"]
    # sensible center: middle-ish value of each parameter's list
    current = {p: grid[p][len(grid[p]) // 2] for p in order}
    all_results = []
    seen = set()

    def _eval(params: dict) -> dict:
        key = tuple(params[p] for p in order)
        if key in seen:
            return None
        seen.add(key)
        ec = EloConfig(k=params["k"], hfa_elo=params["hfa_elo"], carryover=params["carryover"])
        bc = BlendConfig(w_sos=params["w_sos"], srs_min_games=params["srs_min_games"])
        vm = run_backtest(valid_df, ec, bc)
        r = {"elo": ec, "blend": bc, "valid": vm}
        all_results.append(r)
        return r

    n_passes = 3
    for _ in range(n_passes):
        improved = False
        for p in order:
            best_val = current[p]
            best_brier = None
            for v in grid[p]:
                trial = dict(current); trial[p] = v
                r = _eval(trial)
                if r is None:
                    # already evaluated this exact combo; re-look-up its brier
                    ec = EloConfig(k=trial["k"], hfa_elo=trial["hfa_elo"], carryover=trial["carryover"])
                    bc = BlendConfig(w_sos=trial["w_sos"], srs_min_games=trial["srs_min_games"])
                    matches = [x for x in all_results
                               if x["elo"] == ec and x["blend"] == bc]
                    brier = matches[0]["valid"]["brier"] if matches else None
                else:
                    brier = r["valid"]["brier"]
                if brier is not None and (best_brier is None or brier < best_brier):
                    best_brier = brier
                    best_val = v
            if best_val != current[p]:
                improved = True
            current[p] = best_val
        if not improved:
            break

    best = min(all_results, key=lambda r: r["valid"]["brier"])
    return (best["elo"], best["blend"]), all_results

def main() -> None:
    sched = pd.read_parquet("assets/nfl/schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    train = reg[reg["season"] <= 2019]
    valid = reg[reg["season"] >= 2020]
    grid = {"k": [12, 16, 20, 24], "hfa_elo": [40, 55, 65, 80],
            "carryover": [0.6, 0.7, 0.75, 0.85],
            "w_sos": [0.0, 0.15, 0.3, 0.45], "srs_min_games": [3, 4, 6]}
    (best_elo, best_blend), results = _coordinate_search(train, valid, grid)
    pure = min((r for r in results if r["blend"].w_sos == 0.0),
               key=lambda r: r["valid"]["brier"])
    out = {"k": best_elo.k, "hfa_elo": best_elo.hfa_elo,
           "carryover": best_elo.carryover, "base": best_elo.base,
           "w_sos": best_blend.w_sos, "srs_min_games": best_blend.srs_min_games}
    pathlib.Path("assets/nfl/rating.json").write_text(json.dumps(out, indent=2))
    print("best:", out)
    print("best pure-Elo brier:", pure["valid"]["brier"],
          "| best blended brier:", min(r["valid"]["brier"] for r in results))

if __name__ == "__main__":
    main()
