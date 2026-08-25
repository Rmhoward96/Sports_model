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
    """Select by TRAIN-span margin MAE, not Brier. Brier/win_acc are computed
    from e_home (run_backtest uses g["e_home"] straight from run_elo), which
    is pure-Elo and never touched by blend_cfg/w_sos -- Brier is therefore
    identical across every w_sos at a fixed (k, hfa_elo, carryover) and can't
    discriminate the SoS blend. margin_mae (fed by ratings.expected_margin,
    which the blend does move) is the metric that can actually select
    w_sos/srs_min_games. Selecting on train_df (rather than valid_df) also
    keeps the tuning step from peeking at the validation span."""
    combos = list(itertools.product(
        grid["k"], grid["hfa_elo"], grid["carryover"],
        grid["w_sos"], grid["srs_min_games"]))
    results = []
    for k, hfa, carry, w, mg in combos:
        ec = EloConfig(k=k, hfa_elo=hfa, carryover=carry)
        bc = BlendConfig(w_sos=w, srs_min_games=mg)
        tm = run_backtest(train_df, ec, bc)
        vm = run_backtest(valid_df, ec, bc)
        results.append({"elo": ec, "blend": bc, "train": tm, "valid": vm})
    best = min(results, key=lambda r: r["train"]["margin_mae"])
    return (best["elo"], best["blend"]), results

def _coordinate_search(train_df, grid: dict) -> tuple:
    """Tune one parameter at a time over its listed values, holding the
    others at a sensible center, for a few passes, SELECTING ON TRAIN-SPAN
    MARGIN MAE (not Brier -- see the note on tune() above: Brier/win_acc are
    blend-invariant by construction, so only margin_mae can actually surface
    whether w_sos/srs_min_games help). Full Cartesian product over the
    brief's grid (768 combos) was measured too slow given that run_backtest
    recomputes SRS after every game; coordinate search visits
    O(passes * sum(len(values))) configs instead while still covering the
    same per-parameter search space the spec calls for.

    Returns ((best_elo, best_blend), all_results). The pure-Elo counterfactual
    is NOT independently selected here -- main() constructs it by holding
    best_elo's (k, hfa_elo, carryover) fixed and only zeroing w_sos, so the
    "does SoS help" comparison is causal (same Elo params) rather than two
    independently-optimized configs that happen (or don't) to share params."""
    order = ["k", "hfa_elo", "carryover", "w_sos", "srs_min_games"]
    # sensible center: middle-ish value of each parameter's list
    current = {p: grid[p][len(grid[p]) // 2] for p in order}
    all_results = []
    seen: dict = {}

    def _eval(params: dict) -> dict:
        key = tuple(params[p] for p in order)
        if key in seen:
            return seen[key]
        ec = EloConfig(k=params["k"], hfa_elo=params["hfa_elo"], carryover=params["carryover"])
        bc = BlendConfig(w_sos=params["w_sos"], srs_min_games=params["srs_min_games"])
        tm = run_backtest(train_df, ec, bc)
        r = {"elo": ec, "blend": bc, "train": tm}
        seen[key] = r
        all_results.append(r)
        return r

    n_passes = 3
    for _ in range(n_passes):
        improved = False
        for p in order:
            best_val = current[p]
            best_mae = None
            for v in grid[p]:
                trial = dict(current); trial[p] = v
                r = _eval(trial)
                mae = r["train"]["margin_mae"]
                if best_mae is None or mae < best_mae:
                    best_mae = mae
                    best_val = v
            if best_val != current[p]:
                improved = True
            current[p] = best_val
        if not improved:
            break

    best = min(all_results, key=lambda r: r["train"]["margin_mae"])
    return (best["elo"], best["blend"]), all_results

def _team_win_pct_by_season(games_df: pd.DataFrame) -> dict:
    """{(season, team): win_pct}, ties counted as 0.5 wins, over graded games."""
    out: dict = {}
    for season, sdf in games_df.groupby("season"):
        wins: dict = {}; totals: dict = {}
        for _, g in sdf.iterrows():
            hs, as_ = g["home_score"], g["away_score"]
            if pd.isna(hs) or pd.isna(as_):
                continue
            h, a = g["home_team"], g["away_team"]
            totals[h] = totals.get(h, 0) + 1
            totals[a] = totals.get(a, 0) + 1
            if hs > as_:
                wins[h] = wins.get(h, 0) + 1
            elif hs < as_:
                wins[a] = wins.get(a, 0) + 1
            else:
                wins[h] = wins.get(h, 0) + 0.5
                wins[a] = wins.get(a, 0) + 0.5
        for t, tot in totals.items():
            out[(season, t)] = wins.get(t, 0) / tot
    return out

def naive_baselines(reg_df: pd.DataFrame, train_df: pd.DataFrame,
                     valid_df: pd.DataFrame) -> dict:
    """Three naive baselines evaluated on the VALIDATION span, computed in
    code (rather than by hand) so the report's numbers are reproducible from
    committed code:
      1. home-always: predict home wins every time (win_acc + Brier at p=1.0)
      2. prior-season win%: p_home = 0.5 + (home_prior_wp - away_prior_wp)/2,
         clipped to [0.01, 0.99]; teams with no prior-season record default
         to 0.5
      3. naive margin: always predict a constant margin equal to the mean
         home margin over the TRAIN span (2002-2019) -- gives a margin MAE
         baseline to compare the model's margin_mae against, since the
         acceptance bar is "beat baselines on Brier AND margin MAE."
    """
    v = valid_df.dropna(subset=["home_score", "away_score"]).copy()
    n = len(v)
    home_win = (v["home_score"] > v["away_score"]).astype(float)

    home_always = {
        "win_acc": float(home_win.mean()),
        "brier": float(((1.0 - home_win) ** 2).mean()),
        "n": n,
    }

    wp = _team_win_pct_by_season(reg_df)
    p_home_vals = []
    for _, g in v.iterrows():
        season = g["season"]; h, a = g["home_team"], g["away_team"]
        ph = wp.get((season - 1, h), 0.5)
        pa = wp.get((season - 1, a), 0.5)
        p = 0.5 + (ph - pa) / 2.0
        p_home_vals.append(min(max(p, 0.01), 0.99))
    p_home = pd.Series(p_home_vals, index=v.index)
    prior_season_win_pct = {
        "win_acc": float(((p_home >= 0.5) == (home_win == 1.0)).mean()),
        "brier": float(((p_home - home_win) ** 2).mean()),
        "n": n,
    }

    mean_margin = float((train_df["home_score"] - train_df["away_score"]).mean())
    actual_margin = v["home_score"] - v["away_score"]
    naive_margin = {
        "constant_margin": mean_margin,
        "margin_mae": float((actual_margin - mean_margin).abs().mean()),
        "margin_rmse": float(((actual_margin - mean_margin) ** 2).mean() ** 0.5),
        "n": n,
    }

    return {"home_always": home_always, "prior_season_win_pct": prior_season_win_pct,
            "naive_margin": naive_margin}

def main() -> None:
    sched = pd.read_parquet("assets/nfl/schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    train = reg[reg["season"] <= 2019]
    valid = reg[reg["season"] >= 2020]
    grid = {"k": [12, 16, 20, 24], "hfa_elo": [40, 55, 65, 80],
            "carryover": [0.6, 0.7, 0.75, 0.85],
            "w_sos": [0.0, 0.15, 0.3, 0.45], "srs_min_games": [3, 4, 6]}

    (best_elo, best_blend), results = _coordinate_search(train, grid)

    # Enforced-fair OOS verdict: the pure-Elo counterfactual reuses best_elo's
    # (k, hfa_elo, carryover) exactly -- only w_sos is zeroed -- so "does SoS
    # help" is a causal, same-Elo-params comparison, not two independently
    # selected configs that happen to share (or not share) hyperparameters.
    pure_elo = best_elo
    pure_blend = BlendConfig(w_sos=0.0, srs_min_games=best_blend.srs_min_games)

    blend_valid = run_backtest(valid, best_elo, best_blend)
    pure_valid = run_backtest(valid, pure_elo, pure_blend)
    blend_wins = blend_valid["margin_mae"] < pure_valid["margin_mae"]   # strict: a tie does not ship the blend

    if blend_wins:
        final_elo, final_blend, final_valid = best_elo, best_blend, blend_valid
    else:
        final_elo, final_blend, final_valid = pure_elo, pure_blend, pure_valid

    out = {"k": final_elo.k, "hfa_elo": final_elo.hfa_elo,
           "carryover": final_elo.carryover, "base": final_elo.base,
           "w_sos": final_blend.w_sos, "srs_min_games": final_blend.srs_min_games}
    pathlib.Path("assets/nfl/rating.json").write_text(json.dumps(out, indent=2) + "\n")

    baselines = naive_baselines(reg, train, valid)

    print("best blended (train-selected, margin_mae):", {
        "k": best_elo.k, "hfa_elo": best_elo.hfa_elo, "carryover": best_elo.carryover,
        "w_sos": best_blend.w_sos, "srs_min_games": best_blend.srs_min_games})
    print("pure-Elo counterfactual (SAME k/hfa/carryover, w_sos=0):", {
        "k": pure_elo.k, "hfa_elo": pure_elo.hfa_elo, "carryover": pure_elo.carryover})
    print("VALIDATION blended:", blend_valid)
    print("VALIDATION pure-Elo:", pure_valid)
    print("SoS blend beat pure Elo OOS on margin MAE (strict <)?", blend_wins)
    print("naive baselines (validation span):", baselines)
    print("written rating.json:", out, "| final valid metrics:", final_valid)

if __name__ == "__main__":
    main()
