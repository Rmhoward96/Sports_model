"""Fit the results-based power rating (``sportsmodel.context.results_power``).

Per sport, grid-search (cap, hfa, beta, rho) to minimize the walk-forward MAE of
next-week margins (rating diff entering the week + hfa) over the FIT seasons; then
report the HOLDOUT seasons for the fitted rating vs

* point differential + SOS only (same model, beta = 0, uncapped),
* today's rankings rating (CFB ``cfb_power_current``; NFL ``nfl_power`` on
  ``unit_ratings_asof(blend_k=POWER_BLEND_K)`` with its market scale) -- given its
  best constant home edge on the holdout itself (favours the baseline),
* the closing line (reference; not a rating).

Writes ``assets/context/results_power.json`` and prints a markdown report.

  uv run python scripts/fit_results_power.py [--sport cfb|nfl|all] [--quick]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb.priors import load_weights, season_priors  # noqa: E402
from sportsmodel.cfb.teams import load_fbs_ids  # noqa: E402
from sportsmodel.context.results_power import (  # noqa: E402
    PARAMS_PATH, ResultsParams, chain, params_dict, played_games, predict_errors)
from sportsmodel.nfl.teams import normalize_team  # noqa: E402

CFB = ROOT / "assets" / "cfb"
SPEC = {
    "cfb": {"chain": list(range(2015, 2026)), "fit": list(range(2016, 2024)),
            "holdout": [2024, 2025], "sigma": 16.7,
            "grid": {"cap": [None, 21.0, 24.0, 28.0, 35.0], "hfa": [1.5, 2.5, 3.5],
                     "beta": [0.0, 1.0, 2.0, 4.0, 7.0, 10.0], "rho": [0.4, 0.6, 0.8]}},
    "nfl": {"chain": list(range(2005, 2026)), "fit": list(range(2010, 2024)),
            "holdout": [2024, 2025], "sigma": 13.5,
            "grid": {"cap": [None, 14.0, 17.0, 21.0, 28.0], "hfa": [1.0, 1.5, 2.0, 2.5],
                     "beta": [0.0, 1.0, 2.0, 4.0, 6.0, 9.0], "rho": [0.3, 0.45, 0.6, 0.75]}},
}


# ------------------------------------------------------------------------- data
def cfb_priors_points(seasons) -> dict[int, dict[str, float]]:
    """{season: {team: v2 preseason rating in points vs the FBS average}}."""
    df = pd.read_parquet(CFB / "priors.parquet")
    w = load_weights(CFB / "priors_weights.json")
    fbs = {str(t) for t in load_fbs_ids()}
    out = {}
    for s in seasons:
        d = df[df["season"].isin([s - 1, s])]
        rows = {int(y): g.to_dict("records") for y, g in d.groupby("season")}
        pri = season_priors(rows, s, w) if s in rows else {}
        pri = {str(t): float(v) for t, v in pri.items()}
        fb = [v for t, v in pri.items() if t in fbs]
        mu = float(np.mean(fb)) if fb else 0.0
        out[int(s)] = {t: (v - mu) / 25.0 for t, v in pri.items()}
    return out


def cfb_data():
    s = pd.read_parquet(CFB / "schedules.parquet")
    s = s[s["game_type"] == "REG"].rename(columns={"neutral_site": "neutral"})
    return played_games(s), s


def _norm(code) -> str:
    try:
        return normalize_team(str(code))
    except ValueError:
        return str(code)


def nfl_data(seasons):
    from sportsmodel.nfl.nflverse import load_release
    s = load_release("schedules", list(seasons))
    s = s[s["game_type"] == "REG"].copy()
    s["home_team"] = s["home_team"].map(_norm)
    s["away_team"] = s["away_team"].map(_norm)
    s["neutral"] = s["location"].astype(str).str.lower().eq("neutral")
    return played_games(s), s


# ----------------------------------------------------------------------- fitting
def mae(e: pd.DataFrame) -> float:
    return float((e["pred"] - e["margin"]).abs().mean())


def evaluate(games, p, spec, preseason, members, seasons):
    walks = chain(games, p, spec["chain"], preseason=preseason, members=members)
    return predict_errors(games, walks, p, seasons), walks


def fit(sport: str, games, preseason, members, quick: bool):
    spec = SPEC[sport]
    grid = spec["grid"]
    if quick:
        grid = {k: v[:: max(1, len(v) // 2)] for k, v in grid.items()}
    best, tried, by_beta = None, 0, {}
    t0 = time.time()
    for cap, hfa, beta, rho in itertools.product(grid["cap"], grid["hfa"], grid["beta"], grid["rho"]):
        p = ResultsParams(cap=cap, hfa=hfa, beta=beta, sigma=spec["sigma"], rho=rho)
        e, _ = evaluate(games, p, spec, preseason, members, spec["fit"])
        m = mae(e)
        tried += 1
        if best is None or m < best[0]:
            best = (m, p)
        if beta not in by_beta or m < by_beta[beta][0]:
            by_beta[beta] = (m, p)
    print(f"[{sport}] grid {tried} combos in {time.time() - t0:.0f}s; fit MAE {best[0]:.3f} "
          f"with {params_dict(best[1])}", flush=True)
    return best, by_beta


def boot_ci(diff: np.ndarray, n: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    b = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# --------------------------------------------------------------------- baselines
def cfb_today(sched, holdout) -> pd.DataFrame:
    """Walk-forward preds from today's CFB rankings rating (cfb_power_current)."""
    spec = importlib.util.spec_from_file_location("btc", ROOT / "scripts" / "build_team_context.py")
    btc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btc)
    from sportsmodel.context.power import cfb_power_current
    elo_cfg, blend_cfg = btc.load_rating()
    fbs = {str(t) for t in load_fbs_ids()}
    s = sched.copy()
    rows = []
    for season in holdout:
        pri = btc.load_cfb_priors(season)
        g = s[(s["season"] == season)].dropna(subset=["home_score", "away_score"])
        for wk, gw in g.groupby("week"):
            p = cfb_power_current(s, elo_cfg, blend_cfg, (season, int(wk)), priors=pri, fbs=fbs)
            r = p.set_index("team")["rating"]
            for x in gw.itertuples(index=False):
                if x.home_team in r.index and x.away_team in r.index:
                    rows.append({"season": season, "week": int(wk), "home_team": x.home_team,
                                 "away_team": x.away_team, "neutral": bool(x.neutral),
                                 "diff": float(r[x.home_team] - r[x.away_team]),
                                 "margin": float(x.home_score - x.away_score)})
    return pd.DataFrame(rows)


def nfl_today(holdout) -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("btc", ROOT / "scripts" / "build_team_context.py")
    btc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(btc)
    from sportsmodel.context.power import nfl_market_scale, nfl_power
    from sportsmodel.context.units import POWER_BLEND_K, unit_ratings_asof
    src = btc.load_nfl_sources(pd.Timestamp("2025-12-15", tz="UTC"))
    ug, sched = src["unit_games"], src["schedules"]
    rows = []
    for season in holdout:
        scale = nfl_market_scale(ug, sched, season, blend_k=POWER_BLEND_K)
        g = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].dropna(
            subset=["home_score", "away_score"])
        for wk, gw in g.groupby("week"):
            r = nfl_power(unit_ratings_asof(ug, season, int(wk), blend_k=POWER_BLEND_K),
                          ug, scale).set_index("team")["rating"]
            for x in gw.itertuples(index=False):
                h, a = _norm(x.home_team), _norm(x.away_team)
                if h in r.index and a in r.index:
                    rows.append({"season": season, "week": int(wk), "home_team": h, "away_team": a,
                                 "neutral": str(x.location).lower() == "neutral",
                                 "diff": float(r[h] - r[a]),
                                 "margin": float(x.home_score - x.away_score)})
    return pd.DataFrame(rows)


def with_best_hfa(b: pd.DataFrame) -> pd.DataFrame:
    v = (~b["neutral"]).astype(float)
    resid = b["margin"] - b["diff"]
    grid = np.linspace(-1, 6, 141)
    h = grid[np.argmin([np.abs(resid - x * v).mean() for x in grid])]
    return b.assign(pred=b["diff"] + h * v, hfa=h)


def closing(sport: str, games, sched, holdout) -> pd.DataFrame:
    if sport == "cfb":
        l = pd.read_parquet(CFB / "lines.parquet")
        l = l[l["season"].isin(holdout)][["season", "week", "home_team", "away_team", "market_spread"]]
        g = games[games["season"].isin(holdout)].merge(l, on=["season", "week", "home_team", "away_team"])
        m = g["home_score"] - g["away_score"]
        sgn = -1.0 if np.corrcoef(g["market_spread"], m)[0, 1] < 0 else 1.0
        return pd.DataFrame({"pred": sgn * g["market_spread"], "margin": m, "season": g["season"],
                             "week": g["week"], "home_team": g["home_team"], "away_team": g["away_team"]})
    s = sched[sched["season"].isin(holdout)].dropna(subset=["spread_line", "home_score"])
    return pd.DataFrame({"pred": s["spread_line"].astype(float),
                         "margin": (s["home_score"] - s["away_score"]).astype(float),
                         "season": s["season"], "week": s["week"],
                         "home_team": s["home_team"], "away_team": s["away_team"]})


# --------------------------------------------------------------------------- main
def run(sport: str, quick: bool) -> dict:
    spec = SPEC[sport]
    if sport == "cfb":
        games, sched = cfb_data()
        preseason = cfb_priors_points(spec["chain"])
        members = {str(t) for t in load_fbs_ids()}
    else:
        games, sched = nfl_data(spec["chain"])
        preseason, members = None, None
    (fit_mae, p), by_beta = fit(sport, games, preseason, members, quick)
    beta_curve = []
    for b, (m, pb) in sorted(by_beta.items()):
        hb, _ = evaluate(games, pb, spec, preseason, members, spec["holdout"])
        beta_curve.append({"beta": b, "fit_mae": m, "holdout_mae": mae(hb), "cap": pb.cap,
                           "hfa": pb.hfa, "rho": pb.rho})

    hold, _ = evaluate(games, p, spec, preseason, members, spec["holdout"])
    plain_p = ResultsParams(cap=None, hfa=p.hfa, beta=0.0, sigma=p.sigma, rho=p.rho)
    plain, _ = evaluate(games, plain_p, spec, preseason, members, spec["holdout"])
    keys = ["season", "week", "home_team", "away_team"]
    hold_k = hold
    today = cfb_today(sched, spec["holdout"]) if sport == "cfb" else nfl_today(spec["holdout"])
    today = with_best_hfa(today)
    close = closing(sport, games, sched, spec["holdout"])

    def paired(base):
        m = hold_k.merge(base[keys + ["pred"]], on=keys, suffixes=("", "_b"))
        d = (m["pred"] - m["margin"]).abs().to_numpy() - (m["pred_b"] - m["margin"]).abs().to_numpy()
        lo, hi = boot_ci(d)
        return {"n": int(len(m)), "new_mae": float(np.abs(m["pred"] - m["margin"]).mean()),
                "base_mae": float(np.abs(m["pred_b"] - m["margin"]).mean()),
                "diff": float(d.mean()), "ci": [lo, hi]}

    by_phase = {}
    for name, sel in (("weeks_1_4", hold["week"] <= 4), ("weeks_5_plus", hold["week"] >= 5)):
        by_phase[name] = {"new": mae(hold[sel]), "plain": mae(plain[sel])}
    res = {"params": params_dict(p), "fit_seasons": spec["fit"], "holdout_seasons": spec["holdout"],
           "fit_mae": fit_mae, "holdout_mae": mae(hold), "holdout_n": int(len(hold)),
           "point_diff_sos_only_mae": mae(plain),
           "vs_today": paired(today.assign(pred=today["pred"])),
           "today_hfa": float(today["hfa"].iloc[0]) if len(today) else None,
           "vs_close": paired(close), "by_phase": by_phase, "beta_curve": beta_curve}
    return res


def report(results: dict) -> str:
    out = ["# Results-based power rating — fit and holdout", ""]
    for sport, r in results.items():
        p = r["params"]
        vt, vc = r["vs_today"], r["vs_close"]
        out += [f"## {sport.upper()}", "",
                f"Fitted on {r['fit_seasons'][0]}–{r['fit_seasons'][-1]}: cap "
                f"{p['cap'] if p['cap'] is not None else 'none'}, home edge {p['hfa']}, "
                f"win-credit weight {p['beta']}, last-season shrink {p['rho']} "
                f"(sigma {p['sigma']}, prior weight k {p['k']}).", "",
                f"Holdout {r['holdout_seasons']} next-week margin MAE (n={r['holdout_n']}):", "",
                "| Rating | MAE |", "|---|---|",
                f"| **New results rating** | **{r['holdout_mae']:.3f}** |",
                f"| Point differential + SOS only | {r['point_diff_sos_only_mae']:.3f} |",
                f"| Today's rankings rating (same games, n={vt['n']}) | {vt['base_mae']:.3f} "
                f"(new {vt['new_mae']:.3f}; diff {vt['diff']:+.3f}, 95% CI "
                f"[{vt['ci'][0]:+.3f}, {vt['ci'][1]:+.3f}]) |",
                f"| Closing line (reference, n={vc['n']}) | {vc['base_mae']:.3f} "
                f"(new {vc['new_mae']:.3f}) |", "",
                f"By phase: weeks 1–4 new {r['by_phase']['weeks_1_4']['new']:.3f} vs point-diff "
                f"{r['by_phase']['weeks_1_4']['plain']:.3f}; weeks 5+ new "
                f"{r['by_phase']['weeks_5_plus']['new']:.3f} vs "
                f"{r['by_phase']['weeks_5_plus']['plain']:.3f}.", "",
                "Win-credit weight vs accuracy (best other params per weight):", "",
                "| Win credit (pts) | Fit MAE | Holdout MAE |", "|---|---|---|",
                *[f"| {b['beta']:g} | {b['fit_mae']:.3f} | {b['holdout_mae']:.3f} |"
                  for b in r["beta_curve"]], ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", choices=["cfb", "nfl", "all"], default="all")
    ap.add_argument("--quick", action="store_true", help="coarse grid (smoke test)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    sports = ["cfb", "nfl"] if args.sport == "all" else [args.sport]
    results = {s: run(s, args.quick) for s in sports}
    if not args.no_write:
        PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
        for s, r in results.items():
            existing[s] = {**r["params"], "fit": {k: v for k, v in r.items() if k != "params"}}
        PARAMS_PATH.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"wrote {PARAMS_PATH.relative_to(ROOT)}")
    print(report(results))


if __name__ == "__main__":
    main()
