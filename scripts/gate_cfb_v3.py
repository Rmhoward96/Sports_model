"""cfb-ratings-v3 ship gate: v2 baseline first, then v3 vs v2 on held-out 2023-2025, then the
report-only residual check. Writes assets/cfb/v3_gate.json and a markdown summary.

Order (spec section 4):
  1. v2 is run through THIS harness and must reproduce its published held-out numbers (margin
     MAE 12.61, total MAE 13.09, ATS 49.1 % on 2023-2025 FBS-vs-FBS games with a closing spread;
     12.61 / 13.10 / 49.0 % before the 2026-10-06 Arkansas/Missouri/Virginia asset fix)
     within rounding. If it does not, the run STOPS (exit 2) before v3 is scored and nothing is
     written -- reconcile the harness first (compare against scripts/compare_cfb_live_fix.py).
  2. v3 ships only if, on the combined 2023-2025 numbers: margin MAE < v2, total MAE < v2,
     ATS >= v2, O/U >= v2, ML log-loss <= v2 and ECE <= v2 + 0.005. Per-season numbers are
     reported and any season where v3 is worse is flagged; the verdict uses the combined ones.
  3. Residual check (ridge + shallow gradient boosting on v3's 2016-2022 residuals, features v3
     does not use): reported only, nothing ships.
This script never changes which model is live: the repo variable CFB_MODEL_VERSION stays on v2
until the user decides.

Usage:
    uv run python scripts/gate_cfb_v3.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel.cfb import residual, v3, v3_data, v3_gate, v3_table  # noqa: E402
from sportsmodel.nfl.gameline import build_gameline  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
WALK_SEASONS = tuple(range(2015, 2026))
TRAIN_SEASONS = tuple(range(2016, 2023))
EMPTY_MARKET = {"spread_line": None, "total_line": None}


def add_predictions(table: pd.DataFrame, w: v3.V3Weights, gl_v2, gl_v3) -> pd.DataFrame:
    """table + v2_/v3_ margin, total and home-win-prob columns. Both go through build_gameline with
    an empty market (model-only), so the served pred = model - bias and the win prob uses each
    version's own sigma -- exactly what generate_cfb serves."""
    out = table.copy()
    cols = {k: [] for k in ("v2_margin", "v2_total", "v2_wp", "v3_margin", "v3_total", "v3_wp")}
    for r in table.to_dict("records"):
        a = build_gameline(r["margin_v2"], r["total_v2"], EMPTY_MARKET, r["week"], gl_v2)
        m3, t3 = v3.predict_v3(r, w)
        b = build_gameline(m3, t3, EMPTY_MARKET, r["week"], gl_v3)
        for pre, row in (("v2", a), ("v3", b)):
            cols[f"{pre}_margin"].append(row["pred_margin"])
            cols[f"{pre}_total"].append(row["pred_total"])
            cols[f"{pre}_wp"].append(row["home_win_prob"])
    for k, v in cols.items():
        out[k] = v
    return out


def evaluate_gate(scored: pd.DataFrame, expected=v3_gate.V2_EXPECTED, tol=v3_gate.BASELINE_TOL) -> dict:
    """Baseline reproduction (raises BaselineMismatch) then the v3-vs-v2 comparison."""
    ev = v3_gate.eval_set(scored)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    baseline = v3_gate.check_baseline(v2c, expected, tol)
    v3c = v3_gate.metrics(ev, "v3_margin", "v3_total", "v3_wp")
    v2s = v3_gate.by_season(ev, "v2_margin", "v2_total", "v2_wp")
    v3s = v3_gate.by_season(ev, "v3_margin", "v3_total", "v3_wp")
    return {"holdout_seasons": list(v3_gate.HOLDOUT_SEASONS), "n_games": int(len(ev)),
            "baseline": {"expected": expected, "measured": {k: v2c[k] for k in expected},
                         "reproduced": True},
            "v2": {"combined": v2c, "by_season": v2s}, "v3": {"combined": v3c, "by_season": v3s},
            "bias": {"convention": "mean signed error = prediction - actual (positive = over-predicts)",
                     **{name: {"combined": v3_gate.bias(ev, f"{name}_margin", f"{name}_total"),
                               "by_season": v3_gate.bias_by_season(ev, f"{name}_margin", f"{name}_total")}
                        for name in ("v2", "v3")}},
            **v3_gate.verdict(v2c, v3c, v2s, v3s)}


def residual_summary(scored: pd.DataFrame, w: v3.V3Weights, assets: Path) -> dict:
    """Report-only residual check on v3's residuals (train 2016-2022, test = the gate eval set)."""
    sched = v3_data.require_asset("schedules.parquet", assets)
    adv = v3_data.require_asset("advanced_games.parquet", assets)
    means = residual.prior_game_means(adv, sched)
    feats = residual.residual_features(
        scored, means, v3_data.read_asset("cfbd_games.parquet", assets), v3_data.load_priors_rows(assets),
        v3_data.read_asset("prior_ratings.parquet", assets), w.margin.coefs, w.total.coefs, w.points_map.coefs)
    ev = v3_gate.eval_set(scored)
    is_eval = scored["game_pk"].isin(ev["game_pk"]) & scored["season"].isin(v3_gate.HOLDOUT_SEASONS)
    is_train = scored["season"].isin(TRAIN_SEASONS) & scored["actual_margin"].notna()
    keep = is_eval | is_train
    sub = scored[keep]
    return residual.run_residual_check(
        feats[keep], (sub["actual_margin"] - sub["v3_margin"]).to_numpy(),
        (sub["actual_total"] - sub["v3_total"]).to_numpy(), sub["season"].to_numpy(),
        TRAIN_SEASONS, v3_gate.HOLDOUT_SEASONS)


def _fmt(x, nd=3, pct=False) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (
        f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}")


def render_report(res: dict) -> str:
    L = [f"# cfb-ratings-v3 gate ({res['generated']})", "",
         f"Held-out {res['holdout_seasons']}, {res['n_games']} FBS-vs-FBS games with a closing spread.", "",
         "**Live model: v2 (unchanged). Switching to v3 is the user's decision "
         "(repo variable CFB_MODEL_VERSION).**", "",
         f"**Verdict: {'PASS - v3 meets every criterion' if res['ship'] else 'FAIL - v3 does not ship'}**", "",
         "## v2 baseline reproduction", "",
         "| metric | published | measured |", "|---|---|---|"]
    for k, e in res["baseline"]["expected"].items():
        L.append(f"| {k} | {e} | {_fmt(res['baseline']['measured'][k], 4)} |")
    L += ["", "## Criteria (combined)", "", "| criterion | v2 | v3 | pass |", "|---|---|---|---|"]
    for k, c in res["criteria"].items():
        pct = k in ("ats", "ou")
        L.append(f"| {c['label']} | {_fmt(c['v2'], 4, pct)} | {_fmt(c['v3'], 4, pct)} | {'yes' if c['pass'] else 'NO'} |")
    v2c, v3c = res["v2"]["combined"], res["v3"]["combined"]
    L += ["", f"n ATS: v2 {v2c['n_ats']}, v3 {v3c['n_ats']}; n O/U: v2 {v2c['n_ou']}, v3 {v3c['n_ou']}."]
    L += ["", "ATS and O/U are scored on decided games only (pushes and no-pick games excluded), so n ATS / n O/U "
          "are below the game count; one standard error on a combined ATS difference is roughly 0.8 pp.",
          "", "## Per season", "",
          "| season | model | n | n ATS | n O/U | margin MAE | total MAE | ATS | O/U | ML log-loss | ECE |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(res["v2"]["by_season"]):
        for name in ("v2", "v3"):
            m = res[name]["by_season"][s]
            L.append(f"| {s} | {name} | {m['n']} | {m['n_ats']} | {m['n_ou']} | {_fmt(m['margin_mae'])} | {_fmt(m['total_mae'])} | "
                     f"{_fmt(m['ats'], pct=True)} | {_fmt(m['ou'], pct=True)} | {_fmt(m['ml_logloss'], 4)} | "
                     f"{_fmt(m['ml_ece'], 4)} |")
    L += ["", "Seasons where v3 is worse than v2: "
          + (", ".join(f"{f['season']} {f['criterion']}" for f in res["season_flags"]) or "none"), ""]
    L += ["## Bias (mean signed error, prediction - actual; positive = over-predicts; reported, not corrected)", "",
          "| season | model | margin bias | total bias |", "|---|---|---|---|"]
    for s in [*sorted(res["bias"]["v2"]["by_season"]), "all"]:
        for name in ("v2", "v3"):
            b = res["bias"][name]["combined"] if s == "all" else res["bias"][name]["by_season"][s]
            L.append(f"| {s} | {name} | {_fmt(b['margin_bias'], 2)} | {_fmt(b['total_bias'], 2)} |")
    L.append("")
    rc = res.get("residual_check")
    if rc:
        L += ["## Residual check (report only)", ""]
        for tgt in ("margin", "total"):
            r = rc[tgt]
            L.append(f"- {tgt}: ridge R2_oos {_fmt(r['ridge']['r2_oos'], 4)}, MAE change "
                     f"{_fmt(r['ridge']['mae_change'], 4)}; boosting R2_oos {_fmt(r['hgb']['r2_oos'], 4)}, MAE change "
                     f"{_fmt(r['hgb']['mae_change'], 4)}; top ridge features "
                     f"{[f['feature'] for f in r['ridge']['top_features'][:3]]}; v4 candidates "
                     f"{[c['feature'] for c in r['v4_candidates']] or 'none'}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-out", default=str(ASSETS / "v3_gate.json"))
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    inp = v3_data.load_v3_inputs(ASSETS)
    table = v3_table.build_table(v3_data.load_merged_schedule(ASSETS), inp, seasons=WALK_SEASONS)
    w = v3.load_v3_weights(ASSETS / "v3_weights.json")
    scored = add_predictions(table, w, v3.load_gameline_config(ASSETS / "gameline.json"),
                             v3.load_gameline_config(ASSETS / "gameline_v3.json"))
    try:
        res = evaluate_gate(scored)
    except v3_gate.BaselineMismatch as e:
        print(f"STOP: {e}\nNothing written. Reconcile the harness with scripts/compare_cfb_live_fix.py "
              "before comparing v3.", file=sys.stderr)
        return 2
    res["generated"] = date.today().isoformat()
    res["residual_check"] = residual_summary(scored, w, ASSETS)
    res["fit"] = w.meta
    res["live_model"] = "v2 (unchanged); CFB_MODEL_VERSION stays v2 until the user approves this gate"
    res["runtime_s"] = time.time() - t0
    out = Path(args.json_out)
    rep = Path(args.report_out) if args.report_out else (
        ROOT / "docs" / "superpowers" / "reports" / f"{res['generated']}-cfb-v3-gate.md")
    out.write_text(json.dumps(v3_gate.clean(res), indent=2, allow_nan=False) + "\n")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(render_report(res))
    print(render_report(res))
    print(f"wrote {out} and {rep} ({res['runtime_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
