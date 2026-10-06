"""cfb-ratings-v3.1 ship gate: the v3 pipeline REFIT every offseason (spec addendum A, pre-registered).

For each test season S in 2023, 2024, 2025 the efficiency hyperparameters and then the v3 weights /
gameline are fitted on seasons 2016..S-1 only (same grid, objective, features, one-SE rule), the
features are built with that season's eff config, and season S is predicted. The three seasons'
v3.1 predictions are concatenated and judged against v2 exactly as in scripts/gate_cfb_v3.py:
the v2 baseline is reproduced FIRST (exit 2, nothing written, if it is not), then the same six
criteria on the combined 2023-2025 numbers (per-season flags reported, not part of the verdict).
No bias correction, no new features, no grid changes. The report additionally shows n ATS / n O/U,
paired standard errors of the v3.1 - v2 differences and the per-season bias table.

Writes assets/cfb/v31_gate.json, docs/superpowers/reports/<date>-cfb-v31-gate.md and the per-season
fitted configs assets/cfb/v31/<S>/{eff_config,v3_weights,gameline_v3}.json. It never changes which
model is live (CFB_MODEL_VERSION stays v2). v3.1 is judged ONCE: if it fails, v2 stays and this
holdout is retired.

Usage:
    uv run python scripts/gate_cfb_v31.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import gate_cfb_v3 as gate  # noqa: E402
from sportsmodel.cfb import v3, v3_data, v3_gate, v3_table, v31  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
WALK_SEASONS = tuple(range(2015, 2026))
V3_GATE_JSON = ASSETS / "v3_gate.json"
V31_LABEL = "v3.1"


def score_fits(fits, gl_v2) -> pd.DataFrame:
    """Concatenate the per-season predictions: each season's rows scored with ITS OWN fitted weights
    and gameline (columns v3_* hold the v3.1 predictions; v2_* the served v2). Same code path as
    gate_cfb_v3.add_predictions."""
    return pd.concat([gate.add_predictions(f.table, f.weights, gl_v2, v3.gameline_from_dict(f.gameline))
                      for f in fits], ignore_index=True)


def _relabel(res: dict) -> dict:
    """gate_cfb_v3.evaluate_gate names the candidate 'v3'; in this gate it is v3.1 ('v31')."""
    res = dict(res)
    res["v31"] = res.pop("v3")
    res["criteria"] = {k: {("v31" if kk == "v3" else kk): vv for kk, vv in c.items()}
                       | {"label": c["label"].replace("v3 ", f"{V31_LABEL} ", 1)} for k, c in res["criteria"].items()}
    res["season_flags"] = [{("v31" if k == "v3" else k): v for k, v in f.items()} for f in res["season_flags"]]
    res["bias"] = {("v31" if k == "v3" else k): v for k, v in res["bias"].items()}
    return res


def evaluate_v31(scored: pd.DataFrame, expected=v3_gate.V2_EXPECTED, tol=v3_gate.BASELINE_TOL) -> dict:
    """Baseline reproduction (raises BaselineMismatch) then v3.1 vs v2, plus paired standard errors."""
    res = gate.evaluate_gate(scored, expected, tol)
    ev = v3_gate.eval_set(scored)
    res = _relabel(res)
    res["paired"] = {"convention": "v3.1 - v2; matched pairs per game (ATS / O/U on games decided for both models; "
                                   "log-loss on games with a decision)",
                     "combined": v3_gate.paired_diffs(ev, "v3", "v2"),
                     "by_season": {int(s): v3_gate.paired_diffs(g, "v3", "v2") for s, g in ev.groupby("season")}}
    return res


def _fmt(x, nd=3, pct=False) -> str:
    return gate._fmt(x, nd, pct)


def _pp(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:+.2f}"


def render_report(res: dict) -> str:
    v2c, v31c = res["v2"]["combined"], res["v31"]["combined"]
    L = [f"# cfb-ratings-v3.1 gate - rolling-origin refit ({res['generated']})", "",
         f"Held-out {res['holdout_seasons']}, {res['n_games']} FBS-vs-FBS games with a closing spread. "
         "Method pre-registered in spec addendum A: for each test season S the full v3 pipeline "
         "(efficiency hyperparameters, points map, margin/total blends, sigmas) is refit on 2016..S-1 only.", "",
         "**Live model: v2 (unchanged). Switching is the user's decision (repo variable CFB_MODEL_VERSION).**", "",
         f"**Verdict: {'PASS - v3.1 meets every criterion' if res['ship'] else 'FAIL - v3.1 does not ship (v2 stays live; this holdout is retired)'}**",
         "", "## v2 baseline reproduction", "", "| metric | published | measured |", "|---|---|---|"]
    for k, e in res["baseline"]["expected"].items():
        L.append(f"| {k} | {e} | {_fmt(res['baseline']['measured'][k], 4)} |")
    L += ["", "## Criteria (combined)", "", "| criterion | v2 | v3.1 | pass |", "|---|---|---|---|"]
    for k, c in res["criteria"].items():
        pct = k in ("ats", "ou")
        L.append(f"| {c['label']} | {_fmt(c['v2'], 4, pct)} | {_fmt(c['v31'], 4, pct)} | {'yes' if c['pass'] else 'NO'} |")
    L += ["", f"n ATS: v2 {v2c['n_ats']}, v3.1 {v31c['n_ats']}; n O/U: v2 {v2c['n_ou']}, v3.1 {v31c['n_ou']}. "
          "ATS and O/U are scored on decided games only (pushes and no-pick games excluded).", "",
          "## Paired differences (v3.1 - v2)", "",
          "| scope | metric | n pairs | diff | paired SE | z |", "|---|---|---|---|---|---|"]
    for scope, pr in [("all", res["paired"]["combined"]),
                      *((str(s), p) for s, p in sorted(res["paired"]["by_season"].items()))]:
        for k, label, pct in (("ats", "ATS (pp)", True), ("ou", "O/U (pp)", True), ("ml_logloss", "ML log-loss", False)):
            d = pr[k]
            fmt = _pp if pct else (lambda x: "n/a" if x is None or x != x else f"{x:+.4f}")
            z = "n/a" if d["z"] is None else f"{d['z']:+.2f}"
            L.append(f"| {scope} | {label} | {d['n']} | {fmt(d['diff'])} | {fmt(d['se']).lstrip('+')} | {z} |")
    L += ["", "## Per season", "",
          "| season | model | n | n ATS | n O/U | margin MAE | total MAE | ATS | O/U | ML log-loss | ECE |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(res["v2"]["by_season"]):
        for name in ("v2", "v31"):
            m = res[name]["by_season"][s]
            L.append(f"| {s} | {'v3.1' if name == 'v31' else name} | {m['n']} | {m['n_ats']} | {m['n_ou']} | "
                     f"{_fmt(m['margin_mae'])} | {_fmt(m['total_mae'])} | {_fmt(m['ats'], pct=True)} | "
                     f"{_fmt(m['ou'], pct=True)} | {_fmt(m['ml_logloss'], 4)} | {_fmt(m['ml_ece'], 4)} |")
    L += ["", "Seasons where v3.1 is worse than v2: "
          + (", ".join(f"{f['season']} {f['criterion']}" for f in res["season_flags"]) or "none"), ""]
    L += ["## Bias (mean signed error, prediction - actual; positive = over-predicts; reported, not corrected)", "",
          "| season | model | n | margin bias | total bias |", "|---|---|---|---|---|"]
    for s in [*sorted(res["bias"]["v2"]["by_season"]), "all"]:
        for name in ("v2", "v31"):
            b = res["bias"][name]["combined"] if s == "all" else res["bias"][name]["by_season"][s]
            L.append(f"| {s} | {'v3.1' if name == 'v31' else name} | {b['n']} | {_fmt(b['margin_bias'], 2)} | "
                     f"{_fmt(b['total_bias'], 2)} |")
    L += ["", "## Per-season fits (train 2016..S-1)", "",
          "Margin = c + a1*margin_v2 + a2*margin_eff + a3*prior_margin + h*non_neutral + context; "
          "total = c + b1*total_eff + b2*total_v2 + context.", "",
          "| S | n train | a1 | a2 | a3 | h | c (margin) | b1 | b2 | c (total) | sigma margin | sigma total | "
          "context kept | eff config |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, f in sorted(res["fits"].items()):
        m, t = f["margin"]["coefs"], f["total"]["coefs"]
        kept = [f"{k}={v:+.3f}" for blend in (m, t) for k, v in blend.items()
                if k not in ("margin_v2", "margin_eff", "prior_margin", "non_neutral", "total_eff", "total_v2") and v != 0.0]
        e = f["eff_config"]
        L.append(f"| {s} | {f['n_train']} | {m['margin_v2']:.3f} | {m['margin_eff']:.3f} | {m['prior_margin']:.3f} | "
                 f"{m['non_neutral']:+.3f} | {f['margin']['intercept']:+.2f} | {t['total_eff']:.3f} | {t['total_v2']:.3f} | "
                 f"{f['total']['intercept']:+.2f} | {f['sigma_margin']:.2f} | {f['sigma_total']:.2f} | "
                 f"{', '.join(kept) or 'none'} | ridge {e['ridge']:g}, hl {e['half_life_games']:g}, floor "
                 f"{e['prior_floor']:g}, k0 {e['k0']:g}, k_ret {e['k_ret']:g}, k_tal {e['k_tal']:g} |")
    ref = res.get("v3_reference")
    if ref:
        L += ["", "## Reference: the fixed-weights v3 gate (train 2016-2022 once; historical record)", "",
              "| metric | v3 (fixed) | v3.1 |", "|---|---|---|"]
        for k in ("margin_mae", "total_mae", "ats", "ou", "ml_logloss", "ml_ece"):
            pct = k in ("ats", "ou")
            L.append(f"| {k} | {_fmt(ref[k], 4, pct)} | {_fmt(v31c[k], 4, pct)} |")
    return "\n".join(L) + "\n"


def fit_summary(f: v31.SeasonFit) -> dict:
    return {"eff_config": {k: float(v) for k, v in asdict(f.eff_cfg).items()},
            "points_map": f.weights.points_map.to_dict(), "margin": f.weights.margin.to_dict(),
            "total": f.weights.total.to_dict(), "n_train": f.weights.meta["n_train"],
            "train_seasons": f.weights.meta["train_seasons"], "sigma_margin": f.weights.meta["sigma_margin"],
            "sigma_total": f.weights.meta["sigma_total"],
            "lasso_alpha": {"margin": f.weights.meta["margin"]["lasso_alpha"],
                            "total": f.weights.meta["total"]["lasso_alpha"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-out", default=str(ASSETS / "v31_gate.json"))
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--fit-dir", default=str(ASSETS / "v31"))
    args = ap.parse_args(argv)
    t0 = time.time()
    inp = v3_data.load_v3_inputs(ASSETS)
    frame = v3_data.load_merged_schedule(ASSETS)
    gl_v2 = v3.load_gameline_config(ASSETS / "gameline.json")

    # 1. v2 baseline FIRST, before any v3.1 fit (v2's columns do not depend on the efficiency config)
    probe = v3_table.build_table(frame, inp, seasons=v3_gate.HOLDOUT_SEASONS)
    w0 = v3.load_v3_weights(ASSETS / "v3_weights.json")
    try:
        gate.evaluate_gate(gate.add_predictions(probe, w0, gl_v2, v3.load_gameline_config(ASSETS / "gameline_v3.json")))
    except v3_gate.BaselineMismatch as e:
        print(f"STOP: {e}\nNothing written. Reconcile the harness with scripts/compare_cfb_live_fix.py "
              "before comparing v3.1.", file=sys.stderr)
        return 2
    print(f"v2 baseline reproduced ({time.time() - t0:.0f}s); refitting per season", flush=True)

    # 2. rolling-origin refit + prediction, one season at a time
    fits = []
    for s in v31.TEST_SEASONS:
        f = v31.fit_season(s, frame, inp)
        print(f"  S={s}: train {f.weights.meta['train_seasons'][0]}-{f.weights.meta['train_seasons'][-1]} "
              f"n={f.weights.meta['n_train']} eff={f.eff_cfg} ({time.time() - t0:.0f}s)", flush=True)
        fits.append(f)
    scored = score_fits(fits, gl_v2)

    # 3. the pre-registered gate
    try:
        res = evaluate_v31(scored)
    except v3_gate.BaselineMismatch as e:
        print(f"STOP: {e}\nNothing written.", file=sys.stderr)
        return 2
    res["generated"] = date.today().isoformat()
    res["fits"] = {f.season: fit_summary(f) for f in fits}
    if V3_GATE_JSON.exists():
        res["v3_reference"] = json.loads(V3_GATE_JSON.read_text())["v3"]["combined"]
    res["live_model"] = "v2 (unchanged); CFB_MODEL_VERSION stays v2 until the user approves"
    res["runtime_s"] = time.time() - t0
    out = Path(args.json_out)
    rep = Path(args.report_out) if args.report_out else (
        ROOT / "docs" / "superpowers" / "reports" / f"{res['generated']}-cfb-v31-gate.md")
    for f in fits:
        v31.write_season_outputs(f, args.fit_dir)
    out.write_text(json.dumps(v3_gate.clean(res), indent=2, allow_nan=False) + "\n")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(render_report(res))
    print(render_report(res))
    print(f"wrote {out}, {rep} and {args.fit_dir}/<S>/ ({res['runtime_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
