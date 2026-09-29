"""v2 (matchup / defensive injuries / QB profile) vs v1 props-ML gate (spec §6).

Compares the two model versions' SERVED composites -- the post-calibration
records ``scripts/train_props_ml_b.py`` writes for the selected per-market
sources: ``data/props_ml/served__<tag>.parquet`` (v1: ``a_gate.json`` +
``ROLE_SUBSETS``) and ``served__<tag>_v2.parquet`` (``PROPS_ML_GATE_NAME=_v2``:
``a_gate_v2.json`` + ``ROLE_SUBSETS_V2``) -- on the verdict seasons 2024 and
2025 (tuning and rung decisions used 2021-2023), plus a game-level check from
two ``backtest_sim_nfl.run_backtest`` runs of the kept A of each version
(``gate_ml_game_lines.run_ml_side``; same games, per-game seeded streams).

Ship rule (``v2_checks.verdict``, as Props-2):

- RPS: ``props_eval.rung_decision`` of v2 vs v1 passes on 2024-2025 pooled AND
  on 2025 alone (season-week-home cluster bootstrap, per-market RPS and ECE
  guards);
- ECE: every market's v2 PIT decile ECE <= v1's + 0.005 (pooled);
- game level: win / cover / over Brier and margin / total MAE of v2 <= v1 x
  1.005 (games scorable for both);
- sub-check 1 (required) -- QB-change games (``qb_changed`` = 1 or
  ``qb_ratio_ypa`` outside [0.95, 1.05]): pass_yds RPS improves with a
  season-week cluster bootstrap CI excluding 0;
- sub-check 2 (required) -- matchup response in BOTH directions by opponent
  ``mx_pass_minus_rush`` quintile (``v2_checks.matchup_response``);
- spot checks (reported only): ``SPOT_CHECKS`` (Keenum PHI @ CHI 2026-09-28),
  shown when the served files hold a graded record for it.

Env: PROPS_ML_SEASONS (default 2024,2025; must include 2025 to pass),
PROPS_ML_N_SIMS (default 1000), PROPS_ML_RESUME (1 -> reuse the game-level
checkpoints ``data/props_ml/game_records__<tag>{,_v2}.parquet`` when their
sidecar meta matches; the v1 one is shared with ``gate_ml_game_lines.py``).

Outputs: ``assets/nfl/props_ml/v2_gate.json`` and
``docs/superpowers/reports/<date>-matchup-qb-gate.md`` (seasons other than
2024-2025 add ``__<tag>``). No DB writes.

PURE / IO split: everything except ``main()`` is unit tested
(tests/scripts/test_gate_props_ml_v2.py; stubbed backtest and model fits).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel.model import game_gate  # noqa: E402
from sportsmodel.model.props_ml import v2_checks as vc  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tpm = _load_script("train_props_ml")
gml = _load_script("gate_ml_game_lines")

VERDICT_SEASONS: tuple[int, ...] = (2024, 2025)
FINAL_SEASON = vc.FINAL_SEASON
V2_NAME = "_v2"
DEFAULT_N_SIMS = 1000
N_BOOT = 1000
BOOT_SEED = 0

DATA_DIR = tpm.DATA_DIR
SERVED_V1 = DATA_DIR / f"served__{tpm.DEFAULT_TAG}.parquet"
SERVED_V2 = DATA_DIR / f"served__{tpm.DEFAULT_TAG}{V2_NAME}.parquet"
A_GATE_V1 = tpm.GATE_PATH
A_GATE_V2 = tpm.GATE_PATH.with_name(f"{tpm.GATE_PATH.stem}{V2_NAME}{tpm.GATE_PATH.suffix}")
GATE_PATH = ROOT / "assets" / "nfl" / "props_ml" / "v2_gate.json"
REPORT_DIR = tpm.REPORT_DIR
DEFAULT_TAG = tpm.run_tag(list(VERDICT_SEASONS), tpm.REFIT_WEEKS)

# Reported spot checks: the (season, team, opponent, position) player row(s) of
# the player table; a record is shown when the served files hold it graded.
SPOT_CHECKS: tuple[dict, ...] = (
    {"label": "Keenum PHI @ CHI 2026-09-28", "season": 2026, "team": "PHI", "opponent": "CHI",
     "position": "QB", "market": "pass_yds"},
)


# ---- pure helpers -----------------------------------------------------------------------

def output_paths(tag: str, run_date: str, *, gate_path: Path = GATE_PATH,
                 report_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    """(gate json, report md); tags other than the 2024-2025 default get ``__<tag>``."""
    sfx = "" if tag == DEFAULT_TAG else f"__{tag}"
    gate_path = Path(gate_path)
    return (gate_path.with_name(f"{gate_path.stem}{sfx}{gate_path.suffix}"),
            Path(report_dir) / f"{run_date}-matchup-qb-gate{sfx}.md")


def check_gate_names(a_gate_v1: Mapping, a_gate_v2: Mapping) -> None:
    """ValueError unless v1's A gate is unnamed and v2's is ``_v2``."""
    if str(a_gate_v1.get("gate_name", "")) != "":
        raise ValueError(f"v1 A gate has gate_name {a_gate_v1.get('gate_name')!r}; expected none "
                         "(a_gate.json)")
    if str(a_gate_v2.get("gate_name", "")) != V2_NAME:
        raise ValueError(f"v2 A gate has gate_name {a_gate_v2.get('gate_name', '')!r}; expected "
                         f"{V2_NAME!r} (a_gate{V2_NAME}.json)")


def spot_checks(served_v1: pd.DataFrame, served_v2: pd.DataFrame, feats: pd.DataFrame,
                specs=SPOT_CHECKS) -> list[dict]:
    """One row per served (v2) record of each spec's player-week(s): means,
    RPS and actual of both versions; ``graded`` when the actual is known. A
    spec without such a record gives one ``graded: False`` row."""
    out = []
    key = ["season", "week", "player_id", "market"]
    for sp in specs:
        f = feats[(feats["season"] == sp["season"]) & (feats["team"] == sp["team"])
                  & (feats["opponent"] == sp["opponent"]) & (feats["position"] == sp["position"])]
        want = f[["season", "week", "player_id"]].assign(market=sp["market"])
        hits = want.merge(served_v2, on=key).merge(served_v1, on=key, suffixes=("_v2", "_v1"))
        if hits.empty:
            out.append({"label": sp["label"], "graded": False,
                        "note": "no served record (not in the gate's backtest seasons)"})
            continue
        for r in hits.to_dict("records"):
            out.append({"label": sp["label"], "season": int(r["season"]), "week": int(r["week"]),
                        "player_id": str(r["player_id"]), "market": sp["market"],
                        "mean_v1": float(r["mean_v1"]), "mean_v2": float(r["mean_v2"]),
                        "rps_v1": float(r["rps_v1"]), "rps_v2": float(r["rps_v2"]),
                        "actual": float(r["actual_v2"]),
                        "graded": bool(pd.notna(r["actual_v2"]))})
    return out


# ---- report -----------------------------------------------------------------------------

def _f(x, spec: str) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def _verdict_text(gate: dict) -> str:
    v = gate["verdict"]
    seasons = ", ".join(map(str, gate["seasons"]))
    if v["pass"]:
        return (f"**Verdict: PASS.** The v2 served composite (matchup / defensive injuries / QB "
                f"profile) beats v1 on {seasons} pooled and on {FINAL_SEASON} alone, with ECE "
                "within +0.005 per market, game-level metrics within +0.5 %, and both required "
                "sub-checks (QB-change games, two-way matchup response) passing.")
    return ("**Verdict: FAIL.** The v2 served composite does not meet the ship rule against v1 "
            f"on {seasons}: " + "; ".join(v["reasons"]) + ". v1 stays served.")


def render_report(gate: dict) -> str:
    v, ident = gate["verdict"], gate["identity"]
    lines = [f"# Matchup / QB-profile gate (v2 vs v1) — {gate['run_date']}", "",
             _verdict_text(gate), "",
             "## Served composite RPS (base = v1, cand = v2)", "",
             f"{', '.join(map(str, gate['seasons']))} pooled:", ""]
    lines += tpm._decision_block(v["pooled"])
    lines += ["", f"{FINAL_SEASON} alone:", ""]
    lines += (tpm._decision_block(v["season_2025"]) if v.get("season_2025")
              else [f"No {FINAL_SEASON} records."])
    lines += ["", "## ECE per market (v2 ≤ v1 + 0.005)", "",
              "| market | ECE v1 | ECE v2 | result |", "|---|---:|---:|---|"]
    for m, e in v["ece"].items():
        lines.append(f"| {m} | {_f(e['v1'], '.4f')} | {_f(e['v2'], '.4f')} | "
                     f"{'pass' if e['pass'] else 'FAIL'} |")
    g, ci = v["game"], gate["game_ci"]["metrics"]
    lines += ["", f"## Game level (v2 ≤ v1 × {vc.GAME_TOL})", "",
              "| metric | n | v1 | v2 | v2 − v1 (95% CI, season-week clusters) | result |",
              "|---|---:|---:|---:|---|---|"]
    for m, x in g["metrics"].items():
        c = ci.get(m, {})
        lines.append(f"| {m} | {x['n']} | {_f(x['v1'], '.4f')} | {_f(x['v2'], '.4f')} | "
                     f"{_f(c.get('diff'), '+.4f')} [{_f(c.get('lo'), '+.4f')}, "
                     f"{_f(c.get('hi'), '+.4f')}] | {'pass' if x['pass'] else 'FAIL'} |")
    q = gate["qb_change"]
    lines += ["", "## Sub-check 1: QB-change games (required)", "",
              "pass_yds records whose team-week has `qb_changed` = 1 or `qb_ratio_ypa` outside "
              f"[{vc.QB_RATIO_BAND[0]}, {vc.QB_RATIO_BAND[1]}].", "",
              f"- n {q['n']} records in {q.get('n_clusters', 0)} season-weeks; RPS v1 "
              f"{_f(q['rps_v1'], '.4f')} → v2 {_f(q['rps_v2'], '.4f')}; v2 − v1 "
              f"{_f(q['diff'], '+.4f')} (95% CI [{_f(q['lo'], '+.4f')}, {_f(q['hi'], '+.4f')}]).",
              f"- QB-change sub-check: **{'PASS' if q['pass'] else 'FAIL'}** (CI must lie below 0).",
              ""]
    mu = gate["matchup"]
    lines += ["## Sub-check 2: Matchup response, both directions (required)", "",
              "Quintiles of the opponent defense's as-of `mx_pass_minus_rush` (+ = pass D weak "
              "relative to run D; edges " + ", ".join(_f(e, '+.3f') for e in mu["edges"])
              + f"; {mu['n_records']} records with a baseline). dev = Σ projection / Σ player "
              "baseline (mean of his previous 8 played games) − 1.", "",
              "| group | quintile | n | pred dev v1 | pred dev v2 | actual dev |",
              "|---|---|---:|---:|---:|---:|"]
    for t in mu["table"]:
        lines.append(f"| {t['group']} | {t['quintile']} | {t['n']} | "
                     f"{_f(t['pred_dev_v1'], '+.3f')} | {_f(t['pred_dev_v2'], '+.3f')} | "
                     f"{_f(t['act_dev'], '+.3f')} |")
    lines += ["", f"- Top quintile (RB down, QB/WR/TE up, signs = actual): "
              f"{'pass' if mu['pass_top'] else 'FAIL'}; bottom quintile (the reverse): "
              f"{'pass' if mu['pass_bottom'] else 'FAIL'}.",
              f"- Correlation of pred dev with actual dev over the 20 cells: v1 "
              f"{_f(mu['corr_v1'], '+.3f')}, v2 {_f(mu['corr_v2'], '+.3f')}.",
              f"- Matchup response sub-check: **{'PASS' if mu['pass'] else 'FAIL'}**.", "",
              "## Spot checks", "",
              "| check | week | player | market | mean v1 | mean v2 | actual | RPS v1 | RPS v2 |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for s in gate["spot_checks"]:
        if s.get("graded"):
            lines.append(f"| {s['label']} | {s['week']} | {s['player_id']} | {s['market']} | "
                         f"{s['mean_v1']:.1f} | {s['mean_v2']:.1f} | {s['actual']:.0f} | "
                         f"{s['rps_v1']:.3f} | {s['rps_v2']:.3f} |")
        else:
            note = f" ({s['note']})" if s.get("note") else ""
            lines.append(f"| {s['label']} | – | – | – | – | – | not graded{note} | – | – |")
    cov = gate["coverage"]
    lines += ["", "## Run and identity", "",
              f"- Verdict seasons {', '.join(map(str, gate['seasons']))} (tuning and rung "
              "decisions: 2021–2023); served files "
              f"`{gate['served_files']['v1']}` vs `{gate['served_files']['v2']}`; "
              f"{gate['n_paired']} paired served records.",
              f"- Game level: {gate['n_sims']} sims per game, base seed {ident['seed']} "
              f"(per-game seeded streams); refit weeks {', '.join(map(str, gate['refit_weeks']))}; "
              f"games scheduled {cov['schedule_games']}, v1 {cov['ml_games']}, v2 "
              f"{cov['elo_games']}, paired {cov['paired_games']}.",
              "- Kept A: v1 " + ", ".join(gate["a_gates"]["v1"]["kept"]) + "; v2 "
              + ", ".join(gate["a_gates"]["v2"]["kept"]) + ".",
              f"- Code: git {ident.get('git_head')}; features: player sha256 "
              f"{ident['player_features']['sha256'][:12]}, team sha256 "
              f"{ident['team_features']['sha256'][:12]}.",
              f"- Elapsed: {gate['elapsed_s'] / 60:.1f} min.", "",
              "## Caveats", "",
              "- v2's per-market sources, blend and calibration rungs were chosen by its own B "
              "ladder on 2021–2025 (as v1's were); this gate scores 2024–2025 only.",
              "- The player baseline is the player's own recent actuals, so both versions' "
              "deviations include regression to the mean; the check compares their SIGNS and "
              "how well they track the actual deviations, not their level.",
              "", f"pass = {gate['pass']}", ""]
    return "\n".join(lines)


# ---- runner -----------------------------------------------------------------------------

def run_gate_v2(env: Mapping[str, str], *, bsn, player_tbl: pd.DataFrame,
                team_tbl: pd.DataFrame, identity: dict, a_gate_v1: Mapping, a_gate_v2: Mapping,
                served_v1: pd.DataFrame, served_v2: pd.DataFrame, data_dir: Path = DATA_DIR,
                gate_path: Path = GATE_PATH, report_dir: Path = REPORT_DIR,
                log: Callable[[str], None] = print, run_date: str | None = None,
                served_files: Mapping[str, str] | None = None) -> dict:
    """Game-level runs (v1 / v2 kept A) + served-composite checks -> verdict;
    writes the gate json and report; returns the gate dict."""
    t0 = time.time()
    check_gate_names(a_gate_v1, a_gate_v2)
    raw = env.get("PROPS_ML_SEASONS")
    seasons = [int(x) for x in raw.split(",") if x.strip()] if raw else list(VERDICT_SEASONS)
    n_sims = int(env.get("PROPS_ML_N_SIMS") or DEFAULT_N_SIMS)
    resume = env.get("PROPS_ML_RESUME") == "1"
    refit_weeks = tpm.REFIT_WEEKS
    tag = tpm.run_tag(seasons, refit_weeks)
    run_date = run_date or date.today().isoformat()

    # served composites first: a key mismatch aborts before hours of backtests
    s1 = served_v1[served_v1["season"].isin(seasons)].reset_index(drop=True)
    s2 = served_v2[served_v2["season"].isin(seasons)].reset_index(drop=True)
    paired = vc.paired_served(s1, s2)
    log(f"stage=served tag={tag} seasons={seasons} paired={len(paired)}")

    seed = int(bsn.SIM_SEED)
    fetch_seasons = bsn.backtest_fetch_seasons(seasons)
    log(f"stage=fetch seasons={fetch_seasons}")
    sources = bsn.fetch_backtest_sources(fetch_seasons)
    identity = {**identity, "prod": dict(tpm.PROD), "seed": seed,
                "backtest_sources": tpm.sources_fingerprint(sources)}
    games = gml.schedule_games(sources["schedules"], seasons)
    game_recs, a_gates = {}, {}
    for label, a_gate, sfx in (("v1", a_gate_v1, ""), ("v2", a_gate_v2, V2_NAME)):
        toggles, tuned = gml.a_config(a_gate, seasons)
        a_gates[label] = {"kept": [r for r in tpm.LADDER if r in toggles],
                          "tuned": {str(k): list(v) for k, v in tuned.items()}}
        meta = {"n_sims": n_sims, "seasons": seasons, "toggles": sorted(toggles),
                "refit_weeks": list(refit_weeks), "q_weight": tpm.Q_WEIGHT,
                "tuned": {str(k): list(v) for k, v in tuned.items()},
                "margin_half_range": gml.MARGIN_HALF_RANGE, "total_max": gml.TOTAL_MAX,
                "identity": identity}
        log(f"stage=game-{label} toggles={sorted(toggles)} tuned={tuned}")
        recs, _, _ = gml.run_ml_side(
            bsn=bsn, seasons=seasons, n_sims=n_sims, player_tbl=player_tbl, team_tbl=team_tbl,
            toggles=toggles, tuned=tuned, games=games, sources=sources, seed=seed,
            refit_weeks=refit_weeks, ckpt=Path(data_dir) / f"game_records__{tag}{sfx}.parquet",
            meta=meta, resume=resume, log=log, label=f"game-{label}")
        game_recs[label] = recs
    gml.check_lines(game_recs["v2"], game_recs["v1"])
    cov = gml.coverage(games.keys(), game_recs["v1"], game_recs["v2"])
    game_diffs = game_gate.paired_diffs(game_recs["v2"], game_recs["v1"])  # _ml = v2, _elo = v1
    game_ci = game_gate.gate_decision(game_diffs, n_boot=N_BOOT, seed=BOOT_SEED)

    log("stage=checks: ECE, QB-change, matchup response, verdict")
    ece_v1, ece_v2 = vc.ece_by_market(s1), vc.ece_by_market(s2)
    qb = vc.qb_change_check(paired, vc.qb_change_mask(paired, player_tbl), n_boot=N_BOOT,
                            seed=BOOT_SEED)
    matchup = vc.matchup_response(s1, s2, player_tbl)
    verdict = vc.verdict(paired, ece_v1, ece_v2, game_diffs, qb, matchup,
                         final_season=FINAL_SEASON)
    log(f"VERDICT pass={verdict['pass']} reasons={verdict['reasons']}")

    gate = {"run_date": run_date, "seasons": seasons, "n_sims": n_sims, "run_tag": tag,
            "refit_weeks": list(refit_weeks), "identity": identity, "a_gates": a_gates,
            "served_files": dict(served_files or {"v1": SERVED_V1.name, "v2": SERVED_V2.name}),
            "n_paired": int(len(paired)), "verdict": verdict, "pass": bool(verdict["pass"]),
            "game": verdict["game"], "game_ci": {"metrics": game_ci["metrics"]},
            "coverage": cov, "qb_change": qb, "matchup": matchup,
            "spot_checks": spot_checks(served_v1, served_v2, player_tbl),
            "elapsed_s": time.time() - t0}
    out_gate, out_report = output_paths(tag, run_date, gate_path=gate_path, report_dir=report_dir)
    text = gml.gate_json(gate)
    out_gate.parent.mkdir(parents=True, exist_ok=True)
    out_gate.write_text(text)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(render_report(json.loads(text)))  # same values as the json
    log(f"stage=write {out_gate} and {out_report}")
    return gate


# ---- IO ---------------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[+{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)

    for p in (tpm.PLAYER_PATH, tpm.TEAM_PATH, A_GATE_V1, A_GATE_V2, SERVED_V1, SERVED_V2):
        if not p.exists():
            raise SystemExit(f"missing {p}: run the v1 and v2 A / B ladders first")
    identity = {"player_features": tpm.file_fingerprint(tpm.PLAYER_PATH),
                "team_features": tpm.file_fingerprint(tpm.TEAM_PATH), "git_head": tpm.git_head()}
    log(f"git={identity['git_head'][:12]}")
    run_gate_v2(os.environ, bsn=tpm._load_backtest(), player_tbl=pd.read_parquet(tpm.PLAYER_PATH),
                team_tbl=pd.read_parquet(tpm.TEAM_PATH), identity=identity,
                a_gate_v1=json.loads(A_GATE_V1.read_text()),
                a_gate_v2=json.loads(A_GATE_V2.read_text()),
                served_v1=pd.read_parquet(SERVED_V1), served_v2=pd.read_parquet(SERVED_V2),
                log=log)


if __name__ == "__main__":
    main()
