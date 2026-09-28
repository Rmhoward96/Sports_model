"""Walk-forward game-level gate: ML sim vs the legacy Elo game model (NFL).

Decides whether the ML sim replaces the Elo/SoS game model for NFL
moneyline, spread and total (plan 2026-09-28-ml-only-nfl, Task 2).

ML side
-------
The Props-1 A configuration (kept toggles + per-season tuned (decay,
max_iter) from ``assets/nfl/props_ml/a_gate.json``) on the production sim
settings (``train_props_ml.PROD``). For test season S, models are refit
before each refit week r (``REFIT_WEEKS``) with ``learned.fit_models(...,
upto=(S, r))``, which trains only on rows strictly before (S, r); a game in
week w uses block ``refit_block(w)`` via ``train_props_ml.make_hook``.
``backtest_sim_nfl.run_backtest`` simulates every completed REG game with a
per-game seeded stream; its ``on_game`` callback turns each game's sims into
one record (``ml_record``): home win / cover / over probabilities from the
sim's margin and total pmfs (``sim.engine``; margin half-range
``MARGIN_HALF_RANGE`` and total max ``TOTAL_MAX`` wide enough that no NFL
line is affected by the engine's tail clipping), predicted margin/total =
sim means. A spec_hook error aborts the run (``guard_hook``); a game the
backtest skips for another reason is reported as ML-missing coverage.

Elo side
--------
``backtest_nfl_gameline.per_game_predictions`` with the committed configs
(``assets/nfl/{rating,gameline}.json``, as ``generate_nfl.py`` loads them),
computed as its two halves (``_raw_model_predictions`` then ``_apply_gl``)
so the one walk also yields the informational "as served" variant (no
closing-line shrink, ``served_gl_cfg``). Its input schedule is the committed
history before the fetch window plus the SAME nflverse schedules the ML
backtest used (``elo_history``), so both models see identical closing lines.
Cover/over probabilities are Normal around the per-game pred margin/total
with the committed ``GameLineConfig`` sigmas; win prob is the per-game one.

Pairing / decision
------------------
Both sides' records take spread_line / total_line and actuals from the same
schedule rows; ``check_lines`` aborts if a paired game's lines or actuals
differ (NaN == NaN). Games missing on either side are reported
(``coverage``) and excluded. ``game_gate.gate_decision`` on the paired
frame gives the verdict (Brier ML <= Elo, MAE ML <= 1.01 x Elo; 95%
season-week cluster bootstrap CIs).

Env
---
PROPS_ML_SEASONS  comma list of test seasons (default 2021..2025)
PROPS_ML_N_SIMS   sims per game (default 1000)
PROPS_ML_RESUME   1 -> reuse ``data/props_ml/game_records__<tag>.parquet``
                  when its sidecar meta (config, tuned values, feature
                  fingerprints, sources fingerprint, git HEAD, PROD, seed)
                  matches exactly

Outputs: the default run (2021-2025, every 4 weeks) writes
``assets/nfl/props_ml/game_gate.json`` and
``docs/superpowers/reports/<date>-ml-game-gate.md``; other tags add
``__<tag>``. No DB writes.

PURE / IO split: everything except ``main()`` is unit tested
(tests/scripts/test_gate_ml_game_lines.py, stubbed backtest + Elo walk).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sportsmodel.model import game_gate  # noqa: E402
from sportsmodel.nfl.gameline import GameLineConfig  # noqa: E402
from sportsmodel.nfl.shrink import ShrinkParams  # noqa: E402
from sportsmodel.nfl.teams import normalize_team  # noqa: E402
from sportsmodel.sim import engine  # noqa: E402
from sportsmodel.sim.nfl import learned  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tpm = _load_script("train_props_ml")          # Props-1 harness (hook, fingerprints, checkpoints)
GL = _load_script("backtest_nfl_gameline")    # Elo walk-forward (per_game_predictions halves)

TEST_SEASONS = list(tpm.TEST_SEASONS)
DEFAULT_TAG = tpm.DEFAULT_TAG
DEFAULT_N_SIMS = 1000
MARGIN_HALF_RANGE = 80   # engine.margin_pmf default 25 would clip real NFL margins
TOTAL_MAX = 150          # engine.total_pmf default 30 would truncate NFL totals
N_BOOT = 1000
BOOT_SEED = 0

DATA_DIR = tpm.DATA_DIR
PLAYER_PATH, TEAM_PATH = tpm.PLAYER_PATH, tpm.TEAM_PATH
A_GATE_PATH = tpm.GATE_PATH
GATE_PATH = ROOT / "assets" / "nfl" / "props_ml" / "game_gate.json"
REPORT_DIR = ROOT / "docs" / "superpowers" / "reports"
NFL_ASSETS = ROOT / "assets" / "nfl"

_ZERO_SHRINK = ShrinkParams(start=0.0, floor=0.0, decay=0.0)


# ---- pure helpers -----------------------------------------------------------------------

def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _line(x) -> float:
    """A closing line / actual as float, NaN when missing (None or NaN)."""
    return float("nan") if _is_missing(x) else float(x)


def _key(r: Mapping) -> tuple[int, int, str]:
    return int(r["season"]), int(r["week"]), str(r["home"])


def _fmt_key(k) -> str:
    return f"{k[0]} wk{k[1]} {k[2]}"


def a_config(a_gate: Mapping, seasons: Iterable[int]) -> tuple[frozenset[str], dict[int, tuple[float, int]]]:
    """(kept toggles, {season: (decay, max_iter)}) from ``a_gate.json`` (ruling 2).

    ValueError when nothing was kept or a test season has no tuned values.
    """
    kept = frozenset(a_gate.get("kept") or [])
    if not kept:
        raise ValueError("a_gate.json kept no rung; there is no A configuration to gate")
    tuned_raw = a_gate.get("tuned") or {}
    tuned: dict[int, tuple[float, int]] = {}
    for s in seasons:
        v = tuned_raw.get(str(s))
        if v is None:
            raise ValueError(f"a_gate.json has no tuned (decay, max_iter) for season {s}; "
                             f"tuned seasons: {sorted(tuned_raw)}")
        tuned[int(s)] = (float(v[0]), int(v[1]))
    return kept, tuned


def schedule_games(schedules: pd.DataFrame, seasons: Iterable[int]) -> dict[tuple[int, int, str], dict]:
    """``(season, week, home) -> {away, spread_line, total_line, actual_margin,
    actual_total}`` for completed REG games of ``seasons`` (teams normalized;
    missing lines NaN). The single source of both models' lines and actuals."""
    s = schedules[schedules["season"].isin(list(seasons))
                  & (schedules["game_type"] == "REG")
                  & schedules["home_score"].notna() & schedules["away_score"].notna()]
    out: dict[tuple[int, int, str], dict] = {}
    for r in s.sort_values(["season", "week"], kind="stable").itertuples(index=False):
        key = (int(r.season), int(r.week), normalize_team(r.home_team))
        if key in out:
            raise ValueError(f"duplicate schedule game {_fmt_key(key)}")
        hs, as_ = float(r.home_score), float(r.away_score)
        out[key] = {"away": normalize_team(r.away_team), "spread_line": _line(r.spread_line),
                    "total_line": _line(r.total_line), "actual_margin": hs - as_,
                    "actual_total": hs + as_}
    return out


def ml_record(season: int, week: int, home: str, sims, game: Mapping) -> dict:
    """One game-gate record from a game's sims (engine pmfs; ruling 4).

    game_gate's pmf ``offset`` is the margin value at index 0, the negation of
    ``engine.margin_pmf``'s stored ``"offset"``.
    """
    m = engine.margin_pmf(sims, half_range=MARGIN_HALF_RANGE)
    t = engine.total_pmf(sims, max_total=TOTAL_MAX)
    first = -int(m["offset"])
    scores = engine.pred_scores(sims)
    spread, total = _line(game["spread_line"]), _line(game["total_line"])
    return {"season": int(season), "week": int(week), "home": str(home), "away": str(game["away"]),
            "win_prob": game_gate.win_prob_pmf(m["pmf"], first),
            "cover_prob": game_gate.cover_prob_pmf(m["pmf"], first, spread),
            "over_prob": game_gate.over_prob_pmf(t, total),
            "pred_margin": float(scores["pred_margin"]), "pred_total": float(scores["pred_total"]),
            "actual_margin": float(game["actual_margin"]), "actual_total": float(game["actual_total"]),
            "spread_line": spread, "total_line": total}


def elo_history(history: pd.DataFrame, fetched: pd.DataFrame, fetch_seasons: Iterable[int]) -> pd.DataFrame:
    """The Elo walk's schedule: committed history for seasons before the fetch
    window (Elo warm-up) plus the fetched nflverse schedules inside it (the
    rows the ML backtest used). REG games only; teams normalized."""
    lo, hi = min(fetch_seasons), max(fetch_seasons)
    parts = [history[history["season"] < lo], fetched[fetched["season"].between(lo, hi)]]
    out = pd.concat(parts, ignore_index=True)
    out = out[out["game_type"] == "REG"].copy()
    for col in ("home_team", "away_team"):
        out[col] = out[col].map(normalize_team)
    return out.sort_values(["season", "week"], kind="stable").reset_index(drop=True)


def served_gl_cfg(gl_cfg: GameLineConfig) -> GameLineConfig:
    """``gl_cfg`` without closing-line shrink (what ``generate_nfl.py`` serves:
    model-only, since it passes no market line). Informational comparator."""
    return dataclasses.replace(gl_cfg, w_margin=_ZERO_SHRINK, w_total=_ZERO_SHRINK)


def elo_records(preds: Iterable[Mapping], gl_cfg: GameLineConfig, seasons: Iterable[int]) -> list[dict]:
    """Game-gate records from ``per_game_predictions`` rows of ``seasons``.

    Cover/over probs are Normal(pred, gameline sigma) above the closing line
    (ruling 3); win prob is the per-game prediction's.
    """
    keep = {int(s) for s in seasons}
    out = []
    for p in preds:
        if int(p["season"]) not in keep:
            continue
        spread, total = _line(p["spread_line"]), _line(p["total_line"])
        out.append({"season": int(p["season"]), "week": int(p["week"]),
                    "home": normalize_team(p["home_team"]), "away": normalize_team(p["away_team"]),
                    "win_prob": float(p["win_prob"]),
                    "cover_prob": game_gate.cover_prob_normal(p["pred_margin"], gl_cfg.sigma_margin, spread),
                    "over_prob": game_gate.over_prob_normal(p["pred_total"], gl_cfg.sigma_total, total),
                    "pred_margin": float(p["pred_margin"]), "pred_total": float(p["pred_total"]),
                    "actual_margin": float(p["actual_margin"]), "actual_total": float(p["actual_total"]),
                    "spread_line": spread, "total_line": total})
    return out


_CHECKED = ("spread_line", "total_line", "actual_margin", "actual_total")


def check_lines(ml_recs: Iterable[Mapping], elo_recs: Iterable[Mapping]) -> int:
    """Number of paired games; RuntimeError if any paired game's closing lines
    or actuals differ between the two models (NaN == NaN)."""
    ml = {_key(r): r for r in ml_recs}
    bad = []
    n = 0
    for r in elo_recs:
        k = _key(r)
        if k not in ml:
            continue
        n += 1
        for f in _CHECKED:
            a, b = _line(ml[k].get(f)), _line(r.get(f))
            if not ((math.isnan(a) and math.isnan(b)) or a == b):
                bad.append(f"{_fmt_key(k)} {f}: ML {a} vs Elo {b}")
    if bad:
        raise RuntimeError(f"{len(bad)} paired line/actual mismatch(es) between the ML and Elo "
                           "records (they must come from the same schedule rows):\n  "
                           + "\n  ".join(bad[:20]))
    return n


def coverage(schedule_keys: Iterable[tuple[int, int, str]], ml_recs: Iterable[Mapping],
             elo_recs: Iterable[Mapping]) -> dict:
    """Game counts per side and the scheduled games each side is missing."""
    sched = set(schedule_keys)
    ml = {_key(r) for r in ml_recs}
    elo = {_key(r) for r in elo_recs}
    return {"schedule_games": len(sched), "ml_games": len(ml), "elo_games": len(elo),
            "paired_games": len(ml & elo),
            "ml_missing": [list(k) for k in sorted(sched - ml)],
            "elo_missing": [list(k) for k in sorted(sched - elo)]}


def output_paths(tag: str, run_date: str, *, gate_path: Path = GATE_PATH,
                 report_dir: Path = REPORT_DIR) -> tuple[Path, Path]:
    """(gate json, report md); tags other than DEFAULT_TAG get ``__<tag>``."""
    suffix = "" if tag == DEFAULT_TAG else f"__{tag}"
    gate_path = Path(gate_path)
    return (gate_path.with_name(f"{gate_path.stem}{suffix}{gate_path.suffix}"),
            Path(report_dir) / f"{run_date}-ml-game-gate{suffix}.md")


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def gate_json(gate: dict) -> str:
    """Strict JSON (NaN -> null, numpy -> native)."""
    return json.dumps(_nan_to_none(tpm.to_jsonable(gate)), indent=2, allow_nan=False) + "\n"


# ---- report -----------------------------------------------------------------------------

_RULE = {"win_brier": "ML ≤ Elo", "cover_brier": "ML ≤ Elo", "over_brier": "ML ≤ Elo",
         "margin_mae": f"ML ≤ {game_gate.MAE_TOLERANCE} × Elo",
         "total_mae": f"ML ≤ {game_gate.MAE_TOLERANCE} × Elo"}


def _f(x, spec: str) -> str:
    return "n/a" if _is_missing(x) else format(x, spec)


def _metric_ok(metric: str, v: dict) -> bool:
    if metric in ("margin_mae", "total_mae"):
        return v["ml"] <= game_gate.MAE_TOLERANCE * v["elo"]
    return v["ml"] <= v["elo"]


def _decision_table(decision: dict) -> list[str]:
    lines = ["| metric | paired n | ML | Elo | ML − Elo | 95% CI (cluster bootstrap) | rule | result |",
             "|---|---:|---:|---:|---:|---|---|---|"]
    for metric in game_gate.METRICS:
        v = decision["metrics"].get(metric)
        if v is None:
            continue
        ok = v["n"] > 0 and _metric_ok(metric, v)
        lines.append(f"| {metric} | {v['n']} | {_f(v['ml'], '.4f')} | {_f(v['elo'], '.4f')} | "
                     f"{_f(v['diff'], '+.4f')} | [{_f(v['lo'], '+.4f')}, {_f(v['hi'], '+.4f')}] | "
                     f"{_RULE[metric]} | {'pass' if ok else 'FAIL'} |")
    return lines


def _verdict(gate: dict) -> str:
    seasons = ", ".join(str(s) for s in gate["seasons"])
    n = gate["coverage"]["paired_games"]
    if gate["pass"]:
        return (f"**Verdict: PASS.** The ML sim meets the game-level gate against the Elo game "
                f"model on seasons {seasons} ({n} paired games): win, cover and over Brier are no "
                f"worse than Elo's and margin/total MAE are within {game_gate.MAE_TOLERANCE}× of "
                "Elo's. The ML sim may replace the Elo model for NFL moneyline, spread and total.")
    reasons = "; ".join(gate["decision"]["reasons"])
    return (f"**Verdict: FAIL.** The ML sim does not meet the game-level gate against the Elo "
            f"game model on seasons {seasons} ({n} paired games). Failing: {reasons}. The Elo "
            "model stays the NFL game-line source.")


def _shrink_txt(sp) -> str:
    return f"start {sp['start']}, floor {sp['floor']}, decay {sp['decay']}"


def render_report(gate: dict) -> str:
    """Markdown report; the verdict is the first paragraph after the title."""
    ident, cov, elo_cfg = gate["identity"], gate["coverage"], gate["elo_config"]
    gl = elo_cfg.get("gameline", {})
    sigma_m, sigma_t = gate["elo_sigmas"]["margin"], gate["elo_sigmas"]["total"]
    shrink = ""
    if "w_margin" in gl and "w_total" in gl:
        shrink = (f" (margin w(week): {_shrink_txt(gl['w_margin'])}; total w(week): "
                  f"{_shrink_txt(gl['w_total'])})")
    lines = [f"# ML sim vs Elo — NFL game-level gate — {gate['run_date']}", "",
             _verdict(gate), "",
             "## Paired comparison", "",
             "ML sim (walk-forward Props-1 A configuration) vs the Elo comparator "
             "(`backtest_nfl_gameline.per_game_predictions`, committed configs) on the same "
             "games; ML − Elo < 0 favours the ML sim. Gate: Brier point estimates ML ≤ Elo; "
             f"MAE ML ≤ {game_gate.MAE_TOLERANCE} × Elo.", ""]
    lines += _decision_table(gate["decision"])
    lines += ["", "## Caveats", "",
              "- **In-sample edge for Elo.** Elo's configuration (`assets/nfl/rating.json`, "
              "`assets/nfl/gameline.json`: K/HFA/carryover, SoS blend, sigmas "
              f"{sigma_m:.3f}/{sigma_t:.3f} and the closing-line shrink curves) was fit by "
              "`backtest_nfl_elo.py` / `backtest_nfl_gameline.py` on seasons ≤ 2019 and selected "
              "against 2020+ as its validation span, so these gate seasons informed Elo's "
              "configuration. Treat Elo's numbers here as in-sample; this biases the gate toward "
              "NOT switching.",
              "- **The Elo comparator uses the closing line.** `per_game_predictions` shrinks "
              f"Elo's margin/total toward each game's closing spread/total by week{shrink}; "
              "production `generate_nfl.py` serves Elo model-only (no market line). The gated "
              "comparator is therefore stronger than what is served; the as-served (no-shrink) "
              "comparison is below for information only.",
              "- The ML A configuration (kept toggles, per-season tuned decay/max_iter) was "
              "selected in the props-ML A gate on these same seasons (player props, not game "
              "lines).",
              f"- ML probabilities are Monte Carlo estimates from {gate['n_sims']} sims per game.",
              "", "## Informational: Elo as served (no closing-line shrink)", ""]
    lines += _decision_table(gate["elo_served"])
    lines += ["", f"(pass under the gate rules: {gate['elo_served']['pass']}; not the verdict)",
              "", "## Coverage", "",
              f"- Scheduled completed REG games: {cov['schedule_games']}; ML sim: "
              f"{cov['ml_games']}; Elo: {cov['elo_games']}; paired: {cov['paired_games']}.",
              "- Paired n per metric (games where both models are scorable): "
              + ", ".join(f"{m} {v['n']}" for m, v in gate["decision"]["metrics"].items()) + ".",
              f"- ML-missing ({len(cov['ml_missing'])}): "
              + (", ".join(_fmt_key(k) for k in cov["ml_missing"]) or "none") + ".",
              f"- Elo-missing ({len(cov['elo_missing'])}): "
              + (", ".join(_fmt_key(k) for k in cov["elo_missing"]) or "none") + ".",
              "- Missing games are excluded from the paired comparison. Both models take their "
              "closing lines and actuals from the same nflverse schedule rows (checked per game).",
              "", "## Standalone metrics (each model's own games)", "",
              "| model | games | win Brier | cover Brier (n) | over Brier (n) | margin MAE | total MAE |",
              "|---|---:|---:|---|---|---:|---:|"]
    for name, m in (("ML sim", gate["ml_metrics"]), ("Elo", gate["elo_metrics"])):
        lines.append(f"| {name} | {m['n_win']} | {_f(m['win_brier'], '.4f')} | "
                     f"{_f(m['cover_brier'], '.4f')} ({m['n_cover']}) | "
                     f"{_f(m['over_brier'], '.4f')} ({m['n_over']}) | "
                     f"{_f(m['margin_mae'], '.3f')} | {_f(m['total_mae'], '.3f')} |")
    ms = gate["ml_stats"]
    lines += ["", "## Run and identity", "",
              f"- Test seasons: {', '.join(str(s) for s in gate['seasons'])}; sims per game: "
              f"{gate['n_sims']}; base seed {ident['seed']} (per-game seeded streams).",
              f"- Refit weeks {', '.join(str(w) for w in gate['refit_weeks'])}: models for "
              "(S, r) train only on rows strictly before (S, r).",
              f"- A configuration: toggles {', '.join(gate['toggles'])}; tuned (decay, max_iter) "
              + ", ".join(f"{s}: {v[0]}/{v[1]}" for s, v in sorted(gate["tuned"].items()))
              + f"; learned-share questionable multiplier {gate['q_weight']}.",
              "- Production sim settings: "
              + ", ".join(f"{k} {v}" for k, v in ident["prod"].items()) + ".",
              f"- Code: git {ident.get('git_head')}.",
              f"- Features: player sha256 {ident['player_features']['sha256'][:12]} "
              f"({ident['player_features']['size']} B), team sha256 "
              f"{ident['team_features']['sha256'][:12]} ({ident['team_features']['size']} B); "
              f"same files a_gate.json was tuned on: {ident['a_gate_features_match']}.",
              "- Backtest sources: "
              + ", ".join(f"{k} {v['rows']} rows ({v['hash'][:12]})"
                          for k, v in ident["backtest_sources"].items()) + ".",
              f"- Elo configs: rating {json.dumps(elo_cfg.get('rating', {}), sort_keys=True)}; "
              f"gameline {json.dumps(gl, sort_keys=True)}.",
              f"- ML run: {ms.get('games')} games in {ms.get('seconds', 0.0) / 60:.1f} min; "
              f"team-sides with share fallback {ms.get('share_fallbacks')}; "
              f"{'from checkpoint' if gate['resumed'] else 'fresh'} "
              f"(`game_records__{gate['run_tag']}.parquet`).",
              f"- Elapsed: {gate['elapsed_s'] / 60:.1f} min.", "",
              f"pass = {gate['pass']}", ""]
    return "\n".join(lines)


# ---- runner -----------------------------------------------------------------------------

def run_gate(env: Mapping[str, str], *, bsn, player_tbl: pd.DataFrame, team_tbl: pd.DataFrame,
             identity: dict, a_gate: Mapping, history_sched: pd.DataFrame,
             elo_walk: Callable[[pd.DataFrame], list[dict]], gl_cfg: GameLineConfig,
             elo_config: dict, data_dir: Path = DATA_DIR, gate_path: Path = GATE_PATH,
             report_dir: Path = REPORT_DIR, log: Callable[[str], None] = print,
             run_date: str | None = None) -> dict:
    """ML-sim records (walk-forward, checkpointed) + Elo records -> paired gate;
    writes the gate json and report; returns the gate dict.

    ``bsn`` is the ``backtest_sim_nfl`` module; ``elo_walk(schedule)`` is
    ``backtest_nfl_gameline._raw_model_predictions`` bound to the committed
    Elo/blend configs (``_apply_gl`` with ``gl_cfg`` completes
    ``per_game_predictions``).
    """
    t0 = time.time()
    env_seasons = env.get("PROPS_ML_SEASONS")
    seasons = ([int(x) for x in env_seasons.split(",") if x.strip()]
               if env_seasons else list(TEST_SEASONS))
    n_sims = int(env.get("PROPS_ML_N_SIMS") or DEFAULT_N_SIMS)
    resume = env.get("PROPS_ML_RESUME") == "1"
    refit_weeks = tpm.REFIT_WEEKS
    tag = tpm.run_tag(seasons, refit_weeks)
    run_date = run_date or date.today().isoformat()
    toggles, tuned = a_config(a_gate, seasons)

    seed = int(bsn.SIM_SEED)
    fetch_seasons = bsn.backtest_fetch_seasons(seasons)
    log(f"stage=fetch seasons={fetch_seasons}")
    sources = bsn.fetch_backtest_sources(fetch_seasons)
    a_ident = a_gate.get("identity", {})
    identity = {**identity, "prod": dict(tpm.PROD), "seed": seed,
                "backtest_sources": tpm.sources_fingerprint(sources)}
    features_match = all(
        (a_ident.get(k) or {}).get("sha256") == (identity.get(k) or {}).get("sha256")
        for k in ("player_features", "team_features"))
    games = schedule_games(sources["schedules"], seasons)
    log(f"stage=setup tag={tag} seasons={seasons} n_sims={n_sims} resume={resume} "
        f"toggles={sorted(toggles)} tuned={tuned} scheduled_games={len(games)} seed={seed} "
        f"a_gate_features_match={features_match}")

    meta = {"n_sims": n_sims, "seasons": seasons, "toggles": sorted(toggles),
            "refit_weeks": list(refit_weeks), "q_weight": tpm.Q_WEIGHT,
            "tuned": {str(k): list(v) for k, v in tuned.items()},
            "margin_half_range": MARGIN_HALF_RANGE, "total_max": TOTAL_MAX, "identity": identity}
    ckpt = Path(data_dir) / f"game_records__{tag}.parquet"
    got = tpm.load_records(ckpt, meta) if resume else None
    if got is not None:
        ml_recs, ml_stats = got
        log(f"stage=backtest: loaded {len(ml_recs)} ML game records from checkpoint {ckpt.name}")
    else:
        if resume:
            log(f"stage=backtest: no matching checkpoint {ckpt.name}, running")
        models: dict = {}
        for s in seasons:
            decay, max_iter = tuned[s]
            for r in refit_weeks:
                log(f"stage=fit season={s} block={r} toggles={sorted(toggles)} "
                    f"decay={decay} max_iter={max_iter}")
                models[(s, r)] = learned.fit_models(player_tbl, team_tbl, toggles, upto=(s, r),
                                                    test_season=s, decay=decay, max_iter=max_iter)
        hook = tpm.make_hook(models, player_tbl, team_tbl, tpm.questionable_index(player_tbl),
                             refit_weeks=refit_weeks, q_weight=tpm.Q_WEIGHT)
        hook, errors = tpm.guard_hook(hook)
        ml_recs = []
        start = time.time()
        state = {"block": None}

        def on_game(season, week, home, away, sims):
            key = (int(season), int(week), str(home))
            if key not in games:
                raise RuntimeError(f"backtest simulated {_fmt_key(key)}, which is not a scheduled "
                                   "completed REG game")
            blk = (key[0], tpm.refit_block(key[1], refit_weeks))
            if blk != state["block"]:
                state["block"] = blk
                log(f"stage=backtest season={blk[0]} block={blk[1]} week={key[1]} "
                    f"games_so_far={len(ml_recs)} run_min={(time.time() - start) / 60:.1f}")
            ml_recs.append(ml_record(key[0], key[1], key[2], sims, games[key]))

        log(f"stage=backtest start n_sims={n_sims}")
        bsn.run_backtest(seasons, n_sims, seed=seed, on_game=on_game, spec_hook=hook,
                         sources=sources, **tpm.PROD)
        tpm.raise_hook_errors(errors, "ml-game")
        ml_stats = {"seconds": time.time() - start, "games": len(ml_recs),
                    "share_fallbacks": sum(int(m.share_fallbacks) for m in models.values())}
        tpm.check_share_fallbacks(ml_stats["share_fallbacks"], len(ml_recs), "ml-game")
        tpm.save_records(ckpt, ml_recs, meta, ml_stats)
        log(f"stage=backtest done: {len(ml_recs)} games in {ml_stats['seconds'] / 60:.1f} min "
            f"-> {ckpt.name}")

    log("stage=elo walk-forward (per_game_predictions, committed configs)")
    raw = elo_walk(elo_history(history_sched, sources["schedules"], fetch_seasons))
    elo_recs = elo_records(GL._apply_gl(raw, gl_cfg), gl_cfg, seasons)
    served_recs = elo_records(GL._apply_gl(raw, served_gl_cfg(gl_cfg)), gl_cfg, seasons)
    log(f"stage=elo done: {len(elo_recs)} Elo game records")

    log("stage=gate pairing + cluster bootstrap")
    check_lines(ml_recs, elo_recs)
    cov = coverage(games.keys(), ml_recs, elo_recs)
    decision = game_gate.gate_decision(game_gate.paired_diffs(ml_recs, elo_recs),
                                       n_boot=N_BOOT, seed=BOOT_SEED)
    served = game_gate.gate_decision(game_gate.paired_diffs(ml_recs, served_recs),
                                     n_boot=N_BOOT, seed=BOOT_SEED)
    for metric, v in decision["metrics"].items():
        log(f"GATE {metric}: n={v['n']} ML {v['ml']:.4f} Elo {v['elo']:.4f} diff {v['diff']:+.4f} "
            f"CI [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    log(f"GATE coverage: schedule {cov['schedule_games']} ML {cov['ml_games']} Elo "
        f"{cov['elo_games']} paired {cov['paired_games']}; pass={decision['pass']} "
        f"reasons={decision['reasons']}")

    gate = {"run_date": run_date, "seasons": seasons, "n_sims": n_sims, "run_tag": tag,
            "refit_weeks": list(refit_weeks), "toggles": sorted(toggles),
            "tuned": {str(k): list(v) for k, v in tuned.items()}, "q_weight": tpm.Q_WEIGHT,
            "identity": {**identity, "a_gate_features_match": features_match},
            "elo_config": elo_config,
            "elo_sigmas": {"margin": gl_cfg.sigma_margin, "total": gl_cfg.sigma_total},
            "coverage": cov, "decision": decision, "pass": bool(decision["pass"]),
            "elo_served": served, "ml_metrics": game_gate.game_metrics(ml_recs),
            "elo_metrics": game_gate.game_metrics(elo_recs), "ml_stats": ml_stats,
            "resumed": got is not None, "elapsed_s": time.time() - t0}
    out_gate, out_report = output_paths(tag, run_date, gate_path=gate_path, report_dir=report_dir)
    out_gate.parent.mkdir(parents=True, exist_ok=True)
    out_gate.write_text(gate_json(gate))
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(render_report(json.loads(gate_json(gate))))  # same values as the json
    log(f"stage=write {out_gate} and {out_report}")
    return gate


# ---- IO ---------------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[+{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)

    if not PLAYER_PATH.exists() or not TEAM_PATH.exists():
        raise SystemExit(f"missing {PLAYER_PATH} / {TEAM_PATH}: build them first with "
                         "`uv run python scripts/build_player_features.py`")
    from sportsmodel.nfl import config as nfl_config

    identity = {"player_features": tpm.file_fingerprint(PLAYER_PATH),
                "team_features": tpm.file_fingerprint(TEAM_PATH), "git_head": tpm.git_head()}
    log(f"git={identity['git_head'][:12]}")
    elo_cfg, blend_cfg = nfl_config.load_rating()
    gl_cfg = nfl_config.load_gameline()
    elo_config = {"rating": json.loads((NFL_ASSETS / "rating.json").read_text()),
                  "gameline": json.loads((NFL_ASSETS / "gameline.json").read_text())}
    run_gate(os.environ, bsn=tpm._load_backtest(), player_tbl=pd.read_parquet(PLAYER_PATH),
             team_tbl=pd.read_parquet(TEAM_PATH), identity=identity,
             a_gate=json.loads(A_GATE_PATH.read_text()),
             history_sched=pd.read_parquet(NFL_ASSETS / "schedules.parquet"),
             elo_walk=lambda sched: GL._raw_model_predictions(sched, elo_cfg, blend_cfg),
             gl_cfg=gl_cfg, elo_config=elo_config, log=log)


if __name__ == "__main__":
    main()
