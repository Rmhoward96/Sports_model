"""Walk-forward ablation-ladder harness for the props-ML A gate (spec §4).

Runs the current NFL sim (production config) as the baseline, then climbs the
ladder ``volume -> efficiency -> context -> market``: each rung's candidate
(``next_candidate(kept, rung)``) swaps learned inputs into every game's spec
via ``backtest_sim_nfl.run_backtest``'s ``spec_hook``. On the baseline's fixed
projected-usage population a rung passes iff (a) ``rung_decision`` against the
CURRENTLY KEPT configuration's records passes (baseline until a rung passes)
AND (b) no market's PIT decile ECE exceeds the BASELINE's by more than 0.005
(``baseline_ece_check``; stops calibration drifting across rungs). A failed
rung is skipped and the ladder continues. The final gate compares kept vs
baseline on all seasons and on 2025 alone; ``final_pass`` needs both and a
non-empty kept set.

Walk-forward / leakage
----------------------
For test season S, models are refit before each refit week r
(``REFIT_WEEKS``; ``PROPS_ML_WEEKLY_REFIT=1`` -> every week) with
``fit_models(..., upto=(S, r))``, which trains only on rows strictly before
(S, r). A game in week w uses the models of block ``refit_block(w)``.
``(decay, max_iter)`` is tuned ONCE per test season (``learned.tune`` with
toggles ``{"volume"}``, seasons < S only) and reused for every rung.
``run_backtest`` seeds every game from (seed, season, week, home, away):
per-game seeded random streams (game-level alignment across runs).

No silently dropped games
-------------------------
``run_backtest`` skips (and only prints) a game whose spec/sim raises. The
hook is wrapped (``guard_hook``) so any hook error aborts the run afterwards,
and every candidate run (fresh or from a checkpoint) must cover exactly the
baseline's (season, week, home) games (``check_coverage``), else abort.

Env
---
PROPS_ML_SEASONS      comma list of test seasons (default 2021..2025)
PROPS_ML_N_SIMS       sims per game (default 1000)
PROPS_ML_WEEKLY_REFIT 1 -> refit before every week 1..18
PROPS_ML_RESUME       1 -> load a run's records checkpoint
                      (data/props_ml/records_<name>.parquet) instead of
                      re-running it, when its sidecar meta (config, tuned
                      values, feature-file sha256s, git HEAD, PROD, seed)
                      matches exactly

Outputs: ``assets/nfl/props_ml/a_gate.json`` and
``docs/superpowers/reports/<run date>-props-ml-a-gate.md``.

PURE / IO split: everything except ``main()`` / ``_load_backtest`` is unit
tested (tests/scripts/test_train_props_ml.py); ``main()`` (nflverse fetches,
fits, backtests) is not.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel.model.props_eval import (  # noqa: E402
    decile_ece,
    paired_frame,
    population_from_baseline,
    rung_decision,
)
from sportsmodel.sim.nfl import learned  # noqa: E402

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
REFIT_WEEKS = (1, 5, 9, 13, 17)
WEEKLY_REFIT_WEEKS = tuple(range(1, 19))
LADDER = ("volume", "efficiency", "context", "market")
PROD = dict(season_decay=0.4, questionable_weight=0.75, home_field=0.07, ratings_weight=0.5)
# Questionable multiplier on LEARNED shares: 1.0 (none). The count models train
# on the sim's population (stubs as 0) with st_questionable as a feature, so a
# further down-weight would double-count. Baseline shares keep PROD's 0.75.
Q_WEIGHT = 1.0
TUNE_TOGGLES = frozenset({"volume"})
FINAL_SEASON = 2025
DEFAULT_N_SIMS = 1000
ECE_TOL = 0.005  # same tolerance rung_decision applies vs the kept config

DATA_DIR = ROOT / "data" / "props_ml"
PLAYER_PATH = DATA_DIR / "player_week_features.parquet"
TEAM_PATH = DATA_DIR / "team_week_features.parquet"
GATE_PATH = ROOT / "assets" / "nfl" / "props_ml" / "a_gate.json"
REPORT_DIR = ROOT / "docs" / "superpowers" / "reports"

ALIGNMENT = "per-game seeded random streams (game-level alignment across runs)"


# ---- pure helpers: schedule / ladder / hook ---------------------------------------------

def refit_block(week: int, refit_weeks: tuple[int, ...] = REFIT_WEEKS) -> int:
    """The largest refit week <= ``week``."""
    blocks = [r for r in refit_weeks if r <= week]
    if not blocks:
        raise ValueError(f"week {week} precedes the first refit week {min(refit_weeks)}")
    return max(blocks)


def refit_weeks_from_env(env: Mapping[str, str]) -> tuple[int, ...]:
    """``PROPS_ML_WEEKLY_REFIT=1`` -> weekly refits, else ``REFIT_WEEKS``."""
    return WEEKLY_REFIT_WEEKS if env.get("PROPS_ML_WEEKLY_REFIT") == "1" else REFIT_WEEKS


def next_candidate(kept: frozenset[str], rung: str) -> frozenset[str]:
    """``kept | {rung}``; every rung implies ``volume`` (learned inputs are
    only ever used on top of learned volume), so if ``volume`` failed a later
    rung is still evaluated as ``{"volume", rung}``."""
    return frozenset(kept | {rung, "volume"})


def questionable_index(player_tbl: pd.DataFrame) -> dict[tuple[int, int, str], set[str]]:
    """``(season, week, team) -> {player_id}`` of rows with ``st_questionable == 1``."""
    q = player_tbl[player_tbl["st_questionable"] == 1]
    out: dict[tuple[int, int, str], set[str]] = {}
    for (s, w, t), g in q.groupby(["season", "week", "team"]):
        out[(int(s), int(w), str(t))] = set(g["player_id"])
    return out


def make_hook(models_by_block: Mapping, player_tbl: pd.DataFrame, team_tbl: pd.DataFrame,
              injuries_q: Mapping[tuple[int, int, str], set[str]], *,
              refit_weeks: tuple[int, ...] = REFIT_WEEKS, q_weight: float = Q_WEIGHT
              ) -> Callable:
    """``spec_hook(season, week, home, away, spec)`` for ``run_backtest``.

    Uses ``models_by_block[(season, refit_block(week))]`` (KeyError if that
    block was never fitted) and passes ``learned.apply_to_spec`` only the
    (season, week) feature rows of the two teams (empty frames when absent)
    plus the union of both teams' questionable player_ids.
    """
    p_by_week = {(int(s), int(w)): g for (s, w), g in player_tbl.groupby(["season", "week"])}
    t_by_week = {(int(s), int(w)): g for (s, w), g in team_tbl.groupby(["season", "week"])}
    p_empty, t_empty = player_tbl.iloc[0:0], team_tbl.iloc[0:0]

    def spec_hook(season, week, home, away, spec):
        models = models_by_block[(season, refit_block(week, refit_weeks))]
        teams = (home, away)
        p = p_by_week.get((season, week), p_empty)
        t = t_by_week.get((season, week), t_empty)
        questionable = (injuries_q.get((season, week, home), set())
                        | injuries_q.get((season, week, away), set()))
        return learned.apply_to_spec(spec, models, p[p["team"].isin(teams)],
                                     t[t["team"].isin(teams)], questionable, q_weight)

    return spec_hook


def guard_hook(hook: Callable) -> tuple[Callable, list[str]]:
    """Wrap ``hook`` so every exception is recorded (with the game) and re-raised.

    ``run_backtest`` catches per-game exceptions and skips the game; the
    returned ``errors`` list lets the caller make that fatal afterwards
    (``raise_hook_errors``) instead of scoring on a non-random subset.
    """
    errors: list[str] = []

    def guarded(season, week, home, away, spec):
        try:
            return hook(season, week, home, away, spec)
        except Exception as exc:
            errors.append(f"{season} wk{week} {away}@{home}: {type(exc).__name__}: {exc}")
            raise

    return guarded, errors


def raise_hook_errors(errors: list[str], name: str) -> None:
    """RuntimeError listing the games whose spec_hook raised, if any."""
    if errors:
        shown = "\n  ".join(errors[:20])
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise RuntimeError(f"run={name}: spec_hook raised for {len(errors)} game(s); "
                           f"aborting (they would be silently dropped):\n  {shown}{more}")


def game_set(records: list[dict]) -> set[tuple[int, int, str]]:
    """The (season, week, home) games present in a run's records."""
    return {(int(r["season"]), int(r["week"]), str(r["home"])) for r in records}


def check_coverage(base_games: set[tuple[int, int, str]], cand_records: list[dict],
                   name: str) -> int:
    """Number of games; RuntimeError unless the candidate covers exactly ``base_games``."""
    got = game_set(cand_records)
    missing, extra = sorted(base_games - got), sorted(got - base_games)
    if missing or extra:
        raise RuntimeError(
            f"run={name}: game coverage differs from the baseline -- "
            f"missing {len(missing)}: {missing[:20]}; extra {len(extra)}: {extra[:20]}")
    return len(got)


# ---- pure helpers: decisions ------------------------------------------------------------

def baseline_ece_check(base_recs: list[dict], cand_recs: list[dict], population: set,
                       tol: float = ECE_TOL) -> dict:
    """Per-market PIT decile ECE of the candidate vs the BASELINE.

    Fails a market when ``ece_cand > ece_base + tol``; complements
    ``rung_decision`` (which compares against the kept configuration) so
    calibration cannot drift by up to ``tol`` at every rung.
    """
    df = paired_frame(base_recs, cand_recs, population)
    per_market: dict[str, dict] = {}
    reasons: list[str] = []
    if not df.empty:
        for market, g in df.groupby("market"):
            eb, ec = float(decile_ece(g["pit_b"].to_numpy())), float(decile_ece(g["pit_c"].to_numpy()))
            per_market[str(market)] = {"n": int(len(g)), "ece_base": eb, "ece_cand": ec}
            if ec > eb + tol:
                reasons.append(f"market {market}: ece_cand ({ec:.4f}) > baseline ece "
                               f"({eb:.4f}) + {tol}")
    return {"pass": not reasons, "reasons": reasons, "per_market": per_market}


def rung_passes(decision: dict, base_check: dict) -> bool:
    """A rung passes iff it beats the kept config AND keeps calibration vs baseline."""
    return bool(decision["pass"]) and bool(base_check["pass"])


# ---- pure helpers: identity / checkpoints ----------------------------------------------

def file_fingerprint(path: Path) -> dict:
    """``{"size", "sha256"}`` of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"size": path.stat().st_size, "sha256": h.hexdigest()}


def git_head() -> str:
    """``git rev-parse HEAD`` of the repo, or ``"unknown"``."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- identity is best-effort
        return "unknown"


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def to_jsonable(obj):
    """Recursively convert numpy scalars / tuples / sets to native JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj)]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def gate_json(gate: dict) -> str:
    """``a_gate.json`` text with real booleans and native numbers."""
    return json.dumps(to_jsonable(gate), indent=2) + "\n"


def save_records(path: Path, records: list[dict], meta: dict, stats: dict) -> None:
    """Checkpoint a run's records (parquet) with a sidecar ``{meta, stats}`` JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)
    _meta_path(path).write_text(json.dumps(to_jsonable({"meta": meta, "stats": stats}),
                                           sort_keys=True))


def load_records(path: Path, meta: dict) -> tuple[list[dict], dict] | None:
    """``(records, stats)`` from a checkpoint whose sidecar meta equals ``meta``, else None."""
    mp = _meta_path(path)
    if not path.exists() or not mp.exists():
        return None
    side = json.loads(mp.read_text())
    if side.get("meta") != json.loads(json.dumps(to_jsonable(meta), sort_keys=True)):
        return None
    return pd.read_parquet(path).to_dict("records"), side.get("stats", {})


# ---- report -----------------------------------------------------------------------------

def _fmt_decision_line(d: dict) -> str:
    return (f"skill {d['skill']:+.4f} (95% CI [{d['lo']:+.4f}, {d['hi']:+.4f}]) -> "
            f"{'PASS' if d['pass'] else 'FAIL'}")


def _decision_block(d: dict) -> list[str]:
    lines = [f"- Pooled relative RPS skill: **{d['skill']:+.4f}** "
             f"(95% clustered-bootstrap CI [{d['lo']:+.4f}, {d['hi']:+.4f}])",
             f"- Decision: **{'PASS' if d['pass'] else 'FAIL'}**", "",
             "| market | n | RPS base → cand | ECE base → cand |",
             "|---|---:|---|---|"]
    for m, v in sorted(d["per_market"].items()):
        lines.append(f"| {m} | {v['n']} | {v['rps_b']:.4f} → {v['rps_c']:.4f} | "
                     f"{v['ece_b']:.4f} → {v['ece_c']:.4f} |")
    lines.append("")
    if d["reasons"]:
        lines.append("Fail reasons:")
        lines.extend(f"- {r}" for r in d["reasons"])
    else:
        lines.append("Fail reasons: none.")
    return lines


def _baseline_check_block(c: dict) -> list[str]:
    lines = [f"Calibration vs baseline (ECE may not exceed the current sim's by more than "
             f"{ECE_TOL}): **{'PASS' if c['pass'] else 'FAIL'}**", "",
             "| market | n | ECE baseline → cand |", "|---|---:|---|"]
    for m, v in sorted(c["per_market"].items()):
        lines.append(f"| {m} | {v['n']} | {v['ece_base']:.4f} → {v['ece_cand']:.4f} |")
    lines.append("")
    if c["reasons"]:
        lines.append("Fail reasons:")
        lines.extend(f"- {r}" for r in c["reasons"])
    else:
        lines.append("Fail reasons: none.")
    return lines


def _verdict(gate: dict) -> str:
    kept, final, seasons = gate["kept"], gate["final"], gate["seasons"]
    single = list(seasons) == [FINAL_SEASON]
    pooled = (f"season {FINAL_SEASON}" if single
              else f"pooled seasons {', '.join(str(s) for s in seasons)}")
    if final["pass"]:
        also = "" if single else f" and on {FINAL_SEASON} alone"
        return (f"**Verdict: PASS.** The props-ML A gate passes. Kept rungs: "
                f"{', '.join(kept)}. The kept configuration beats the current sim on "
                f"{pooled}{also}.")
    if not kept:
        return ("**Verdict: FAIL.** The props-ML A gate fails: no rung was kept "
                "(every rung failed its checks), so the current sim stays.")
    why = []
    if not final["all"]["pass"]:
        reasons = "; ".join(final["all"]["reasons"])
        why.append(f"it does not pass on {pooled} ({reasons})")
    f25 = final.get("season_2025")
    if f25 is None:
        why.append(f"season {FINAL_SEASON} was not in the run")
    elif not single and not f25["pass"]:
        why.append(f"it does not pass on {FINAL_SEASON} alone ({'; '.join(f25['reasons'])})")
    return (f"**Verdict: FAIL.** The props-ML A gate fails. Kept rungs: {', '.join(kept)}, "
            f"but against the current sim {' and '.join(why)}.")


def render_report(gate: dict) -> str:
    """Markdown report for an ``a_gate.json`` dict; verdict in the first paragraph."""
    final = gate["final"]
    seasons = ", ".join(str(s) for s in gate["seasons"])
    ident = gate["identity"]
    lines = [f"# Props-ML A gate — {gate['run_date']}", "", _verdict(gate), "",
             "## Run", "",
             f"- Test seasons: {seasons}; sims per game: {gate['n_sims']}; base seed "
             f"{ident['seed']} with {ALIGNMENT}.",
             f"- Refit schedule: {gate['refit_schedule']} (refit weeks "
             f"{', '.join(str(w) for w in gate['refit_weeks'])}); models for (S, r) "
             "train only on rows strictly before (S, r).",
             "- Baseline: current sim, production config "
             f"({', '.join(f'{k} {v}' for k, v in ident['prod'].items())}); "
             f"{gate['n_games_baseline']} games. Every candidate covered exactly these games.",
             "- Population: baseline projected-usage gate (fixed for every candidate).",
             "- A rung passes iff it beats the currently kept configuration "
             "(rung_decision) AND no market's ECE exceeds the baseline's by more than "
             f"{ECE_TOL}.",
             f"- Code: git {ident['git_head']}; features: player sha256 "
             f"{ident['player_features']['sha256'][:12]} ({ident['player_features']['size']} B), "
             f"team sha256 {ident['team_features']['sha256'][:12]} "
             f"({ident['team_features']['size']} B).",
             f"- Elapsed: {gate['elapsed_s'] / 60:.1f} min.", "",
             "Tuned (decay, max_iter) per test season (tuned once with toggles "
             "{volume}, reused for every rung):", "",
             "| season | decay | max_iter |", "|---|---:|---:|"]
    for s, (dec, it) in sorted(gate["tuned"].items()):
        lines.append(f"| {s} | {dec} | {it} |")
    lines += ["", "## Ladder", ""]
    for step in gate["ladder"]:
        against = ", ".join(step["compared_against"]) or "baseline (current sim)"
        lines += [f"### {step['rung']}", "",
                  f"- Candidate toggles: {', '.join(step['toggles'])}",
                  f"- Compared against: {against}",
                  f"- Games: {step['n_games']}; records: {step['n_records']}; sides with "
                  f"share fallback: {step['share_fallbacks']}; run time "
                  f"{step['seconds'] / 60:.1f} min"]
        if step.get("note"):
            lines.append(f"- Note: {step['note']}")
        lines += ["", "Versus the kept configuration:", ""]
        lines += _decision_block(step["decision"])
        lines += [""] + _baseline_check_block(step["baseline_ece"])
        lines += ["", f"Rung result: **{'PASS' if step['pass'] else 'FAIL'}**. "
                  f"Kept after this rung: {', '.join(step['kept_after']) or 'none'}", ""]
    lines += ["## Final gate (kept vs current sim)", "",
              f"### {'Season ' + seasons if len(gate['seasons']) == 1 else 'Pooled seasons ' + seasons}",
              ""]
    lines += _decision_block(final["all"])
    lines += ["", f"### {FINAL_SEASON} alone", ""]
    if final.get("season_2025") is None:
        lines.append(f"Season {FINAL_SEASON} was not part of this run.")
    elif list(gate["seasons"]) == [FINAL_SEASON]:
        lines.append("Single-season run: identical to the comparison above.")
    else:
        lines += _decision_block(final["season_2025"])
    lines += ["", f"final_pass = {final['pass']}", ""]
    return "\n".join(lines)


# ---- IO ------------------------------------------------------------------------------

def _load_backtest():
    """``backtest_sim_nfl`` is a script, not a package: load it by path (the
    same way ``build_cover_dataset._build_sim_lookup`` does)."""
    path = Path(__file__).resolve().parent / "backtest_sim_nfl.py"
    spec = importlib.util.spec_from_file_location("backtest_sim_nfl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[+{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)

    if not PLAYER_PATH.exists() or not TEAM_PATH.exists():
        raise SystemExit(f"missing {PLAYER_PATH} / {TEAM_PATH}: build them first with "
                         "`uv run python scripts/build_player_features.py`")
    player_tbl = pd.read_parquet(PLAYER_PATH)
    team_tbl = pd.read_parquet(TEAM_PATH)

    env_seasons = os.environ.get("PROPS_ML_SEASONS")
    seasons = ([int(s) for s in env_seasons.split(",") if s.strip()]
               if env_seasons else list(TEST_SEASONS))
    n_sims = int(os.environ.get("PROPS_ML_N_SIMS") or DEFAULT_N_SIMS)
    refit_weeks = refit_weeks_from_env(os.environ)
    schedule = "weekly" if refit_weeks == WEEKLY_REFIT_WEEKS else "every 4 weeks"
    resume = os.environ.get("PROPS_ML_RESUME") == "1"

    bsn = _load_backtest()
    seed = int(bsn.SIM_SEED)
    identity = {"player_features": file_fingerprint(PLAYER_PATH),
                "team_features": file_fingerprint(TEAM_PATH),
                "git_head": git_head(), "prod": dict(PROD), "seed": seed}
    log(f"seasons={seasons} n_sims={n_sims} refit={schedule} {list(refit_weeks)} "
        f"resume={resume} player_rows={len(player_tbl)} team_rows={len(team_tbl)} "
        f"git={identity['git_head'][:12]} seed={seed}")
    injuries_q = questionable_index(player_tbl)

    def run(name: str, meta: dict, hook=None, models=None, base_games=None
            ) -> tuple[list[dict], dict]:
        """Run (or resume) one backtest; returns (records, stats). Aborts on
        hook errors or on game coverage differing from the baseline."""
        path = DATA_DIR / f"records_{name}.parquet"
        if resume:
            got = load_records(path, meta)
            if got is not None:
                recs, stats = got
                if base_games is not None:
                    check_coverage(base_games, recs, name)
                log(f"run={name}: loaded {len(recs)} records from checkpoint {path.name}")
                return recs, stats
            log(f"run={name}: no matching checkpoint, running")
        start = time.time()
        state = {"key": None, "games": 0}

        def on_game(season, week, home, away, sims):
            state["games"] += 1
            key = (season, refit_block(week, refit_weeks))
            if key != state["key"]:
                state["key"] = key
                log(f"run={name} season={season} block={key[1]} week={week} "
                    f"games_so_far={state['games']}")

        errors: list[str] = []
        if hook is not None:
            hook, errors = guard_hook(hook)
        recs: list[dict] = []
        log(f"run={name}: backtest start")
        bsn.run_backtest(seasons, n_sims, seed=seed, on_game=on_game, spec_hook=hook,
                         record=recs, **PROD)
        raise_hook_errors(errors, name)
        if base_games is not None:
            check_coverage(base_games, recs, name)
        stats = {"seconds": time.time() - start, "games": state["games"],
                 "share_fallbacks": sum(m.share_fallbacks for m in (models or {}).values())}
        save_records(path, recs, meta, stats)
        log(f"run={name}: {state['games']} games, {len(recs)} records in "
            f"{stats['seconds'] / 60:.1f} min -> {path.name}")
        return recs, stats

    base_meta = {"n_sims": n_sims, "seasons": seasons, "toggles": [], "identity": identity}
    base_recs, _ = run("baseline", base_meta)
    base_games = game_set(base_recs)
    population = population_from_baseline(base_recs)
    log(f"baseline games: {len(base_games)}; population: {len(population)} "
        "(season, week, player, market) keys")

    tuned: dict[int, tuple[float, int]] = {}
    for s in seasons:
        tuned[s] = learned.tune(player_tbl, s, TUNE_TOGGLES)
        log(f"tuned season={s}: decay={tuned[s][0]} max_iter={tuned[s][1]}")

    kept: frozenset[str] = frozenset()
    kept_recs = base_recs
    ladder_out = []
    for rung in LADDER:
        toggles = next_candidate(kept, rung)
        note = None
        if rung != "volume" and "volume" not in kept:
            note = (f"volume was not kept; {rung} evaluated as "
                    f"{{{', '.join(sorted(toggles))}}} anyway")
        meta = {"n_sims": n_sims, "seasons": seasons, "toggles": sorted(toggles),
                "refit_weeks": list(refit_weeks),
                "tuned": {str(s): list(v) for s, v in tuned.items()}, "identity": identity}
        path = DATA_DIR / f"records_{rung}.parquet"
        got = load_records(path, meta) if resume else None
        if got is not None:
            cand_recs, stats = got
            check_coverage(base_games, cand_recs, rung)
            log(f"run={rung}: loaded {len(cand_recs)} records from checkpoint {path.name}")
        else:
            models: dict = {}
            for s in seasons:
                decay, max_iter = tuned[s]
                for r in refit_weeks:
                    log(f"rung={rung} toggles={sorted(toggles)} season={s} block={r}: fit")
                    models[(s, r)] = learned.fit_models(
                        player_tbl, team_tbl, toggles, upto=(s, r), test_season=s,
                        decay=decay, max_iter=max_iter)
            hook = make_hook(models, player_tbl, team_tbl, injuries_q,
                             refit_weeks=refit_weeks, q_weight=Q_WEIGHT)
            cand_recs, stats = run(rung, meta, hook, models=models, base_games=base_games)
        d = rung_decision(paired_frame(kept_recs, cand_recs, population))
        bchk = baseline_ece_check(base_recs, cand_recs, population)
        passed = rung_passes(d, bchk)
        against = sorted(kept)
        if passed:
            kept, kept_recs = toggles, cand_recs
        reasons = list(d["reasons"]) + list(bchk["reasons"])
        log(f"DECISION rung={rung} toggles={sorted(toggles)} vs "
            f"{against or 'baseline'}: {_fmt_decision_line(d)}; calibration vs baseline "
            f"{'PASS' if bchk['pass'] else 'FAIL'} => {'PASS' if passed else 'FAIL'}"
            + (f" reasons={reasons}" if reasons else "")
            + f" | kept={sorted(kept) or 'none'}")
        ladder_out.append({"rung": rung, "toggles": sorted(toggles), "compared_against": against,
                           "decision": d, "baseline_ece": bchk, "pass": passed,
                           "kept_after": sorted(kept), "note": note,
                           "n_records": len(cand_recs), "n_games": len(game_set(cand_recs)),
                           "share_fallbacks": int(stats.get("share_fallbacks", 0)),
                           "seconds": float(stats.get("seconds", 0.0))})

    log("final gate: kept vs baseline")
    final_all = rung_decision(paired_frame(base_recs, kept_recs, population))
    final_25 = None
    if FINAL_SEASON in seasons:
        final_25 = rung_decision(paired_frame(
            [r for r in base_recs if r["season"] == FINAL_SEASON],
            [r for r in kept_recs if r["season"] == FINAL_SEASON], population))
    final_pass = bool(bool(kept) and final_all["pass"] and final_25 is not None
                      and final_25["pass"])
    log(f"FINAL all seasons: {_fmt_decision_line(final_all)}")
    if final_25 is not None:
        log(f"FINAL {FINAL_SEASON}: {_fmt_decision_line(final_25)}")
    log(f"FINAL kept={sorted(kept) or 'none'} final_pass={final_pass}")

    run_date = date.today().isoformat()
    gate = {"run_date": run_date, "seasons": seasons, "n_sims": n_sims,
            "refit_schedule": schedule, "refit_weeks": list(refit_weeks),
            "alignment": ALIGNMENT, "identity": identity,
            "tuned": {str(s): list(v) for s, v in tuned.items()},
            "n_games_baseline": len(base_games),
            "ladder": ladder_out, "kept": sorted(kept),
            "final": {"pass": final_pass, "all": final_all, "season_2025": final_25},
            "elapsed_s": time.time() - t0}
    GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_PATH.write_text(gate_json(gate))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{run_date}-props-ml-a-gate.md"
    report_path.write_text(render_report(gate))
    log(f"wrote {GATE_PATH} and {report_path}")


if __name__ == "__main__":
    main()
