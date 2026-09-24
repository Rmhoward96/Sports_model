"""Walk-forward ablation-ladder harness for the props-ML A gate (spec §4).

Runs the current NFL sim (production config) as the baseline, then climbs the
ladder ``volume -> efficiency -> context -> market``: each rung's candidate
(``next_candidate(kept, rung)``) swaps learned inputs into every game's spec
via ``backtest_sim_nfl.run_backtest``'s ``spec_hook`` and is compared, on the
baseline's fixed projected-usage population, against the CURRENTLY KEPT
configuration's records (baseline until a rung passes). A failed rung is
skipped and the ladder continues. The final gate compares kept vs baseline on
all seasons and on 2025 alone; ``final_pass`` needs both and a non-empty kept
set.

Walk-forward / leakage
----------------------
For test season S, models are refit before each refit week r
(``REFIT_WEEKS``; ``PROPS_ML_WEEKLY_REFIT=1`` -> every week) with
``fit_models(..., upto=(S, r))``, which trains only on rows strictly before
(S, r). A game in week w uses the models of block ``refit_block(w)``.
``(decay, max_iter)`` is tuned ONCE per test season (``learned.tune`` with
toggles ``{"volume"}``, seasons < S only) and reused for every rung. Every run
uses the same seed (common random numbers).

Env
---
PROPS_ML_SEASONS      comma list of test seasons (default 2021..2025)
PROPS_ML_N_SIMS       sims per game (default 1000)
PROPS_ML_WEEKLY_REFIT 1 -> refit before every week 1..18
PROPS_ML_RESUME       1 -> load a run's records checkpoint
                      (data/props_ml/records_<name>.parquet) instead of
                      re-running it, when its sidecar meta matches

Outputs: ``assets/nfl/props_ml/a_gate.json`` and
``docs/superpowers/reports/<run date>-props-ml-a-gate.md``.

PURE / IO split: ``refit_block``, ``refit_weeks_from_env``, ``next_candidate``,
``questionable_index``, ``make_hook``, ``save_records``/``load_records`` and
``render_report`` are unit tested (tests/scripts/test_train_props_ml.py);
``main()`` (nflverse fetches, fits, backtests) is not.
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

import pandas as pd  # noqa: E402

from sportsmodel.model.props_eval import (  # noqa: E402
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
Q_WEIGHT = 0.75  # production questionable down-weight for learned shares
TUNE_TOGGLES = frozenset({"volume"})
FINAL_SEASON = 2025
DEFAULT_N_SIMS = 1000

DATA_DIR = ROOT / "data" / "props_ml"
PLAYER_PATH = DATA_DIR / "player_week_features.parquet"
TEAM_PATH = DATA_DIR / "team_week_features.parquet"
GATE_PATH = ROOT / "assets" / "nfl" / "props_ml" / "a_gate.json"
REPORT_DIR = ROOT / "docs" / "superpowers" / "reports"


# ---- pure helpers -------------------------------------------------------------------

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


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def save_records(path: Path, records: list[dict], meta: dict) -> None:
    """Checkpoint a run's records (parquet) with a sidecar meta JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)
    _meta_path(path).write_text(json.dumps(meta, sort_keys=True))


def load_records(path: Path, meta: dict) -> list[dict] | None:
    """Records from a checkpoint whose sidecar meta equals ``meta``, else None."""
    mp = _meta_path(path)
    if not path.exists() or not mp.exists():
        return None
    if json.loads(mp.read_text()) != json.loads(json.dumps(meta, sort_keys=True)):
        return None
    return pd.read_parquet(path).to_dict("records")


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


def render_report(gate: dict) -> str:
    """Markdown report for an ``a_gate.json`` dict; verdict in the first paragraph."""
    kept = gate["kept"]
    final = gate["final"]
    seasons = ", ".join(str(s) for s in gate["seasons"])
    if final["pass"]:
        verdict = (f"**Verdict: PASS.** The props-ML A gate passes. Kept rungs: "
                   f"{', '.join(kept)}. The kept configuration beats the current sim "
                   f"on pooled seasons {seasons} and on {FINAL_SEASON} alone.")
    elif not kept:
        verdict = ("**Verdict: FAIL.** The props-ML A gate fails: no rung was kept "
                   "(every rung failed its comparison), so the current sim stays.")
    else:
        why = []
        if not final["all"]["pass"]:
            why.append(f"it does not pass on pooled seasons {seasons}")
        f25 = final.get("season_2025")
        if f25 is None:
            why.append(f"season {FINAL_SEASON} was not in the run")
        elif not f25["pass"]:
            why.append(f"it does not pass on {FINAL_SEASON} alone")
        verdict = (f"**Verdict: FAIL.** The props-ML A gate fails. Kept rungs: "
                   f"{', '.join(kept)}, but against the current sim "
                   f"{' and '.join(why)}.")

    lines = [f"# Props-ML A gate — {gate['run_date']}", "", verdict, "",
             "## Run", "",
             f"- Test seasons: {seasons}; sims per game: {gate['n_sims']} (same seed for "
             "every run — common random numbers).",
             f"- Refit schedule: {gate['refit_schedule']} (refit weeks "
             f"{', '.join(str(w) for w in gate['refit_weeks'])}); models for (S, r) "
             "train only on rows strictly before (S, r).",
             "- Baseline: current sim, production config "
             "(season_decay 0.4, questionable_weight 0.75, home_field 0.07, ratings_weight 0.5).",
             "- Population: baseline projected-usage gate (fixed for every candidate).",
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
                  f"- Records: {step['n_records']}; sides with share fallback: "
                  f"{step['share_fallbacks']}; run time {step['seconds'] / 60:.1f} min"]
        if step.get("note"):
            lines.append(f"- Note: {step['note']}")
        lines += _decision_block(step["decision"])
        lines += ["", f"Kept after this rung: {', '.join(step['kept_after']) or 'none'}", ""]
    lines += ["## Final gate (kept vs current sim)", "", "### All seasons", ""]
    lines += _decision_block(final["all"])
    lines += ["", f"### {FINAL_SEASON} alone", ""]
    if final.get("season_2025") is None:
        lines.append(f"Season {FINAL_SEASON} was not part of this run.")
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
    log(f"seasons={seasons} n_sims={n_sims} refit={schedule} {list(refit_weeks)} "
        f"resume={resume} player_rows={len(player_tbl)} team_rows={len(team_tbl)}")

    bsn = _load_backtest()
    injuries_q = questionable_index(player_tbl)

    def run(name: str, meta: dict, hook=None) -> tuple[list[dict], float]:
        path = DATA_DIR / f"records_{name}.parquet"
        if resume:
            recs = load_records(path, meta)
            if recs is not None:
                log(f"run={name}: loaded {len(recs)} records from checkpoint {path.name}")
                return recs, 0.0
        start = time.time()
        state = {"key": None, "games": 0}

        def on_game(season, week, home, away, sims):
            state["games"] += 1
            key = (season, refit_block(week, refit_weeks))
            if key != state["key"]:
                state["key"] = key
                log(f"run={name} season={season} block={key[1]} week={week} "
                    f"games_so_far={state['games']}")

        recs: list[dict] = []
        log(f"run={name}: backtest start")
        bsn.run_backtest(seasons, n_sims, on_game=on_game, spec_hook=hook, record=recs, **PROD)
        secs = time.time() - start
        save_records(path, recs, meta)
        log(f"run={name}: {state['games']} games, {len(recs)} records in "
            f"{secs / 60:.1f} min -> {path.name}")
        return recs, secs

    base_meta = {"n_sims": n_sims, "seasons": seasons, "toggles": []}
    base_recs, _ = run("baseline", base_meta)
    population = population_from_baseline(base_recs)
    log(f"population: {len(population)} (season, week, player, market) keys")

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
                "tuned": {str(s): list(v) for s, v in tuned.items()}}
        path = DATA_DIR / f"records_{rung}.parquet"
        models: dict = {}
        cand_recs = load_records(path, meta) if resume else None
        if cand_recs is not None:
            log(f"run={rung}: loaded {len(cand_recs)} records from checkpoint {path.name}")
            secs = 0.0
        else:
            for s in seasons:
                decay, max_iter = tuned[s]
                for r in refit_weeks:
                    log(f"rung={rung} toggles={sorted(toggles)} season={s} block={r}: fit")
                    models[(s, r)] = learned.fit_models(
                        player_tbl, team_tbl, toggles, upto=(s, r), test_season=s,
                        decay=decay, max_iter=max_iter)
            hook = make_hook(models, player_tbl, team_tbl, injuries_q,
                             refit_weeks=refit_weeks, q_weight=Q_WEIGHT)
            cand_recs, secs = run(rung, meta, hook)
        fallbacks = sum(m.share_fallbacks for m in models.values())
        d = rung_decision(paired_frame(kept_recs, cand_recs, population))
        against = sorted(kept)
        if d["pass"]:
            kept, kept_recs = toggles, cand_recs
        log(f"DECISION rung={rung} toggles={sorted(toggles)} vs "
            f"{against or 'baseline'}: {_fmt_decision_line(d)}"
            + (f" reasons={d['reasons']}" if d["reasons"] else "")
            + f" | kept={sorted(kept) or 'none'}")
        ladder_out.append({"rung": rung, "toggles": sorted(toggles), "compared_against": against,
                           "decision": d, "kept_after": sorted(kept), "note": note,
                           "n_records": len(cand_recs), "share_fallbacks": fallbacks,
                           "seconds": secs})

    log("final gate: kept vs baseline")
    final_all = rung_decision(paired_frame(base_recs, kept_recs, population))
    final_25 = None
    if FINAL_SEASON in seasons:
        final_25 = rung_decision(paired_frame(
            [r for r in base_recs if r["season"] == FINAL_SEASON],
            [r for r in kept_recs if r["season"] == FINAL_SEASON], population))
    final_pass = bool(kept) and final_all["pass"] and final_25 is not None and final_25["pass"]
    log(f"FINAL all seasons: {_fmt_decision_line(final_all)}")
    if final_25 is not None:
        log(f"FINAL {FINAL_SEASON}: {_fmt_decision_line(final_25)}")
    log(f"FINAL kept={sorted(kept) or 'none'} final_pass={final_pass}")

    run_date = date.today().isoformat()
    gate = {"run_date": run_date, "seasons": seasons, "n_sims": n_sims,
            "refit_schedule": schedule, "refit_weeks": list(refit_weeks),
            "tuned": {str(s): list(v) for s, v in tuned.items()},
            "ladder": ladder_out, "kept": sorted(kept),
            "final": {"pass": final_pass, "all": final_all, "season_2025": final_25},
            "elapsed_s": time.time() - t0}
    GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_PATH.write_text(json.dumps(gate, indent=2, default=float) + "\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{run_date}-props-ml-a-gate.md"
    report_path.write_text(render_report(gate))
    log(f"wrote {GATE_PATH.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
