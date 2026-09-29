"""Offline B / blend / calibration ladder for the props-ML Props-2 gate (spec §3-§4).

B, the blend and calibration only change per-player marginals, so they are
gated OFFLINE on stored pmfs from ONE baseline run and ONE kept-A run (both
``run_backtest(..., record_pmf=True)``; plan ruling 1), on all seven markets:

1. Kept A toggles and the per-season tuned (decay, max_iter) come from
   ``assets/nfl/props_ml/a_gate.json`` (the Props-1 verdict).
2. Population = ``population_from_baseline(baseline)`` (7 markets, fixed).
3. A re-check: kept A vs baseline; a market whose RPS A worsens by > 1%
   starts with ``source = "baseline"`` (ruling 3): the kept composite serves
   the baseline's records for it until the final per-market decision.
4. B out-of-fold: for test season S and refit block r, ``fit_market`` on rows
   strictly before (S, r) (unweighted, ``max_iter`` 150; kept-A feature
   columns) predicts that block's population records.
5. +B blend rung: weight on A for season S = ``choose_weight`` over OOF
   records of test seasons < S (none -> 1.0, i.e. pure A).
6. +calibration rung: per market PIT map (Platt for anytime_td) for season S
   fit on the kept pipeline's pre-calibration OOF records of seasons < S
   (none -> identity).
   A rung passes as in Props-1: ``rung_decision`` vs the kept composite AND
   ``baseline_ece_check``; a failed rung is skipped.
   B trains and predicts only on each market's pre-game role subset
   (``dist_models.ROLE_SUBSETS``; ``ROLE_SUBSETS_V2`` for gate name ``_v2``,
   ``role_subsets``); an out-of-role record gets B := A.
7. Final: per market ``source = "ml"`` iff the kept ML pipeline's RPS <= the
   baseline's AND its ECE <= the baseline's + 0.005 on all seasons; the
   served composite (those sources) must beat the baseline pooled AND on 2025
   alone. The UNSELECTED decision (ladder sources, no per-market swap) is
   recorded next to it: the selected one is optimistic by construction. ``w_final`` = ``choose_weight`` on
   all seasons' OOF rows that have a B pmf (1.0 if the blend was not kept);
   ``calibrate`` = calibration rung kept.
8. Serving calibration maps (``calibration.json``, committed): each OOF row's
   pre-calibration pmf is recomputed at its market's ``w_final``
   (``apply_pipeline``, identity calib) and the PIT map / Platt is fit on ALL
   OOF rows of every ``calibrate`` market (>= 300 PITs / >= 100 per class,
   else identity). pipeline.json and calibration.json record ``data_end``, the
   last (season, week) of the OOF rows: the weekly quick gate only scores
   weeks after it.

Outputs: ``b_gate.json``, ``pipeline.json`` + ``calibration.json`` (consumed by
fit_props_ml_final),
``docs/superpowers/reports/<date>-props-ml-b-gate.md`` (other run tags are
suffixed), ``data/props_ml/oof_b7__<tag>.parquet`` -- the final pipeline's
PRE-calibration OOF records (Task 6 fits its calibration maps on them) -- and
``data/props_ml/served__<tag>.parquet``: the SELECTED served composite's
post-calibration records (``SERVED_COLS``; one row per population key; ``mean``
is the expectation of the served pmf), which ``scripts/gate_props_ml_v2.py``
compares across model versions.

Checkpoints ``records_{baseline,kept_a}__<tag>__b7.parquet`` and
``b_oof__<tag>__b7.parquet`` (``PROPS_ML_RESUME=1``) carry the identity:
git HEAD, feature fingerprints, sources fingerprint, PROD, seed, record_pmf.
Env: PROPS_ML_SEASONS, PROPS_ML_N_SIMS, PROPS_ML_WEEKLY_REFIT, PROPS_ML_RESUME
(as in scripts/train_props_ml.py, whose helpers this script reuses),
PROPS_ML_DECIDE_SEASONS (``tpm.decide_seasons_from_env``; default every run
season): every SELECTION decision -- the A re-check's starting sources, the
blend and calibration rung decisions and the final per-market sources -- uses
only the records / population keys of those seasons, while per-season blend
weights and calibration maps stay walk-forward and the final pass is still
scored on all seasons pooled and on 2025 alone; and PROPS_ML_GATE_NAME, one
of ``GATE_NAMES`` (anything else raises ValueError):

- ``""``: v1 -- ``a_gate.json``, v1 ``ROLE_SUBSETS``, unsuffixed files (the
  committed v1 outputs);
- ``"_v1cmp"``: the v1 comparison run for the v2 gate -- ``a_gate.json``, v1
  ``ROLE_SUBSETS``, every output, checkpoint, OOF and served file suffixed
  ``_v1cmp`` (never the committed v1 files);
- ``"_v2"``: ``a_gate_v2.json``, ``ROLE_SUBSETS_V2`` everywhere B trains,
  predicts and records roles, every file suffixed ``_v2``.

The A gate json passed in must carry the ``gate_name`` its table entry
expects (none / "" for ``a_gate.json``).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel.model.props_eval import (  # noqa: E402
    decile_ece,
    pit_pmf,
    pit_uniform,
    population_from_baseline,
    rps_pmf,
    rung_decision,
)
from sportsmodel.model.props_ml.blend import BINARY_MARKET, choose_weight  # noqa: E402
from sportsmodel.model.props_ml.dist_models import (  # noqa: E402
    ROLE_SUBSETS,
    ROLE_SUBSETS_V2,
    fit_market,
    in_role,
    predict_pmfs,
    role_conditions,
)
from sportsmodel.model.props_ml.pit_calibration import (  # noqa: E402
    IDENTITY_KNOTS,
    IDENTITY_PLATT,
    fit_pit_map,
    fit_platt,
)
from sportsmodel.model.props_ml.pipeline import blend_calibrate  # noqa: E402
from sportsmodel.sim.nfl import learned  # noqa: E402


def _load_script(name: str):
    """Scripts are not a package: load one by path (Ruling P2)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tpm = _load_script("train_props_ml")

MARKETS: tuple[str, ...] = ("pass_yds", "rush_yds", "rec_yds", "receptions", "rush_att",
                            "pass_tds", "anytime_td")
TAG_SUFFIX = "b7"
B_DECAY = 1.0      # B fits unweighted (Task 2 ruling)
B_MAX_ITER = 150   # fixed for B (Task 2 ruling); A keeps a_gate.json's tuned values
A_WORSE_TOL = 1.01
MAX_B_MISSING = 0.02
A_GATE_PATH = tpm.GATE_PATH
GATE_PATH = A_GATE_PATH.with_name("b_gate.json")
PIPELINE_PATH = A_GATE_PATH.with_name("pipeline.json")
CALIBRATION_PATH = A_GATE_PATH.with_name("calibration.json")
ECE_TOL = tpm.ECE_TOL
# P3 OOF records (PRE-calibration, at the per-season OOF weights ``w``). Task 6
# must recompute pre-calibration values at ``w_final`` via ``apply_pipeline``
# from pmf_a / pmf_b before fitting its final maps. ``pmf_b`` is the effective
# B: the A pmf for an out-of-role record (``in_role`` False; B := A).
OOF_COLS = ("season", "week", "home", "player_id", "market", "w", "pit_pre", "p_pre", "actual",
            "pmf_a", "pmf_b", "in_role")
# The selected served composite's post-calibration records (``served_frame``).
SERVED_COLS = ("season", "week", "home", "player_id", "market", "mean", "rps", "pit", "actual")
V2_GATE_NAME = "_v2"
V1CMP_GATE_NAME = "_v1cmp"
# PROPS_ML_GATE_NAME -> (A gate json name, its expected ``gate_name``, B role table)
GATE_NAMES: dict[str, tuple[str, str, dict]] = {
    "": ("a_gate.json", "", ROLE_SUBSETS),
    V1CMP_GATE_NAME: ("a_gate.json", "", ROLE_SUBSETS),
    V2_GATE_NAME: ("a_gate_v2.json", V2_GATE_NAME, ROLE_SUBSETS_V2),
}


# ---- pure helpers ---------------------------------------------------------------------

def rec_key(r: Mapping) -> tuple:
    return (r["season"], r["week"], r["player_id"], r["market"])


def gate_entry(gate_name: str) -> tuple[str, str, dict]:
    """``GATE_NAMES[gate_name]``; ValueError for any other name."""
    if gate_name not in GATE_NAMES:
        raise ValueError(f"PROPS_ML_GATE_NAME {gate_name!r} is not one of "
                         f"{sorted(GATE_NAMES)}")
    return GATE_NAMES[gate_name]


def role_subsets(gate_name: str) -> dict:
    """B's role table for the gate name (Ruling P1): v1 ``ROLE_SUBSETS`` for
    "" and ``_v1cmp``, ``ROLE_SUBSETS_V2`` for ``_v2``."""
    return gate_entry(gate_name)[2]


def a_gate_path(gate_name: str) -> Path:
    """The A gate json this ladder builds on (``a_gate.json`` for "" and
    ``_v1cmp``, ``a_gate_v2.json`` for ``_v2``)."""
    return A_GATE_PATH.with_name(gate_entry(gate_name)[0])


def served_path(data_dir: Path, base_tag: str, gate_name: str = "") -> Path:
    """The served composite's records: ``served__<tag>{gate_name}.parquet``."""
    return Path(data_dir) / f"served__{base_tag}{gate_name}.parquet"


def pmf_mean(pmf) -> float:
    """Expectation of a pmf over 0..K (anytime_td's ``[1-p, p]`` -> p)."""
    p = np.asarray(pmf, dtype=float)
    return float((np.arange(len(p)) * p).sum() / p.sum())


def served_frame(recs, population: set) -> pd.DataFrame:
    """``SERVED_COLS`` rows of served records (``pmf`` = the served, post-
    calibration pmf; ``mean`` = its expectation). RuntimeError unless there is
    exactly one row per population key."""
    df = pd.DataFrame([{**{c: r[c] for c in SERVED_COLS if c != "mean"},
                        "mean": pmf_mean(r["pmf"])} for r in recs], columns=list(SERVED_COLS))
    keys = [rec_key(r) for r in recs]
    if len(set(keys)) != len(keys) or set(keys) != set(population):
        raise RuntimeError(f"served composite has {len(keys)} records ({len(set(keys))} unique) "
                           f"for {len(population)} population keys: it must be one per key")
    return df


def a_config(a_gate: Mapping, seasons: list[int]) -> tuple[frozenset[str], dict[int, tuple]]:
    """(kept A toggles, {season: (decay, max_iter)}) from the A gate json."""
    missing = [s for s in seasons if str(s) not in a_gate["tuned"]]
    if missing:
        raise ValueError(f"a_gate.json has no tuned (decay, max_iter) for seasons {missing}")
    tuned = {s: (float(a_gate["tuned"][str(s)][0]), int(a_gate["tuned"][str(s)][1]))
             for s in seasons}
    return frozenset(a_gate["kept"]), tuned


def market_sources(per_market: Mapping, markets, tol: float,
                   ece_tol: float | None = None) -> dict[str, str]:
    """``"baseline"`` for a market whose candidate RPS exceeds ``tol`` x the
    baseline's, whose ECE exceeds the baseline's + ``ece_tol`` (when given),
    or that has no scored rows; else ``"ml"``. A re-check: ``tol`` 1.01;
    final decision: ``tol`` 1.0 and ``ece_tol`` 0.005."""
    out = {}
    for m in markets:
        v = per_market.get(m)
        ok = v is not None and v["rps_c"] <= tol * v["rps_b"]
        if ok and ece_tol is not None:
            ok = v["ece_c"] <= v["ece_b"] + ece_tol
        out[m] = "ml" if ok else "baseline"
    return out


def apply_pipeline(records_a, b_pmfs: Mapping, weights: Mapping[str, float],
                   calib: Mapping | None = None) -> list[dict]:
    """A -> blend with B -> (optional) calibration, re-scored per record. PURE.

    Offline use: ``records_a`` are dicts with season, week, home, player_id,
    market, pmf (A) and ``actual`` (required -- every record is re-scored),
    and ``calib``, when given, must hold a map for every market present.
    Inputs are not mutated. ``b_pmfs``: ``rec_key -> B pmf``; a record without one keeps its
    A pmf (``b_missing``; ``w`` NaN). ``weights``: market -> weight on A.
    anytime_td is a logit blend of P(>=1) = 1 - pmf[0] and always leaves as
    ``[1-p, p]``. ``calib``: market -> PIT knots (Platt ``(a, b)`` for
    anytime_td); None = no calibration. Each output carries the key fields,
    actual, pmf_a, pmf_b, w, pmf (final), p_pre (anytime_td P(>=1) before
    calibration, else NaN), pit_pre, rps, pit, b_missing.
    """
    out = []
    for r in records_a:
        m, key = r["market"], rec_key(r)
        pb = b_pmfs.get(key)
        w = float(weights[m]) if pb is not None else float("nan")
        pre, post = blend_calibrate(r["pmf"], pb, w, m, None if calib is None else calib[m])
        a, u = r["actual"], pit_uniform(*key)
        pit = pit_pmf(post, a, u)
        out.append({"season": r["season"], "week": r["week"], "home": r["home"],
                    "player_id": r["player_id"], "market": m, "actual": a, "pmf_a": r["pmf"],
                    "pmf_b": pb, "w": w, "pmf": post,
                    "p_pre": float(pre[1]) if m == BINARY_MARKET else float("nan"),
                    "pit_pre": pit if calib is None else pit_pmf(pre, a, u),
                    "rps": rps_pmf(post, a), "pit": pit, "b_missing": pb is None})
    return out


def apply_by_season(records_a, b_pmfs, weights_by_season: Mapping,
                    calib_by_season: Mapping | None = None) -> list[dict]:
    """``apply_pipeline`` per season with that season's weights / maps."""
    by_season = defaultdict(list)
    for r in records_a:
        by_season[r["season"]].append(r)
    out = []
    for s, recs in sorted(by_season.items()):
        out += apply_pipeline(recs, b_pmfs, weights_by_season[s],
                              None if calib_by_season is None else calib_by_season[s])
    return out


def _by_market(rows) -> dict[str, list]:
    g = defaultdict(list)
    for r in rows:
        g[r["market"]].append(r)
    return g


def season_weights(rows, seasons, markets) -> dict[int, dict[str, float]]:
    """``{S: {m: w}}``: ``choose_weight`` over rows (pmf_a, pmf_b, actual) of
    test seasons < S that have a B pmf; 1.0 (pure A) when there are none."""
    g = _by_market(r for r in rows if r["pmf_b"] is not None)
    out = {}
    for s in seasons:
        out[s] = {}
        for m in markets:
            prior = [r for r in g.get(m, []) if r["season"] < s]
            out[s][m] = float(choose_weight(prior, m)) if prior else 1.0
    return out


def identity_calibration(market: str):
    return IDENTITY_PLATT if market == BINARY_MARKET else IDENTITY_KNOTS.copy()


def fit_calibration(rows, market: str):
    """Platt ``(a, b)`` on (p_pre, actual) for anytime_td, else PIT knots on
    pit_pre (the guards in ``fit_platt`` / ``fit_pit_map`` return identity
    when there is too little data)."""
    if market == BINARY_MARKET:
        return fit_platt([r["p_pre"] for r in rows], [int(r["actual"]) for r in rows])
    return fit_pit_map([r["pit_pre"] for r in rows])


def season_calibrations(rows, seasons, markets) -> dict[int, dict]:
    """``{S: {m: map}}`` fit on pre-calibration rows of test seasons < S;
    identity when there are none."""
    g = _by_market(rows)
    out = {}
    for s in seasons:
        out[s] = {}
        for m in markets:
            prior = [r for r in g.get(m, []) if r["season"] < s]
            out[s][m] = fit_calibration(prior, m) if prior else identity_calibration(m)
    return out


def calib_json(c, market: str):
    """A map as JSON: Platt ``[a, b]`` for anytime_td, else knots ``[[u], [g]]``."""
    if market == BINARY_MARKET:
        return [float(c[0]), float(c[1])]
    return np.asarray(c, dtype=float).tolist()


def final_weights(rows, markets, blend_kept: bool) -> dict[str, float]:
    """``w_final``: ``choose_weight`` over each market's rows (in role) that
    HAVE a B pmf; 1.0 when the blend was not kept or there are none."""
    g = _by_market(r for r in rows if r["pmf_b"] is not None)
    return {m: (float(choose_weight(g[m], m)) if blend_kept and g.get(m) else 1.0)
            for m in markets}


def final_calibration(rows, markets: Mapping) -> dict[str, dict]:
    """Serving maps ``{m: {calibrate, kind, map, n}}`` (``calibration.json``).
    A ``calibrate`` market's map is fit on ALL its OOF rows after recomputing
    each row's pre-calibration pmf at the market's ``w_final`` via
    ``apply_pipeline`` (rows carry pmf_a / pmf_b / actual; the stored values are
    at the per-season OOF weights); others get identity with n 0."""
    g = _by_market(rows)
    out = {}
    for m, v in markets.items():
        kind = "platt" if m == BINARY_MARKET else "pit"
        recs = g.get(m, []) if v["calibrate"] else []
        pre = apply_pipeline([{**r, "pmf": r["pmf_a"]} for r in recs],
                             {rec_key(r): r["pmf_b"] for r in recs if r["pmf_b"] is not None},
                             {m: float(v["w_final"])}) if recs else []
        c = fit_calibration(pre, m) if pre else identity_calibration(m)
        out[m] = {"calibrate": bool(v["calibrate"]), "kind": kind, "map": calib_json(c, m),
                  "n": len(pre)}
    return out


def calib_label(c, market: str):
    """JSON summary of one map: ``"identity"``, ``"pit"`` or ``[a, b]``."""
    if market == BINARY_MARKET:
        return "identity" if tuple(c) == IDENTITY_PLATT else [float(c[0]), float(c[1])]
    k = np.asarray(c, dtype=float)
    return "identity" if np.allclose(k[0], k[1]) else "pit"


def composite(base_recs, ml_recs, sources: Mapping[str, str]) -> list[dict]:
    """Served records: the ML record for ``"ml"`` markets, the baseline's otherwise."""
    return ([r for r in ml_recs if sources.get(r["market"]) == "ml"]
            + [r for r in base_recs if sources.get(r["market"]) == "baseline"])


def per_market_scores(df: pd.DataFrame) -> dict[str, dict]:
    """``{m: {n, rps_b, rps_c, ece_b, ece_c}}`` of a paired frame (no bootstrap)."""
    return {str(m): {"n": len(g), "rps_b": float(g["rps_b"].mean()),
                     "rps_c": float(g["rps_c"].mean()),
                     "ece_b": float(decile_ece(g["pit_b"].to_numpy())),
                     "ece_c": float(decile_ece(g["pit_c"].to_numpy()))}
            for m, g in df.groupby("market")}


def pipeline_config(kept_a, tuned: Mapping, sources: Mapping, w_final: Mapping,
                    calibrate: bool, data_end: tuple[int, int], *, final_pass: bool,
                    unselected_pass: bool, run_tag: str, git: str | None) -> dict:
    """``pipeline.json``: ``{kept_a_toggles, tuned: {season: [decay, max_iter]},
    markets: {m: {source, w_final, calibrate}}, data_end: [season, week],
    final_pass, unselected_pass, run_tag, git}``. ``final_pass`` is the
    SELECTED final decision (per-market sources) -- ``fit_props_ml_final.py``
    refuses to fit or publish unless it is true; ``unselected_pass`` is the
    unselected (A re-check sources) decision, recorded for review only."""
    return {"kept_a_toggles": [r for r in tpm.LADDER if r in kept_a],
            "tuned": {str(s): [float(d), int(i)] for s, (d, i) in sorted(tuned.items())},
            "markets": {m: {"source": str(sources[m]), "w_final": float(w_final[m]),
                            "calibrate": bool(calibrate)} for m in MARKETS},
            "data_end": [int(data_end[0]), int(data_end[1])],
            "final_pass": bool(final_pass), "unselected_pass": bool(unselected_pass),
            "run_tag": str(run_tag), "git": None if git is None else str(git)}


def oof_frame(ml_recs, out_of_role: set) -> pd.DataFrame:
    """P3 OOF records (pre-calibration) for Task 6 (see ``OOF_COLS``)."""
    def f32(x):
        return None if x is None else np.asarray(x, dtype=np.float32)
    return pd.DataFrame([{**{c: r[c] for c in OOF_COLS[:-3]}, "pmf_a": f32(r["pmf_a"]),
                          "pmf_b": f32(r["pmf_b"]), "in_role": rec_key(r) not in out_of_role}
                         for r in ml_recs], columns=list(OOF_COLS))


def output_paths(tag: str, run_date: str, *, gate_path: Path = GATE_PATH,
                 pipeline_path: Path = PIPELINE_PATH, report_dir: Path = tpm.REPORT_DIR,
                 gate_name: str = "") -> tuple[Path, Path, Path, Path]:
    """(b_gate json, pipeline json, calibration json -- next to pipeline.json --,
    report md); non-default tags are suffixed; a non-empty ``gate_name`` (e.g.
    ``_v2``) is appended last."""
    sfx = ("" if tag == tpm.DEFAULT_TAG else f"__{tag}") + gate_name
    gate_path, pipeline_path = Path(gate_path), Path(pipeline_path)
    return (gate_path.with_name(f"{gate_path.stem}{sfx}{gate_path.suffix}"),
            pipeline_path.with_name(f"{pipeline_path.stem}{sfx}{pipeline_path.suffix}"),
            pipeline_path.with_name(f"calibration{sfx}.json"),
            Path(report_dir) / f"{run_date}-props-ml-b-gate{sfx}.md")


def rung_step(name: str, kept_comp, cand_comp, base_recs, population) -> dict:
    """Props-1 rung rule: ``rung_decision`` vs the kept composite AND no
    market's ECE above the baseline's + tolerance."""
    d = rung_decision(tpm.checked_paired_frame(kept_comp, cand_comp, population, base_recs, name))
    bchk = tpm.baseline_ece_check(base_recs, cand_comp, population)
    return {"rung": name, "decision": d, "baseline_ece": bchk, "pass": tpm.rung_passes(d, bchk)}


def season_decision(base_recs, cand, population, season: int, name: str) -> dict | None:
    """``rung_decision`` of cand vs baseline on one season (None if absent)."""
    base_s = [r for r in base_recs if r["season"] == season]
    if not base_s:
        return None
    pop = {k for k in population if k[0] == season}
    return rung_decision(tpm.checked_paired_frame(
        base_s, [r for r in cand if r["season"] == season], pop, base_s, name))


def b_oof(player_tbl: pd.DataFrame, pop_recs, seasons, refit_weeks, cols: list[str],
          market_max: Mapping[str, int], log: Callable[[str], None],
          subsets: dict = ROLE_SUBSETS) -> tuple[dict, set]:
    """``(rec_key -> B pmf, out-of-role keys)`` for population records: for
    each (S, r) block with records, ``fit_market`` on rows before (S, r) and
    predict that block's IN-ROLE records (``in_role``) from their (season,
    week, player_id) feature rows. A record whose feature row is out of role
    is returned in the set (B := A downstream); one without a feature row is
    in neither (missing). ``subsets``: the role table (``role_subsets``)."""
    blocks = defaultdict(list)
    for r in pop_recs:
        blocks[(r["season"], tpm.refit_block(r["week"], refit_weeks))].append(r)
    cond_cols = [c for spec in subsets.values() if spec is not None
                 for c, _, _ in role_conditions(spec)]
    role_cols = list(dict.fromkeys(
        c for c in ("position", "p_pos", "p_y_pass_att_ewm", "p_y_carries_ewm",
                    "p_y_targets_ewm", *cond_cols) if c in player_tbl.columns and c not in cols))
    feats = player_tbl[["season", "week", "player_id"] + cols + role_cols]
    out: dict = {}
    out_of_role: set = set()
    t0 = time.time()
    for s in seasons:
        for rw in refit_weeks:
            recs = _by_market(blocks.get((s, rw), []))
            if not recs:
                continue
            for m in MARKETS:
                if not recs.get(m):
                    continue
                keys = pd.DataFrame([{"season": r["season"], "week": r["week"],
                                      "player_id": r["player_id"]} for r in recs[m]])
                rows = keys.merge(feats, on=["season", "week", "player_id"], how="inner")
                role = in_role(rows, m, subsets).to_numpy()
                out_of_role |= {(int(a), int(b), str(c), m) for a, b, c in
                                zip(rows["season"][~role], rows["week"][~role],
                                    rows["player_id"][~role])}
                rows = rows[role]
                if rows.empty:
                    continue
                model = fit_market(player_tbl, m, cols, upto=(s, rw), test_season=s,
                                   decay=B_DECAY, max_iter=B_MAX_ITER, subsets=subsets)
                for (rs, rwk, pid), pmf in zip(
                        zip(rows["season"], rows["week"], rows["player_id"]),
                        predict_pmfs(model, rows, int(market_max.get(m, 1)))):
                    out[(int(rs), int(rwk), str(pid), m)] = pmf
            log(f"stage=b-oof season={s} block={rw}: {sum(map(len, recs.values()))} records, "
                f"{(time.time() - t0) / 60:.1f} min in B so far")
    return out, out_of_role


# ---- report ------------------------------------------------------------------------------

def _verdict(gate: dict) -> str:
    f = gate["final"]
    ml = [m for m, v in gate["pipeline"]["markets"].items() if v["source"] == "ml"]
    rungs = ", ".join(gate["kept_rungs"]) or "none (pure kept A)"
    if f["pass"]:
        return (f"**Verdict: PASS.** The props-ML B gate passes. Kept B-ladder rungs: {rungs}; "
                f"markets served from ML: {', '.join(ml)}. The served pipeline beats the "
                "current sim on all test seasons pooled and on 2025 alone.")
    why = []
    if not f["all"]["pass"]:
        why.append(f"pooled: {'; '.join(f['all']['reasons'])}")
    if f.get("season_2025") is None:
        why.append("season 2025 was not in the run")
    elif not f["season_2025"]["pass"]:
        why.append(f"2025 alone: {'; '.join(f['season_2025']['reasons'])}")
    return (f"**Verdict: FAIL.** The props-ML B gate fails. Kept B-ladder rungs: {rungs}; "
            f"markets served from ML: {', '.join(ml) or 'none'}. Against the current sim "
            f"({' | '.join(why)}).")


def _season_table(title: str, by_season: Mapping, fmt: Callable) -> list[str]:
    lines = [title, "", "| season | " + " | ".join(MARKETS) + " |",
             "|---|" + "---|" * len(MARKETS)]
    for s, v in sorted(by_season.items()):
        lines.append(f"| {s} | " + " | ".join(fmt(v[m]) for m in MARKETS) + " |")
    return lines + [""]


def _rung_section(title: str, step: dict) -> list[str]:
    return ([f"Versus the kept composite ({title}):", ""] + tpm._decision_block(step["decision"])
            + [""] + tpm._baseline_check_block(step["baseline_ece"])
            + ["", f"Rung result: **{'PASS' if step['pass'] else 'FAIL'}**.", ""])


CAVEATS = tpm.CAVEATS + (
    "- Per-market sources are chosen on the decide seasons (a market is served from ML "
    "only if its own RPS there beats the baseline's); when those include the final-gate "
    "seasons the final gate is not an independent check of that choice.",
    "- B models are fit unweighted with max_iter 150 (runtime rulings); A keeps the "
    "per-season tuned values from a_gate.json.",
    "- The B role thresholds (dist_models.ROLE_SUBSETS) were chosen after a 2025 smoke run: "
    "that design freedom was exercised on the final-gate season, so the 2025 result is not "
    "fully independent of it.",
    "- Serving simulates DEFAULT_N_SIMS (10,000) sims per game (generate_sim_nfl.py) where "
    "this gate scored 1000 per game; the served pmfs are smoother than the gated ones "
    "(same pipeline, less Monte Carlo noise).",
)


def render_report(gate: dict) -> str:
    ident, b = gate["identity"], gate["b_oof"]
    f = gate["final"]
    lines = [f"# Props-ML B gate — {gate['run_date']}", "", _verdict(gate), "", "## Run", "",
             f"- Test seasons: {', '.join(map(str, gate['seasons']))}; sims per game: "
             f"{gate['n_sims']}; base seed {ident['seed']} with {tpm.ALIGNMENT}.",
             f"- Refit weeks {', '.join(map(str, gate['refit_weeks']))} (A and B models for "
             "(S, r) train only on rows strictly before (S, r)). Run tag: "
             f"{gate['run_tag']}"
             + (f"; gate name {gate['gate_name']} (a_gate{gate['gate_name']}.json)"
                if gate.get("gate_name") else "") + ".",
             f"- Decide seasons (A re-check sources, rung decisions, final per-market "
             f"sources): {', '.join(map(str, gate.get('decide_seasons', gate['seasons'])))}; "
             "blend weights and calibration maps stay walk-forward per season.",
             f"- Kept A toggles ({a_gate_path(gate.get('gate_name', '')).name}): "
             f"{', '.join(gate['pipeline']['kept_a_toggles'])}.",
             f"- Baseline: current sim ({', '.join(f'{k} {v}' for k, v in ident['prod'].items())}); "
             f"{gate['n_games_baseline']} games; the kept-A run covered exactly these.",
             "- Population: baseline projected-usage gate, 7 markets: "
             + ", ".join(f"{m} {n}" for m, n in gate["n_population"].items()) + ".",
             "- Blend weights and calibration maps for season S use only out-of-fold records "
             "of test seasons < S; the first test season is pure A with identity calibration.",
             "- anytime_td is a 2-bin pmf, so its RPS column is its Brier score.",
             f"- Code: git {ident['git_head']}; features: player sha256 "
             f"{ident['player_features']['sha256'][:12]}, team sha256 "
             f"{ident['team_features']['sha256'][:12]}.",
             f"- Elapsed: {gate['elapsed_s'] / 60:.1f} min.", "",
             "## A re-check (kept A vs current sim, 7 markets)", ""]
    lines += tpm._decision_block(gate["a_recheck"]["decision"])
    lines += ["", "Starting sources: " + ", ".join(f"{m} {s}" for m, s in
                                                   gate["a_recheck"]["sources"].items()), "",
              "## B out-of-fold", "",
              f"- {b['fits']} fits (decay {B_DECAY}, max_iter {B_MAX_ITER}) in "
              f"{b['seconds'] / 60:.1f} min; records without a B feature row (kept A): "
              + ", ".join(f"{m} {n}" for m, n in b["missing"].items()) + ".",
              "- B trains and predicts only on each market's pre-game role subset "
              "(pass: QB with pass-att EWM >= 10"
              + ("; ROLE_SUBSETS_V2: also a pass attempt in the last 10 games"
                 if "conds" in (b.get("role_subsets") or {}).get("pass_yds", {}) else "")
              + "; rush: carries EWM >= 3; receiving: WR/TE/RB "
              "with targets EWM >= 2; anytime_td: all). Out-of-role records get B := A: "
              + ", ".join(f"{m} {n}" for m, n in b["out_of_role"].items()) + ".", "",
              "## +B blend", ""]
    lines += _season_table("Weight on A per test season:", gate["blend"]["weights"],
                           lambda w: f"{w:.1f}")
    lines += _rung_section("blend", gate["blend"])
    lines += ["## +calibration", ""]
    lines += _season_table("Map per test season (PIT isotonic; Platt [a, b] for anytime_td):",
                           gate["calibration"]["maps"],
                           lambda c: c if isinstance(c, str) else f"[{c[0]:.2f}, {c[1]:.2f}]")
    lines += _rung_section("calibration", gate["calibration"])
    nan = float("nan")
    lines += ["## Final gate (served pipeline vs current sim)", "",
              f"A market is served from ML iff its RPS <= the baseline's AND its ECE <= the "
              f"baseline's + {ECE_TOL} on the decide seasons "
              f"({', '.join(map(str, gate.get('decide_seasons', gate['seasons'])))}; the table "
              "shows those scores).", "",
              "| market | n | RPS base | RPS ML | ECE base | ECE ML | source | w_final | calibrate |",
              "|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for m, v in gate["pipeline"]["markets"].items():
        pm = f["ml_per_market"].get(m, {"n": 0, "rps_b": nan, "rps_c": nan, "ece_b": nan,
                                        "ece_c": nan})
        lines.append(f"| {m} | {pm['n']} | {pm['rps_b']:.4f} | {pm['rps_c']:.4f} | "
                     f"{pm['ece_b']:.4f} | {pm['ece_c']:.4f} | {v['source']} | "
                     f"{v['w_final']:.1f} | {v['calibrate']} |")
    un = f["unselected"]
    for title, d in (("Selected (per-market sources above)", f),
                     ("Unselected (ML wherever the ladder kept it; no per-market swap)", un)):
        lines += ["", f"### {title}", "", "All seasons:", ""] + tpm._decision_block(d["all"])
        lines += ["", "2025 alone:", ""]
        lines += (tpm._decision_block(d["season_2025"]) if d.get("season_2025") is not None
                  else ["Season 2025 was not part of this run."])
        lines += ["", f"pass = {d['pass']}"]
    lines += ["", "The selected verdict is optimistic by construction where its per-market "
              "sources were chosen on seasons it is scored on (the decide seasons above); the "
              "unselected decision is not.",
              "", f"final_pass = {f['pass']} (unselected: {un['pass']})", "",
              "## Caveats", "", *CAVEATS, ""]
    return "\n".join(lines)


# ---- ladder ---------------------------------------------------------------------------------

def run_b_ladder(env: Mapping[str, str], *, bsn, player_tbl: pd.DataFrame,
                 team_tbl: pd.DataFrame, identity: dict, a_gate: Mapping,
                 data_dir: Path = tpm.DATA_DIR, gate_path: Path = GATE_PATH,
                 pipeline_path: Path = PIPELINE_PATH, report_dir: Path = tpm.REPORT_DIR,
                 log: Callable[[str], None] = print, run_date: str | None = None) -> dict:
    """Baseline + kept-A runs, B OOF, blend and calibration rungs, final gate;
    writes b_gate / pipeline / calibration / report / OOF and served parquets;
    returns the gate dict. ``PROPS_ML_GATE_NAME`` suffixes every file and picks
    the role table (``role_subsets``); ``a_gate`` must carry the same
    ``gate_name`` (none = "")."""
    t0 = time.time()
    gate_name = tpm.gate_name_from_env(env)
    a_file, a_name, subsets = gate_entry(gate_name)
    if str(a_gate.get("gate_name") or "") != a_name:
        raise ValueError(f"PROPS_ML_GATE_NAME {gate_name!r} needs {a_file} (gate_name "
                         f"{a_name!r}); the A gate passed has gate_name "
                         f"{a_gate.get('gate_name', '')!r}")
    raw = env.get("PROPS_ML_SEASONS")
    seasons = [int(x) for x in raw.split(",") if x.strip()] if raw else list(tpm.TEST_SEASONS)
    decide = tpm.decide_seasons_from_env(env, seasons)
    decide_set = set(decide)

    def dec(recs):
        """Only the records of the decide seasons (selection decisions)."""
        return [r for r in recs if r["season"] in decide_set]
    n_sims = int(env.get("PROPS_ML_N_SIMS") or tpm.DEFAULT_N_SIMS)
    refit_weeks = tpm.refit_weeks_from_env(env)
    resume = env.get("PROPS_ML_RESUME") == "1"
    base_tag = tpm.run_tag(seasons, refit_weeks)
    tag = f"{base_tag}__{TAG_SUFFIX}"
    run_date = run_date or date.today().isoformat()
    kept_a, tuned = a_config(a_gate, seasons)
    tuned_json = {str(k): list(v) for k, v in tuned.items()}
    seed = int(bsn.SIM_SEED)
    log(f"stage=sources fetching once for seasons {bsn.backtest_fetch_seasons(seasons)}")
    sources = bsn.fetch_backtest_sources(bsn.backtest_fetch_seasons(seasons))
    identity = {**identity, "prod": dict(tpm.PROD), "seed": seed,
                "backtest_sources": tpm.sources_fingerprint(sources)}
    log(f"tag={tag} seasons={seasons} n_sims={n_sims} refit={list(refit_weeks)} resume={resume} "
        f"kept_a={sorted(kept_a)} gate_name={gate_name!r} decide_seasons={decide}")
    injuries_q = tpm.questionable_index(player_tbl)

    def run(name: str, toggles: frozenset[str]) -> tuple[list[dict], dict]:
        meta = {"n_sims": n_sims, "seasons": seasons, "toggles": sorted(toggles),
                "refit_weeks": list(refit_weeks), "q_weight": tpm.Q_WEIGHT,
                "tuned": tuned_json if toggles else {}, "record_pmf": True, "identity": identity}
        path = tpm.checkpoint_path(data_dir, name, tag, gate_name)
        got = tpm.load_records(path, meta) if resume else None
        if got is not None:
            log(f"stage=run-{name}: loaded {len(got[0])} records from {path.name}")
            return got
        hook, models, errors = None, {}, []
        if toggles:
            for x in seasons:
                for r in refit_weeks:
                    log(f"stage=fit-a season={x} block={r}")
                    models[(x, r)] = learned.fit_models(
                        player_tbl, team_tbl, toggles, upto=(x, r), test_season=x,
                        decay=tuned[x][0], max_iter=tuned[x][1])
            hook, errors = tpm.guard_hook(tpm.make_hook(models, player_tbl, team_tbl, injuries_q,
                                                        refit_weeks=refit_weeks,
                                                        q_weight=tpm.Q_WEIGHT))
        start, state, recs = time.time(), {"key": None, "games": 0}, []

        def on_game(season, week, home, away, sims):
            state["games"] += 1
            key = (season, tpm.refit_block(week, refit_weeks))
            if key != state["key"]:
                state["key"] = key
                log(f"stage=run-{name} season={season} block={key[1]} games={state['games']} "
                    f"{(time.time() - start) / 60:.1f} min")

        bsn.run_backtest(seasons, n_sims, seed=seed, on_game=on_game, spec_hook=hook,
                         record=recs, sources=sources, record_pmf=True, **tpm.PROD)
        tpm.raise_hook_errors(errors, name)
        stats = {"seconds": time.time() - start, "games": state["games"],
                 "share_fallbacks": sum(m.share_fallbacks for m in models.values())}
        tpm.save_records(path, recs, meta, stats)
        log(f"stage=run-{name}: {state['games']} games, {len(recs)} records in "
            f"{stats['seconds'] / 60:.1f} min")
        return recs, stats

    base_recs, _ = run("baseline", frozenset())
    a_recs, a_stats = run("kept_a", kept_a)
    base_games = tpm.game_set(base_recs)
    tpm.check_coverage(base_games, a_recs, "kept_a")
    tpm.check_share_fallbacks(int(a_stats.get("share_fallbacks", 0)), len(base_games), "kept_a")
    population = population_from_baseline(base_recs)
    # memory: only population records (and their pmfs) are needed from here on
    base_recs = [r for r in base_recs if rec_key(r) in population]
    pop_a = [r for r in a_recs if rec_key(r) in population]
    del a_recs
    n_pop = {m: sum(1 for k in population if k[3] == m) for m in MARKETS}
    log(f"stage=population {n_pop}")

    base_dec = dec(base_recs)
    pop_dec = {k for k in population if k[0] in decide_set}
    log(f"stage=a-recheck (decide seasons {decide})")
    a_step = rung_step("a_recheck", base_dec, dec(pop_a), base_dec, pop_dec)
    a_sources = market_sources(a_step["decision"]["per_market"], MARKETS, A_WORSE_TOL)
    log(f"A re-check: skill {a_step['decision']['skill']:+.4f}; sources {a_sources}")

    cols = learned.feature_columns(player_tbl, kept_a)
    b_meta = {"n_sims": n_sims, "seasons": seasons, "refit_weeks": list(refit_weeks),
              "kept_a": sorted(kept_a), "b_decay": B_DECAY, "b_max_iter": B_MAX_ITER,
              "role_subsets": subsets, "identity": identity}
    b_path = Path(data_dir) / f"b_oof__{tag}{gate_name}.parquet"
    got = tpm.load_records(b_path, b_meta) if resume else None
    if got is not None:
        b_pmfs = {rec_key(r): r["pmf_b"] for r in got[0] if r["in_role"]}
        out_of_role, b_stats = {rec_key(r) for r in got[0] if not r["in_role"]}, got[1]
        log(f"stage=b-oof: loaded {len(b_pmfs)} B pmfs from {b_path.name}")
    else:
        start = time.time()
        b_pmfs, out_of_role = b_oof(player_tbl, pop_a, seasons, refit_weeks, cols,
                                    bsn.MARKET_MAX, log, subsets)
        b_stats = {"seconds": time.time() - start,
                   "fits": len({(k[0], tpm.refit_block(k[1], refit_weeks), k[3])
                                for k in b_pmfs})}
        tpm.save_records(b_path, [{"season": k[0], "week": k[1], "player_id": k[2], "market": k[3],
                                   "pmf_b": np.asarray(v, dtype=np.float32), "in_role": True}
                                  for k, v in b_pmfs.items()]
                         + [{"season": k[0], "week": k[1], "player_id": k[2], "market": k[3],
                             "pmf_b": None, "in_role": False} for k in out_of_role],
                         b_meta, b_stats)
    in_role_n = {m: sum(1 for r in pop_a if r["market"] == m and rec_key(r) not in out_of_role)
                 for m in MARKETS}
    missing = {m: sum(1 for r in pop_a if r["market"] == m and rec_key(r) not in b_pmfs
                      and rec_key(r) not in out_of_role) for m in MARKETS}
    oor = {m: sum(1 for k in out_of_role if k[3] == m) for m in MARKETS}
    bad = {m: n for m, n in missing.items() if n > MAX_B_MISSING * max(in_role_n[m], 1)}
    if bad:
        raise RuntimeError(f"B feature rows missing for > {MAX_B_MISSING:.0%} of in-role "
                           f"population records in {sorted(bad)}: missing {missing} of "
                           f"{in_role_n} -- rebuild the feature table")
    # an out-of-role record gets B := A (blend output = A); it is not "missing"
    b_eff = {**b_pmfs, **{rec_key(r): r["pmf"] for r in pop_a if rec_key(r) in out_of_role}}

    def with_b(rows):
        return [r for r in rows if rec_key(r) not in out_of_role]

    ones = {s: {m: 1.0 for m in MARKETS} for s in seasons}
    kept_ml, kept_w, kept_rungs = apply_by_season(pop_a, b_eff, ones), ones, []
    kept_comp = composite(base_recs, kept_ml, a_sources)

    log("stage=blend weights + rung")
    weights = season_weights(with_b(kept_ml), seasons, MARKETS)
    cand = apply_by_season(pop_a, b_eff, weights)
    blend = {**rung_step("blend", dec(kept_comp), dec(composite(base_recs, cand, a_sources)),
                         base_dec, pop_dec), "weights": {str(k): v for k, v in weights.items()}}
    log(f"DECISION rung=blend: {tpm._fmt_decision_line(blend['decision'])} => "
        f"{'PASS' if blend['pass'] else 'FAIL'} weights={weights}")
    if blend["pass"]:
        kept_ml, kept_w, kept_rungs = cand, weights, ["blend"]
        kept_comp = composite(base_recs, kept_ml, a_sources)
    pre_calib = kept_ml

    log("stage=calibration maps + rung")
    calibs = season_calibrations(pre_calib, seasons, MARKETS)
    cand = apply_by_season(pop_a, b_eff, kept_w, calibs)
    calib = {**rung_step("calibration", dec(kept_comp),
                         dec(composite(base_recs, cand, a_sources)), base_dec, pop_dec),
             "maps": {str(s): {m: calib_label(c, m) for m, c in v.items()}
                      for s, v in calibs.items()}}
    log(f"DECISION rung=calibration: {tpm._fmt_decision_line(calib['decision'])} => "
        f"{'PASS' if calib['pass'] else 'FAIL'}")
    if calib["pass"]:
        kept_ml, kept_rungs = cand, kept_rungs + ["calibration"]

    log("stage=final gate")
    # per-market sources: decided on the decide seasons only
    ml_pm = per_market_scores(tpm.checked_paired_frame(base_dec, dec(kept_ml), pop_dec, base_dec,
                                                       "final-ml"))
    sources = market_sources(ml_pm, MARKETS, 1.0, ece_tol=ECE_TOL)

    def final_decision(src: Mapping[str, str], name: str) -> dict:
        served = composite(base_recs, kept_ml, src)
        d_all = rung_decision(tpm.checked_paired_frame(base_recs, served, population, base_recs,
                                                       name))
        d25 = (season_decision(base_recs, served, population, tpm.FINAL_SEASON, f"{name}:2025")
               if tpm.FINAL_SEASON in seasons else None)
        return {"all": d_all, "season_2025": d25, "sources": dict(src),
                "pass": bool(d_all["pass"] and d25 is not None and d25["pass"])}

    selected, unselected = final_decision(sources, "final"), final_decision(a_sources, "final-un")
    w_final = final_weights(with_b(pre_calib), MARKETS, "blend" in kept_rungs)
    data_end = max((int(r["season"]), int(r["week"])) for r in pre_calib)
    pipeline = pipeline_config(kept_a, tuned, sources, w_final, "calibration" in kept_rungs,
                               data_end, final_pass=selected["pass"],
                               unselected_pass=unselected["pass"], run_tag=tag,
                               git=identity.get("git_head"))
    log("stage=serving calibration maps at w_final (all OOF rows)")
    serving_cal = {"data_end": list(data_end),
                   "markets": final_calibration(pre_calib, pipeline["markets"])}
    for label, f in (("selected", selected), ("unselected", unselected)):
        log(f"FINAL {label} all: {tpm._fmt_decision_line(f['all'])}; sources {f['sources']}")
        if f["season_2025"] is not None:
            log(f"FINAL {label} {tpm.FINAL_SEASON}: {tpm._fmt_decision_line(f['season_2025'])}")
    log(f"FINAL kept_rungs={kept_rungs} final_pass={selected['pass']} "
        f"(unselected {unselected['pass']})")

    oof_path = Path(data_dir) / f"oof_b7__{base_tag}{gate_name}.parquet"
    oof_frame(pre_calib, out_of_role).to_parquet(oof_path, index=False)
    served_out = served_path(data_dir, base_tag, gate_name)
    served_frame(composite(base_recs, kept_ml, sources), population).to_parquet(served_out,
                                                                                 index=False)
    gate = {"run_date": run_date, "seasons": seasons, "n_sims": n_sims,
            "refit_weeks": list(refit_weeks), "run_tag": tag, "gate_name": gate_name,
            "decide_seasons": decide, "identity": identity,
            "tuned": tuned_json, "n_games_baseline": len(base_games), "n_population": n_pop,
            "a_recheck": {**a_step, "sources": a_sources},
            "b_oof": {**b_stats, "fits": int(b_stats.get("fits", 0)), "missing": missing,
                      "out_of_role": oor, "role_subsets": subsets},
            "blend": blend, "calibration": calib, "kept_rungs": kept_rungs,
            "final": {**selected, "ml_per_market": ml_pm, "unselected": unselected},
            "pipeline": pipeline, "oof_path": str(oof_path), "served_path": str(served_out),
            "elapsed_s": time.time() - t0}
    out_gate, out_pipe, out_cal, out_report = output_paths(
        base_tag, run_date, gate_path=gate_path, pipeline_path=pipeline_path,
        report_dir=report_dir, gate_name=gate_name)
    for path, text in ((out_gate, tpm.gate_json(gate)), (out_pipe, tpm.gate_json(pipeline)),
                       (out_cal, tpm.gate_json(serving_cal)), (out_report, render_report(gate))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    log(f"wrote {out_gate}, {out_pipe}, {out_cal}, {out_report}, {oof_path}, {served_out}")
    return gate


def main() -> None:
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[+{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)

    tpm.require_gate_tables()   # before anything runs: serving-param tables leak the verdict seasons
    a_path = a_gate_path(tpm.gate_name_from_env(os.environ))  # ValueError on an unknown name
    for p in (tpm.PLAYER_PATH, tpm.TEAM_PATH, a_path):
        if not p.exists():
            raise SystemExit(f"missing {p}: build the features / run the A gate first")
    identity = {"player_features": tpm.file_fingerprint(tpm.PLAYER_PATH),
                "team_features": tpm.file_fingerprint(tpm.TEAM_PATH), "git_head": tpm.git_head()}
    log(f"git={identity['git_head'][:12]}")
    run_b_ladder(os.environ, bsn=tpm._load_backtest(), player_tbl=pd.read_parquet(tpm.PLAYER_PATH),
                 team_tbl=pd.read_parquet(tpm.TEAM_PATH), identity=identity,
                 a_gate=json.loads(a_path.read_text()), log=log)


if __name__ == "__main__":
    main()
