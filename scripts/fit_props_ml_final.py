"""Final fit, model artifacts and the weekly quick gate for props-ML (spec §4-§5).

Inputs: the COMMITTED ``assets/nfl/props_ml/pipeline.json`` and
``calibration.json`` (both written by ``train_props_ml_b.py``: per-market
source / ``w_final`` / calibrate, the serving calibration maps fit on the
ladder's OOF rows, and ``data_end`` = the last OOF (season, week)), plus
``a_gate.json`` and the feature tables. No OOF records are read here.

Final fit (default, ``--holdout-weeks 0``)
-----------------------------------------
Fits the gated pipeline on every completed game in the feature table and
writes ``data/props_ml/models/<version>/`` atomically (temp dir, then swap):

* ``learned.joblib`` -- A (``learned.fit_models``) with the kept toggles and
  the MOST RECENT test season's tuned ``(decay, max_iter)`` from
  ``a_gate.json``;
* ``b_<market>.joblib`` -- B (``dist_models.fit_market``, unweighted,
  ``max_iter`` 150, in-role rows only) for every market that READS B:
  ``source == "ml"`` and ``w_final < 1`` (``b_markets``);
* ``calibration.json`` -- the committed serving maps, copied;
* ``props_ml_config.json`` -- ``pipeline.json`` + ``trained_through``,
  ``fit_upto``, ``git``, ``features`` (player/team fingerprints),
  ``created_at``, ``a_fit``, ``b_fit``, ``b_markets``, ``q_weight``,
  ``role_subsets``, ``market_max``, ``feature_columns`` and ``artifact_files``
  -- everything serving needs without importing the training scripts.

Loading / verifying the artifacts (``load_artifacts``, ``required_columns``,
``calib_for_apply``, ``b_markets``, the file names) lives in
``sportsmodel.model.props_ml.artifacts`` (shared with live serving,
``sim.nfl.ml_serving``); this script re-exports those names.

Versions (``--gate-name``)
--------------------------
* none (default): ``nfl-sim-ml-v1`` -- ``pipeline.json`` / ``calibration.json``
  / ``a_gate.json``, v1 ``ROLE_SUBSETS``, written to
  ``data/props_ml/models/nfl-sim-ml-v1/``; the config is exactly v1's (no
  ``model_version`` key: a config without one IS v1).
* ``_v2``: ``nfl-sim-ml-v2`` -- ``pipeline_v2.json`` / ``calibration_v2.json``
  / ``a_gate_v2.json`` (re-tunes nothing), ``ROLE_SUBSETS_V2``, written to
  ``data/props_ml/models/nfl-sim-ml-v2/`` with ``"model_version":
  "nfl-sim-ml-v2"`` and ``"qb_profile_params"`` = the ``serving`` block of
  ``assets/nfl/props_ml/qb_profile_params.json`` (the final fit is refused
  when that block is missing: run ``tune_qb_profile.py --mode serving``).
  Live serving builds the v2 feature tables with those (H, k).

``trained_through`` = the last (season, week) with a played label in the player
table; every model trains on rows strictly before ``trained_through + 1 week``
(``fit_upto``).

Quick gate (``--holdout-weeks N``; plan ruling 9)
------------------------------------------------
Holdout = the last N completed (season, week)s STRICTLY AFTER pipeline.json's
``data_end`` (the blend weights and calibration maps saw every OOF week up to
it); none left -> the gate fails. A and B are fit only on data strictly before
the first holdout week. The baseline (current sim) and the pipeline (A via
``spec_hook`` + offline B / blend / calibration via ``apply_pipeline``, the
per-market sources) run on the holdout weeks alone (``run_backtest`` with the
schedules filtered to them, ``record_pmf``). Pass iff the pooled relative-RPS
skill point estimate >= 0 AND no market's RPS > 1.05 x the baseline's AND the
coverage / share-fallback / B-feature checks pass. ``quick_gate.json`` is
removed at entry and written on EVERY exit path (an exception writes a failing
record, then re-raises); exit 1 on failure. No model artifact is written.

Refusal (both modes): ``pipeline_refusal`` -- ``pipeline.json``'s
``final_pass`` (the B gate's SELECTED final decision) must be a JSON ``true``
and at least one market must have source ``"ml"``; otherwise ``main`` prints
``REFUSED: <reason>`` and exits 1 before fitting anything, ``run_final_fit``
raises, and the quick gate fails with that reason. A failed B gate is never
fit or published.

Pure parts are unit tested (tests/scripts/test_fit_props_ml_final.py);
``main`` / ``_load_inputs`` (file IO) are not.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel.model.props_eval import population_from_baseline, rung_decision  # noqa: E402
from sportsmodel.model.props_ml import artifacts  # noqa: E402,F401  (re-exported)
from sportsmodel.model.props_ml.artifacts import (  # noqa: E402,F401  (re-exported)
    CALIB_FILE,
    CONFIG_FILE,
    FORMAT_VERSION,
    LEARNED_FILE,
    ML_V1,
    ML_V2,
    ROLE_TABLES,
    Artifacts,
    _is_artifact,
    b_markets,
    calib_for_apply,
    load_artifacts,
    required_columns,
    verify_feature_columns,
)
from sportsmodel.model.props_ml.blend import BINARY_MARKET  # noqa: E402
from sportsmodel.model.props_ml.dist_models import (  # noqa: E402
    ROLE_SUBSETS,
    fit_market,
    in_role,
    predict_pmfs,
    role_conditions,
)
from sportsmodel.sim.nfl import learned  # noqa: E402


def _load_script(name: str):
    """Scripts are not a package: load one by path (Ruling P2)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tpb = _load_script("train_props_ml_b")
tpm = tpb.tpm

MODEL_ROOT = tpm.DATA_DIR / "models"
QB_PARAMS_PATH = ROOT / "assets" / "nfl" / "props_ml" / "qb_profile_params.json"
# --gate-name -> the served model version it fits (the v1 comparison run
# ``_v1cmp`` is a gate input only: never fit or served)
GATE_VERSIONS = {"": ML_V1, tpb.V2_GATE_NAME: ML_V2}
QUICK_GATE_FILE = "quick_gate.json"
QUICK_MIN_SKILL = 0.0
QUICK_MAX_RPS_RATIO = 1.05
KEY = ["season", "week", "player_id"]


def gate_version(gate_name: str) -> str:
    """The model version a ``--gate-name`` fits; ValueError for any other name."""
    if gate_name not in GATE_VERSIONS:
        raise ValueError(f"--gate-name {gate_name!r} is not one of {sorted(GATE_VERSIONS)}")
    return GATE_VERSIONS[gate_name]


def gate_subsets(gate_name: str) -> dict:
    """B's role table for the gate name's version (Ruling P1)."""
    return ROLE_TABLES[gate_version(gate_name)]


def model_dir(version: str) -> Path:
    """``data/props_ml/models/<version>/``."""
    if version not in ROLE_TABLES:
        raise ValueError(f"unknown props-ML model version {version!r}")
    return MODEL_ROOT / version


MODEL_DIR = model_dir(ML_V1)


def input_paths(gate_name: str) -> tuple[Path, Path, Path]:
    """(pipeline json, calibration json, A gate json) of the gate name."""
    gate_version(gate_name)
    return (tpb.PIPELINE_PATH.with_name(f"pipeline{gate_name}.json"),
            tpb.CALIBRATION_PATH.with_name(f"calibration{gate_name}.json"),
            tpb.a_gate_path(gate_name))


def serving_qb_block(path: Path | None = None) -> dict:
    """The ``serving`` block of qb_profile_params.json (fit on 2021 -> latest);
    RuntimeError when it is absent."""
    path = QB_PARAMS_PATH if path is None else Path(path)
    block = json.loads(Path(path).read_text()).get("serving")
    if not block:
        raise RuntimeError(f"{path} has no 'serving' block -- run "
                           "`uv run python scripts/tune_qb_profile.py --mode serving` first")
    return dict(block)


# ---- pure helpers: weeks / config inputs --------------------------------------------------

def labelled_weeks(player_tbl: pd.DataFrame) -> list[tuple[int, int]]:
    """Sorted (season, week)s with at least one played row: not a stub and
    some ``y_*`` label present (future / upcoming rows have no labels)."""
    ycols = [c for c in player_tbl.columns if c.startswith("y_")]
    played = player_tbl[ycols].notna().any(axis=1)
    if "is_stub" in player_tbl.columns:
        played &= ~player_tbl["is_stub"].fillna(False).astype(bool)
    wk = player_tbl.loc[played, ["season", "week"]].drop_duplicates()
    return sorted((int(s), int(w)) for s, w in zip(wk["season"], wk["week"]))


def trained_through(player_tbl: pd.DataFrame) -> tuple[int, int]:
    """The last (season, week) with played labels."""
    weeks = labelled_weeks(player_tbl)
    if not weeks:
        raise ValueError("the player feature table has no played (labelled) rows")
    return weeks[-1]


def new_labelled_weeks(player_tbl: pd.DataFrame, after: tuple[int, int]) -> list[tuple[int, int]]:
    """Labelled (season, week)s STRICTLY AFTER ``after`` (e.g. the published
    config's ``trained_through``). PURE."""
    return [w for w in labelled_weeks(player_tbl) if w > (int(after[0]), int(after[1]))]


def next_week(sw: tuple[int, int]) -> tuple[int, int]:
    """``(season, week + 1)``: the exclusive ``upto`` covering ``sw``."""
    return (int(sw[0]), int(sw[1]) + 1)


def holdout_weeks(player_tbl: pd.DataFrame, n: int, after: tuple[int, int]
                  ) -> list[tuple[int, int]]:
    """The last ``n`` completed (season, week)s STRICTLY AFTER ``after``
    (fewer if fewer remain; [] if none). ValueError for ``n < 1`` or when no
    completed week precedes the holdout (nothing to train on)."""
    if n < 1:
        raise ValueError(f"holdout needs n >= 1 weeks, got {n}")
    weeks = labelled_weeks(player_tbl)
    out = [w for w in weeks if w > tuple(after)][-n:]
    if out and not any(w < out[0] for w in weeks):
        raise ValueError(f"no completed week before the holdout {out}: nothing to train on")
    return out


def pipeline_refusal(pipeline: Mapping) -> str | None:
    """Why this pipeline.json must not be fit / published, or None. PURE.
    Refuses unless ``final_pass`` is exactly ``True`` (the selected final B
    gate decision passed) and at least one market's source is ``"ml"``."""
    fp = pipeline.get("final_pass")
    if fp is not True:
        return (f"pipeline.json final_pass is {fp!r} (the B gate's selected final decision did "
                f"not pass, or pipeline.json predates final_pass): a failed B gate is never fit "
                f"or published")
    if not any(str(v.get("source")) == "ml" for v in pipeline.get("markets", {}).values()):
        return 'pipeline.json has no market with source "ml": nothing for the ML path to serve'
    return None


def pipeline_data_end(pipeline: Mapping) -> tuple[int, int]:
    if "data_end" not in pipeline:
        raise ValueError("pipeline.json has no data_end: re-run train_props_ml_b.py")
    return (int(pipeline["data_end"][0]), int(pipeline["data_end"][1]))


def check_calibration(pipeline: Mapping, calibration: Mapping) -> None:
    """ValueError unless calibration.json matches pipeline.json (same
    ``data_end``, markets and calibrate flags) -- both come from one ladder run."""
    got = {m: v["calibrate"] for m, v in calibration["markets"].items()}
    want = {m: v["calibrate"] for m, v in pipeline["markets"].items()}
    if list(calibration.get("data_end", [])) != list(pipeline_data_end(pipeline)) or got != want:
        raise ValueError(f"calibration.json (data_end {calibration.get('data_end')}, calibrate "
                         f"{got}) does not match pipeline.json (data_end {pipeline['data_end']}, "
                         f"calibrate {want}): re-run train_props_ml_b.py")


def a_fit_params(a_gate: Mapping) -> dict:
    """``{season, decay, max_iter}``: the most recent test season's tuned values."""
    season = max(int(s) for s in a_gate["tuned"])
    decay, max_iter = a_gate["tuned"][str(season)]
    return {"season": season, "decay": float(decay), "max_iter": int(max_iter)}


def kept_toggles(pipeline: Mapping, a_gate: Mapping) -> frozenset[str]:
    """pipeline.json's kept A toggles; ValueError unless they equal a_gate.json's
    and are non-empty (there is no learned A to fit otherwise)."""
    kept = frozenset(pipeline["kept_a_toggles"])
    if kept != frozenset(a_gate["kept"]):
        raise ValueError(f"pipeline.json kept_a_toggles {sorted(kept)} != a_gate.json kept "
                         f"{sorted(a_gate['kept'])}: re-run train_props_ml_b.py")
    if not kept:
        raise ValueError("no kept A toggles: nothing to fit")
    return kept


def role_columns(player_tbl: pd.DataFrame, markets, subsets: Mapping = ROLE_SUBSETS) -> list[str]:
    """Columns ``in_role`` reads for ``markets`` under the role table
    ``subsets`` (every condition column; sorted)."""
    cols = set()
    for m in markets:
        spec = subsets[m]
        if spec is None:
            continue
        cols.update(c for c, _, _ in role_conditions(spec))
        if spec["positions"] is not None:
            cols.add("position" if "position" in player_tbl.columns else "p_pos")
    return sorted(cols)


def holdout_sources(sources: Mapping, weeks) -> dict:
    """Shallow copy of the backtest sources whose ``schedules`` holds only the
    holdout (season, week)s (``run_backtest`` walks the schedule's games)."""
    sch, want = sources["schedules"], {(int(s), int(w)) for s, w in weeks}
    keep = pd.Series([(int(s), int(w)) in want for s, w in zip(sch["season"], sch["week"])],
                     index=sch.index)
    return {**sources, "schedules": sch[keep]}


# ---- pure helpers: B fit / predict ----------------------------------------------------------

def fit_b_models(player_tbl: pd.DataFrame, cols: list[str], markets, upto: tuple[int, int],
                 log: Callable[[str], None], subsets: Mapping = ROLE_SUBSETS) -> dict:
    """``{m: MarketModel}``: ``fit_market`` on in-role (``subsets``) rows
    strictly before ``upto``, unweighted, ``max_iter`` 150 (the ladder's B
    settings)."""
    out = {}
    for m in markets:
        t0 = time.time()
        out[m] = fit_market(player_tbl, m, cols, upto=upto, test_season=int(upto[0]),
                            decay=tpb.B_DECAY, max_iter=tpb.B_MAX_ITER, subsets=subsets)
        log(f"stage=fit-b market={m} upto={upto} {time.time() - t0:.1f}s")
    return out


def predict_b(models: Mapping, player_tbl: pd.DataFrame, recs, market_max: Mapping,
              subsets: Mapping = ROLE_SUBSETS) -> tuple[dict, set]:
    """``(rec_key -> B pmf, out-of-role keys)`` for records of markets in
    ``models`` from their (season, week, player_id) feature rows; a record
    whose feature row is out of role (``subsets``) is in the set (B := A);
    one without a feature row is in neither (missing)."""
    feats = player_tbl.drop_duplicates(KEY)
    by_m = tpb._by_market(recs)
    out, oor = {}, set()
    for m, model in models.items():
        if not by_m.get(m):
            continue
        keys = pd.DataFrame([{k: r[k] for k in KEY} for r in by_m[m]])
        rows = keys.merge(feats, on=KEY, how="inner")
        role = in_role(rows, m, subsets).to_numpy()
        oor |= {(int(s), int(w), str(p), m) for s, w, p in
                zip(rows["season"][~role], rows["week"][~role], rows["player_id"][~role])}
        rows = rows[role]
        if rows.empty:
            continue
        for s, w, p, pmf in zip(rows["season"], rows["week"], rows["player_id"],
                                predict_pmfs(model, rows, int(market_max.get(m, 1)))):
            out[(int(s), int(w), str(p), m)] = pmf
    return out, oor


def b_missing_reasons(recs, b_pmfs: Mapping, out_of_role: set, markets) -> list[str]:
    """A reason per market whose in-role records lack a B feature row for
    more than ``MAX_B_MISSING`` of them."""
    reasons = []
    for m in markets:
        keys = [tpb.rec_key(r) for r in recs if r["market"] == m]
        inr = [k for k in keys if k not in out_of_role]
        miss = sum(1 for k in inr if k not in b_pmfs)
        if miss > tpb.MAX_B_MISSING * max(len(inr), 1):
            reasons.append(f"market {m}: B feature rows missing for {miss} of {len(inr)} "
                           "in-role records -- rebuild the feature table")
    return reasons


# ---- pure helpers: config / artifacts --------------------------------------------------------

def _learned_columns(lm) -> dict:
    def cols(m):
        return None if m is None else list(m.cols)
    return {"player": list(lm.player_cols), "team": list(lm.team_cols),
            "fitted": {"team_pass": cols(lm.team_pass), "team_rush": cols(lm.team_rush),
                       "targets": cols(lm.targets), "carries": cols(lm.carries),
                       "eff": ({n: cols(m) for n, m in lm.eff.items()} if lm.eff else None)}}


def build_config(pipeline: Mapping, *, learned_models, b_models: Mapping, a_fit: Mapping,
                 trained_through: tuple[int, int], market_max: Mapping, role_cols: list[str],
                 identity: Mapping, created_at: str, gate_name: str = "",
                 qb_profile_params: Mapping | None = None) -> dict:
    """``props_ml_config.json``: pipeline.json + everything serving needs.
    v1 (no gate name) is exactly today's config; ``_v2`` adds
    ``model_version`` and ``qb_profile_params`` (required: ValueError) and
    records ``ROLE_SUBSETS_V2``."""
    version = gate_version(gate_name)
    extra = {}
    if version != ML_V1:
        if not qb_profile_params:
            raise ValueError(f"{version} config needs qb_profile_params (the serving block)")
        extra = {"model_version": version, "qb_profile_params": dict(qb_profile_params)}
    return {**pipeline, **extra, "format_version": FORMAT_VERSION,
            "trained_through": [int(trained_through[0]), int(trained_through[1])],
            "fit_upto": list(next_week(trained_through)),
            "git": identity.get("git_head"), "pipeline_git": pipeline.get("git"),
            "features": {"player": identity.get("player_features"),
                         "team": identity.get("team_features")},
            "created_at": created_at, "a_fit": dict(a_fit),
            "b_fit": {"decay": tpb.B_DECAY, "max_iter": tpb.B_MAX_ITER},
            "b_markets": list(b_models), "q_weight": tpm.Q_WEIGHT,
            "role_subsets": ROLE_TABLES[version],
            "market_max": {**{m: int(k) for m, k in market_max.items()}, BINARY_MARKET: 1},
            "feature_columns": {"learned": _learned_columns(learned_models),
                                "b": {m: list(model.cols) for m, model in b_models.items()},
                                "role": list(role_cols)},
            "artifact_files": {"learned": LEARNED_FILE, "calibration": CALIB_FILE,
                               "b": {m: f"b_{m}.joblib" for m in b_models}}}


def save_artifacts(out_dir: Path, learned_models, b_models: Mapping, calibration: Mapping,
                   config: Mapping) -> None:
    """Write the model files into a temp dir next to ``out_dir``, carry over
    its non-artifact files (``quick_gate.json``), then swap the directories:
    a failure while writing leaves the previous artifacts in place, and stale
    ``b_*.joblib`` never survive."""
    out = Path(out_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.new-", dir=out.parent))
    old = out.with_name(f".{out.name}.old")
    try:
        joblib.dump(learned_models, tmp / LEARNED_FILE)
        for m, model in b_models.items():
            joblib.dump(model, tmp / f"b_{m}.joblib")
        (tmp / CALIB_FILE).write_text(tpm.gate_json(dict(calibration)))
        (tmp / CONFIG_FILE).write_text(tpm.gate_json(dict(config)))
        if out.exists():
            for p in out.iterdir():
                if p.is_file() and not _is_artifact(p.name):
                    shutil.copy2(p, tmp / p.name)
            shutil.rmtree(old, ignore_errors=True)
            out.rename(old)
            try:
                tmp.rename(out)
            except BaseException:
                old.rename(out)
                raise
            shutil.rmtree(old, ignore_errors=True)
        else:
            tmp.rename(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- quick-gate decision -------------------------------------------------------------------

def quick_gate_decision(per_market: Mapping, skill: float | None, check_reasons=()) -> dict:
    """Pass iff no check failed, some market was scored, the pooled skill
    point estimate exists and is >= 0, and no market's RPS > 1.05 x the
    baseline's."""
    reasons = list(check_reasons)
    if not reasons:
        if not per_market:
            reasons.append("no scored records")
        if skill is None:
            reasons.append("pooled relative RPS skill unavailable")
        elif skill < QUICK_MIN_SKILL:
            reasons.append(f"pooled relative RPS skill {skill:+.4f} < {QUICK_MIN_SKILL}")
        for m, v in sorted(per_market.items()):
            if v["rps_c"] > QUICK_MAX_RPS_RATIO * v["rps_b"]:
                reasons.append(f"market {m}: RPS {v['rps_c']:.4f} > {QUICK_MAX_RPS_RATIO} x "
                               f"baseline {v['rps_b']:.4f}")
    return {"pass": not reasons, "reasons": reasons}


# ---- runs ------------------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_final_fit(*, bsn, player_tbl: pd.DataFrame, team_tbl: pd.DataFrame, pipeline: Mapping,
                  a_gate: Mapping, calibration: Mapping, identity: Mapping,
                  out_dir: Path = MODEL_DIR, log: Callable[[str], None] = print,
                  created_at: str | None = None, gate_name: str = "",
                  qb_profile_params: Mapping | None = None) -> dict:
    """Fit A and B on every completed week; write the artifacts (with the
    committed calibration maps); return the config. RuntimeError (nothing
    fit or written) when ``pipeline_refusal`` names a reason, or for v2
    (``gate_name`` ``_v2``) without the serving ``qb_profile_params``."""
    refusal = pipeline_refusal(pipeline)
    if refusal:
        raise RuntimeError(refusal)
    subsets = gate_subsets(gate_name)
    if gate_version(gate_name) != ML_V1 and not qb_profile_params:
        raise RuntimeError("the v2 final fit needs the 'serving' qb_profile_params block -- run "
                           "`uv run python scripts/tune_qb_profile.py --mode serving` first")
    kept, a_fit = kept_toggles(pipeline, a_gate), a_fit_params(a_gate)
    check_calibration(pipeline, calibration)
    tt = trained_through(player_tbl)
    upto = next_week(tt)
    log(f"stage=fit-a trained_through={tt} upto={upto} toggles={sorted(kept)} {a_fit}")
    t0 = time.time()
    lm = learned.fit_models(player_tbl, team_tbl, kept, upto=upto, test_season=int(upto[0]),
                            decay=a_fit["decay"], max_iter=a_fit["max_iter"])
    log(f"stage=fit-a done {time.time() - t0:.1f}s")
    need_b = b_markets(pipeline)
    b_models = fit_b_models(player_tbl, learned.feature_columns(player_tbl, kept), need_b, upto,
                            log, subsets)
    config = build_config(pipeline, learned_models=lm, b_models=b_models, a_fit=a_fit,
                          trained_through=tt, market_max=bsn.MARKET_MAX,
                          role_cols=role_columns(player_tbl, need_b, subsets), identity=identity,
                          created_at=created_at or _now(), gate_name=gate_name,
                          qb_profile_params=qb_profile_params)
    save_artifacts(out_dir, lm, b_models, calibration, config)
    log(f"wrote {out_dir}: {gate_version(gate_name)} learned + B {need_b} + calibration + config "
        f"(trained_through={list(tt)})")
    return config


def _quick_gate_body(n_weeks: int, rec: dict, *, bsn, player_tbl, team_tbl, pipeline, a_gate,
                     calibration, n_sims, log, gate_name: str = "") -> dict:
    """The quick gate's evaluation; fills ``rec`` (holdout) as it goes."""
    subsets = gate_subsets(gate_name)
    refusal = pipeline_refusal(pipeline)
    if refusal:
        return {"pass": False, "reasons": [refusal]}
    kept, a_fit = kept_toggles(pipeline, a_gate), a_fit_params(a_gate)
    check_calibration(pipeline, calibration)
    data_end = pipeline_data_end(pipeline)
    rec["data_end"] = list(data_end)
    weeks = holdout_weeks(player_tbl, n_weeks, after=data_end)
    if not weeks:
        return {"pass": False, "reasons": [f"no completed weeks after ladder data_end {data_end}"]}
    rec["holdout_weeks"] = [list(w) for w in weeks]
    start, seasons = weeks[0], sorted({s for s, _ in weeks})
    rec["fit_upto"] = list(start)
    sources_map = {m: v["source"] for m, v in pipeline["markets"].items()}
    weights = {m: float(v["w_final"]) for m, v in pipeline["markets"].items()}
    log(f"stage=quick-gate holdout={weeks} data_end={data_end} fit_upto={start} n_sims={n_sims}")

    log(f"stage=fit-a upto={start} toggles={sorted(kept)} {a_fit}")
    lm = learned.fit_models(player_tbl, team_tbl, kept, upto=start, test_season=int(start[0]),
                            decay=a_fit["decay"], max_iter=a_fit["max_iter"])
    need_b = b_markets(pipeline)
    b_models = fit_b_models(player_tbl, learned.feature_columns(player_tbl, kept), need_b, start,
                            log, subsets)
    fetch = bsn.backtest_fetch_seasons(seasons)
    log(f"stage=sources fetching {fetch}")
    sources = holdout_sources(bsn.fetch_backtest_sources(fetch), weeks)
    hook, errors = tpm.guard_hook(tpm.make_hook({(s, 1): lm for s in seasons}, player_tbl,
                                                team_tbl, tpm.questionable_index(player_tbl),
                                                refit_weeks=(1,), q_weight=tpm.Q_WEIGHT))
    runs = {}
    for name, h in (("baseline", None), ("pipeline", hook)):
        recs: list[dict] = []
        st = time.time()
        bsn.run_backtest(seasons, n_sims, seed=int(bsn.SIM_SEED), on_game=lambda *a: None,
                         spec_hook=h, record=recs, sources=sources, record_pmf=True, **tpm.PROD)
        runs[name] = recs
        log(f"stage=run-{name}: {len(tpm.game_set(recs))} games, {len(recs)} records "
            f"{(time.time() - st) / 60:.1f} min")

    base_recs = runs["baseline"]
    base_games = tpm.game_set(base_recs)
    checks: list[str] = []
    for check in (lambda: tpm.raise_hook_errors(errors, "quick_pipeline"),
                  lambda: tpm.check_coverage(base_games, runs["pipeline"], "quick_pipeline"),
                  lambda: tpm.check_share_fallbacks(int(lm.share_fallbacks), len(base_games),
                                                    "quick_pipeline")):
        try:
            check()
        except RuntimeError as exc:
            checks.append(str(exc))

    population = population_from_baseline(base_recs)
    d = {"skill": None, "lo": None, "hi": None, "per_market": {}}
    if not checks:
        base_pop = [r for r in base_recs if tpb.rec_key(r) in population]
        pop_a = [r for r in runs["pipeline"] if tpb.rec_key(r) in population]
        b_pmfs, oor = predict_b(b_models, player_tbl, pop_a, bsn.MARKET_MAX, subsets)
        checks += b_missing_reasons(pop_a, b_pmfs, oor, need_b)
        b_eff = {**b_pmfs, **{tpb.rec_key(r): r["pmf"] for r in pop_a if tpb.rec_key(r) in oor}}
        ml = tpb.apply_pipeline(pop_a, b_eff, weights, calib_for_apply(calibration))
        try:
            df = tpm.checked_paired_frame(base_pop, tpb.composite(base_pop, ml, sources_map),
                                          population, base_pop, "quick")
            rd = rung_decision(df)
            d = {k: rd[k] for k in ("skill", "lo", "hi", "per_market")}
        except RuntimeError as exc:
            checks.append(str(exc))
    return {"n_games": len(base_games),
            "n_population": {m: sum(1 for k in population if k[3] == m) for m in sources_map},
            "sources": sources_map, "weights": weights, "b_markets": need_b,
            "calibrate": {m: v["calibrate"] for m, v in calibration["markets"].items()},
            "a_fit": a_fit, "kept_a_toggles": pipeline["kept_a_toggles"], **d,
            **quick_gate_decision(d["per_market"], d["skill"], checks)}


def run_quick_gate(n_weeks: int, *, bsn, player_tbl: pd.DataFrame, team_tbl: pd.DataFrame,
                   pipeline: Mapping, a_gate: Mapping, calibration: Mapping,
                   identity: Mapping, out_dir: Path = MODEL_DIR, n_sims: int = 1000,
                   log: Callable[[str], None] = print, created_at: str | None = None,
                   gate_name: str = "") -> dict:
    """Score the pipeline fit without the last ``n_weeks`` completed weeks
    after ``data_end`` on them vs the baseline. ``quick_gate.json`` is removed
    first and written on every exit path (an exception writes a failing
    record and is re-raised); returns the gate dict."""
    t0 = time.time()
    path = Path(out_dir) / QUICK_GATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    rec = {"mode": "quick_gate", "gate_name": gate_name, "model_version": gate_version(gate_name),
           "created_at": created_at or _now(), "holdout_weeks": None,
           "n_sims": n_sims, "thresholds": {"min_skill": QUICK_MIN_SKILL,
                                            "max_rps_ratio": QUICK_MAX_RPS_RATIO},
           "identity": dict(identity)}

    def write(gate: dict) -> None:
        path.write_text(tpm.gate_json({**gate, "elapsed_s": time.time() - t0}))
        log(f"QUICK GATE {'PASS' if gate['pass'] else 'FAIL'}: skill "
            f"{'n/a' if gate.get('skill') is None else format(gate['skill'], '+.4f')}"
            + ("" if gate["pass"] else f"; reasons: {gate['reasons']}") + f" -> wrote {path}")

    try:
        body = _quick_gate_body(n_weeks, rec, bsn=bsn, player_tbl=player_tbl,
                                team_tbl=team_tbl, pipeline=pipeline, a_gate=a_gate,
                                calibration=calibration, n_sims=n_sims, log=log,
                                gate_name=gate_name)
        gate = {**rec, **body}   # rec gains the holdout while the body runs
    except Exception as exc:
        write({**rec, "pass": False, "reasons": [f"error: {type(exc).__name__}: {exc}"]})
        raise
    write(gate)
    return gate


# ---- IO ----------------------------------------------------------------------------------

def _load_inputs(args) -> dict:
    """Tables, pipeline / calibration / A-gate json (of ``args.gate_name``),
    identity, backtest module."""
    a_gate_path = input_paths(args.gate_name)[2]
    for p in (tpm.PLAYER_PATH, tpm.TEAM_PATH, a_gate_path, args.pipeline, args.calibration):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}: build the features / run the A and B gates first")
    return {"bsn": tpm._load_backtest(), "player_tbl": pd.read_parquet(tpm.PLAYER_PATH),
            "team_tbl": pd.read_parquet(tpm.TEAM_PATH),
            "pipeline": json.loads(Path(args.pipeline).read_text()),
            "calibration": json.loads(Path(args.calibration).read_text()),
            "a_gate": json.loads(a_gate_path.read_text()),
            "identity": {"player_features": tpm.file_fingerprint(tpm.PLAYER_PATH),
                         "team_features": tpm.file_fingerprint(tpm.TEAM_PATH),
                         "git_head": tpm.git_head()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout-weeks", type=int, default=0,
                    help="quick gate on the last N completed weeks after data_end (0 = final fit)")
    ap.add_argument("--n-sims", type=int, default=1000, help="sims per game (quick gate)")
    ap.add_argument("--gate-name", default="", choices=sorted(GATE_VERSIONS),
                    help="'' = nfl-sim-ml-v1 (default); _v2 = nfl-sim-ml-v2 (pipeline_v2.json, "
                         "calibration_v2.json, a_gate_v2.json, ROLE_SUBSETS_V2, serving QB params)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default data/props_ml/models/<version of --gate-name>/")
    ap.add_argument("--pipeline", type=Path, default=None,
                    help="default pipeline{gate-name}.json")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="default calibration{gate-name}.json")
    ap.add_argument("--check-new-weeks", type=Path, metavar="CONFIG",
                    help="only report whether the player table has a labelled week after "
                         "CONFIG's trained_through (last line new_weeks=true|false); fits nothing")
    ap.add_argument("--player-table", type=Path, default=tpm.PLAYER_PATH,
                    help="player feature table for --check-new-weeks")
    args = ap.parse_args(argv)
    pipe_path, cal_path, _ = input_paths(args.gate_name)
    args.pipeline = pipe_path if args.pipeline is None else args.pipeline
    args.calibration = cal_path if args.calibration is None else args.calibration
    out_dir = model_dir(gate_version(args.gate_name)) if args.out_dir is None else args.out_dir
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[+{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)

    if args.check_new_weeks is not None:
        tt = json.loads(Path(args.check_new_weeks).read_text())["trained_through"]
        new = new_labelled_weeks(pd.read_parquet(args.player_table), (tt[0], tt[1]))
        print(f"props-ML: labelled weeks after the published trained_through {list(tt)}: "
              f"{[list(w) for w in new]}", flush=True)
        print(f"new_weeks={'true' if new else 'false'}", flush=True)
        return 0

    inputs = _load_inputs(args)
    refusal = pipeline_refusal(inputs["pipeline"])
    if refusal:
        print(f"REFUSED: {refusal}", flush=True)
        return 1
    if args.holdout_weeks > 0:
        gate = run_quick_gate(args.holdout_weeks, **inputs, out_dir=out_dir,
                              n_sims=args.n_sims, log=log, gate_name=args.gate_name)
        return 0 if gate["pass"] else 1
    qb = None
    if gate_version(args.gate_name) != ML_V1:
        try:
            qb = serving_qb_block()
        except RuntimeError as exc:
            print(f"REFUSED: {exc}", flush=True)
            return 1
    run_final_fit(**inputs, out_dir=out_dir, log=log, gate_name=args.gate_name,
                  qb_profile_params=qb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
