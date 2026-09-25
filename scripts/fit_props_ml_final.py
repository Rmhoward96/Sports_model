"""Final fit, model artifacts and the weekly quick gate for props-ML (spec §4-§5).

Final fit (default, ``--holdout-weeks 0``)
-----------------------------------------
Fits the gated pipeline (``assets/nfl/props_ml/pipeline.json``, written by
``train_props_ml_b.py``) on every completed game in the feature table and
writes ``data/props_ml/models/``:

* ``learned.joblib`` -- A (``learned.fit_models``) with the kept toggles and
  the MOST RECENT test season's tuned ``(decay, max_iter)`` from
  ``a_gate.json``;
* ``b_<market>.joblib`` -- B (``dist_models.fit_market``, unweighted,
  ``max_iter`` 150, in-role rows only) for every market served from ML
  (``source == "ml"``);
* ``calibration.json`` -- per market ``{calibrate, kind, map, n}``: for markets
  with ``calibrate`` the PIT map (Platt for anytime_td) fit on ALL OOF records
  (``data/props_ml/oof_b7__<tag>.parquet``) whose pre-calibration values are
  RECOMPUTED at the market's ``w_final`` via ``apply_pipeline`` (the stored
  values are at the per-season OOF weights); identity otherwise;
* ``props_ml_config.json`` -- ``pipeline.json`` + ``trained_through``,
  ``fit_upto``, ``git``, ``features`` (player/team fingerprints),
  ``created_at``, ``a_fit``, ``b_fit``, ``q_weight``, ``role_subsets``,
  ``market_max``, ``feature_columns`` and ``artifact_files`` -- everything
  serving needs without importing the training scripts. Written LAST.

``trained_through`` = the last (season, week) with a played label in the player
table; every model trains on rows strictly before ``trained_through + 1 week``
(``fit_upto``).

Quick gate (``--holdout-weeks N``; plan ruling 9)
------------------------------------------------
Holdout = the last N completed (season, week)s. A, B and the calibration maps
(and, where the blend is active, the blend weights) are fit ONLY on data
strictly before the first holdout week. The baseline (current sim) and the
pipeline (A via ``spec_hook`` + offline B / blend / calibration via
``apply_pipeline``, per-market sources from pipeline.json) are run on the
holdout weeks alone (``run_backtest`` with the schedules filtered to them,
``record_pmf``). Pass iff the pooled relative-RPS skill point estimate >= 0 AND
no market's RPS > 1.05 x the baseline's AND the coverage / share-fallback /
B-feature checks pass. ``quick_gate.json`` is written either way; exit 1 with
the reasons on failure. No model artifact is written in this mode.

Pure parts are unit tested (tests/scripts/test_fit_props_ml_final.py);
``main`` / ``_load_inputs`` (file IO) are not.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel.model.props_eval import population_from_baseline, rung_decision  # noqa: E402
from sportsmodel.model.props_ml.blend import BINARY_MARKET, choose_weight  # noqa: E402
from sportsmodel.model.props_ml.dist_models import (  # noqa: E402
    ROLE_SUBSETS,
    fit_market,
    in_role,
    predict_pmfs,
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

MODEL_DIR = tpm.DATA_DIR / "models"
OOF_PATH = tpm.DATA_DIR / f"oof_b7__{tpm.DEFAULT_TAG}.parquet"
CONFIG_FILE = "props_ml_config.json"
LEARNED_FILE = "learned.joblib"
CALIB_FILE = "calibration.json"
QUICK_GATE_FILE = "quick_gate.json"
FORMAT_VERSION = 1
QUICK_MIN_SKILL = 0.0
QUICK_MAX_RPS_RATIO = 1.05
KEY = ["season", "week", "player_id"]


# ---- pure helpers: weeks / config inputs --------------------------------------------------

def _before(df: pd.DataFrame, upto: tuple[int, int]) -> pd.Series:
    s, w = upto
    return (df["season"] < s) | ((df["season"] == s) & (df["week"] < w))


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


def next_week(sw: tuple[int, int]) -> tuple[int, int]:
    """``(season, week + 1)``: the exclusive ``upto`` covering ``sw``."""
    return (int(sw[0]), int(sw[1]) + 1)


def holdout_weeks(player_tbl: pd.DataFrame, n: int) -> list[tuple[int, int]]:
    """The last ``n`` completed (season, week)s; at least one earlier
    completed week must remain to train on."""
    weeks = labelled_weeks(player_tbl)
    if n < 1:
        raise ValueError(f"holdout needs n >= 1 weeks, got {n}")
    if len(weeks) <= n:
        raise ValueError(f"only {len(weeks)} completed weeks: nothing left to train on "
                         f"before a {n}-week holdout")
    return weeks[-n:]


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


def ml_markets(pipeline: Mapping) -> list[str]:
    """Markets served from ML (``source == "ml"``), in pipeline order."""
    return [m for m, v in pipeline["markets"].items() if v["source"] == "ml"]


def role_columns(player_tbl: pd.DataFrame, markets) -> list[str]:
    """Columns ``in_role`` reads for ``markets`` (sorted)."""
    cols = set()
    for m in markets:
        spec = ROLE_SUBSETS[m]
        if spec is None:
            continue
        cols.add(spec["col"])
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


# ---- pure helpers: OOF -> weights / calibration maps ---------------------------------------

def _has(x) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def _oof_inputs(oof: pd.DataFrame, market: str, before=None, in_role_only: bool = False
                ) -> tuple[list[dict], dict]:
    """(``apply_pipeline`` records with pmf = pmf_a, ``rec_key -> pmf_b``) of
    one market's OOF rows (optionally strictly before ``before`` / in role)."""
    g = oof[oof["market"] == market]
    if before is not None:
        g = g[_before(g, before)]
    if in_role_only:
        g = g[g["in_role"].astype(bool)]
    recs, b = [], {}
    for r in g.to_dict("records"):
        rec = {"season": int(r["season"]), "week": int(r["week"]), "home": r["home"],
               "player_id": str(r["player_id"]), "market": market,
               "pmf": np.asarray(r["pmf_a"], dtype=float), "actual": float(r["actual"])}
        recs.append(rec)
        if _has(r["pmf_b"]):
            b[tpb.rec_key(rec)] = np.asarray(r["pmf_b"], dtype=float)
    return recs, b


def serving_weights(markets: Mapping, oof: pd.DataFrame | None, before=None) -> dict[str, float]:
    """Weight on A per market. Final fit (``before`` None): pipeline.json's
    ``w_final``. Quick gate: a market whose blend is active (``w_final`` < 1)
    re-chooses its weight on in-role OOF rows with a B pmf strictly before
    ``before`` (none -> 1.0, pure A), so nothing is chosen on the holdout."""
    out = {}
    for m, v in markets.items():
        w = float(v["w_final"])
        if before is None or w >= 1.0:
            out[m] = w
            continue
        if oof is None:
            raise ValueError(f"{m}: the blend is active but no OOF records were given")
        recs, b = _oof_inputs(oof, m, before, in_role_only=True)
        rows = [{"pmf_a": r["pmf"], "pmf_b": b[tpb.rec_key(r)], "actual": r["actual"]}
                for r in recs if tpb.rec_key(r) in b]
        out[m] = float(choose_weight(rows, m)) if rows else 1.0
    return out


def _map_json(c, market: str):
    if market == BINARY_MARKET:
        return [float(c[0]), float(c[1])]
    return np.asarray(c, dtype=float).tolist()


def calibration_maps(markets: Mapping, oof: pd.DataFrame | None, weights: Mapping[str, float],
                     before=None) -> dict[str, dict]:
    """``{m: {calibrate, kind, map, n}}``. A calibrated market's map is fit
    (``train_props_ml_b.fit_calibration``; guards -> identity) on its OOF rows
    (strictly before ``before`` when given) after recomputing each row's
    pre-calibration pmf at ``weights[m]`` via ``apply_pipeline``; others get
    the identity map with n 0."""
    out = {}
    for m, v in markets.items():
        kind = "platt" if m == BINARY_MARKET else "pit"
        if not v["calibrate"]:
            out[m] = {"calibrate": False, "kind": kind,
                      "map": _map_json(tpb.identity_calibration(m), m), "n": 0}
            continue
        if oof is None:
            raise ValueError(f"{m}: calibrate is on but no OOF records were given "
                             f"(expected {OOF_PATH.name} from train_props_ml_b.py)")
        recs, b = _oof_inputs(oof, m, before)
        pre = tpb.apply_pipeline(recs, b, {m: float(weights[m])}) if recs else []
        c = tpb.fit_calibration(pre, m) if pre else tpb.identity_calibration(m)
        out[m] = {"calibrate": True, "kind": kind, "map": _map_json(c, m), "n": len(pre)}
    return out


def calib_for_apply(calibration: Mapping) -> dict | None:
    """``apply_pipeline``'s ``calib`` (None when no market calibrates)."""
    if not any(v["calibrate"] for v in calibration.values()):
        return None
    return {m: (tuple(v["map"]) if v["kind"] == "platt" else np.asarray(v["map"], dtype=float))
            for m, v in calibration.items()}


# ---- pure helpers: B fit / predict ----------------------------------------------------------

def fit_b_models(player_tbl: pd.DataFrame, cols: list[str], markets, upto: tuple[int, int],
                 log: Callable[[str], None]) -> dict:
    """``{m: MarketModel}``: ``fit_market`` on in-role rows strictly before
    ``upto``, unweighted, ``max_iter`` 150 (the ladder's B settings)."""
    out = {}
    for m in markets:
        t0 = time.time()
        out[m] = fit_market(player_tbl, m, cols, upto=upto, test_season=int(upto[0]),
                            decay=tpb.B_DECAY, max_iter=tpb.B_MAX_ITER)
        log(f"stage=fit-b market={m} upto={upto} {time.time() - t0:.1f}s")
    return out


def predict_b(models: Mapping, player_tbl: pd.DataFrame, recs, market_max: Mapping
              ) -> tuple[dict, set]:
    """``(rec_key -> B pmf, out-of-role keys)`` for records of markets in
    ``models`` from their (season, week, player_id) feature rows; a record
    whose feature row is out of role is in the set (B := A); one without a
    feature row is in neither (missing)."""
    feats = player_tbl.drop_duplicates(KEY)
    by_m = tpb._by_market(recs)
    out, oor = {}, set()
    for m, model in models.items():
        if not by_m.get(m):
            continue
        keys = pd.DataFrame([{k: r[k] for k in KEY} for r in by_m[m]])
        rows = keys.merge(feats, on=KEY, how="inner")
        role = in_role(rows, m).to_numpy()
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
                 identity: Mapping, created_at: str) -> dict:
    """``props_ml_config.json``: pipeline.json + everything serving needs."""
    return {**pipeline, "format_version": FORMAT_VERSION,
            "trained_through": [int(trained_through[0]), int(trained_through[1])],
            "fit_upto": list(next_week(trained_through)),
            "git": identity.get("git_head"),
            "features": {"player": identity.get("player_features"),
                         "team": identity.get("team_features")},
            "created_at": created_at, "a_fit": dict(a_fit),
            "b_fit": {"decay": tpb.B_DECAY, "max_iter": tpb.B_MAX_ITER},
            "q_weight": tpm.Q_WEIGHT, "role_subsets": ROLE_SUBSETS,
            "market_max": {**{m: int(k) for m, k in market_max.items()}, BINARY_MARKET: 1},
            "feature_columns": {"learned": _learned_columns(learned_models),
                                "b": {m: list(model.cols) for m, model in b_models.items()},
                                "role": list(role_cols)},
            "artifact_files": {"learned": LEARNED_FILE, "calibration": CALIB_FILE,
                               "b": {m: f"b_{m}.joblib" for m in b_models}}}


def required_columns(config: Mapping) -> dict[str, list[str]]:
    """``{"player": [...], "team": [...]}``: every column the saved models read
    (fitted columns, role columns, join keys)."""
    fc = config["feature_columns"]
    player = {"season", "week", "player_id", "team", *fc["role"]}
    team = {"season", "week", "team"}
    lf = fc["learned"]["fitted"]
    for n in ("targets", "carries"):
        player |= set(lf[n] or [])
    for cols in (lf["eff"] or {}).values():
        player |= set(cols or [])
    for n in ("team_pass", "team_rush"):
        team |= set(lf[n] or [])
    for cols in fc["b"].values():
        player |= set(cols)
    return {"player": sorted(player), "team": sorted(team)}


def verify_feature_columns(config: Mapping, player_tbl: pd.DataFrame | None = None,
                           team_tbl: pd.DataFrame | None = None) -> None:
    """ValueError naming every required column absent from a given table."""
    req, bad = required_columns(config), []
    for name, tbl in (("player", player_tbl), ("team", team_tbl)):
        if tbl is not None:
            miss = [c for c in req[name] if c not in tbl.columns]
            if miss:
                bad.append(f"{name} table is missing {miss}")
    if bad:
        raise ValueError("props-ML artifacts do not match the feature tables: " + "; ".join(bad))


class Artifacts(NamedTuple):
    config: dict
    learned: object
    b_models: dict
    calibration: dict


def save_artifacts(out_dir: Path, learned_models, b_models: Mapping, calibration: Mapping,
                   config: Mapping) -> None:
    """Write the model files; stale artifacts (config first) are removed so a
    crash never leaves a config pointing at old models; config is written last.
    Other files (``quick_gate.json``) are kept."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in [out / CONFIG_FILE, out / LEARNED_FILE, out / CALIB_FILE, *out.glob("b_*.joblib")]:
        p.unlink(missing_ok=True)
    joblib.dump(learned_models, out / LEARNED_FILE)
    for m, model in b_models.items():
        joblib.dump(model, out / f"b_{m}.joblib")
    (out / CALIB_FILE).write_text(tpm.gate_json(dict(calibration)))
    (out / CONFIG_FILE).write_text(tpm.gate_json(dict(config)))


def load_artifacts(out_dir: Path, player_tbl: pd.DataFrame | None = None,
                   team_tbl: pd.DataFrame | None = None) -> Artifacts:
    """Load what ``save_artifacts`` wrote; with tables, verify the config's
    feature columns exist in them (``verify_feature_columns``)."""
    out = Path(out_dir)
    config = json.loads((out / CONFIG_FILE).read_text())
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"props_ml_config.json format_version {config.get('format_version')} "
                         f"!= {FORMAT_VERSION}")
    if player_tbl is not None or team_tbl is not None:
        verify_feature_columns(config, player_tbl, team_tbl)
    files = config["artifact_files"]
    return Artifacts(config=config, learned=joblib.load(out / files["learned"]),
                     b_models={m: joblib.load(out / f) for m, f in files["b"].items()},
                     calibration=json.loads((out / files["calibration"]).read_text()))


# ---- quick-gate decision -------------------------------------------------------------------

def quick_gate_decision(per_market: Mapping, skill: float | None, check_reasons=()) -> dict:
    """Pass iff no check failed, some market was scored, the pooled skill
    point estimate >= 0 and no market's RPS > 1.05 x the baseline's."""
    reasons = list(check_reasons)
    if not reasons:
        if not per_market:
            reasons.append("no scored records")
        if skill is not None and skill < QUICK_MIN_SKILL:
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
                  a_gate: Mapping, oof: pd.DataFrame | None, identity: Mapping,
                  out_dir: Path = MODEL_DIR, log: Callable[[str], None] = print,
                  created_at: str | None = None) -> dict:
    """Fit A, B and the calibration maps on every completed week; write the
    artifacts; return the config."""
    kept, a_fit = kept_toggles(pipeline, a_gate), a_fit_params(a_gate)
    tt = trained_through(player_tbl)
    upto = next_week(tt)
    weights = serving_weights(pipeline["markets"], oof)
    log(f"stage=calibration maps (OOF rows: {0 if oof is None else len(oof)})")
    calibration = calibration_maps(pipeline["markets"], oof, weights)
    log(f"stage=fit-a trained_through={tt} upto={upto} toggles={sorted(kept)} {a_fit}")
    t0 = time.time()
    lm = learned.fit_models(player_tbl, team_tbl, kept, upto=upto, test_season=int(upto[0]),
                            decay=a_fit["decay"], max_iter=a_fit["max_iter"])
    log(f"stage=fit-a done {time.time() - t0:.1f}s")
    served = ml_markets(pipeline)
    b_models = fit_b_models(player_tbl, learned.feature_columns(player_tbl, kept), served, upto,
                            log)
    config = build_config(pipeline, learned_models=lm, b_models=b_models, a_fit=a_fit,
                          trained_through=tt, market_max=bsn.MARKET_MAX,
                          role_cols=role_columns(player_tbl, served), identity=identity,
                          created_at=created_at or _now())
    save_artifacts(out_dir, lm, b_models, calibration, config)
    log(f"wrote {out_dir}: learned + B {served} + calibration + config "
        f"(trained_through={list(tt)})")
    return config


def run_quick_gate(n_weeks: int, *, bsn, player_tbl: pd.DataFrame, team_tbl: pd.DataFrame,
                   pipeline: Mapping, a_gate: Mapping, oof: pd.DataFrame | None,
                   identity: Mapping, out_dir: Path = MODEL_DIR, n_sims: int = 1000,
                   log: Callable[[str], None] = print, created_at: str | None = None) -> dict:
    """Score the pipeline fit without the last ``n_weeks`` completed weeks on
    them vs the baseline; write ``quick_gate.json``; return the gate dict."""
    t0 = time.time()
    kept, a_fit = kept_toggles(pipeline, a_gate), a_fit_params(a_gate)
    weeks = holdout_weeks(player_tbl, n_weeks)
    start = weeks[0]
    seasons = sorted({s for s, _ in weeks})
    sources_map = {m: v["source"] for m, v in pipeline["markets"].items()}
    log(f"stage=quick-gate holdout={weeks} fit_upto={start} n_sims={n_sims}")

    weights = serving_weights(pipeline["markets"], oof, before=start)
    calibration = calibration_maps(pipeline["markets"], oof, weights, before=start)
    log(f"stage=fit-a upto={start} toggles={sorted(kept)} {a_fit}")
    lm = learned.fit_models(player_tbl, team_tbl, kept, upto=start, test_season=int(start[0]),
                            decay=a_fit["decay"], max_iter=a_fit["max_iter"])
    served = ml_markets(pipeline)
    b_models = fit_b_models(player_tbl, learned.feature_columns(player_tbl, kept), served, start,
                            log)

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
    n_pop = {m: sum(1 for k in population if k[3] == m) for m in pipeline["markets"]}
    d = {"skill": None, "lo": None, "hi": None, "per_market": {}}
    if not checks:
        base_pop = [r for r in base_recs if tpb.rec_key(r) in population]
        pop_a = [r for r in runs["pipeline"] if tpb.rec_key(r) in population]
        b_pmfs, oor = predict_b(b_models, player_tbl, pop_a, bsn.MARKET_MAX)
        checks += b_missing_reasons(pop_a, b_pmfs, oor, served)
        b_eff = {**b_pmfs, **{tpb.rec_key(r): r["pmf"] for r in pop_a if tpb.rec_key(r) in oor}}
        ml = tpb.apply_pipeline(pop_a, b_eff, weights, calib_for_apply(calibration))
        try:
            df = tpm.checked_paired_frame(base_pop, tpb.composite(base_pop, ml, sources_map),
                                          population, base_pop, "quick")
            rd = rung_decision(df)
            d = {k: rd[k] for k in ("skill", "lo", "hi", "per_market")}
        except RuntimeError as exc:
            checks.append(str(exc))
    decision = quick_gate_decision(d["per_market"], d["skill"], checks)
    gate = {"mode": "quick_gate", "created_at": created_at or _now(),
            "holdout_weeks": [list(w) for w in weeks], "fit_upto": list(start),
            "n_sims": n_sims, "n_games": len(base_games), "n_population": n_pop,
            "sources": sources_map, "weights": weights,
            "calibration": {m: {"calibrate": v["calibrate"], "n": v["n"]}
                            for m, v in calibration.items()},
            "a_fit": a_fit, "kept_a_toggles": pipeline["kept_a_toggles"],
            **d, "thresholds": {"min_skill": QUICK_MIN_SKILL, "max_rps_ratio": QUICK_MAX_RPS_RATIO},
            **decision, "identity": dict(identity), "elapsed_s": time.time() - t0}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / QUICK_GATE_FILE).write_text(tpm.gate_json(gate))
    skill = "n/a" if d["skill"] is None else f"{d['skill']:+.4f}"
    log(f"QUICK GATE {'PASS' if decision['pass'] else 'FAIL'}: skill {skill}"
        + ("" if decision["pass"] else f"; reasons: {decision['reasons']}"))
    log(f"wrote {out / QUICK_GATE_FILE}")
    return gate


# ---- IO ----------------------------------------------------------------------------------

def _load_inputs(args) -> dict:
    """Tables, pipeline / A-gate json, OOF records, identity, backtest module."""
    for p in (tpm.PLAYER_PATH, tpm.TEAM_PATH, tpb.A_GATE_PATH, args.pipeline):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}: build the features / run the A and B gates first")
    oof = pd.read_parquet(args.oof) if Path(args.oof).exists() else None
    return {"bsn": tpm._load_backtest(), "player_tbl": pd.read_parquet(tpm.PLAYER_PATH),
            "team_tbl": pd.read_parquet(tpm.TEAM_PATH),
            "pipeline": json.loads(Path(args.pipeline).read_text()),
            "a_gate": json.loads(tpb.A_GATE_PATH.read_text()), "oof": oof,
            "identity": {"player_features": tpm.file_fingerprint(tpm.PLAYER_PATH),
                         "team_features": tpm.file_fingerprint(tpm.TEAM_PATH),
                         "git_head": tpm.git_head()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--holdout-weeks", type=int, default=0,
                    help="quick gate on the last N completed weeks (0 = final fit)")
    ap.add_argument("--n-sims", type=int, default=1000, help="sims per game (quick gate)")
    ap.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--pipeline", type=Path, default=tpb.PIPELINE_PATH)
    ap.add_argument("--oof", type=Path, default=OOF_PATH)
    args = ap.parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[+{(time.time() - t0) / 60:7.1f} min] {msg}", flush=True)

    inputs = _load_inputs(args)
    if args.holdout_weeks > 0:
        gate = run_quick_gate(args.holdout_weeks, **inputs, out_dir=args.out_dir,
                              n_sims=args.n_sims, log=log)
        return 0 if gate["pass"] else 1
    run_final_fit(**inputs, out_dir=args.out_dir, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
