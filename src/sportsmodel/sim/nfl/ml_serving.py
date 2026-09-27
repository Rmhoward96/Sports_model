"""Live props-ML serving: A -> sim -> B -> blend -> calibrate -> map.

The gated pipeline (``scripts/fit_props_ml_final.py`` artifacts in
``data/props_ml/models/``) applied to one upcoming game:

1. ``build_ml_spec`` -- rung A: ``learned.apply_to_spec`` with the saved
   ``LearnedModels`` (team volume, shares, efficiency) on the game's pre-kickoff
   feature rows; the caller simulates it (``kernel.simulate_game``) -> ``sims_ml``.
2. ``ml_player_dists`` -- per active player and ``source == "ml"`` market:
   A = the sim's pmf (``aggregate.nfl_player_prop_dists``); B =
   ``dist_models.predict_pmfs`` only for markets that read B (``b_markets``:
   ``w_final < 1``) and only for in-role players (``dist_models.in_role``), else
   B := A (a player with no feature row keeps A too); blend at ``w_final`` and
   calibrate with calibration.json's maps (``pipeline.blend_calibrate`` -- the
   step the offline ladder / quick gate apply, so serving matches what was
   gated); then ``quantile_map.map_game_sims`` rewrites ``sims_ml`` so its
   player draws follow the final pmfs (parlays / team totals read the same
   numbers), and the returned dists are the mapped sims' marginals.
   ``source == "baseline"`` markets are omitted -- the caller serves the
   current sim for those. With ``gate`` (the caller's
   ``props_eval.gate_population`` of the CURRENT sim's dist means), only the
   gated ``(player_id, market)``s get targets / are mapped / are returned:
   the ML pipeline was gated on that population only.

Leakage: nothing here reads data by itself; only the pre-kickoff feature rows
the caller passes are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sportsmodel.model.props_ml import artifacts as art_mod
from sportsmodel.model.props_ml.artifacts import Artifacts, b_markets, calib_for_apply
from sportsmodel.model.props_ml.dist_models import in_role, predict_pmfs
from sportsmodel.model.props_ml.pipeline import blend_calibrate
from sportsmodel.model.props_ml.quantile_map import map_game_sims
from sportsmodel.sim.nfl import learned
from sportsmodel.sim.nfl.aggregate import nfl_player_prop_dists
from sportsmodel.sim.nfl.spec import NflGameSims, NflGameSpec


def load_artifacts(model_dir: Path, player_tbl: pd.DataFrame | None = None,
                   team_tbl: pd.DataFrame | None = None) -> Artifacts | None:
    """``artifacts.load_artifacts`` for serving: None (and a printed reason)
    instead of raising when the artifacts are missing, unloadable or
    incompatible (format version, role subsets, B model set, scikit-learn
    version, or -- with tables -- a required feature column absent)."""
    d = Path(model_dir)
    if not (d / art_mod.CONFIG_FILE).is_file():
        print(f"props-ML: no {art_mod.CONFIG_FILE} in {d}; ML serving disabled", flush=True)
        return None
    try:
        return art_mod.load_artifacts(d, player_tbl, team_tbl)
    except Exception as exc:  # noqa: BLE001 -- any failure means: serve without ML
        print(f"props-ML: artifacts in {d} unloadable or incompatible "
              f"({type(exc).__name__}: {exc}); ML serving disabled", flush=True)
        return None


def _questionable(player_rows: pd.DataFrame) -> set[str]:
    return set(player_rows.loc[player_rows["st_questionable"] == 1, "player_id"].astype(str))


def build_ml_spec(spec: NflGameSpec, artifacts: Artifacts, feature_rows: pd.DataFrame,
                  team_rows: pd.DataFrame) -> NflGameSpec:
    """Rung A on ``spec``: ``learned.apply_to_spec`` with the saved models, the
    game's (both teams') upcoming feature rows, the players flagged
    ``st_questionable == 1`` and the config's ``q_weight``."""
    return learned.apply_to_spec(spec, artifacts.learned, feature_rows, team_rows,
                                 questionable=_questionable(feature_rows),
                                 q_weight=float(artifacts.config["q_weight"]))


def _b_pmfs(artifacts: Artifacts, rows: pd.DataFrame, a: dict, markets, market_max
            ) -> dict[tuple[str, str], np.ndarray]:
    """``(player_id, market) -> effective B pmf`` for the B markets: the model's
    pmf for in-role players, the A pmf (B := A) for out-of-role ones; players
    without a feature row are absent (the caller keeps A)."""
    out: dict[tuple[str, str], np.ndarray] = {}
    for m in markets:
        role = in_role(rows, m).to_numpy()
        for pid in rows["player_id"][~role]:
            if m in a.get(pid, {}):
                out[(pid, m)] = np.asarray(a[pid][m]["pmf"], dtype=float)
        inr = rows[role]
        kmax = int(market_max.get(m, 1))   # config market_max has anytime_td: 1
        for pid, pmf in zip(inr["player_id"], predict_pmfs(artifacts.b_models[m], inr, kmax)):
            out[(pid, m)] = pmf
    return out


def serving_targets(a: dict, b: dict, weights: dict, calib: dict | None, markets, players,
                    gate: set[tuple[str, str]] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """The final (pre quantile-map) pmfs: ``{player_id: {market: pmf}}``. PURE.

    ``a``: ``{player_id: {market: {"pmf": [...]}}}`` (the ML sims' marginals);
    ``b``: ``{(player_id, market): effective B pmf}`` (absent -> keep A);
    ``weights``: market -> ``w_final``; ``calib``: ``calib_for_apply``'s maps
    or None. Each (player, market) goes through ``pipeline.blend_calibrate``
    -- exactly what ``train_props_ml_b.apply_pipeline`` applies offline (the
    ladder / quick gate), so serving matches what was gated (parity-tested).
    Only ``gate`` keys when a gate is given; built in the given
    (players, markets) order."""
    targets: dict[str, dict[str, np.ndarray]] = {}
    for pid in players:
        for m in markets:
            if m not in a.get(pid, {}) or (gate is not None and (pid, m) not in gate):
                continue
            _, post = blend_calibrate(a[pid][m]["pmf"], b.get((pid, m)), weights[m], m,
                                      None if calib is None else calib[m])
            targets.setdefault(pid, {})[m] = post
    return targets


def ml_player_dists(spec: NflGameSpec, sims_ml: NflGameSims, feature_rows: pd.DataFrame,
                    team_rows: pd.DataFrame, artifacts: Artifacts,
                    rng: np.random.Generator,
                    gate: set[tuple[str, str]] | None = None) -> dict[str, dict[str, dict]]:
    """``{player_id: {market: {"kind": "pmf", "pmf": [...], "mean": float}}}``
    for every active player of ``spec`` (in ``sims_ml``) and every
    ``source == "ml"`` market (see the module docstring) -- restricted to the
    ``(player_id, market)`` keys in ``gate`` when given (live serving always
    passes it; None = no restriction). Ungated player-markets get no target,
    so their sims draws are left unmapped.

    ``sims_ml`` (the simulation of ``build_ml_spec(spec, ...)``) is UPDATED IN
    PLACE: its ``player_stats`` is replaced by the quantile-mapped copy (the
    input arrays are not mutated; unmapped stats and scores keep their array
    objects), so the returned dists equal its marginals. Targets are built
    in sorted (player_id, market) order, so a fixed ``rng`` state gives a
    deterministic result. ``team_rows`` is unused here (A already went into
    ``build_ml_spec``); it is accepted to keep the call symmetric.
    """
    cfg = artifacts.config
    ml = sorted(m for m, v in cfg["markets"].items() if v["source"] == "ml")
    if not ml:
        return {}
    market_max = {m: int(k) for m, k in cfg["market_max"].items()}
    active = sorted({p.player_id for p in (*spec.home_players, *spec.away_players)}
                    & set(sims_ml.player_stats))
    a = {pid: d for pid, d in nfl_player_prop_dists(sims_ml, market_max).items() if pid in active}
    if gate is not None:   # B is only needed for gated players
        active = [pid for pid in active if any((pid, m) in gate for m in ml)]
    rows = feature_rows[feature_rows["player_id"].isin(active)].drop_duplicates("player_id")
    b = _b_pmfs(artifacts, rows, a, [m for m in b_markets(cfg) if m in ml], market_max)
    calib = calib_for_apply(artifacts.calibration)
    weights = {m: float(cfg["markets"][m]["w_final"]) for m in ml}
    targets = serving_targets(a, b, weights, calib, ml, active, gate)
    if not targets:
        return {}
    mapped = map_game_sims(sims_ml, targets, rng)
    sims_ml.player_stats = mapped.player_stats
    final = nfl_player_prop_dists(sims_ml, market_max)
    return {pid: {m: final[pid][m] for m in ms} for pid, ms in targets.items()}
