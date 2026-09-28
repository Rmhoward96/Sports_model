"""Props-ML model artifacts: file names, load and verify (props ML, Props-2).

``scripts/fit_props_ml_final.py`` writes ``data/props_ml/models/``
(``save_artifacts``): ``learned.joblib`` (rung-A ``LearnedModels``),
``b_<market>.joblib`` (B ``MarketModel`` for every market that READS B:
``source == "ml"`` and ``w_final < 1``), ``calibration.json`` (the committed
serving maps) and ``props_ml_config.json`` (pipeline + everything serving
needs). The script and live serving (``sim.nfl.ml_serving``) both load and
verify them through this module, so the checks live in one place.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Mapping, NamedTuple

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from sportsmodel.model.props_ml.dist_models import ROLE_SUBSETS

CONFIG_FILE = "props_ml_config.json"
LEARNED_FILE = "learned.joblib"
CALIB_FILE = "calibration.json"
FORMAT_VERSION = 1


class Artifacts(NamedTuple):
    config: dict
    learned: object
    b_models: dict
    calibration: dict


def _jsonable(x):
    """``x`` as it reads back from JSON (tuples -> lists)."""
    return json.loads(json.dumps(x))


def _is_artifact(name: str) -> bool:
    """A file ``save_artifacts`` owns (replaced on every save)."""
    return name in (CONFIG_FILE, LEARNED_FILE, CALIB_FILE) or (
        name.startswith("b_") and name.endswith(".joblib"))


def b_markets(pipeline: Mapping) -> list[str]:
    """Markets whose served pmf READS B: ``source == "ml"`` and ``w_final < 1``."""
    return [m for m, v in pipeline["markets"].items()
            if v["source"] == "ml" and float(v["w_final"]) < 1.0]


def calib_for_apply(calibration: Mapping) -> dict | None:
    """``{market: map}`` from calibration.json -- Platt ``(a, b)`` or PIT knots
    ``(2, K)`` -- or None when no market calibrates (``apply_pipeline`` /
    ``pipeline.blend_calibrate`` then skip calibration)."""
    ms = calibration["markets"]
    if not any(v["calibrate"] for v in ms.values()):
        return None
    return {m: (tuple(v["map"]) if v["kind"] == "platt" else np.asarray(v["map"], dtype=float))
            for m, v in ms.items()}


def required_columns(config: Mapping) -> dict[str, list[str]]:
    """``{"player": [...], "team": [...]}``: every column the saved models and
    the A hook read (fitted columns, role columns, ``st_questionable``, keys)."""
    fc = config["feature_columns"]
    player = {"season", "week", "player_id", "team", "opponent", "st_questionable", *fc["role"]}
    team = {"season", "week", "team", "opponent"}
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


def _joblib_load(path: Path):
    """``joblib.load`` raising ValueError naming the file when it cannot be
    unpickled or was pickled under another scikit-learn version (its
    ``InconsistentVersionWarning``). A missing file stays ``OSError``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except OSError:
            raise
        except Exception as exc:
            raise ValueError(f"{Path(path).name}: cannot load ({type(exc).__name__}: {exc})"
                             ) from exc


def load_artifacts(out_dir: Path, player_tbl: pd.DataFrame | None = None,
                   team_tbl: pd.DataFrame | None = None) -> Artifacts:
    """Load what ``save_artifacts`` wrote. ValueError on a format mismatch, a
    config ``role_subsets`` differing from ``dist_models.ROLE_SUBSETS``, a B
    model set differing from the markets that read B, a model pickled under
    another scikit-learn version or unreadable, or (with tables) a required
    feature column missing (``verify_feature_columns``). Missing files raise
    ``OSError``."""
    out = Path(out_dir)
    config = json.loads((out / CONFIG_FILE).read_text())
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"props_ml_config.json format_version {config.get('format_version')} "
                         f"!= {FORMAT_VERSION}")
    if config.get("role_subsets") != _jsonable(ROLE_SUBSETS):
        raise ValueError("props_ml_config.json role_subsets differ from dist_models.ROLE_SUBSETS: "
                         "refit the models")
    files = config["artifact_files"]
    if set(files["b"]) != set(b_markets(config)):
        raise ValueError(f"B models {sorted(files['b'])} != markets reading B "
                         f"{sorted(b_markets(config))}")
    if player_tbl is not None or team_tbl is not None:
        verify_feature_columns(config, player_tbl, team_tbl)
    return Artifacts(config=config, learned=_joblib_load(out / files["learned"]),
                     b_models={m: _joblib_load(out / f) for m, f in files["b"].items()},
                     calibration=json.loads((out / files["calibration"]).read_text()))
