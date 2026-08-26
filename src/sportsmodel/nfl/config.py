"""Loaders that read the committed P1/P2/P3 calibration JSONs into config dataclasses.

Each loader reads `assets/nfl/{rating,gameline,props}.json` (committed, fitted
values) rather than letting the producer fall back to the dataclass defaults
baked into elo.py/gameline.py/props.py, which are illustrative-only.
"""
from __future__ import annotations
import json, pathlib
from .elo import EloConfig
from .ratings import BlendConfig
from .gameline import GameLineConfig
from .shrink import ShrinkParams
from .props import PropConfig

_ASSETS = pathlib.Path(__file__).resolve().parents[3] / "assets" / "nfl"

def _load(name: str) -> dict:
    return json.loads((_ASSETS / name).read_text())

def load_rating() -> tuple[EloConfig, BlendConfig]:
    j = _load("rating.json")
    return (EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"], base=j["base"]),
            BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"]))

def load_gameline() -> GameLineConfig:
    j = _load("gameline.json")
    return GameLineConfig(sigma_margin=j["sigma_margin"], sigma_total=j["sigma_total"],
                          offset=j["offset"], total_max=j["total_max"],
                          w_margin=ShrinkParams(**j["w_margin"]), w_total=ShrinkParams(**j["w_total"]))

def load_props() -> PropConfig:
    j = _load("props.json")
    return PropConfig(sigma=j["sigma"], nb_var_mult=j["nb_var_mult"], mean_mult=j["mean_mult"])
