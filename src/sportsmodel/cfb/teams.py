"""CFB team registry.

FBS opponents are tracked individually by ESPN team id. Every non-FBS opponent
(FCS, D2, D3, etc.) collapses to a single pseudo-team, FCS, since the rating
engine doesn't need to distinguish among them. The FBS id set is snapshotted
once from ESPN into assets/cfb/fbs_teams.json (see that file for provenance).
"""
from __future__ import annotations

import json

from .. import config

FCS = "FCS"

_FBS_PATH = config.PROJECT_ROOT / "assets" / "cfb" / "fbs_teams.json"
_cache: set[str] | None = None


def load_fbs_ids() -> set[str]:
    """ESPN team ids (as strings) for current FBS programs; {} if the asset is absent."""
    global _cache
    if _cache is None:
        _cache = set(json.loads(_FBS_PATH.read_text())) if _FBS_PATH.exists() else set()
    return _cache


def normalize(espn_team_id) -> str:
    """Coerce to str; pass through FBS ids, collapse everything else to FCS."""
    team_id = str(espn_team_id)
    return team_id if team_id in load_fbs_ids() else FCS
