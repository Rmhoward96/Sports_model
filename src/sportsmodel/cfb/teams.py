"""CFB team registry.

FBS opponents are tracked individually by ESPN team id. Every non-FBS opponent
(FCS, D2, D3, etc.) collapses to a single pseudo-team, FCS, since the rating
engine doesn't need to distinguish among them. The FBS id set is snapshotted
once from ESPN into assets/cfb/fbs_teams.json (see that file for provenance).
"""
from __future__ import annotations

import json
import unicodedata

from .. import config

FCS = "FCS"

_FBS_PATH = config.PROJECT_ROOT / "assets" / "cfb" / "fbs_teams.json"
_cache: set[str] | None = None
_names_cache: dict[str, str] | None = None


def load_fbs_ids() -> set[str]:
    """ESPN team ids (as strings) for current FBS programs; {} if the asset is absent."""
    global _cache
    if _cache is None:
        _cache = set(json.loads(_FBS_PATH.read_text())) if _FBS_PATH.exists() else set()
    return _cache


def _load_names() -> dict[str, str]:
    """{espn_id: displayName} from fbs_teams.json ({} if absent)."""
    global _names_cache
    if _names_cache is None:
        _names_cache = json.loads(_FBS_PATH.read_text()) if _FBS_PATH.exists() else {}
    return _names_cache


def normalize(espn_team_id) -> str:
    """Coerce to str; pass through FBS ids, collapse everything else to FCS."""
    team_id = str(espn_team_id)
    return team_id if team_id in load_fbs_ids() else FCS


def _norm(s: str) -> str:
    """Lowercase, strip accents + punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    kept = "".join(c if (c.isalnum() or c == " ") else (" " if c not in "&'.,()-" else "") for c in s)
    return " ".join(kept.split())


# CFBD "school" name (normalized) -> ESPN id, for cases the unique-prefix match can't
# resolve: (a) a bare school name that is a prefix of longer siblings (Texas -> Longhorns,
# not Texas A&M/State/Tech), and (b) CFBD names that differ from our ESPN displayName.
_CFBD_ALIASES = {
    # ambiguous bare names -> the flagship program
    "arizona": "12", "colorado": "38", "florida": "57", "georgia": "61",
    "iowa": "2294", "kansas": "2305", "louisiana": "309", "miami": "2390",
    "michigan": "130", "new mexico": "167", "ohio": "195", "oklahoma": "201",
    "oregon": "2483", "texas": "251", "utah": "254", "washington": "264",
    # CFBD spelling differs from ESPN displayName
    "connecticut": "41", "appalachian state": "2026",
    "southern mississippi": "2572", "louisiana monroe": "2433",
    "umass": "113", "sam houston state": "2534",
}


def cfbd_to_espn(name: str) -> str | None:
    """Map a collegefootballdata.com school name to our ESPN team id, or None.

    Alias table first (ambiguous prefixes + spelling differences), then a UNIQUE
    normalized prefix match against the FBS displayNames. Multiple matches -> None,
    so an ambiguous name is never silently mis-assigned.
    """
    n = _norm(name)
    if not n:
        return None
    if n in _CFBD_ALIASES:
        return _CFBD_ALIASES[n]
    matches = [eid for eid, disp in _load_names().items()
               if (dn := _norm(disp)) == n or dn.startswith(n + " ")]
    return matches[0] if len(matches) == 1 else None
