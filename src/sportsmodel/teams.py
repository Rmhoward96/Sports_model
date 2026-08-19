"""MLBAM team id -> Statcast team abbreviation crosswalk.

The schedule (MLB StatsAPI) uses numeric MLBAM team ids; Statcast uses string
abbreviations. Validated against the distinct home_team values in the 2026 Statcast
data (30 teams). Note the fiddly ones: ATH (Athletics), AZ, CWS, KC, SD, SF, TB, WSH.
"""
from __future__ import annotations

MLBAM_TO_STATCAST: dict[int, str] = {
    108: "LAA", 109: "AZ", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
    120: "WSH", 121: "NYM", 133: "ATH", 134: "PIT", 135: "SD", 136: "SEA",
    137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def statcast_abbrev(mlbam_team_id) -> str | None:
    if mlbam_team_id is None:
        return None
    return MLBAM_TO_STATCAST.get(int(mlbam_team_id))
