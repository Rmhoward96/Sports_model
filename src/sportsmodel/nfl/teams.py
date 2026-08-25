TEAMS = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
})

# Historical/alternate codes → current franchise code.
_ALIASES = {
    "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "SDG": "LAC",
    "STL": "LA", "JAC": "JAX", "LVR": "LV", "KAN": "KC", "GNB": "GB",
    "NWE": "NE", "NOR": "NO", "TAM": "TB", "SFO": "SF",
}


def normalize_team(abbr: str) -> str:
    a = (abbr or "").strip().upper()
    a = _ALIASES.get(a, a)
    if a not in TEAMS:
        raise ValueError(f"unknown NFL team abbreviation: {abbr!r}")
    return a
