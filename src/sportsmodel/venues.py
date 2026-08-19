"""Ballpark locations + roof status, keyed by Statcast team abbreviation.

Used to fetch game-time weather (Open-Meteo) for the home park. `outdoor=False`
means a fixed dome or a park that plays climate-controlled by default (no weather
effect applied). Retractable-roof parks that are usually open in summer are marked
outdoor=True. Coordinates are approximate stadium centers — fine for a forecast.

Approximate and best-effort: roof status is a judgment call and the wind-direction
refinement (needs park orientation) is future work. Note ATH plays in West
Sacramento (Sutter Health Park) as of 2025-26.
"""
from __future__ import annotations

# abbrev: (latitude, longitude, outdoor)
PARKS: dict[str, tuple[float, float, bool]] = {
    "LAA": (33.80, -117.88, True),  "AZ": (33.45, -112.07, True),
    "BAL": (39.28, -76.62, True),   "BOS": (42.35, -71.10, True),
    "CHC": (41.95, -87.66, True),   "CIN": (39.10, -84.51, True),
    "CLE": (41.50, -81.69, True),   "COL": (39.76, -104.99, True),
    "DET": (42.34, -83.05, True),   "HOU": (29.76, -95.36, True),
    "KC": (39.05, -94.48, True),    "LAD": (34.07, -118.24, True),
    "WSH": (38.87, -77.01, True),   "NYM": (40.76, -73.85, True),
    "ATH": (38.58, -121.51, True),  "PIT": (40.45, -80.01, True),
    "SD": (32.71, -117.16, True),   "SEA": (47.59, -122.33, True),
    "SF": (37.78, -122.39, True),   "STL": (38.62, -90.19, True),
    "TB": (27.77, -82.65, False),   "TEX": (32.75, -97.08, True),
    "TOR": (43.64, -79.39, True),   "MIN": (44.98, -93.28, True),
    "PHI": (39.91, -75.17, True),   "ATL": (33.89, -84.47, True),
    "CWS": (41.83, -87.63, True),   "MIA": (25.78, -80.22, False),
    "NYY": (40.83, -73.93, True),   "MIL": (43.03, -87.97, True),
}


def park(abbrev: str | None) -> tuple[float, float, bool] | None:
    return PARKS.get(abbrev) if abbrev else None
