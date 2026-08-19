"""Game-time weather via Open-Meteo (free, no key). Best-effort with graceful fallback.

Returns the forecast high for the game date as a temperature proxy (we don't store
exact first-pitch times yet). Any failure returns None so prediction never breaks on
a weather hiccup. Wind-direction handling is future work.
"""
from __future__ import annotations

import httpx

_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_game_temp(lat: float, lon: float, game_date: str) -> float | None:
    """Forecast high (°F) at a park for game_date (YYYY-MM-DD), or None on any failure."""
    try:
        with httpx.Client(timeout=8) as client:
            resp = client.get(_URL, params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
                "start_date": str(game_date), "end_date": str(game_date),
            })
            resp.raise_for_status()
            temps = resp.json().get("daily", {}).get("temperature_2m_max", [])
            return float(temps[0]) if temps else None
    except Exception:
        return None
