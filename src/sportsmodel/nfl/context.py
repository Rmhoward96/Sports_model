"""Per team-game context features (rest, travel, time zones, roof/surface,
weather) and market features (spread, total, implied team total) for the
props-ML feature table. PURE except `load_stadiums` (reads a committed asset).

Leakage: every value here is known before kickoff (schedule, rest days, roof,
the line). Historical weather is the recorded game weather (live will use a
forecast -- a stated train/serve difference in the spec).
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from sportsmodel.nfl.teams import normalize_team

_STADIUMS = Path(__file__).resolve().parents[3] / "assets" / "nfl" / "stadiums.json"
_INDOOR_TEMP_F, _INDOOR_WIND_MPH = 70.0, 0.0
_PACIFIC = {"America/Los_Angeles"}


def load_stadiums() -> dict[str, dict]:
    return json.loads(_STADIUMS.read_text())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _utc_offset_h(tz: str, day: str) -> float:
    return ZoneInfo(tz).utcoffset(datetime.fromisoformat(f"{day}T12:00:00")).total_seconds() / 3600


def _home_stadiums(games: pd.DataFrame) -> dict[tuple[int, str], str]:
    """(season, team) -> the stadium the team played most home games in that
    season -- data-driven, so relocations (OAK->LV, SD->LAC) and
    international 'home' games resolve correctly."""
    def _mode_first(s: pd.Series) -> str:
        s_clean = s[pd.notna(s)]  # Skip NaN stadium_ids
        if len(s_clean) == 0:
            return ""
        counts = s_clean.value_counts()
        top = counts.max()
        return next(v for v in s_clean if counts[v] == top)   # ties resolve to earliest game by row order

    g = games.sort_values(["season", "week"])
    h = g.groupby(["season", "home_team"])["stadium_id"].agg(_mode_first)
    return {(int(k[0]), str(k[1])): v for k, v in h.items()}


def _indoor(roof: str, stadium_roof: str) -> float:
    r = ""
    if pd.notna(roof):
        r = str(roof).strip().lower()
    if r in ("dome", "closed"):
        return 1.0
    if r in ("outdoors", "open"):
        return 0.0
    # blank roof (future games): known from the stadium unless retractable
    if stadium_roof == "dome":
        return 1.0
    if stadium_roof == "outdoors":
        return 0.0
    return float("nan")


def team_game_context(schedules: pd.DataFrame, stadiums: dict[str, dict]) -> pd.DataFrame:
    g = schedules[schedules["game_type"] == "REG"].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(normalize_team)
    homes = _home_stadiums(g)
    rows = []
    for r in g.itertuples(index=False):
        st = stadiums.get(r.stadium_id, {})
        indoor = _indoor(r.roof, st.get("roof", ""))
        if indoor == 1.0:
            temp, wind = _INDOOR_TEMP_F, _INDOOR_WIND_MPH
        elif indoor == 0.0:
            temp = float(r.temp) if pd.notna(r.temp) else float("nan")
            wind = float(r.wind) if pd.notna(r.wind) else float("nan")
        else:
            temp = wind = float("nan")
        if pd.isna(r.surface):
            turf = float("nan")
        else:
            surface_str = str(r.surface).strip().lower()
            if surface_str == "grass":
                turf = 0.0
            elif surface_str == "":
                turf = float("nan")
            else:
                turf = 1.0
        spread = float(r.spread_line) if pd.notna(r.spread_line) else float("nan")
        total = float(r.total_line) if pd.notna(r.total_line) else float("nan")
        gametime_str = r.gametime if pd.notna(r.gametime) else "13:00"
        kick_h = int(str(gametime_str).split(":")[0])
        for side, team, opp, rest, opp_rest in (
            ("home", r.home_team, r.away_team, r.home_rest, r.away_rest),
            ("away", r.away_team, r.home_team, r.away_rest, r.home_rest),
        ):
            base_id = homes.get((int(r.season), team))
            base = stadiums.get(base_id or "", {})
            if st and base:
                travel = haversine_km(base["lat"], base["lon"], st["lat"], st["lon"])
                tz_shift = abs(_utc_offset_h(st["tz"], str(r.gameday)) - _utc_offset_h(base["tz"], str(r.gameday)))
            else:
                travel = tz_shift = float("nan")
            # Check if venue is in Eastern time zone by comparing UTC offsets
            is_eastern = False
            if st.get("tz"):
                try:
                    eastern_offset = _utc_offset_h("America/New_York", str(r.gameday))
                    venue_offset = _utc_offset_h(st.get("tz"), str(r.gameday))
                    is_eastern = venue_offset == eastern_offset
                except (ValueError, KeyError):
                    pass
            sign = 1.0 if side == "home" else -1.0
            rows.append({
                "season": int(r.season), "week": int(r.week), "team": team, "opponent": opp,
                "is_home": 1 if side == "home" else 0, "stadium_id": r.stadium_id,
                "cx_rest": float(rest), "cx_rest_diff": float(rest) - float(opp_rest),
                # unknown rest -> unknown flags (NaN), not "not short / not off a bye"
                "cx_short_week": float(rest <= 5) if pd.notna(rest) else float("nan"),
                "cx_off_bye": float(rest >= 13) if pd.notna(rest) else float("nan"),
                "cx_travel_km": round(travel, 1) if not math.isnan(travel) else travel,
                "cx_tz_shift": tz_shift,
                "cx_west_early": float(base.get("tz") in _PACIFIC and kick_h < 14 and is_eastern),
                "cx_indoor": indoor, "cx_turf": turf, "cx_temp": temp, "cx_wind": wind,
                "cx_div": float(r.div_game),
                "mk_spread": sign * spread, "mk_total": total,
                "mk_implied": (total + sign * spread) / 2 if not (math.isnan(total) or math.isnan(spread)) else float("nan"),
            })
    return pd.DataFrame(rows)
