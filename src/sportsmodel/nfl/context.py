"""Per team-game context features (rest, travel, time zones, roof/surface,
weather) and market features (spread, total, implied team total) for the
props-ML feature table. PURE except `load_stadiums` (reads a committed asset)
and `fetch_hourly_forecast` (the only network call; injected into
`fill_forecast_weather`, so tests pass a fake).

Leakage: every value here is known before kickoff (schedule, rest days, roof,
the line). Historical weather is the recorded game weather; live serving fills the
still-NaN outdoor rows with the Open-Meteo kickoff-hour forecast
(`fill_forecast_weather`) -- a stated train/serve difference in the spec.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

_STADIUMS = Path(__file__).resolve().parents[3] / "assets" / "nfl" / "stadiums.json"
_INDOOR_TEMP_F, _INDOOR_WIND_MPH = 70.0, 0.0
_PACIFIC = {"America/Los_Angeles"}
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_FORECAST_DAYS = 7
_ET = "America/New_York"                     # nflverse gameday/gametime are US-Eastern local


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


def fetch_hourly_forecast(lat: float, lon: float) -> dict | None:
    """Open-Meteo hourly forecast (degF, mph, UTC timestamps) for a venue --
    the real `fetch` for `fill_forecast_weather`. None on any failure, so a
    weather hiccup leaves the features NaN instead of breaking serving."""
    try:
        with httpx.Client(timeout=8) as client:
            resp = client.get(_FORECAST_URL, params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "UTC", "forecast_days": _FORECAST_DAYS + 1,
            })
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _num(v) -> float:
    try:
        return float("nan") if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return float("nan")


def parse_kickoff_forecast(payload: dict, kickoff_utc: pd.Timestamp) -> tuple[float, float]:
    """(temp degF, wind mph) at the forecast hour nearest kickoff. Values are
    returned as-is (units were requested in degF / mph). NaN when the payload
    lacks the series or has no hour within 1h of kickoff (ties -> earlier hour)."""
    nan = (float("nan"), float("nan"))
    hourly = (payload or {}).get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return nan
    t = pd.to_datetime(pd.Series(times), errors="coerce")
    t = t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")
    gap = (t - _utc(kickoff_utc)).abs()
    if gap.isna().all() or gap.min() > pd.Timedelta(hours=1):
        return nan
    i = int(gap.idxmin())

    def at(key: str) -> float:
        vals = hourly.get(key) or []
        return _num(vals[i]) if i < len(vals) else float("nan")

    return at("temperature_2m"), at("wind_speed_10m")


def _game_kickoffs(games: pd.DataFrame) -> dict[tuple[int, int, str], pd.Timestamp]:
    """(season, week, team) -> kickoff UTC for REG games (both sides)."""
    g = games[games["game_type"] == "REG"]
    local = pd.to_datetime(g["gameday"].astype(str) + " " + g["gametime"].fillna("13:00").astype(str),
                           errors="coerce")
    ko = local.dt.tz_localize(_ET, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    out: dict[tuple[int, int, str], pd.Timestamp] = {}
    for season, week, home, away, k in zip(g["season"], g["week"], g["home_team"], g["away_team"], ko):
        if pd.isna(k):
            continue
        for team in (home, away):
            try:
                out[(int(season), int(week), normalize_team(str(team)))] = k
            except ValueError:
                continue
    return out


def fill_forecast_weather(ctx: pd.DataFrame, stadiums: dict[str, dict], games: pd.DataFrame,
                          fetch: Callable[[float, float], dict | None], *,
                          now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Live serving: fill NaN `cx_temp` / `cx_wind` from the kickoff-hour
    forecast for rows not known to be indoor (`cx_indoor != 1`) whose kickoff
    is within the next 7 days of `now` (default: current UTC). `fetch(lat,
    lon)` returns an Open-Meteo hourly payload (or None) and is called at most
    once per venue, only for eligible rows. Returns a copy; other columns and
    ineligible rows are untouched."""
    out = ctx.copy()
    now = _utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
    kickoffs = _game_kickoffs(games)
    payloads: dict[tuple[float, float], dict | None] = {}
    for idx, r in out.iterrows():
        if not (pd.isna(r["cx_temp"]) or pd.isna(r["cx_wind"])) or r["cx_indoor"] == 1:
            continue
        ko = kickoffs.get((int(r["season"]), int(r["week"]), str(r["team"])))
        if ko is None or not (now <= ko <= now + pd.Timedelta(days=_FORECAST_DAYS)):
            continue
        st = stadiums.get(r["stadium_id"]) if pd.notna(r["stadium_id"]) else None
        if not st:
            continue
        loc = (st["lat"], st["lon"])
        if loc not in payloads:
            payloads[loc] = fetch(*loc)
        if not payloads[loc]:
            continue
        temp, wind = parse_kickoff_forecast(payloads[loc], ko)
        if pd.isna(r["cx_temp"]):
            out.at[idx, "cx_temp"] = temp
        if pd.isna(r["cx_wind"]):
            out.at[idx, "cx_wind"] = wind
    return out
