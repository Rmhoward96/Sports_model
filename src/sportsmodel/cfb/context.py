"""Game-day context features for cfb-ratings-v3 (weather, travel, time zones, rest, talent gap). PURE.

Sign convention: margin = home - away, so every *_diff feature is (away team's burden) minus
(home team's burden): a positive value should push the margin UP. Totals features are
non-negative magnitudes. Everything unknown is 0.0 (never fabricated); `weather_missing` flags a
game with no usable weather reading so the fit can absorb that group's mean shift (it is a
nuisance column: the fitted model drops its coefficient at prediction time).

Leak rule: nothing here reads a result. Weather/venue/schedule facts are known (or forecast)
before kickoff; rest uses only earlier start dates; talent is the season's preseason composite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

WIND_MPH = 15.0               # wind above this slows the passing game / scoring
COLD_F = 40.0                 # temperatures below this suppress scoring
TRAVEL_MID_MI, TRAVEL_FAR_MI = 500.0, 1500.0
TZ_SHIFT_HOURS = 2.0
SHORT_WEEK_DAYS, BYE_DAYS = 6, 13
TALENT_HALF_LIFE_GAMES = 3.0

MARGIN_CTX = ("travel_mid_diff", "travel_far_diff", "tz_east_diff", "tz_west_diff",
              "short_week_diff", "bye_diff", "talent_gap")
TOTAL_CTX = ("wind_excess", "cold_excess", "precip")
NUISANCE = ("weather_missing",)
CONTEXT_COLS = MARGIN_CTX + TOTAL_CTX + NUISANCE


@dataclass(frozen=True)
class ContextAssets:
    venues: dict = field(default_factory=dict)       # venue_id -> {"lat", "lon", "tz", "dome"}
    game_venue: dict = field(default_factory=dict)   # game_id -> venue_id
    home_venue: dict = field(default_factory=dict)   # team -> {season: venue_id} (non-neutral home games)
    weather: dict = field(default_factory=dict)      # game_id -> weather row dict
    talent_z: dict = field(default_factory=dict)     # (season, team) -> z-score of the talent composite


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def utc_offset_hours(tz: str, when_iso: str) -> float:
    """UTC offset (hours, DST-aware) of IANA zone `tz` at ISO instant `when_iso`; NaN if unknown."""
    try:
        when = datetime.fromisoformat(when_iso.replace("Z", "+00:00"))
        return when.astimezone(ZoneInfo(tz)).utcoffset().total_seconds() / 3600.0
    except Exception:  # noqa: BLE001 -- unknown zone / unparseable date -> unknown, never guessed
        return float("nan")


def utc_shift_hours(origin: dict, dest: dict, when_iso: str) -> float:
    """Time-zone shift (hours, + = traveled east) from `origin` to `dest` venue. When both venues have
    a usable IANA timezone the shift is DST-aware; if EITHER lacks one, BOTH use the longitude zone
    round(lon / 15) so a DST-aware offset is never compared with a standard-time one. NaN when a
    needed longitude is missing (-> a zero adjustment downstream)."""
    a, b = utc_offset_hours(origin.get("tz") or "", when_iso), utc_offset_hours(dest.get("tz") or "", when_iso)
    if not (math.isnan(a) or math.isnan(b)):
        return b - a
    lons = (origin.get("lon"), dest.get("lon"))
    if not all(_ok(x) for x in lons):
        return float("nan")
    # round(lon / 15) is a known approximation near zone boundaries (e.g. Ann Arbor, Lubbock can land
    # one zone off). It applies only to venues with a null timezone, and only to the shift feature.
    return float(round(lons[1] / 15.0) - round(lons[0] / 15.0))


def build_context_assets(venues: pd.DataFrame | None, meta: pd.DataFrame | None,
                         weather: pd.DataFrame | None, talent: pd.DataFrame | None) -> ContextAssets:
    """Index the committed frames (any may be None -> that feature family is neutral)."""
    v = {}
    if venues is not None:
        for r in venues.itertuples():
            v[int(r.venue_id)] = {"lat": r.latitude, "lon": r.longitude,
                                  "tz": r.timezone if isinstance(r.timezone, str) else "",
                                  "dome": r.dome == 1.0}
    gv, hv = {}, {}
    if meta is not None:
        m = meta.dropna(subset=["venue_id"])
        gv = {int(g): int(x) for g, x in zip(m["game_id"], m["venue_id"])}
        home = m[~m["neutral_site"].astype(bool)]
        mode = home.groupby(["home_team", "season"])["venue_id"].agg(lambda s: int(s.mode().iloc[0]))
        for (team, season), vid in mode.items():
            hv.setdefault(team, {})[int(season)] = int(vid)
    wx = {}
    if weather is not None:
        wx = {int(r["game_id"]): r for r in weather.to_dict("records")}
        for gid, r in wx.items():                  # weather rows know their own venue (incl. forecasts)
            if _ok(r.get("venue_id")):
                gv.setdefault(gid, int(r["venue_id"]))
    tz = {}
    if talent is not None:
        for season, sdf in talent.dropna(subset=["talent"]).groupby("season"):
            vals = sdf["talent"].astype(float)
            sd = vals.std(ddof=0)
            for team, x in zip(sdf["team"], vals):
                tz[(int(season), str(team))] = float((x - vals.mean()) / sd) if sd > 0 else 0.0
    return ContextAssets(v, gv, hv, wx, tz)


def home_venue_of(assets: ContextAssets, team: str, season: int):
    """The team's home venue id for `season` (latest earlier season if that one is unknown)."""
    seasons = assets.home_venue.get(team, {})
    for s in sorted((s for s in seasons if s <= season), reverse=True):
        return seasons[s]
    return None


def weather_features(row: dict | None, venue: dict | None) -> dict:
    """wind_excess (mph over 15), cold_excess (F under 40), precip (CFBD units, >= 0).
    A dome / indoor game is a known zero. No usable reading -> zeros + weather_missing=1."""
    if (row is not None and bool(row.get("game_indoors"))) or (venue is not None and venue.get("dome")):
        return {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0, "weather_missing": 0.0}
    wind, temp, prec = (row.get(k) if row else None for k in ("wind_speed", "temperature", "precipitation"))
    if not any(_ok(x) for x in (wind, temp, prec)):
        return {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0, "weather_missing": 1.0}
    return {"wind_excess": max(0.0, wind - WIND_MPH) if _ok(wind) else 0.0,
            "cold_excess": max(0.0, COLD_F - temp) if _ok(temp) else 0.0,
            "precip": max(0.0, prec) if _ok(prec) else 0.0, "weather_missing": 0.0}


def _burden(assets: ContextAssets, team: str, season: int, game_vid, when_iso: str) -> dict:
    """One team's travel burden: distance bucket flags and time-zone shift flags (0 = unknown/none)."""
    zero = {"mid": 0.0, "far": 0.0, "east": 0.0, "west": 0.0}
    hv = home_venue_of(assets, team, season)
    if game_vid is None or hv is None or hv == game_vid:
        return zero
    a, b = assets.venues.get(hv), assets.venues.get(game_vid)
    if not a or not b or not (_ok(a["lat"]) and _ok(a["lon"]) and _ok(b["lat"]) and _ok(b["lon"])):
        return zero
    d = haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])
    shift = utc_shift_hours(a, b, when_iso)   # + = traveled east
    shift = 0.0 if math.isnan(shift) else shift
    return {"mid": float(TRAVEL_MID_MI <= d < TRAVEL_FAR_MI), "far": float(d >= TRAVEL_FAR_MI),
            "east": float(shift >= TZ_SHIFT_HOURS), "west": float(shift <= -TZ_SHIFT_HOURS)}


def travel_features(assets: ContextAssets, season: int, home: str, away: str, game_vid, when_iso: str) -> dict:
    h, a = (_burden(assets, t, season, game_vid, when_iso) for t in (home, away))
    return {"travel_mid_diff": a["mid"] - h["mid"], "travel_far_diff": a["far"] - h["far"],
            "tz_east_diff": a["east"] - h["east"], "tz_west_diff": a["west"] - h["west"]}


def rest_features(rest_home, rest_away) -> dict:
    """short_week_diff / bye_diff = away flag - home flag. Unknown rest (season opener) = no flag."""
    def flags(r):
        return (float(_ok(r) and r <= SHORT_WEEK_DAYS), float(_ok(r) and r >= BYE_DAYS))
    (sh, bh), (sa, ba) = flags(rest_home), flags(rest_away)
    return {"short_week_diff": sa - sh, "bye_diff": ba - bh}


def talent_gap(assets: ContextAssets, season: int, home: str, away: str, games_home: int, games_away: int) -> float:
    """(talent z home - talent z away), fading with games played: 0.5 ** (mean games / 3)."""
    zh, za = assets.talent_z.get((season, home)), assets.talent_z.get((season, away))
    if zh is None or za is None:
        return 0.0
    return (zh - za) * 0.5 ** (((games_home + games_away) / 2.0) / TALENT_HALF_LIFE_GAMES)


def rest_table(frame: pd.DataFrame) -> dict:
    """{(game_pk, team): US-Eastern calendar days since that team's previous game of the season (NaN for an opener).
    Reads start dates only, so unscored (upcoming) games are fine."""
    d = frame[["season", "game_pk", "start_date", "home_team", "away_team"]].copy()
    d["t"] = pd.to_datetime(d["start_date"], utc=True, errors="coerce")
    long = pd.concat([d.rename(columns={"home_team": "team"})[["season", "game_pk", "team", "t"]],
                      d.rename(columns={"away_team": "team"})[["season", "game_pk", "team", "t"]]])
    long = long.sort_values(["season", "team", "t"])
    prev = long.groupby(["season", "team"])["t"].shift(1)
    # Count US Eastern calendar days (a Sat 10:30pm ET kickoff is already Sunday in UTC, which would make
    # the next Saturday look like 6 days). Drop the tz after normalising so DST changes cannot skew it.
    def et_day(t):
        return t.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    days = (et_day(long["t"]) - et_day(prev)).dt.days
    return {(int(g), str(t)): (float(x) if pd.notna(x) else float("nan"))
            for g, t, x in zip(long["game_pk"], long["team"], days)}


def context_features(game: dict, assets: ContextAssets, rests: dict, games_home: int, games_away: int) -> dict:
    """All CONTEXT_COLS for one game dict (season, game_pk, home_team, away_team, start_date)."""
    pk, season = int(game["game_pk"]), int(game["season"])
    h, a = game["home_team"], game["away_team"]
    vid = assets.game_venue.get(pk)
    if vid is None and not game.get("neutral_site"):
        vid = home_venue_of(assets, h, season)
    when = str(game.get("start_date") or "")
    out = {**travel_features(assets, season, h, a, vid, when),
           **rest_features(rests.get((pk, h), float("nan")), rests.get((pk, a), float("nan"))),
           "talent_gap": talent_gap(assets, season, h, a, games_home, games_away),
           **weather_features(assets.weather.get(pk), assets.venues.get(vid) if vid is not None else None)}
    return {c: out[c] for c in CONTEXT_COLS}
