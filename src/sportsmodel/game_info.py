"""game_info rows (pure): venue + weather (+ CFB line score) for the games of the upcoming window.

One row per (sport, game_pk) for games that kick off between now - WINDOW_BEFORE and now + WINDOW_AFTER,
written by scripts/build_game_info.py into Supabase `game_info` (db/migration_site_panels.sql); the game
page's hero line, weather insight and Projected Game Flow actuals read it.

* CFB: weather + venue link from CFBD `/games/weather` (forecast before kickoff, observation after),
  city / state / dome from assets/cfb/venues.parquet, the final quarter scores from ESPN's scoreboard
  (Q1-Q4 only; overtime is dropped).
* NFL: ESPN `/summary` gameInfo (`nfl.espn.parse_game_info`); weather_kind is always 'forecast' (ESPN's
  block is not known to be an observation).
Indoor games carry no weather (the site says "Indoors"). A missing field stays None, never guessed.
"""
from __future__ import annotations

import math
from typing import Callable

import pandas as pd

WINDOW_BEFORE = pd.Timedelta(days=2)
WINDOW_AFTER = pd.Timedelta(days=10)
WEATHER_FIELDS = ("temp_f", "wind_mph", "precip_chance", "precip_in", "conditions")


def utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def in_window(kickoff, now) -> bool:
    """now - 2 days <= kickoff <= now + 10 days (both inclusive); False for an unreadable kickoff."""
    try:
        k, n = utc(kickoff), utc(now)
    except (ValueError, TypeError):
        return False
    if pd.isna(k):
        return False
    return n - WINDOW_BEFORE <= k <= n + WINDOW_AFTER


def _num(v):
    """float, or None for None / NaN / non-numbers."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _text(v):
    s = "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()
    return s or None


def _blank(sport: str, pk: int, source: str, now) -> dict:
    return {"sport": sport, "game_pk": int(pk), "venue_name": None, "city": None, "state": None,
            "indoor": None, "temp_f": None, "wind_mph": None, "precip_chance": None, "precip_in": None,
            "conditions": None, "weather_kind": None, "source": source,
            "captured_at": utc(now).to_pydatetime(), "line_score": None}


def cfb_game_info(games: list[dict], line_scores: dict[int, dict], weather: pd.DataFrame,
                  venues: pd.DataFrame | None, now) -> list[dict]:
    """CFB rows. `games` = ESPN `cfb.espn.parse_schedule` dicts (game_pk, commence_time); `line_scores` =
    `cfb.espn.parse_line_scores`; `weather` = `cfbd_games.parse_weather_window` (only has_fbs rows are
    used); `venues` = venues.parquet (venue_id, name, city, state, dome) or None. A game with neither a
    weather row nor a line score is skipped. weather_kind is 'observed' once the game has kicked off."""
    now = utc(now)
    wx = {}
    if weather is not None and len(weather):
        for r in weather[weather["has_fbs"]].drop_duplicates("game_id", keep="last").to_dict("records"):
            wx[int(r["game_id"])] = r
    vmap = {} if venues is None or venues.empty else venues.drop_duplicates("venue_id").set_index("venue_id").to_dict("index")
    out = []
    for g in games:
        pk, kick = int(g["game_pk"]), g.get("commence_time") or g.get("start_date")
        if not in_window(kick, now):
            continue
        w, ls = wx.get(pk), line_scores.get(pk)
        if w is None and ls is None:
            continue
        row = _blank("cfb", pk, "cfbd", now)
        if ls is not None:
            row["line_score"] = {"home": list(ls["home"]), "away": list(ls["away"])}
        if w is not None:
            vid = _num(w.get("venue_id"))
            v = vmap.get(int(vid)) if vid is not None else None
            row["venue_name"] = _text(v["name"]) if v else _text(w.get("venue"))
            if v:
                row["city"], row["state"] = _text(v.get("city")), _text(v.get("state"))
            dome = _num(v.get("dome")) if v else None
            row["indoor"] = bool(w.get("game_indoors")) or dome == 1.0
            if not row["indoor"]:
                row.update(temp_f=_num(w.get("temperature")), wind_mph=_num(w.get("wind_speed")),
                           precip_in=_num(w.get("precipitation")), conditions=_text(w.get("condition")))
                if any(row[f] is not None for f in WEATHER_FIELDS):
                    row["weather_kind"] = "observed" if utc(kick) <= now else "forecast"
        out.append(row)
    return out


def nfl_game_info(games: list[dict], fetch_info: Callable[[int], dict], now,
                  warn: Callable[[str], None] = lambda m: None) -> list[dict]:
    """NFL rows. `games` = `nfl.espn.parse_schedule` dicts; `fetch_info(game_pk)` = `nfl.espn.fetch_game_info`.
    A game whose summary call raises is skipped with a warning (one bad call never loses the slate)."""
    now = utc(now)
    out, seen = [], set()
    for g in games:
        pk = int(g["game_pk"])
        if pk in seen or not in_window(g.get("commence_time"), now):
            continue
        seen.add(pk)
        try:
            info = fetch_info(pk)
        except Exception as exc:  # noqa: BLE001
            warn(f"nfl: summary for {pk} unavailable ({type(exc).__name__})")
            continue
        row = _blank("nfl", pk, "espn", now)
        # True when ESPN says indoors; False only when it named the venue (so the verdict is real);
        # None for a blank summary, so a stored True is never overwritten by a guess.
        indoor = True if info.get("indoor") else (False if info.get("venue_name") else None)
        row.update(venue_name=info.get("venue_name"), city=info.get("city"), state=info.get("state"),
                   indoor=indoor, temp_f=_num(info.get("temp_f")),
                   wind_mph=_num(info.get("wind_mph")), precip_chance=_num(info.get("precip_chance")),
                   conditions=_text(info.get("conditions")))
        if any(row[f] is not None for f in WEATHER_FIELDS):
            row["weather_kind"] = "forecast"
        out.append(row)
    return out
