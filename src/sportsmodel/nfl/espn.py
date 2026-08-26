from __future__ import annotations
import httpx
from .teams import normalize_team

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

def _competitors(event) -> dict:
    comp = event["competitions"][0]["competitors"]
    return {c["homeAway"]: c for c in comp}

def parse_schedule(payload) -> list[dict]:
    out = []
    for ev in payload.get("events", []):
        c = _competitors(ev)
        out.append({
            "game_pk": int(ev["id"]),
            "commence_time": ev["date"],
            "home_team": normalize_team(c["home"]["team"]["abbreviation"]),
            "away_team": normalize_team(c["away"]["team"]["abbreviation"]),
            "home_name": c["home"]["team"].get("displayName"),
            "away_name": c["away"]["team"].get("displayName"),
            "status": ev["status"]["type"]["name"],
        })
    return out

def parse_final(event) -> dict | None:
    if event["status"]["type"]["name"] != "STATUS_FINAL":
        return None
    c = _competitors(event)
    return {"home_score": int(c["home"]["score"]),
            "away_score": int(c["away"]["score"]), "final": True}

def parse_inactives(payload) -> list[str]:
    names = []
    for ev in payload.get("events", []):
        for c in ev.get("competitions", [{}])[0].get("competitors", []):
            for inj in c.get("injuries", []):
                status = (inj.get("status") or "").lower()
                ath = inj.get("athlete", {})
                if status in {"out", "inactive"} and ath.get("displayName"):
                    names.append(ath["displayName"])
    return names

def fetch_schedule(season: int, week: int, season_type: int = 2) -> list[dict]:
    r = httpx.get(f"{_BASE}/scoreboard",
                  params={"dates": season, "seasontype": season_type, "week": week},
                  timeout=20)
    r.raise_for_status()
    return parse_schedule(r.json())

def parse_current_week(payload) -> dict:
    """(season, week, season_type) from a scoreboard payload fetched with no
    week/season params -- ESPN returns the live current week for such a call, so
    this stays correct across the regular-season -> postseason transition
    (season_type flips 2 -> 3 and week resets to 1) without any date math.
    """
    return {
        "season": int(payload["season"]["year"]),
        "week": int(payload["week"]["number"]),
        "season_type": int(payload["season"]["type"]),
    }

def fetch_current_week() -> dict:
    r = httpx.get(f"{_BASE}/scoreboard", timeout=20)
    r.raise_for_status()
    return parse_current_week(r.json())

def fetch_final(event_id: int) -> dict | None:
    r = httpx.get(f"{_BASE}/summary", params={"event": event_id}, timeout=20)
    r.raise_for_status()
    data = r.json()
    ev = data.get("header", {}).get("competitions", [{}])[0]
    # summary shape differs from scoreboard; adapt via the header competition
    status = ev.get("status", {}).get("type", {}).get("name")
    if status != "STATUS_FINAL":
        return None
    comp = {c["homeAway"]: c for c in ev.get("competitors", [])}
    return {"home_score": int(comp["home"]["score"]),
            "away_score": int(comp["away"]["score"]), "final": True}

def fetch_inactives(event_id: int) -> list[str]:
    r = httpx.get(f"{_BASE}/summary", params={"event": event_id}, timeout=20)
    r.raise_for_status()
    return parse_inactives(r.json())
