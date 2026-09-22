from __future__ import annotations
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .teams import normalize_team

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, params: dict | None = None) -> Any:
    """GET {_BASE}{path} and return parsed JSON, retrying transient failures.

    ESPN's edge occasionally drops a TLS handshake or returns a 5xx; a single
    blip used to fail the whole grading batch. tenacity retries any raised
    exception (connect/read timeouts and raise_for_status errors) 3 times with
    exponential backoff (same retry policy the CFB adapter uses)."""
    r = httpx.get(f"{_BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _competitors(event) -> dict:
    comp = event["competitions"][0]["competitors"]
    return {c["homeAway"]: c for c in comp}

def _market_from_odds(odds: list | None) -> dict:
    """Home spread + total from an ESPN odds list (scoreboard competitions[].odds
    or summary pickcenter -- same element shape). `spread` is the HOME line (home
    favored -> negative), `overUnder` is the total. First provider carrying each;
    both independently nullable."""
    odds = odds or []
    spread = next((o.get("spread") for o in odds if o.get("spread") is not None), None)
    total = next((o.get("overUnder") for o in odds if o.get("overUnder") is not None), None)
    return {"market_spread": float(spread) if spread is not None else None,
            "market_total": float(total) if total is not None else None}

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
            **_market_from_odds(ev["competitions"][0].get("odds")),
        })
    return out

def parse_final(event) -> dict | None:
    if event["status"]["type"]["name"] != "STATUS_FINAL":
        return None
    c = _competitors(event)
    return {"home_score": int(c["home"]["score"]),
            "away_score": int(c["away"]["score"]), "final": True}

def parse_market(summary) -> dict:
    """Closing market line from a /summary payload's `pickcenter` block.

    ESPN's `spread` is the HOME team's line (home favored -> negative, e.g.
    'TB -3' -> -3.0; home underdog -> positive, e.g. away 'SF -5.5' -> +5.5),
    and `overUnder` is the total. Books are listed per provider; take the first
    provider that carries each number (they agree within a point at close).
    Either can be missing -- a stale game has an empty pickcenter, and some
    games post a spread but no total -- so both fields are independently
    nullable. Returns {"market_spread": float|None, "market_total": float|None}."""
    return _market_from_odds(summary.get("pickcenter"))

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

def parse_injuries(payload) -> list[dict]:
    """ESPN league-wide injuries (/injuries) -> ``[{team, player, status}]``.
    PURE. `team` is ESPN's team displayName (maps to our abbrev via the same
    crosswalk as predictions team names); `status` is ESPN's designation
    ("Out"/"Doubtful"/"Questionable"/"Injured Reserve"/...). ESPN updates these
    far faster than nflverse's weekly report parquet, so it catches a
    recently-injured starter mid-week before nflverse lists them."""
    out: list[dict] = []
    for team in payload.get("injuries", []) or []:
        name = team.get("displayName")
        if not name:
            continue
        for it in team.get("injuries", []) or []:
            ath = (it.get("athlete") or {}).get("displayName")
            status = it.get("status")
            if ath and status:
                out.append({"team": name, "player": ath, "status": str(status)})
    return out


def fetch_injuries() -> list[dict]:
    return parse_injuries(_get("/injuries"))


def fetch_schedule(season: int, week: int, season_type: int = 2) -> list[dict]:
    return parse_schedule(_get("/scoreboard",
                               {"dates": season, "seasontype": season_type, "week": week}))

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
    return parse_current_week(_get("/scoreboard"))

def target_week(cur: dict) -> dict:
    """The (season, week, season_type) the NFL pipeline should PRICE, given the
    live current week `cur` (from parse/fetch_current_week).

    Regular season (2) and postseason (3): price the live current week as-is.
    Preseason (1) or offseason (4): look ahead to regular-season Week 1 -- the
    games the odds market actually prices. Preseason games are not in the
    `americanfootball_nfl` odds feed, so pricing the current preseason week would
    match no odds; Week-1 lines are already posted, so we target those instead.
    Once real Week 1 arrives ESPN reports season_type 2 / week 1 and this returns
    the live week again, tracking the season forward with no look-ahead.
    """
    st = int(cur["season_type"])
    if st in (2, 3):
        return {"season": int(cur["season"]), "week": int(cur["week"]), "season_type": st}
    return {"season": int(cur["season"]), "week": 1, "season_type": 2}

def advance_if_complete(tw: dict, games: list[dict]) -> dict:
    """PURE. If `tw` is a regular-season week whose every game in `games` is
    STATUS_FINAL, return the NEXT regular-season week (capped at 18); otherwise
    return `tw` unchanged.

    ESPN keeps its "current week" on the just-played week from Monday night until
    it rolls forward mid-week, so without this the pipeline re-prices a finished
    week (0 upcoming games -> stale sims/props) for a day or two after MNF. Once
    the week is complete we look ahead so the next week's upcoming games get
    priced/simmed right away. An empty `games` list (schedule not posted) leaves
    `tw` untouched.
    """
    if int(tw["season_type"]) != 2 or not games:
        return tw
    if tw["week"] < 18 and all(g.get("status") == "STATUS_FINAL" for g in games):
        return {"season": int(tw["season"]), "week": int(tw["week"]) + 1, "season_type": 2}
    return tw


def resolve_target_week() -> dict:
    tw = target_week(fetch_current_week())
    if int(tw["season_type"]) == 2:
        tw = advance_if_complete(
            tw, fetch_schedule(tw["season"], tw["week"], season_type=2))
    return tw

def fetch_final(event_id: int) -> dict | None:
    data = _get("/summary", {"event": event_id})
    ev = data.get("header", {}).get("competitions", [{}])[0]
    # summary shape differs from scoreboard; adapt via the header competition
    status = ev.get("status", {}).get("type", {}).get("name")
    if status != "STATUS_FINAL":
        return None
    comp = {c["homeAway"]: c for c in ev.get("competitors", [])}
    # Same payload also carries the closing line (pickcenter), so grade the
    # model's spread/total picks against the market with no extra request.
    return {"home_score": int(comp["home"]["score"]),
            "away_score": int(comp["away"]["score"]), "final": True,
            **parse_market(data)}

def fetch_inactives(event_id: int) -> list[str]:
    return parse_inactives(_get("/summary", {"event": event_id}))
