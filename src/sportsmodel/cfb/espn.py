"""ESPN college-football schedule/score adapter.

Mirrors sportsmodel.nfl.espn's structure and STATUS_FINAL gate. The only
differences: the base path is college-football, and teams are normalized via
cfb.teams.normalize (FBS ESPN team id passthrough, everything else -> "FCS").
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .teams import normalize

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, params: dict | None = None) -> Any:
    """GET {_BASE}{path} and return parsed JSON, retrying transient failures.

    ESPN's edge occasionally drops a TLS handshake or returns a 5xx; a single
    blip used to fail the whole grading batch. tenacity retries any raised
    exception (connect/read timeouts and raise_for_status errors) 3 times with
    exponential backoff -- same retry policy as nfl.espn._get."""
    r = httpx.get(f"{_BASE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _competitors(event) -> dict:
    comp = event["competitions"][0]["competitors"]
    return {c["homeAway"]: c for c in comp}


def _score(competitor: dict) -> int | None:
    raw = competitor.get("score")
    if raw is None or raw == "":
        return None
    return int(raw)


def _market_from_odds(odds: list | None) -> dict:
    """Home spread + total from an ESPN odds list (scoreboard competitions[].odds
    or summary pickcenter -- same element shape). `spread` is the HOME line (home
    favored -> negative), `overUnder` is the total. Take the first provider that
    carries each; both independently nullable."""
    odds = odds or []
    spread = next((o.get("spread") for o in odds if o.get("spread") is not None), None)
    total = next((o.get("overUnder") for o in odds if o.get("overUnder") is not None), None)
    return {"market_spread": float(spread) if spread is not None else None,
            "market_total": float(total) if total is not None else None}


def _conf_id(competitor: dict) -> str | None:
    """ESPN conference id (string) for a scoreboard competitor, None if absent
    (FCS opponents and some payloads omit it)."""
    cid = competitor["team"].get("conferenceId")
    return str(cid) if cid not in (None, "") else None


def parse_schedule(payload) -> list[dict]:
    default_season = payload.get("season", {}).get("year")
    default_week = payload.get("week", {}).get("number")
    out = []
    for ev in payload.get("events", []):
        c = _competitors(ev)
        season = ev.get("season", {}).get("year", default_season)
        week = ev.get("week", {}).get("number", default_week)
        comp0 = ev["competitions"][0]
        out.append({
            "game_pk": int(ev["id"]),
            "home_team": normalize(c["home"]["team"]["id"]),
            "away_team": normalize(c["away"]["team"]["id"]),
            "home_name": c["home"]["team"].get("displayName"),
            "away_name": c["away"]["team"].get("displayName"),
            "home_score": _score(c["home"]),
            "away_score": _score(c["away"]),
            "commence_time": ev["date"],
            "start_date": ev["date"],
            "neutral_site": bool(comp0.get("neutralSite")),
            "conference_game": bool(comp0.get("conferenceCompetition")),
            "home_conf": _conf_id(c["home"]),
            "away_conf": _conf_id(c["away"]),
            "status": ev["status"]["type"]["name"],
            "week": week,
            "season": season,
            **_market_from_odds(comp0.get("odds")),
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

    Identical shape to nfl.espn.parse_market: `spread` is the HOME team's line
    (home favored -> negative), `overUnder` is the total. Take the first
    provider carrying each; both are independently nullable (stale games have
    an empty pickcenter). Returns {"market_spread": float|None, "market_total": float|None}."""
    return _market_from_odds(summary.get("pickcenter"))


def fetch_schedule(season: int, week: int, season_type: int = 2) -> list[dict]:
    return parse_schedule(_get("/scoreboard",
                               {"dates": season, "seasontype": season_type,
                                "week": week, "groups": 80}))


def parse_current_week(payload) -> dict:
    """(season, week, season_type) from a scoreboard payload fetched with no
    week/season params -- ESPN returns the live current week for such a call,
    same as nfl.espn.parse_current_week (identical payload shape)."""
    return {
        "season": int(payload["season"]["year"]),
        "week": int(payload["week"]["number"]),
        "season_type": int(payload["season"]["type"]),
    }


def fetch_current_week() -> dict:
    return parse_current_week(_get("/scoreboard"))


def fetch_final(event_id: int) -> dict | None:
    """Final score for one event via the summary endpoint, or None if not final yet.

    Mirrors nfl.espn.fetch_final: the summary payload's shape differs from the
    scoreboard's, so adapt via header.competitions[0] rather than parse_final
    (which expects a scoreboard-shaped event).
    """
    data = _get("/summary", {"event": event_id})
    ev = data.get("header", {}).get("competitions", [{}])[0]
    status = ev.get("status", {}).get("type", {}).get("name")
    if status != "STATUS_FINAL":
        return None
    comp = {c["homeAway"]: c for c in ev.get("competitors", [])}
    # Same payload also carries the closing line (pickcenter), so grade the
    # model's spread/total picks against the market with no extra request.
    return {"home_score": int(comp["home"]["score"]),
            "away_score": int(comp["away"]["score"]), "final": True,
            **parse_market(data)}


def fetch_scoreboard(season: int, week: int, season_type: int = 2) -> dict:
    """The raw FBS scoreboard payload for one week (`fetch_schedule` parses the same call)."""
    return _get("/scoreboard", {"dates": season, "seasontype": season_type, "week": week, "groups": 80})


def parse_line_scores(payload) -> dict[int, dict]:
    """{game_pk: {"home": [q1, q2, q3, q4], "away": [...]}} for every STATUS_FINAL event of a scoreboard
    payload whose two competitors both carry at least four `linescores` periods (overtime is dropped).
    Anything else is omitted. NOT verified against a live payload (the shape read is ESPN's usual
    competitors[].linescores[].value); the Task 2 [NETWORK] probe confirms it."""
    out: dict[int, dict] = {}
    for ev in payload.get("events", []):
        try:
            if ev["status"]["type"]["name"] != "STATUS_FINAL":
                continue
            sides = {}
            for c in ev["competitions"][0]["competitors"]:
                vals = [p.get("value") for p in (c.get("linescores") or [])]
                if len(vals) < 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vals[:4]):
                    raise ValueError("no usable line score")
                sides[c["homeAway"]] = [int(v) for v in vals[:4]]
            if set(sides) == {"home", "away"}:
                out[int(ev["id"])] = sides
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out
