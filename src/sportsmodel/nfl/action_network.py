"""Action Network public betting splits via the Apify actor
`zen-studio/action-network-odds` (the only splits SOURCE; the
`nfl_betting_splits` table + `market_features` consume it downstream).

Each actor dataset item is one game with `homeTeam.abbreviation`,
`awayTeam.abbreviation`, `startTime` (ISO), and a `consensus` object holding
`moneyline`/`spread`/`total`, each with a `sides` array whose entries carry
`side` ("home"/"away"/"over"/"under"), `ticketPercent` (share of bets),
`moneyPercent` (share of handle), and `sharpGap` (money - ticket).

`parse_action_network_splits` is the PURE seam (items -> split rows, unit
tested). `fetch_splits` is the thin IO wrapper around Apify's
run-sync-get-dataset-items endpoint (needs APIFY_TOKEN; not unit tested).

NOTE: this is a scraper (Action Network has no first-party API), so field names
follow the actor's documented output and should be re-checked if a live run
returns something unexpected.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from sportsmodel.nfl.teams import normalize_team

ACTOR_ID = "zen-studio~action-network-odds"
_RUN_SYNC_URL = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"

# Action Network consensus market keys -> our nfl_betting_splits market codes
# (they already line up; explicit for clarity + to ignore anything else).
_MARKETS = {"moneyline": "moneyline", "spread": "spread", "total": "total"}
_SIDES = {"home", "away", "over", "under"}


def _norm_abbr(abbr: str | None) -> str | None:
    if not abbr:
        return None
    try:
        return normalize_team(str(abbr).strip())
    except Exception:  # noqa: BLE001 -- an unknown abbr shouldn't abort the batch
        return str(abbr).strip().upper()


def parse_action_network_splits(items) -> list[dict]:
    """PURE. Actor dataset items -> split rows, one per (game, market, side):
    ``{away_abbr, home_abbr, start_time, market, side, cash_pct, ticket_pct}``
    with team abbreviations normalized to our canonical codes. `cash_pct` =
    moneyPercent (handle), `ticket_pct` = ticketPercent (bets). Markets/sides
    we don't recognize are skipped, as is a side missing BOTH percentages. The
    caller resolves game_pk (team+date) and attaches captured_at."""
    out: list[dict] = []
    for item in items or []:
        home_team = item.get("homeTeam") or {}
        away_team = item.get("awayTeam") or {}
        home = _norm_abbr(home_team.get("abbreviation"))
        away = _norm_abbr(away_team.get("abbreviation"))
        # Team display names too: NFL resolves game_pk by normalized abbrev, but
        # CFB's ESPN slate keys on team ID (not abbrev), so the CFB capture
        # matches on name instead. Additive -- NFL ignores these.
        home_name = home_team.get("displayName") or home_team.get("name")
        away_name = away_team.get("displayName") or away_team.get("name")
        start = item.get("startTime")
        consensus = item.get("consensus") or {}
        for an_market, market in _MARKETS.items():
            block = consensus.get(an_market) or {}
            for s in block.get("sides") or []:
                side = str(s.get("side") or "").strip().lower()
                if side not in _SIDES:
                    continue
                cash, tickets = s.get("moneyPercent"), s.get("ticketPercent")
                if cash is None and tickets is None:
                    continue
                out.append({
                    "away_abbr": away, "home_abbr": home,
                    "away_name": away_name, "home_name": home_name,
                    "start_time": start,
                    "market": market, "side": side,
                    "cash_pct": cash, "ticket_pct": tickets,
                })
    return out


def attach_game_pks(rows: list[dict], index: dict) -> list[dict]:
    """PURE. Resolve each split row's ESPN game_pk via
    `index[(home_abbr, away_abbr, date)] -> game_pk`, where date is the UTC day
    of `start_time`. Rows with no matching slate game are dropped (their game
    isn't in our current slate). Returns rows with `game_pk` added."""
    out: list[dict] = []
    for r in rows:
        date = (r.get("start_time") or "")[:10]
        gp = index.get((r.get("home_abbr"), r.get("away_abbr"), date))
        if gp is None:
            continue
        out.append({**r, "game_pk": gp})
    return out


def _norm_name(name) -> str:
    """Lowercased, whitespace-collapsed team name for cross-source matching."""
    return " ".join(str(name or "").lower().split())


def attach_game_pks_by_name(rows: list[dict], index: dict) -> list[dict]:
    """PURE. Like `attach_game_pks` but matches on normalized team NAMES + UTC
    date: ``index[(home_name_norm, away_name_norm, date)] -> game_pk``. Used for
    CFB, whose ESPN slate keys on team ID (not the actor's abbreviations), so the
    display names are the reliable join key. Rows with no match are dropped."""
    out: list[dict] = []
    for r in rows:
        date = (r.get("start_time") or "")[:10]
        key = (_norm_name(r.get("home_name")), _norm_name(r.get("away_name")), date)
        gp = index.get(key)
        if gp is None:
            continue
        out.append({**r, "game_pk": gp})
    return out


_LEAGUE_TO_SPORT = {"nfl": "nfl", "ncaaf": "cfb"}


def parse_team_records(items) -> list[dict]:
    """PURE. The actor's standings item(s) (includeStandings: true) -> one row
    per team: {sport, season, an_team_name, abbr, records}, where records maps
    each category (ats, ats_road, over_under_home, units_fav, last_5, ...) to
    {w, l, p, o, u}: wins/losses, pushes (draws, else ties), overs/unders
    (over_under_* only; units_* carry the unit value in w). Game items and
    rows without a team name are skipped."""
    out: list[dict] = []
    for item in items or []:
        if item.get("recordType") != "standings":
            continue
        sport = _LEAGUE_TO_SPORT.get(str(item.get("league") or "").lower())
        if sport is None:
            continue
        for row in item.get("standings") or []:
            team = row.get("team") or {}
            name = team.get("name")
            if not name:
                continue
            records = {}
            for rec in row.get("records") or []:
                cat = rec.get("category")
                if not cat:
                    continue
                push = rec.get("draws") if rec.get("draws") is not None else rec.get("ties")
                records[cat] = {"w": rec.get("wins"), "l": rec.get("losses"), "p": push,
                                "o": rec.get("overs"), "u": rec.get("unders")}
            out.append({"sport": sport, "season": item.get("season"), "an_team_name": name,
                        "abbr": team.get("abbreviation"), "records": records})
    return out


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8),
       retry=retry_if_exception_type(httpx.TransportError), reraise=True)
def fetch_splits(token: str, leagues=("nfl",), game_status=("scheduled",),
                 extra_input: dict | None = None) -> list[Any]:
    """Run the actor synchronously and return its dataset items (a list). IO;
    not unit tested. Splits are cheap consensus data (no props/line movement
    pulled) to keep actor cost/quota down.

    Only transient transport errors are retried; an HTTP error status is raised
    immediately as a RuntimeError carrying the status code AND the actor's
    response body, which is where Apify explains a bad token/quota/input (a bare
    raise_for_status hides that behind a generic message)."""
    payload = {
        "leagues": list(leagues),
        "gameStatus": list(game_status),
        "includeProps": False,
        "includeLineMovement": False,
        "includeExpertPicks": False,
        "includeInjuries": False,
        "includeStandings": False,
    }
    if extra_input:
        payload.update(extra_input)
    r = httpx.post(_RUN_SYNC_URL, params={"token": token}, json=payload, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(
            f"Apify actor {ACTOR_ID} returned HTTP {r.status_code}: {r.text[:800]}")
    return r.json()
