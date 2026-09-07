"""SportsDataIO adapter: live CFB injuries (+ team crosswalk) for the news
agent.

Pure response parsers for the SportsDataIO CFB endpoints that feed the
decision desk's news agent. Field names mirror SportsDataIO's REAL CFB
schemas, verified against their published OpenAPI swagger (the prior
implementation guessed at endpoint paths/fields and 404'd live):

  - Injuries: `GET /v3/cfb/scores/json/InjuredPlayers` (NOT `/Injuries`,
    which does not exist in this API). Returns a `Player[]` array. Relevant
    fields (all nullable in practice): `FirstName`, `LastName` (there is NO
    single "Name" field), `Team` (the team ABBREVIATION/Key, e.g. "SMU"),
    `TeamID`, `Position`, `InjuryStatus` (e.g. Probable/Questionable/
    Doubtful/Out), `InjuryBodyPart`, `InjuryNotes`, `InjuryStartDate`.

  - Teams (abbreviation -> school crosswalk): `GET
    /v3/cfb/scores/json/Teams`. Returns a `Team[]` array: `TeamID` (int),
    `Key` (abbreviation, e.g. "SMU"), `School` (e.g. "SMU", "Florida
    State"), `Name` (mascot), `TeamLogoUrl`. `InjuredPlayers` keys rows by
    this same `Key` abbreviation, not the school name, so this endpoint is
    what lets a later ingest step rekey injuries onto a school name (and
    from there onto ESPN's display name -- see scripts/desk_inputs.py).

There is NO News endpoint and NO usable weather endpoint in the CFB API --
`parse_news`/`parse_weather` (and their fixtures) have been removed rather
than kept as parsers for endpoints that don't exist.

`_get` is the only network-touching piece here, and it is used solely by a
later ingest step's main() -- never by the parse_* functions above. It
mirrors `cfb.cfbd._get` / `cfb.espn._get`'s retry policy (tenacity: 3
attempts, exponential backoff) but uses SportsDataIO's subscription-key
header auth (`Ocp-Apim-Subscription-Key: <key>`) rather than a Bearer token
or a `?key=` query param -- SportsDataIO documents both the header and the
query-param forms as equivalent; the header keeps the key out of any logged
URL/query string.

`_get` takes the API key as a parameter rather than reading
`SPORTSDATA_API_KEY` itself -- the env lookup (and fail-fast if unset) is a
later ingest script's main()'s job, mirroring `cfb.cfbd._get`. This keeps
`_get` pure of env access, same as the parse_* functions above.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_BASE = "https://api.sportsdata.io/v3/cfb"

# Endpoint paths (see module docstring). Exposed as constants so
# scripts/desk_inputs.py can stay adapter-generic across sports.
INJURED_PLAYERS_PATH = "/scores/json/InjuredPlayers"
TEAMS_PATH = "/scores/json/Teams"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, api_key: str, params: dict | None = None) -> Any:
    """GET {_BASE}{path} with SportsDataIO subscription-key auth and return
    parsed JSON, retrying transient failures.

    `api_key` is passed in by the caller (a later ingest script's main(),
    which reads `SPORTSDATA_API_KEY` once and fails fast if it's unset)
    rather than read from the environment here. tenacity retries any raised
    exception (connect/read timeouts and raise_for_status errors) 3 times
    with exponential backoff -- same retry policy as cfb.cfbd._get /
    cfb.espn._get."""
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    r = httpx.get(f"{_BASE}{path}", params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_injuries(payload) -> dict[str, list[dict]]:
    """Injury report grouped by team ABBREVIATION, from SportsDataIO's real
    CFB `InjuredPlayers` endpoint (list of `Player` objects: `FirstName`,
    `LastName`, `Team`, `TeamID`, `Position`, `InjuryStatus`,
    `InjuryBodyPart`, `InjuryNotes`, `InjuryStartDate`, ...).

    Rows with a null `Team` are skipped (nothing to key them by). Per
    player:
      - "player": "{FirstName} {LastName}", stripped (either half may be
        missing/null -- treated as empty rather than crashing).
      - "position": Position
      - "status": InjuryStatus (e.g. "Out"/"Questionable"/"Doubtful"/
        "Probable")
      - "note": InjuryBodyPart and InjuryNotes joined with " - ", skipping
        whichever is null, or None when both are null.

    Note this dict is keyed by SportsDataIO's team ABBREVIATION (e.g.
    "SMU"), not a school name -- callers that need to join onto ESPN
    display names must first rekey through `parse_teams`'s
    abbreviation -> School map (see scripts/desk_inputs.py)."""
    out: dict[str, list[dict]] = {}
    for row in payload:
        team = row.get("Team")
        if team is None:
            continue
        first = row.get("FirstName") or ""
        last = row.get("LastName") or ""
        player = f"{first} {last}".strip()
        body_part, notes = row.get("InjuryBodyPart"), row.get("InjuryNotes")
        parts = [p for p in (body_part, notes) if p is not None]
        note = " - ".join(parts) if parts else None
        out.setdefault(team, []).append({
            "player": player,
            "position": row.get("Position"),
            "status": row.get("InjuryStatus"),
            "note": note,
        })
    return out


def parse_teams(payload) -> dict[str, str]:
    """Abbreviation ({Key}) -> school name ({School}) crosswalk, from
    SportsDataIO's real CFB `Teams` endpoint (list of `Team` objects:
    `TeamID`, `Key`, `School`, `Name`, `TeamLogoUrl`, ...).

    Rows with a null `Key` or `School` are skipped -- nothing usable to map.
    This is what lets a later ingest step rekey `parse_injuries`'s
    abbreviation-keyed dict onto a school name, which is in turn matched
    (by prefix, since SportsDataIO's school name and ESPN's displayName
    differ, e.g. "Florida State" vs "Florida State Seminoles") onto ESPN's
    display name -- see scripts/desk_inputs.py's `_rekey_by_espn_name`."""
    out: dict[str, str] = {}
    for row in payload:
        key, school = row.get("Key"), row.get("School")
        if key is None or school is None:
            continue
        out[key] = school
    return out
