"""SportsDataIO adapter: live CFB injuries/news/weather for the news agent.

Pure response parsers for the SportsDataIO CFB endpoints that feed the
decision desk's news agent (injuries, headlines, game-day weather). Every
parser is a pure function over already-decoded JSON -- no network, no DB, no
file reads. Field names mirror SportsDataIO's real CFB schemas:

  - Injuries (`/scores/json/Injuries` or similar): rows carry `Team`, `Name`,
    `Position`, `Status`, `BodyPart`, `Practice` (all nullable in practice).
  - News (`/scores/json/News`): rows carry `Title`, `Content`, `Updated`, and
    a nullable `Team` (single team abbreviation, or null for a
    league-wide/general story).
  - Weather: game rows carry `GameID`, `ForecastTempLow`/`ForecastTempHigh`,
    `ForecastWindSpeed`, `ForecastDescription` (free text; no numeric precip
    field exists in the schema, so precipitation is inferred from keywords
    in the description), and a nested `Stadium.Type` (`"Outdoor"`, `"Dome"`,
    or `"RetractableDome"`) used to derive the `dome` flag. All of these are
    null for a long-range forecast that hasn't populated yet, or (in the
    dome-stadium case) because forecast fields are meaningless indoors.

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

# Keywords in SportsDataIO's free-text `ForecastDescription` that indicate
# precipitation. There is no numeric precip-probability field in the CFB
# weather schema, so this substring match is the documented proxy.
_PRECIP_KEYWORDS = ("rain", "snow", "shower", "storm", "sleet", "drizzle")

# Stadium.Type values that mean "not exposed to weather".
_DOME_TYPES = {"Dome", "RetractableDome"}


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
    """Injury report grouped by team, from SportsDataIO's CFB Injuries
    endpoint (list of {"Team", "Name", "Position", "Status", "BodyPart",
    "Practice", ...}).

    Rows with a null `Team` are skipped (nothing to key them by). Per player:
      - "player": Name
      - "position": Position
      - "status": Status (e.g. "Out"/"Questionable"/"Doubtful"/"Probable")
      - "note": "{BodyPart} - {Practice}" when both are present, whichever
        one is present alone, or None when both are null (real SportsDataIO
        rows are often missing one or both)."""
    out: dict[str, list[dict]] = {}
    for row in payload:
        team = row.get("Team")
        if team is None:
            continue
        body_part, practice = row.get("BodyPart"), row.get("Practice")
        parts = [p for p in (body_part, practice) if p is not None]
        note = " - ".join(parts) if parts else None
        out.setdefault(team, []).append({
            "player": row.get("Name"),
            "position": row.get("Position"),
            "status": row.get("Status"),
            "note": note,
        })
    return out


def parse_news(payload) -> list[dict]:
    """Headlines, from SportsDataIO's CFB News endpoint (list of {"Title",
    "Content", "Updated", "Team", ...}).

    Per story:
      - "headline": Title
      - "teams": [Team] when Team is present, else [] (a null Team means a
        league-wide/general story not tied to one team)
      - "published": Updated
      - "summary": Content"""
    out = []
    for row in payload:
        team = row.get("Team")
        out.append({
            "headline": row.get("Title"),
            "teams": [team] if team is not None else [],
            "published": row.get("Updated"),
            "summary": row.get("Content"),
        })
    return out


def parse_weather(payload) -> dict:
    """Game-day weather keyed by GameID, from SportsDataIO's CFB game/weather
    schema (list of {"GameID", "ForecastTempLow", "ForecastTempHigh",
    "ForecastWindSpeed", "ForecastDescription", "Stadium": {"Type", ...}}).

    Rows with a null GameID are skipped. Per game:
      - "temp": average of ForecastTempLow/ForecastTempHigh when both are
        present, whichever one is present alone, or None when both are null
        (e.g. forecast not yet populated).
      - "wind": ForecastWindSpeed, passed through as-is (may be None).
      - "precip": True/False from a keyword match against
        ForecastDescription (see _PRECIP_KEYWORDS), or None when
        ForecastDescription itself is null (no forecast text to check --
        e.g. domed games, where SportsDataIO often leaves forecast fields
        null since they're meaningless indoors).
      - "dome": True when the nested Stadium.Type is "Dome" or
        "RetractableDome", False for "Outdoor", or None when Stadium/Type is
        missing."""
    out: dict = {}
    for row in payload:
        game_id = row.get("GameID")
        if game_id is None:
            continue

        low, high = row.get("ForecastTempLow"), row.get("ForecastTempHigh")
        if low is not None and high is not None:
            temp = (low + high) / 2
        else:
            temp = low if low is not None else high

        description = row.get("ForecastDescription")
        precip = (
            None if description is None
            else any(kw in description.lower() for kw in _PRECIP_KEYWORDS)
        )

        stadium_type = (row.get("Stadium") or {}).get("Type")
        dome = None if stadium_type is None else stadium_type in _DOME_TYPES

        out[game_id] = {
            "temp": temp,
            "wind": row.get("ForecastWindSpeed"),
            "precip": precip,
            "dome": dome,
        }
    return out
