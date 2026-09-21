"""SportsDataIO adapter: live NFL injuries (+ team crosswalk) for the news
agent. NFL sibling of cfb/sportsdata.py.

Field names mirror SportsDataIO's REAL NFL schemas, verified against their
published OpenAPI swagger:

  - Injuries: `GET /v3/nfl/projections/json/InjuredPlayers` (note the
    `/projections/` base -- the NFL injuries feed is NOT under `/scores/`
    like CFB). Returns a `Player[]` array with the SAME relevant fields the
    CFB `Player` carries: `FirstName`, `LastName` (no single "Name" field),
    `Team` (the team ABBREVIATION/Key, e.g. "PHI"), `TeamID`, `Position`,
    `InjuryStatus` (Probable/Questionable/Doubtful/Out), `InjuryBodyPart`,
    `InjuryNotes` -- all nullable in practice.

  - Teams (abbreviation -> full name crosswalk): `GET
    /v3/nfl/scores/json/Teams`. Returns a `Team[]` array: `Key`
    (abbreviation, e.g. "PHI"), `FullName` (e.g. "Philadelphia Eagles"),
    `City`, `Name` (mascot). NOTE: unlike the CFB `Team`, there is NO
    `School` field -- use `FullName`, which equals ESPN's NFL displayName,
    so desk_inputs.py's `_rekey_by_espn_name` prefix match joins injuries
    onto games with no new logic.

parse_injuries is deliberately a near-duplicate of the CFB parser rather
than a shared helper: the repo keeps a full per-sport module (cfb/espn.py
and nfl/espn.py already duplicate substantially), and isolating the sports
is preferred here over a shared abstraction.

`_get` takes the API key as a parameter (main()'s job to read
SPORTSDATA_API_KEY and fail fast) and uses SportsDataIO's subscription-key
header auth, mirroring cfb.sportsdata._get exactly.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_BASE = "https://api.sportsdata.io/v3/nfl"

# Endpoint paths (see module docstring). Exposed as constants so
# scripts/desk_inputs.py can stay adapter-generic across sports. NOTE the
# NFL injuries path is under /projections/, unlike CFB's /scores/.
INJURED_PLAYERS_PATH = "/projections/json/InjuredPlayers"
TEAMS_PATH = "/scores/json/Teams"
# Betting splits (public cash%/ticket%). SportsDataIO's Betting product; the
# per-game splits feed. Path templated by the provider's game id, resolved by
# the capture script. NOTE: this endpoint + its field names are written to
# SportsDataIO's published Betting swagger but MUST be confirmed against a live
# payload once the Betting tier is active (the tier is not yet enabled).
BETTING_SPLITS_PATH = "/odds/json/BettingSplitsByScoreID/{score_id}"

# BettingMarketType (SportsDataIO) -> our market code.
_SPLIT_MARKET_MAP = {"point spread": "spread", "spread": "spread",
                     "total points": "total", "total": "total", "over/under": "total",
                     "moneyline": "moneyline"}
# BettingOutcomeType/label -> our side code.
_SPLIT_SIDE_MAP = {"home": "home", "away": "away", "over": "over", "under": "under"}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, api_key: str, params: dict | None = None) -> Any:
    """GET {_BASE}{path} with SportsDataIO subscription-key auth and return
    parsed JSON, retrying transient failures (tenacity: 3 attempts,
    exponential backoff -- same policy as cfb.sportsdata._get)."""
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    r = httpx.get(f"{_BASE}{path}", params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_injuries(payload) -> dict[str, list[dict]]:
    """Injury report grouped by team ABBREVIATION, from SportsDataIO's NFL
    `InjuredPlayers` endpoint (list of `Player` objects). Same shape and
    field names as the CFB parser.

    Rows with a null `Team` are skipped. Per player:
      - "player": "{FirstName} {LastName}", stripped (either half may be null).
      - "position": Position
      - "status": InjuryStatus
      - "note": InjuryBodyPart and InjuryNotes joined with " - ", skipping
        whichever is null, or None when both are null.

    Keyed by SportsDataIO's team ABBREVIATION (e.g. "PHI") -- callers that
    need ESPN display names first rekey through `parse_teams`'s
    abbreviation -> FullName map (see scripts/desk_inputs.py)."""
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


def parse_betting_splits(payload) -> list[dict]:
    """Public betting splits for ONE game, from SportsDataIO's Betting-splits
    feed. PURE. Returns rows ``{market, side, cash_pct, ticket_pct}`` where
    ``market`` in {spread,total,moneyline}, ``side`` in {home,away,over,under},
    ``cash_pct`` = MoneyPercentage (% of money) and ``ticket_pct`` =
    BetPercentage (% of tickets). The caller (capture script) attaches
    ``game_pk``/``commence_time``/``captured_at``.

    Accepts either the game object ``{"BettingMarketSplits": [...]}`` or the
    bare ``BettingMarketSplits`` list. Each market split carries
    ``BettingMarketType`` and a list of per-outcome splits (``BettingBetSplits``
    or ``BettingSplits``) with ``BettingOutcomeType``/``Name`` +
    ``MoneyPercentage`` + ``BetPercentage``. Markets/outcomes we don't map are
    skipped, and a split missing both percentages is skipped.

    CAVEAT: field names follow SportsDataIO's published Betting swagger but are
    UNVERIFIED against a live payload (the Betting tier is not yet active) --
    confirm and adjust the key names here once a real response is available."""
    splits = payload.get("BettingMarketSplits", payload) if isinstance(payload, dict) else payload
    out: list[dict] = []
    for ms in splits or []:
        market = _SPLIT_MARKET_MAP.get(str(ms.get("BettingMarketType", "")).strip().lower())
        if market is None:
            continue
        outcomes = ms.get("BettingBetSplits") or ms.get("BettingSplits") or []
        for o in outcomes:
            label = str(o.get("BettingOutcomeType") or o.get("Name") or "").strip().lower()
            side = _SPLIT_SIDE_MAP.get(label)
            if side is None:
                continue
            cash, tickets = o.get("MoneyPercentage"), o.get("BetPercentage")
            if cash is None and tickets is None:
                continue
            out.append({"market": market, "side": side,
                        "cash_pct": cash, "ticket_pct": tickets})
    return out


def parse_teams(payload) -> dict[str, str]:
    """Abbreviation ({Key}) -> full name ({FullName}) crosswalk, from
    SportsDataIO's NFL `Teams` endpoint (list of `Team` objects). NFL's
    `Team` has NO `School` field (unlike CFB) -- `FullName` (e.g.
    "Philadelphia Eagles") is the ESPN-displayName-equivalent used to rekey
    abbreviation-keyed injuries onto ESPN display names (see
    scripts/desk_inputs.py's `_rekey_by_espn_name`).

    Rows with a null `Key` or `FullName` are skipped."""
    out: dict[str, str] = {}
    for row in payload:
        key, full = row.get("Key"), row.get("FullName")
        if key is None or full is None:
            continue
        out[key] = full
    return out
