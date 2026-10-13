"""One-off probe: does Pinnacle publish NFL player props through The Odds API?

Costs about 8 credits (the 8 NFL prop markets x regions=eu x ONE event; listing events is free). It picks the
next upcoming NFL event, asks for its props from the EU region only, and prints a verdict:

    PROBE_VERDICT=pinnacle_has_props | pinnacle_absent | inconclusive

Run it through .github/workflows/probe-pinnacle-props.yml (ODDS_API_KEY lives in GitHub secrets, not in .env).
It never touches the database.

    ODDS_API_KEY=... uv run python scripts/probe_pinnacle_props.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import sports
from sportsmodel.ingest import odds
from sportsmodel.ledger.probe import pinnacle_props_verdict


def next_event(events: list[dict], now: datetime) -> dict | None:
    """The soonest event that has not started yet."""
    upcoming = [(c, e) for e in events if (c := odds.parse_commence(e.get("commence_time"))) and c > now]
    return min(upcoming, key=lambda t: t[0])[1] if upcoming else None


def run(now: datetime, fetch_events, fetch_props, remaining, last_cost) -> dict:
    """The probe on injected IO (`fetch_events() -> list`, `fetch_props(event_id) -> dict`,
    `remaining()` / `last_cost()` -> header strings or None). Returns the report dict."""
    event = next_event(fetch_events(), now)
    if event is None:
        return {"verdict": "inconclusive", "reason": "no upcoming NFL event", "credits_last": None,
                "credits_remaining": remaining()}
    payload = fetch_props(event["id"])
    return {**pinnacle_props_verdict(payload), "event": f"{event.get('away_team')} @ {event.get('home_team')}",
            "commence_time": event.get("commence_time"), "credits_last": last_cost(),
            "credits_remaining": remaining()}


def main() -> int:
    if not os.getenv("ODDS_API_KEY"):
        print("::error::ODDS_API_KEY is not set")
        return 1
    cfg = sports.get("nfl")
    markets = list(cfg.prop_market_map.values())
    report = run(
        datetime.now(timezone.utc),
        lambda: odds.fetch_events(cfg),
        lambda eid: odds.fetch_event_props(eid, markets, cfg, regions="eu"),
        lambda: odds.last_requests_remaining,
        lambda: odds.last_requests_last)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    print(f"PROBE_VERDICT={report['verdict']}")
    print(f"PROBE_CREDITS_LAST={report['credits_last']}")
    print(f"PROBE_CREDITS_REMAINING={report['credits_remaining']}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"### Pinnacle props probe\n\n**{report['verdict']}**\n\n```json\n{text}\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
