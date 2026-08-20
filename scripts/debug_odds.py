"""One-off diagnostic: dump the RAW The Odds API player-prop response (no DB writes).

Prints, for the first few upcoming events, each bookmaker's market key, how we map it
(_ODDS_TO_OURS), and the raw outcomes (name / description / point / price). This shows
exactly what the API returns under `batter_hits` etc. vs what we store, so we can tell
whether a mislabel is in the API payload or in our ingest.

Usage:
    uv run python scripts/debug_odds.py [player_substring] [n_events]
Requires ODDS_API_KEY.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ingest import odds

# Focus on the two markets in question — cheap on credits.
MARKETS = ["batter_hits", "batter_home_runs", "batter_hits_runs_rbis"]


def main() -> None:
    needle = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    n_events = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    events = odds.fetch_events()
    print(f"{len(events)} upcoming events; requesting markets={MARKETS}")
    if not events:
        return

    for ev in events[:n_events]:
        print(f"\n=== EVENT {ev.get('away_team')} @ {ev.get('home_team')} "
              f"(commence {ev.get('commence_time')}) ===")
        try:
            ep = odds.fetch_event_props(ev["id"], MARKETS)
        except Exception as e:
            print(f"  fetch failed: {e}")
            continue
        for bk in ep.get("bookmakers", []):
            for m in bk.get("markets", []):
                our = odds._ODDS_TO_OURS.get(m["key"])
                printed = False
                for o in m.get("outcomes", []):
                    desc = (o.get("description") or "")
                    if needle and needle not in desc.lower():
                        continue
                    if not printed:
                        print(f"  [{bk['key']}] api_key={m['key']!r} -> our={our!r}")
                        printed = True
                    print(f"      name={o.get('name')!r:8} desc={desc!r:22} "
                          f"point={o.get('point')} price={o.get('price')}")

    print(f"\ncredits remaining: {odds.last_requests_remaining}")


if __name__ == "__main__":
    main()
