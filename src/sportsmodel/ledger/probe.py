"""Does Pinnacle publish player props through The Odds API? Pure verdict over ONE event-odds payload
(`GET /v4/sports/{sport}/events/{id}/odds?regions=eu&markets=<prop markets>`)."""
from __future__ import annotations

PINNACLE = "pinnacle"


def pinnacle_props_verdict(payload: dict | None) -> dict:
    """Classify one event-odds payload.

      pinnacle_has_props  Pinnacle returned at least one prop outcome -- keep regions=us,eu for prop captures.
      pinnacle_absent     other books returned props but Pinnacle returned none -- conclusive: prop captures
                          use regions=us only (half the credits).
      inconclusive        NO book returned any prop (props not posted yet, or the event is too far out) --
                          re-run closer to kickoff before deciding.
    """
    books: dict[str, int] = {}
    pin_markets: list[str] = []
    for bk in (payload or {}).get("bookmakers") or []:
        n = 0
        for m in bk.get("markets") or []:
            k = len(m.get("outcomes") or [])
            n += k
            if bk.get("key") == PINNACLE and k:
                pin_markets.append(m.get("key"))
        books[bk.get("key")] = books.get(bk.get("key"), 0) + n
    pin_outcomes = books.get(PINNACLE, 0)
    if pin_outcomes > 0:
        verdict = "pinnacle_has_props"
    elif any(n > 0 for n in books.values()):
        verdict = "pinnacle_absent"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "pinnacle_outcomes": pin_outcomes, "pinnacle_markets": sorted(set(pin_markets)),
            "bookmaker_outcomes": dict(sorted(books.items()))}
