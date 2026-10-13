"""Pre-kick close capture: window selection, credit budget guard and prop-market planning. PURE -- no IO.

Window. A run captures the events that kick off in WINDOW_MIN..WINDOW_MAX minutes and have no final capture yet.
The spec's outer bound was 35 minutes; it is 25 here because the weekly gate asks for a close within 30 minutes of
kickoff, and the first run to see an event at 31-35 minutes would fail it. With a 10-minute cadence a 20-minute
window still holds two runs, so one skipped or late GitHub cron firing loses nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..ingest.odds import parse_commence
from ..serving.props_ev import MLB_SIM_TO_ODDS_MARKET, SIM_TO_ODDS_MARKET

WINDOW_MIN = 5
WINDOW_MAX = 25
PREFILTER_MIN = 60               # DB look-ahead that decides whether a run is idle (no network at all)
# The close capture is the highest-priority use of Odds API credits, so its floors sit BELOW the regular
# ingest's (odds.INGEST_PROPS_FLOOR=2500, odds.INGEST_LINES_FLOOR=800): the regular pulls run out first.
BUDGET_PROPS_FLOOR = 600         # remaining credits below this: skip prop captures
BUDGET_ALL_FLOOR = 200           # remaining credits below this: skip everything
LINES_REGIONS = "us,eu"
DEFAULT_PROP_REGIONS = "us,eu"
SIM_TO_ODDS_BY_SPORT = {"nfl": SIM_TO_ODDS_MARKET, "mlb": MLB_SIM_TO_ODDS_MARKET}


def minutes_to_kick(commence, now: datetime) -> float | None:
    """Minutes from `now` to `commence` (ISO string or datetime); None when unparseable."""
    c = parse_commence(commence) if isinstance(commence, str) else commence
    if c is None:
        return None
    return (c - now).total_seconds() / 60.0


def prefilter_sports(candidates: list[dict], now: datetime, horizon_min: int = PREFILTER_MIN) -> list[str]:
    """Sports (nfl, cfb, mlb order) with at least one DB-known kickoff in (now, now + horizon]. None -> the run
    is idle and must exit before any network call."""
    live = set()
    for c in candidates:
        m = minutes_to_kick(c.get("commence_time"), now)
        if m is not None and 0 < m <= horizon_min:
            live.add(c.get("sport"))
    return [s for s in ("nfl", "cfb", "mlb") if s in live]


def select_targets(games: list[dict], captured: set[tuple], sport: str, kind: str, now: datetime,
                   lo: float = WINDOW_MIN, hi: float = WINDOW_MAX) -> list[dict]:
    """Schedule rows (game_pk, commence_time, ...) kicking off in [lo, hi] minutes with no final capture of
    `kind` yet. `captured` is a set of (sport, game_pk, kind) for the rows already stored as 'ok' or 'empty'
    (a 'failed' attempt is not final: the next run retries it)."""
    out = []
    for g in games:
        m = minutes_to_kick(g.get("commence_time"), now)
        if m is None or not (lo <= m <= hi):
            continue
        if (sport, g["game_pk"], kind) in captured:
            continue
        out.append({**g, "minutes_to_kick": m})
    return out


def prop_regions(env: dict) -> str:
    """Regions for prop captures: CLOSE_PROP_REGIONS (repo variable), default 'us,eu'. Set it to 'us' once the
    Pinnacle probe says Pinnacle publishes no props -- that halves the prop credits."""
    raw = (env.get("CLOSE_PROP_REGIONS") or "").strip().lower().replace(" ", "")
    parts = [p for p in raw.split(",") if p in ("us", "eu")]
    return ",".join(parts) if parts else DEFAULT_PROP_REGIONS


def n_regions(regions: str) -> int:
    return len([r for r in regions.split(",") if r])


def estimated_cost(n_markets: int, regions: str) -> int:
    """The Odds API bills markets x regions per call."""
    return n_markets * n_regions(regions)


def parse_credits(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def split_credits(total: int, n: int) -> list[int]:
    """Spread a call's cost over `n` event rows so the rows sum to the cost exactly."""
    if n <= 0:
        return []
    base, extra = divmod(max(int(total), 0), n)
    return [base + (1 if i < extra else 0) for i in range(n)]


@dataclass(frozen=True)
class BudgetDecision:
    lines: bool
    props: bool
    warning: str | None = None


def budget_decision(remaining: int | None) -> BudgetDecision:
    """< 200 credits remaining: skip everything; < 600: skip props, keep game lines; otherwise capture all.
    An unreadable header fails open (the capture is the point) but says so."""
    if remaining is None:
        return BudgetDecision(True, True, "x-requests-remaining unavailable; capturing without a budget check")
    if remaining < BUDGET_ALL_FLOOR:
        return BudgetDecision(False, False, f"{remaining} Odds API credits left (< {BUDGET_ALL_FLOOR}): "
                                            "skipping ALL close captures")
    if remaining < BUDGET_PROPS_FLOOR:
        return BudgetDecision(True, False, f"{remaining} Odds API credits left (< {BUDGET_PROPS_FLOOR}): "
                                           "skipping prop close captures")
    return BudgetDecision(True, True, None)


def api_markets_for(sport: str, ledger_markets: set[str], prop_market_map: dict[str, str]) -> list[str]:
    """Odds API market keys to request for one event: only the markets the ledger actually holds picks in
    (ledger markets are the writers' sim codes -> odds-side code -> API key), sorted, unknown ones dropped."""
    to_odds = SIM_TO_ODDS_BY_SPORT.get(sport, {})
    keys = {prop_market_map[to_odds[m]] for m in ledger_markets if m in to_odds and to_odds[m] in prop_market_map}
    return sorted(keys)


def capture_row(*, sport: str, game: dict, event_id: str | None, kind: str, now: datetime, credits: int,
                status: str, regions: str, markets: list[str]) -> dict:
    """One close_capture row (columns of db.LEDGER_COLUMNS['close_capture'])."""
    m = minutes_to_kick(game["commence_time"], now)
    return {"sport": sport, "game_pk": game["game_pk"], "event_id": event_id, "kind": kind,
            "captured_at": now.isoformat(), "commence_time": game["commence_time"],
            "minutes_to_kick": m, "credits_used": int(credits), "status": status, "regions": regions,
            "markets": ",".join(markets)}
