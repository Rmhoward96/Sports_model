"""Capture a true closing price shortly before kickoff (pick-ledger step 1).

Run every ~10 minutes during game windows (.github/workflows/capture-close.yml). Each run:

  1. Idle check, DB only: no kickoff known within the next hour -> exit with ZERO network calls.
  2. For each sport with a nearby kickoff: fetch the live schedule (ESPN / StatsAPI, free), keep the games kicking
     off in WINDOW_MIN..WINDOW_MAX minutes that have no final (ok / empty) close_capture yet. None -> no Odds API call.
  3. Free Odds API events call: reads x-requests-remaining for the budget guard (< 600 skips props, < 200
     skips everything) and gives the event ids.
  4. Game lines: ONE call per sport (h2h / spreads / totals, regions us,eu) stored to odds_snapshot for the
     target games only. Props: one call per target event that holds >= 1 prop pick in the ledger, only for the
     markets those picks are in (regions CLOSE_PROP_REGIONS, default us,eu).
  5. One close_capture row per event and kind (ok / empty / failed). A failed attempt is retried by the next run
     while the event is still in the window; an 'ok' or 'empty' row is final (never replaced, never re-bought).

Credits: lines = 3 markets x 2 regions = 6 per sport-run; props = (markets with picks) x regions per event.

    uv run python scripts/capture_close.py [--dry-run] [--now 2026-10-18T16:40:00Z]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import ingest_odds  # noqa: E402  (scripts/ingest_odds.py: schedule fetchers, event matchers, game lookup)
from sportsmodel import sports  # noqa: E402
from sportsmodel.ingest import odds  # noqa: E402
from sportsmodel.ledger import close  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

SPORT_ORDER = ("nfl", "cfb", "mlb")
LINES_MARKETS = "h2h,spreads,totals"


@dataclass
class Deps:
    """Every side effect of a run, injectable so the whole flow is testable without a network or a database."""
    missing_tables: Callable[[], list[str]]
    load_candidates: Callable[[datetime, datetime], list[dict]]        # (now, horizon_end) -> {sport, game_pk, commence_time}
    load_captured: Callable[[list[int]], set]                           # -> {(sport, game_pk, kind)} of 'ok' rows
    load_prop_markets: Callable[[str, list[int]], dict]                 # (sport, pks) -> {game_pk: {ledger prop markets}}
    fetch_schedule: Callable[[str], list[dict]]
    match: Callable[[str], Callable]                                    # sport -> matcher(event, games) -> game_pk | None
    fetch_events: Callable[[Any], list[dict]]
    fetch_game_odds: Callable[[Any, str], list[dict]]
    fetch_event_props: Callable[[str, list, Any, str], dict]
    remaining: Callable[[], Any]
    last_cost: Callable[[], Any]
    store_snapshots: Callable[[list[dict]], int]
    store_captures: Callable[[list[dict]], int]


def _warn(msg: str) -> None:
    print(f"::warning::capture-close: {msg}", flush=True)


def _call_cost(deps: Deps, n_markets: int, regions: str) -> int:
    got = close.parse_credits(deps.last_cost())
    return got if got is not None else close.estimated_cost(n_markets, regions)


def _flush(deps: Deps, snapshot_rows: list[dict], capture_rows: list[dict], out: dict) -> None:
    """Persist what a single call produced, right now: snapshots first, then the capture rows (a capture row
    implies its prices are stored). Called after the lines call and after EACH props event, so an error or the job
    timeout later in the run never loses data that was already paid for."""
    if snapshot_rows:
        out["snapshots"] += deps.store_snapshots(snapshot_rows)
    if capture_rows:
        out["captures"] += deps.store_captures(capture_rows)
        out["rows"] += [(r["sport"], r["game_pk"], r["kind"], r["status"]) for r in capture_rows]


def _capture_sport(sport: str, now: datetime, deps: Deps, prop_regions: str, dry_run: bool, out: dict) -> None:
    cfg = sports.get(sport)
    schedule = deps.fetch_schedule(sport)
    in_window = [g for g in schedule if (m := close.minutes_to_kick(g.get("commence_time"), now)) is not None
                 and close.WINDOW_MIN <= m <= close.WINDOW_MAX]
    if not in_window:
        return
    pks = [g["game_pk"] for g in in_window]
    captured = deps.load_captured(pks)
    line_targets = close.select_targets(in_window, captured, sport, "lines", now)
    prop_markets = deps.load_prop_markets(sport, pks) if cfg.prop_market_map else {}
    prop_targets, unmapped = [], []
    for g in close.select_targets(in_window, captured, sport, "props", now):
        if not prop_markets.get(g["game_pk"]):
            continue
        (prop_targets if close.api_markets_for(sport, prop_markets[g["game_pk"]], cfg.prop_market_map)
         else unmapped).append(g)
    if not line_targets and not prop_targets and not unmapped:
        return

    if dry_run:
        out["planned"].append({"sport": sport, "lines": [g["game_pk"] for g in line_targets],
                               "props": [g["game_pk"] for g in prop_targets]})
        return

    captured_at = now.isoformat()
    if unmapped:
        # The ledger holds prop picks here but none in a market the Odds API serves: nothing to buy, so record the
        # event 'empty' (final) instead of retrying it every run. No network call, no credits.
        _flush(deps, [], [close.capture_row(sport=sport, game=g, event_id=None, kind="props", now=now, credits=0,
                                            status="empty", regions=prop_regions, markets=[]) for g in unmapped], out)
    if not line_targets and not prop_targets:
        return

    events = deps.fetch_events(cfg)                                    # free; also refreshes the credit header
    decision = close.budget_decision(close.parse_credits(deps.remaining()))
    if decision.warning:
        _warn(decision.warning)
    matcher = deps.match(sport)
    event_by_pk = {}
    for ev in events:
        gp = matcher(ev, schedule)
        if gp is not None:
            event_by_pk[gp] = ev

    if line_targets and decision.lines:
        regions = close.LINES_REGIONS
        target_pks = {g["game_pk"] for g in line_targets}
        snapshot_rows: list[dict] = []
        capture_rows: list[dict] = []
        try:
            line_events = deps.fetch_game_odds(cfg, regions)
            lookup = ingest_odds.build_game_lookup(line_events, schedule, matcher)
            rows = [r for r in odds.parse_game_odds(line_events, lookup, captured_at) if r["game_pk"] in target_pks]
            cost = _call_cost(deps, len(cfg.game_markets), regions)
            per_game = close.split_credits(cost, len(line_targets))
            snapshot_rows += rows
            have = {r["game_pk"] for r in rows}
            for g, credits in zip(line_targets, per_game):
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=(event_by_pk.get(g["game_pk"]) or {}).get("id"), kind="lines",
                    now=now, credits=credits, status="ok" if g["game_pk"] in have else "empty", regions=regions,
                    markets=LINES_MARKETS.split(",")))
            out["credits"] += cost
        except Exception as exc:  # noqa: BLE001 -- one sport's failed call is retried by the next run
            _warn(f"{sport} lines capture failed: {type(exc).__name__}: {exc}")
            snapshot_rows, capture_rows = [], []
            for g in line_targets:
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=(event_by_pk.get(g["game_pk"]) or {}).get("id"), kind="lines",
                    now=now, credits=0, status="failed", regions=regions, markets=LINES_MARKETS.split(",")))
        _flush(deps, snapshot_rows, capture_rows, out)
        decision = close.budget_decision(close.parse_credits(deps.remaining()))   # re-check after the spend

    if prop_targets and decision.props:
        for g in prop_targets:
            ev = event_by_pk.get(g["game_pk"])
            markets = close.api_markets_for(sport, prop_markets[g["game_pk"]], cfg.prop_market_map)
            snapshot_rows, capture_rows = [], []
            if ev is None:
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=None, kind="props", now=now, credits=0,
                    status="failed", regions=prop_regions, markets=markets))
                _warn(f"{sport} {g['game_pk']}: no Odds API event for the props capture")
            else:
                try:
                    payload = deps.fetch_event_props(ev["id"], markets, cfg, prop_regions)
                    rows = odds.parse_prop_odds(payload, g["game_pk"], captured_at, cfg)
                    cost = _call_cost(deps, len(markets), prop_regions)
                    snapshot_rows += rows
                    capture_rows.append(close.capture_row(
                        sport=sport, game=g, event_id=ev["id"], kind="props", now=now, credits=cost,
                        status="ok" if rows else "empty", regions=prop_regions, markets=markets))
                    out["credits"] += cost
                except Exception as exc:  # noqa: BLE001
                    _warn(f"{sport} {g['game_pk']} props capture failed: {type(exc).__name__}: {exc}")
                    snapshot_rows = []
                    capture_rows = [close.capture_row(
                        sport=sport, game=g, event_id=ev.get("id"), kind="props", now=now, credits=0,
                        status="failed", regions=prop_regions, markets=markets)]
            _flush(deps, snapshot_rows, capture_rows, out)           # per event: a later error loses nothing
            if not close.budget_decision(close.parse_credits(deps.remaining())).props:
                _warn("credit floor reached mid-run; remaining prop captures skipped")
                break


def run_capture(now: datetime, deps: Deps, *, prop_regions: str = close.DEFAULT_PROP_REGIONS,
                dry_run: bool = False) -> dict:
    """One run. Returns a summary dict; status 'idle' means nothing was fetched from any network."""
    gone = deps.missing_tables()
    if gone:
        _warn(f"table {gone[0]} missing — run db/migration_pick_ledger.sql")
        return {"status": "no_table"}
    candidates = deps.load_candidates(now, now + timedelta(minutes=close.PREFILTER_MIN))
    live = close.prefilter_sports(candidates, now)
    if not live:
        return {"status": "idle"}
    out: dict = {"status": "ok", "credits": 0, "snapshots": 0, "captures": 0, "rows": [], "planned": []}
    for sport in live:
        try:
            _capture_sport(sport, now, deps, prop_regions, dry_run, out)
        except Exception as exc:  # noqa: BLE001 -- schedule / matching trouble in one sport must not stop the others
            _warn(f"{sport} failed: {type(exc).__name__}: {exc}")
    return out


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db

    def q(sql, params):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def candidates(now, end):
        # predictions_current covers model games; pick_ledger covers games only a market pick touched.
        rows = q("SELECT sport, game_pk, commence_time FROM predictions_current "
                 "WHERE commence_time > %s AND commence_time <= %s "
                 "UNION "
                 "SELECT DISTINCT sport, game_pk, commence_time FROM pick_ledger "
                 "WHERE commence_time > %s AND commence_time <= %s", (now, end, now, end))
        return [{"sport": s, "game_pk": pk, "commence_time": c} for s, pk, c in rows]

    def captured(pks):
        rows = q("SELECT sport, game_pk, kind FROM close_capture WHERE status IN ('ok', 'empty') "
                 "AND game_pk = ANY(%s)", (pks,))
        return {(s, pk, k) for s, pk, k in rows}

    def prop_markets(sport, pks):
        rows = q("SELECT DISTINCT game_pk, market FROM pick_ledger WHERE sport = %s AND game_pk = ANY(%s) "
                 "AND player_id IS NOT NULL AND source IN ('ev_prop', 'parlay_leg')", (sport, pks))
        out: dict = {}
        for pk, market in rows:
            out.setdefault(pk, set()).add(market)
        return out

    return Deps(
        missing_tables=lambda: db.missing_ledger_tables(["pick_ledger", "close_capture"]),
        load_candidates=candidates, load_captured=captured, load_prop_markets=prop_markets,
        fetch_schedule=lambda sport: ingest_odds._ESPN_FETCHERS[sport](),
        match=ingest_odds._matcher_for,
        fetch_events=odds.fetch_events, fetch_game_odds=odds.fetch_game_odds,
        fetch_event_props=lambda eid, markets, cfg, regions: odds.fetch_event_props(eid, markets, cfg, regions=regions),
        remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
        store_snapshots=db.upsert_odds_snapshot, store_captures=db.upsert_close_capture)


def _parse_now(value: str) -> datetime:
    """--now as an aware UTC datetime; a naive value (no offset / Z) is read as UTC."""
    dt = odds.parse_commence(value)
    if dt is None:
        raise SystemExit(f"capture-close: cannot parse --now {value!r}")
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="select targets and print the plan; no Odds API call, no writes")
    ap.add_argument("--now", default=None, help="as-of time (ISO, default: now UTC)")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("capture-close: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::capture-close: DATABASE_URL is not set")
        return 1
    now = _parse_now(args.now) if args.now else datetime.now(timezone.utc)
    result = run_capture(now, _db_deps(), prop_regions=close.prop_regions(os.environ), dry_run=args.dry_run)
    print(f"[capture-close] {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
