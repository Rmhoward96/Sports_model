"""Price pull for the 'take these' list: pull prices -> find picks -> store (.github/workflows/take-these-pull.yml).

Every 2 hours (12:00-04:00 UTC; cadence = repo variable TAKE_PULL_EVERY_HOURS), for each of the 8 leagues in order
(nfl, cfb, mlb, nhl, nba, wnba, epl, mls):

  1. A free Odds API /events call. If no event starts within 7 days the league is skipped and costs nothing.
  2. Credit floor: below 1,500 credits remaining (read from the free call's header) this and every later league is
     skipped with a warning. An unknown balance runs.
  3. ONE odds call with `bookmakers` = the 7 list books + pinnacle (8 keys = one billed region) and markets
     h2h, spreads, totals: 3 credits a league, 24 per full pull, about 12 pulls a day. Credit math: spec section 4.
  4. The prices are parsed and the pick engine (sportsmodel.take.picks) runs against Pinnacle's de-vigged fair.
  5. Writes, in this order: take_events (with game_pk for nfl/cfb/mlb), take_prices, take_picks (append-only),
     take_pick_state, then one take_api_calls row with the credits used and remaining. After the last league (skipped
     and failed leagues included) picks on games that have started are ended and raw prices older than 7 days pruned.

A failure in one league warns and moves on; nothing is retried (the next pull is hours away). The run exits 1 only when
every league it attempted failed. Missing tables: warn and exit 0. After each league it prints which of the 8 requested
bookmaker keys appeared and which did not (the go-live check; bet365 has no US key and is not sent).

    uv run python scripts/take_pull.py [--dry-run] [--scheduled] [--league nfl ...]

--dry-run still makes the paid odds calls but writes nothing: it prints the picks it would store, and reports any of the
8 sport keys the Odds API's free /sports list does not carry. SM_LEDGER_OFF also means a dry run.
--scheduled applies the TAKE_PULL_EVERY_HOURS cadence (a bad or empty value counts as 2): outside it, exit 0 with no calls.
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

import ingest_odds  # noqa: E402  (scripts/ingest_odds.py: schedule fetchers and event matchers)
from sportsmodel.ingest import odds  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402
from sportsmodel.take import constants as C  # noqa: E402
from sportsmodel.early.polling import parse_credits  # noqa: E402
from sportsmodel.take.parse import parse_events, parse_prices  # noqa: E402
from sportsmodel.take.picks import find_picks  # noqa: E402

MARKETS = [C.API_MARKET[m] for m in C.MARKETS]                         # h2h, spreads, totals
DEFAULT_COST = len(MARKETS)                                            # 3 credits when the cost header is missing


@dataclass
class Deps:
    """Every side effect of a run, injectable so the whole flow is testable without a network or a database."""
    missing_tables: Callable[[], list[str]]
    fetch_events: Callable[[Any], list[dict]]                          # the free /events call (cfg)
    fetch_odds: Callable[[Any, list[str], list[str]], list[dict]]      # (cfg, bookmakers, markets)
    fetch_sports: Callable[[], list[dict]]                             # free: the Odds API sport list
    remaining: Callable[[], Any]
    last_cost: Callable[[], Any]
    match_game_pks: Callable[[str, list[dict]], dict[str, int]]        # (league, event rows) -> {event_id: game_pk}
    upsert_events: Callable[[list[dict]], int]
    insert_prices: Callable[[list[dict]], int]
    insert_picks: Callable[[list[dict], str], int]
    sync_state: Callable[[str, list[str], list[dict], datetime], None]
    prune_prices: Callable[[datetime], int]
    log_call: Callable[..., None]
    expire_started: Callable[[datetime], int]                          # ends active picks whose game has started (all leagues)


def _warn(msg: str) -> None:
    print(f"::warning::take-pull: {msg}", flush=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def due(now: datetime, every_hours) -> bool:
    """Is this hour a pull hour? `every_hours` comes from a repo variable, so a bad or empty value counts as 2."""
    try:
        n = int(every_hours)
    except (TypeError, ValueError):
        n = 2
    if n < 1:
        n = 2
    return now.hour % n == 0


def match_game_pks(league: str, events: list[dict]) -> dict[str, int]:
    """event_id -> game_pk for the model leagues, using the same schedule fetchers and matchers as ingest_odds.
    Any failure (or a league without a model) gives {}: game_pk is optional."""
    if league not in C.MODEL_LEAGUES:
        return {}
    try:
        games = ingest_odds._ESPN_FETCHERS[league]()
        match = ingest_odds._matcher_for(league)
        out: dict[str, int] = {}
        for e in events:
            raw = {"home_team": e["home_team"], "away_team": e["away_team"],
                   "commence_time": e["commence_time"].isoformat()}          # the matchers read a raw API event
            pk = match(raw, games)
            if pk is not None:
                out[e["event_id"]] = int(pk)
        return out
    except Exception as exc:  # noqa: BLE001 -- unmatched events just have no game_pk
        _warn(f"{league}: game_pk matching failed: {type(exc).__name__}: {exc}")
        return {}


def _check_sport_keys(deps: Deps) -> None:
    try:
        have = {s.get("key") for s in deps.fetch_sports()}
    except Exception as exc:  # noqa: BLE001 -- informational only
        _warn(f"could not read the sport list: {type(exc).__name__}: {exc}")
        return
    unknown = [k for k in C.LEAGUES.values() if k not in have]
    if unknown:
        _warn(f"sport keys the Odds API does not list: {unknown}")


def _pull_league(league: str, now: datetime, deps: Deps, run_id: str, dry_run: bool, out: dict) -> str:
    """One league. Returns 'skipped', 'floor' (stop everything) or 'ok'."""
    cfg = C.league(league)
    horizon = now + timedelta(days=C.EVENTS_HORIZON_DAYS)
    upcoming = [e for e in parse_events(deps.fetch_events(cfg), league) if now < e["commence_time"] <= horizon]
    if not upcoming:
        out["skipped"].append(league)
        return "skipped"
    ok, why = C.budget_ok(deps.remaining(), C.PULL_FLOOR)
    if not ok:
        _warn(why)
        return "floor"
    response = deps.fetch_odds(cfg, list(C.BOOKMAKERS), MARKETS)
    parsed = parse_credits(deps.last_cost())
    cost = parsed if parsed is not None else DEFAULT_COST
    out["credits"] += cost                                             # count it the moment it is spent
    events = parse_events(response, league)
    prices = parse_prices(response, league, now)
    picks = find_picks(prices, events, now)
    seen = [b for b in C.BOOKMAKERS if any(r["book"] == b for r in prices)]
    missing = [b for b in C.BOOKMAKERS if b not in seen]
    print(f"[take-pull] {league} books seen: {seen}; missing: {missing}", flush=True)   # dry run too: the go-live check
    out["leagues"][league] = {"events": len(events), "prices": len(prices), "picks": len(picks), "credits": cost,
                              "books_missing": missing}
    if dry_run:
        out["picks"] += len(picks)
        out["would_write"] += picks
        return "ok"
    pks = deps.match_game_pks(league, events)
    deps.upsert_events([dict(e, game_pk=pks.get(e["event_id"])) for e in events])
    deps.insert_prices(prices)
    deps.insert_picks(picks, run_id)
    deps.sync_state(league, [e["event_id"] for e in events], picks, now)
    deps.log_call("pull", league, "odds", cost, parse_credits(deps.remaining()), missing, run_id)
    out["picks"] += len(picks)                                         # counted only once stored
    return "ok"


def run_pull(now: datetime, deps: Deps, *, run_id: str = "local", dry_run: bool = False,
             leagues: tuple[str, ...] = tuple(C.LEAGUES)) -> dict:
    """One pull of the given leagues. Returns {"status", "leagues", "credits", "picks", "skipped", ...}."""
    out: dict = {"status": "ok", "leagues": {}, "credits": 0, "picks": 0, "skipped": [], "would_write": []}
    gone = deps.missing_tables()
    if gone:
        _warn(f"table {gone[0]} missing — run db/migration_take_these.sql")
        out["status"] = "no_table"
        return out
    if dry_run:
        _check_sport_keys(deps)
    for i, league in enumerate(leagues):
        try:
            state = _pull_league(league, now, deps, run_id, dry_run, out)
        except Exception as exc:  # noqa: BLE001 -- a failed league is not retried; the next league and pull go on
            _warn(f"{league} pull failed: {type(exc).__name__}: {exc}")
            out["leagues"][league] = {"failed": f"{type(exc).__name__}: {exc}"[:200]}   # also after a paid call: a write failure is a failure
            continue
        if state == "floor":
            out["skipped"] += list(leagues[i:])
            break
    if not dry_run:
        try:
            out["expired"] = deps.expire_started(now)          # independent of the pull: skipped and floor-stopped leagues too
        except Exception as exc:  # noqa: BLE001 -- housekeeping must not fail the run
            _warn(f"expiring started picks failed: {type(exc).__name__}: {exc}")
    if not dry_run and out["leagues"]:
        try:
            out["pruned"] = deps.prune_prices(now - timedelta(days=C.PRICE_RETENTION_DAYS))
        except Exception as exc:  # noqa: BLE001 -- housekeeping must not fail the run
            _warn(f"pruning old prices failed: {type(exc).__name__}: {exc}")
    out["remaining"] = deps.remaining()
    return out


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db
    from sportsmodel.take import store

    timeouts = {"connect_timeout": db.LEDGER_CONNECT_TIMEOUT, "statement_timeout_ms": db.LEDGER_STATEMENT_TIMEOUT_MS}
    return Deps(
        missing_tables=lambda: store.missing_tables(**timeouts),
        fetch_events=odds.fetch_events, fetch_odds=odds.fetch_bookmaker_odds, fetch_sports=odds.fetch_sports,
        remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
        match_game_pks=match_game_pks,
        upsert_events=lambda rows: store.upsert_events(rows, **timeouts),
        insert_prices=lambda rows: store.insert_prices(rows, **timeouts),
        insert_picks=lambda picks, run_id: store.insert_picks(picks, run_id, **timeouts),
        sync_state=lambda league, ids, picks, at: store.sync_state(league, ids, picks, at, **timeouts),
        prune_prices=lambda before: store.prune_prices(before, **timeouts),
        log_call=lambda *a: store.log_call(*a, **timeouts),
        expire_started=lambda now: store.expire_started(now, **timeouts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="pull (paid) and print the picks it would store; write nothing")
    ap.add_argument("--scheduled", action="store_true", help="only run in an hour allowed by TAKE_PULL_EVERY_HOURS")
    ap.add_argument("--league", action="append", choices=list(C.LEAGUES), help="limit to this league (repeatable)")
    args = ap.parse_args(argv)
    now = _now()
    if args.scheduled and not due(now, os.getenv("TAKE_PULL_EVERY_HOURS")):
        print(f"take-pull: hour {now.hour} is not a pull hour for TAKE_PULL_EVERY_HOURS={os.getenv('TAKE_PULL_EVERY_HOURS')!r}; nothing to do",
              flush=True)
        return 0
    dry_run = args.dry_run or writes_disabled()
    for name in ("DATABASE_URL", "ODDS_API_KEY"):
        if not os.getenv(name):
            print(f"::error::take-pull: {name} is not set")
            return 1
    leagues = tuple(lg for lg in C.LEAGUES if not args.league or lg in args.league)      # keep LEAGUES order
    result = run_pull(now, _db_deps(), run_id=os.getenv("GITHUB_RUN_ID") or "local", dry_run=dry_run, leagues=leagues)
    for p in result.get("would_write", []):
        line = "" if p["line"] is None else f" {p['line']:+g}"
        print(f"[take-pull] DRY RUN would store: {p['league']} {p['event_id']} {p['market']} {p['side']} "
              f"{p['book']}{line} {p['price']:+d} ev={p['ev']:+.2%} good until {p['good_until']:+d}", flush=True)
    print(f"[take-pull] { {k: v for k, v in result.items() if k != 'would_write'} }", flush=True)
    attempted = [v for v in result["leagues"].values()]
    if attempted and all("failed" in v for v in attempted):
        print("::error::take-pull: every attempted league's pull failed (check ODDS_API_KEY and the books line)", flush=True)
        return 1                                       # a dead pull must turn the run red
    return 0


if __name__ == "__main__":
    sys.exit(main())
