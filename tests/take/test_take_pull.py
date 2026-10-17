"""scripts/take_pull.py: flow, credit floor, write order, dry run, failure handling. Fake Deps only (no network, no DB)."""
import copy
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sportsmodel.take import constants

_SPEC = importlib.util.spec_from_file_location(
    "take_pull", Path(__file__).resolve().parents[2] / "scripts" / "take_pull.py")
take_pull = importlib.util.module_from_spec(_SPEC)
sys.modules["take_pull"] = take_pull        # @dataclass resolves its module through sys.modules
_SPEC.loader.exec_module(take_pull)
run_pull, due, Deps = take_pull.run_pull, take_pull.due, take_pull.Deps

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)          # the fixture's Pinnacle prices are 5 minutes old
NFL = json.loads((FIXTURES / "odds_nfl.json").read_text())
EVENT = {"id": "evt_nfl_1", "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills",
         "commence_time": "2026-11-02T18:00:00Z"}


def with_edge():
    """The fixture with a FanDuel away moneyline far above Pinnacle's: one pick."""
    data = copy.deepcopy(NFL)
    fd = next(b for b in data[0]["bookmakers"] if b["key"] == "fanduel")
    h2h = next(m for m in fd["markets"] if m["key"] == "h2h")
    for o in h2h["outcomes"]:
        if o["name"] == "Buffalo Bills":
            o["price"] = 200
    return data


class Rec:
    """Fake Deps that record every call in order."""

    def __init__(self, *, events=None, odds=None, remaining="5000", cost="3", tables=(), sports=None,
                 pks=None, fail_odds=()):
        self.order, self.args, self.writes = [], {}, {}
        self.events = events if events is not None else {lg: [EVENT] for lg in constants.LEAGUES}
        self.odds = odds if odds is not None else {lg: NFL for lg in constants.LEAGUES}
        self.remaining_value, self.cost, self.tables, self.fail_odds = remaining, cost, list(tables), set(fail_odds)
        self.sports = sports if sports is not None else [{"key": k} for k in constants.LEAGUES.values()]
        self.pks = pks or {}

    def n(self, name):
        return sum(1 for o in self.order if o == name)

    def _note(self, name, *a, **k):
        self.order.append(name)
        self.args.setdefault(name, []).append((a, k))

    def deps(self) -> Deps:
        def fetch_events(cfg):
            self._note("fetch_events", cfg)
            return self.events[cfg.key]

        def fetch_odds(cfg, books, markets):
            self._note("fetch_odds", cfg, books, markets)
            if cfg.key in self.fail_odds:
                raise RuntimeError("boom")
            return self.odds[cfg.key]

        def fetch_sports():
            self._note("fetch_sports")
            if isinstance(self.sports, Exception):
                raise self.sports
            return self.sports

        def match(league, events):
            self._note("match_game_pks", league, events)
            return self.pks.get(league, {})

        def writer(name):
            def write(*a, **k):
                self._note(name, *a, **k)
                return len(a[0]) if a and isinstance(a[0], list) else 0
            return write

        return Deps(
            missing_tables=lambda: self.tables, fetch_events=fetch_events, fetch_odds=fetch_odds,
            fetch_sports=fetch_sports, remaining=lambda: self.remaining_value, last_cost=lambda: self.cost,
            match_game_pks=match, upsert_events=writer("upsert_events"), insert_prices=writer("insert_prices"),
            insert_picks=writer("insert_picks"), sync_state=writer("sync_state"), prune_prices=writer("prune_prices"),
            log_call=writer("log_call"))


def test_pull_skips_league_with_no_events_and_spends_nothing():
    f = Rec(events={"nfl": []})
    out = run_pull(NOW, f.deps(), leagues=("nfl",))
    assert out["skipped"] == ["nfl"] and out["credits"] == 0 and f.n("fetch_odds") == 0


def test_pull_skips_league_whose_events_are_all_past_the_horizon():
    late = {**EVENT, "commence_time": (NOW + timedelta(days=constants.EVENTS_HORIZON_DAYS, minutes=1)).isoformat()}
    f = Rec(events={"nfl": [late]})
    out = run_pull(NOW, f.deps(), leagues=("nfl",))
    assert out["skipped"] == ["nfl"] and f.n("fetch_odds") == 0


def test_pull_requests_eight_keys_three_markets():
    f = Rec()
    run_pull(NOW, f.deps(), leagues=("nfl",))
    (cfg, books, markets), _ = f.args["fetch_odds"][0]
    assert cfg.odds_sport == "americanfootball_nfl" and len(books) == 8 and markets == ["h2h", "spreads", "totals"]
    assert books == list(constants.BOOKMAKERS) and books[-1] == "pinnacle"


def test_pull_stops_below_floor():
    f = Rec(remaining="1499")
    out = run_pull(NOW, f.deps(), leagues=("nfl", "nba"))
    assert f.n("fetch_odds") == 0 and out["credits"] == 0 and out["skipped"] == ["nfl", "nba"]
    assert f.n("fetch_events") == 1          # the free events call is what refreshes the balance; nba is never touched


def test_pull_runs_at_exactly_the_floor_and_when_balance_unknown():
    for remaining in ("1500", None):
        f = Rec(remaining=remaining)
        run_pull(NOW, f.deps(), leagues=("nfl",))
        assert f.n("fetch_odds") == 1


def test_pull_writes_in_order_picks_state_and_logs_credits():
    f = Rec(odds={"nfl": with_edge()}, pks={"nfl": {"evt_nfl_1": 777}}, remaining="4997")
    out = run_pull(NOW, f.deps(), run_id="run9", leagues=("nfl",))
    assert out["credits"] == 3 and out["picks"] == 1
    seq = [o for o in f.order if o in ("upsert_events", "insert_prices", "insert_picks", "sync_state", "log_call",
                                       "prune_prices")]
    assert seq == ["upsert_events", "insert_prices", "insert_picks", "sync_state", "log_call", "prune_prices"]
    (events,), _ = f.args["upsert_events"][0]
    assert events[0]["event_id"] == "evt_nfl_1" and events[0]["game_pk"] == 777
    (picks, run_id), _ = f.args["insert_picks"][0]
    assert len(picks) == 1 and picks[0]["book"] == "fanduel" and run_id == "run9"
    (lg, event_ids, state_picks, pulled_at), _ = f.args["sync_state"][0]
    assert (lg, event_ids, pulled_at) == ("nfl", ["evt_nfl_1"], NOW) and state_picks == picks
    (job, lg, endpoint, used, left, missing, rid), _ = f.args["log_call"][0]
    assert (job, lg, used, rid) == ("pull", "nfl", 3, "run9") and left == 4997
    assert missing == ["betmgm", "williamhill_us", "fanatics", "hardrockbet", "ballybet"]
    (cutoff,), _ = f.args["prune_prices"][0]
    assert cutoff == NOW - timedelta(days=constants.PRICE_RETENTION_DAYS)


def test_pull_event_without_a_matched_game_gets_game_pk_none():
    f = Rec()
    run_pull(NOW, f.deps(), leagues=("nfl",))
    (events,), _ = f.args["upsert_events"][0]
    assert events[0]["game_pk"] is None


def test_credits_are_the_sum_of_each_leagues_last_cost():
    f = Rec(cost="3")
    out = run_pull(NOW, f.deps(), leagues=("nfl", "nba", "nhl"))
    assert out["credits"] == 9 and f.n("fetch_odds") == 3
    assert [a[0][1] for a in f.args["log_call"]] == ["nfl", "nba", "nhl"]


def test_pull_dry_run_writes_nothing():
    f = Rec(odds={"nfl": with_edge()})
    out = run_pull(NOW, f.deps(), dry_run=True, leagues=("nfl",))
    for w in ("upsert_events", "insert_prices", "insert_picks", "sync_state", "log_call", "prune_prices"):
        assert f.n(w) == 0, w
    assert out["picks"] == 1 and len(out["would_write"]) == 1 and f.n("fetch_odds") == 1


def test_dry_run_reports_unknown_sport_keys(capsys):
    listed = [{"key": k} for k in constants.LEAGUES.values() if k not in ("basketball_wnba", "soccer_usa_mls")]
    run_pull(NOW, Rec(sports=listed).deps(), dry_run=True, leagues=())
    out = capsys.readouterr().out
    assert "::warning::take-pull:" in out and "basketball_wnba" in out and "soccer_usa_mls" in out
    assert "americanfootball_nfl" not in out


def test_dry_run_with_every_sport_key_listed_is_quiet_and_a_failed_lookup_only_warns(capsys):
    f = Rec()
    run_pull(NOW, f.deps(), dry_run=True, leagues=())
    assert "::warning" not in capsys.readouterr().out and f.n("fetch_sports") == 1
    run_pull(NOW, Rec(sports=RuntimeError("down")).deps(), dry_run=True, leagues=("nfl",))
    assert "sport list" in capsys.readouterr().out


def test_real_run_never_calls_fetch_sports():
    f = Rec()
    run_pull(NOW, f.deps(), leagues=("nfl",))
    assert f.n("fetch_sports") == 0


def test_books_seen_and_missing_are_printed_even_in_a_dry_run(capsys):
    run_pull(NOW, Rec().deps(), dry_run=True, leagues=("nfl",))
    out = capsys.readouterr().out
    assert "nfl books seen: ['draftkings', 'fanduel', 'pinnacle']" in out and "missing: ['betmgm'" in out


def test_one_failing_league_warns_and_the_rest_go_on(capsys):
    f = Rec(fail_odds={"nfl"})
    out = run_pull(NOW, f.deps(), leagues=("nfl", "nba"))
    assert "failed" in out["leagues"]["nfl"] and "events" in out["leagues"]["nba"]
    assert "::warning::take-pull: nfl pull failed" in capsys.readouterr().out
    assert f.n("log_call") == 1 and f.n("prune_prices") == 1


def test_failed_league_still_counts_the_credits_it_spent():
    f = Rec(fail_odds={"nfl"})
    assert run_pull(NOW, f.deps(), leagues=("nfl",))["credits"] == 0      # the call raised: nothing was billed to us


def test_missing_tables_warn_and_do_nothing(capsys):
    f = Rec(tables=["take_events"])
    out = run_pull(NOW, f.deps(), leagues=("nfl",))
    assert out["status"] == "no_table" and f.order == []
    assert "migration_take_these.sql" in capsys.readouterr().out


def test_main_exit_codes(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "x")
    monkeypatch.setenv("ODDS_API_KEY", "y")
    monkeypatch.setattr(take_pull, "_now", lambda: NOW)
    monkeypatch.setattr(take_pull, "_db_deps", lambda: Rec(fail_odds=set(constants.LEAGUES)).deps())
    assert take_pull.main(["--dry-run"]) == 1                              # every attempted league failed
    monkeypatch.setattr(take_pull, "_db_deps", lambda: Rec().deps())
    assert take_pull.main(["--dry-run", "--league", "nfl"]) == 0
    monkeypatch.setattr(take_pull, "_db_deps", lambda: Rec(tables=["take_events"]).deps())
    assert take_pull.main([]) == 0                                        # missing tables: warn, exit 0


def test_main_scheduled_respects_the_cadence(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "x")
    monkeypatch.setenv("ODDS_API_KEY", "y")
    f = Rec()
    monkeypatch.setattr(take_pull, "_db_deps", lambda: f.deps())
    monkeypatch.setenv("TAKE_PULL_EVERY_HOURS", "2")
    monkeypatch.setattr(take_pull, "_now", lambda: datetime(2026, 11, 1, 13, tzinfo=UTC))
    assert take_pull.main(["--scheduled"]) == 0 and f.order == []
    monkeypatch.setattr(take_pull, "_now", lambda: datetime(2026, 11, 1, 14, tzinfo=UTC))   # EVENT starts Nov 2: in range
    assert take_pull.main(["--scheduled", "--dry-run", "--league", "nfl"]) == 0 and f.n("fetch_odds") == 1


def test_due():
    assert due(datetime(2026, 11, 1, 14, tzinfo=UTC), 2) and not due(datetime(2026, 11, 1, 13, tzinfo=UTC), 2)
    assert due(datetime(2026, 11, 1, 14, tzinfo=UTC), "") is True
    assert due(datetime(2026, 11, 1, 14, tzinfo=UTC), "abc") is True and not due(datetime(2026, 11, 1, 13, tzinfo=UTC), None)
    assert due(datetime(2026, 11, 1, 12, tzinfo=UTC), "4") and not due(datetime(2026, 11, 1, 14, tzinfo=UTC), "4")
    assert due(datetime(2026, 11, 1, 13, tzinfo=UTC), 1) and due(datetime(2026, 11, 1, 13, tzinfo=UTC), 0) is False


# ---- match_game_pks (R3) ----------------------------------------------------------------------

EV_ROW = {"event_id": "evt_nfl_1", "league": "nfl", "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills",
          "commence_time": datetime(2026, 11, 2, 18, 0, tzinfo=UTC)}


def test_match_game_pks_uses_the_leagues_schedule_and_matcher(monkeypatch):
    seen = []
    monkeypatch.setitem(take_pull.ingest_odds._ESPN_FETCHERS, "nfl", lambda: seen.append("fetch") or ["g"])

    def matcher(ev, games):
        seen.append((ev, games))
        return 42

    monkeypatch.setattr(take_pull.ingest_odds, "_matcher_for", lambda lg: matcher)
    assert take_pull.match_game_pks("nfl", [EV_ROW]) == {"evt_nfl_1": 42}
    assert seen[0] == "fetch" and len(seen) == 2          # one schedule fetch per league
    ev, games = seen[1]
    assert ev == {"home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills",
                  "commence_time": "2026-11-02T18:00:00+00:00"} and games == ["g"]


def test_match_game_pks_with_the_real_nfl_matcher(monkeypatch):
    monkeypatch.setitem(take_pull.ingest_odds._ESPN_FETCHERS, "nfl", lambda: [
        {"game_pk": "9001", "home_name": "Kansas City Chiefs", "away_name": "Buffalo Bills",
         "commence_time": "2026-11-02T18:00:00Z"}])
    assert take_pull.match_game_pks("nfl", [EV_ROW]) == {"evt_nfl_1": 9001}


def test_match_game_pks_drops_unmatched_and_returns_empty_on_failure(monkeypatch):
    monkeypatch.setitem(take_pull.ingest_odds._ESPN_FETCHERS, "nfl", lambda: [])
    assert take_pull.match_game_pks("nfl", [EV_ROW]) == {}

    def boom():
        raise RuntimeError("espn down")

    monkeypatch.setitem(take_pull.ingest_odds._ESPN_FETCHERS, "nfl", boom)
    assert take_pull.match_game_pks("nfl", [EV_ROW]) == {}


def test_match_game_pks_does_not_fetch_for_non_model_leagues(monkeypatch):
    monkeypatch.setattr(take_pull.ingest_odds, "_ESPN_FETCHERS", {})
    assert take_pull.match_game_pks("nba", [{**EV_ROW, "league": "nba"}]) == {}


def test_run_pull_asks_for_game_pks_only_for_the_events_it_upserts():
    f = Rec(pks={"nfl": {"evt_nfl_1": 5}})
    run_pull(NOW, f.deps(), leagues=("nfl", "nba"))
    assert [a[0][0] for a in f.args["match_game_pks"]] == ["nfl", "nba"]       # the dep itself decides per league
