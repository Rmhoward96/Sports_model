"""scripts/take_close.py: grade first, then capture Pinnacle closes for picks about to start. Fake Deps only."""
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sportsmodel.take import constants

_SPEC = importlib.util.spec_from_file_location(
    "take_close", Path(__file__).resolve().parents[2] / "scripts" / "take_close.py")
take_close = importlib.util.module_from_spec(_SPEC)
sys.modules["take_close"] = take_close          # @dataclass resolves its module through sys.modules
_SPEC.loader.exec_module(take_close)
run_close, Deps = take_close.run_close, take_close.Deps

NOW = datetime(2026, 11, 2, 17, 50, tzinfo=UTC)
START = NOW + timedelta(minutes=10)                                     # inside the 5-20 minute target window


def pin_event(event_id="e1", start=START, total=47.5):
    return {"id": event_id, "home_team": "Home", "away_team": "Away",
            "commence_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bookmakers": [{"key": "pinnacle", "markets": [
                {"key": "totals", "last_update": "2026-11-02T17:49:00Z", "outcomes": [
                    {"name": "Over", "price": -108, "point": total}, {"name": "Under", "price": -112, "point": total}]},
                {"key": "h2h", "last_update": "2026-11-02T17:49:00Z", "outcomes": [
                    {"name": "Home", "price": -150}, {"name": "Away", "price": 130}]}]}]}


def started_pick(**over):
    close = {"event_id": "e1", "market": "total", "captured_at": NOW - timedelta(minutes=20), "minutes_to_start": 12.0,
             "prices": {"over": {"line": 47.5, "price": -108}, "under": {"line": 47.5, "price": -112}}}
    return {"pick_id": 7, "event_id": "e1", "league": "nhl", "market": "total", "side": "over", "line": 47.5,
            "price": 105, "ref_line": 47.5, "close": close, **over}


class Rec:
    def __init__(self, *, remaining="5000", cost="1", tables=(), ungraded=(), targets=None, odds=None, fail=()):
        self.order, self.args = [], {}
        self.results, self.closes, self.events_up = [], [], []
        self.remaining_value, self.cost, self.tables = remaining, cost, list(tables)
        self.ungraded, self.targets = list(ungraded), targets if targets is not None else {}
        self.odds, self.fail = odds if odds is not None else {}, set(fail)

    def n(self, name):
        return self.order.count(name)

    def _note(self, name, *a):
        self.order.append(name)
        self.args.setdefault(name, []).append(a)

    def deps(self) -> Deps:
        def fetch_odds(cfg, books, markets):
            self._note("fetch_odds", cfg, books, markets)
            if cfg.key in self.fail:
                raise RuntimeError("boom")
            return self.odds.get(cfg.key, [])

        def insert_results(rows):
            self._note("insert_results", rows)
            self.results += rows
            return len(rows)

        def upsert_closes(rows):
            self._note("upsert_closes", rows)
            self.closes += rows
            return len(rows)

        def upsert_events(rows):
            self._note("upsert_events", rows)
            self.events_up += rows
            return len(rows)

        return Deps(
            missing_tables=lambda: self.tables, prime=lambda: self._note("prime") or [],
            fetch_odds=fetch_odds, remaining=lambda: self.remaining_value, last_cost=lambda: self.cost,
            ungraded_started=lambda now: self._note("ungraded_started", now) or self.ungraded,
            insert_results=insert_results, close_targets=lambda now, lo, hi: self._note("close_targets", now, lo, hi)
            or self.targets, upsert_closes=upsert_closes, upsert_events=upsert_events,
            log_call=lambda *a: self._note("log_call", *a))


def test_close_grades_started_picks_even_when_broke():
    f = Rec(remaining="10", ungraded=[started_pick()], targets={"nhl": {"total"}})
    out = run_close(NOW, f.deps())
    assert len(f.results) == 1 and f.n("fetch_odds") == 0 and out["graded"] == 1
    row = f.results[0]
    assert row["pick_id"] == 7 and row["graded_by"] == "clv" and row["graded_at"] == NOW and row["beat"] is True


def test_close_grades_a_pick_with_no_close_as_no_close():
    f = Rec(ungraded=[started_pick(close=None)])
    run_close(NOW, f.deps())
    assert f.results[0]["graded_by"] == "no_close" and f.n("close_targets") == 1


def test_close_does_nothing_more_when_no_targets():
    f = Rec(targets={})
    out = run_close(NOW, f.deps())
    assert f.n("prime") == 0 and f.n("fetch_odds") == 0 and out["credits"] == 0
    assert f.args["close_targets"][0] == (NOW, constants.CLOSE_LO_MIN, constants.CLOSE_HI_MIN)


def test_close_requests_only_needed_markets():
    f = Rec(targets={"nhl": {"total"}, "nba": {"spread", "ml"}}, odds={})
    run_close(NOW, f.deps())
    (cfg, books, markets), (cfg2, books2, markets2) = f.args["fetch_odds"]
    assert cfg.odds_sport == "icehockey_nhl" and books == ["pinnacle"] and markets == ["totals"]
    assert cfg2.odds_sport == "basketball_nba" and books2 == ["pinnacle"] and markets2 == ["h2h", "spreads"]
    assert f.n("prime") == 1


def test_close_stores_a_valid_close_and_logs_credits():
    f = Rec(targets={"nhl": {"total"}}, odds={"nhl": [pin_event()]}, remaining="4990", cost="1")
    out = run_close(NOW, f.deps(), run_id="run3")
    (row,) = f.closes
    assert row == {"event_id": "e1", "market": "total", "captured_at": NOW, "minutes_to_start": 10.0,
                   "prices": {"over": {"line": 47.5, "price": -108}, "under": {"line": 47.5, "price": -112}}}
    assert out["credits"] == 1 and out["closes"] == 1
    assert f.args["log_call"][0] == ("close", "nhl", "odds", 1, 4990, [], "run3")
    assert f.order.index("upsert_events") < f.order.index("upsert_closes") < f.order.index("log_call")


def test_close_only_stores_the_target_markets():
    f = Rec(targets={"nhl": {"total"}}, odds={"nhl": [pin_event()]})
    run_close(NOW, f.deps())
    assert {c["market"] for c in f.closes} == {"total"}


def test_close_uses_response_commence_time():
    late = NOW + timedelta(minutes=70)                                  # stored start 17:55 (target); API now says 19:00
    f = Rec(targets={"nhl": {"total"}}, odds={"nhl": [pin_event(start=late)]})
    run_close(NOW, f.deps())
    assert f.closes == []
    assert [e["commence_time"] for e in f.events_up] == [late]
    assert f.events_up[0]["event_id"] == "e1" and f.events_up[0]["league"] == "nhl"


def test_close_window_is_zero_to_thirty_minutes():
    for minutes, stored in ((0, True), (30, True), (30.5, False), (-1, False)):
        f = Rec(targets={"nhl": {"total"}}, odds={"nhl": [pin_event(start=NOW + timedelta(minutes=minutes))]})
        run_close(NOW, f.deps())
        assert bool(f.closes) is stored, minutes


def test_close_stops_below_the_close_floor_but_still_grades():
    f = Rec(remaining="299", ungraded=[started_pick()], targets={"nhl": {"total"}})
    out = run_close(NOW, f.deps())
    assert f.n("fetch_odds") == 0 and len(f.results) == 1 and out["skipped"] == ["nhl"]
    f = Rec(remaining="300", targets={"nhl": {"total"}})
    run_close(NOW, f.deps())
    assert f.n("fetch_odds") == 1


def test_close_dry_run_writes_nothing(capsys):
    f = Rec(ungraded=[started_pick()], targets={"nhl": {"total"}}, odds={"nhl": [pin_event()]})
    out = run_close(NOW, f.deps(), dry_run=True)
    for w in ("insert_results", "upsert_closes", "upsert_events", "log_call"):
        assert f.n(w) == 0, w
    assert out["graded"] == 1 and out["closes"] == 1 and f.n("fetch_odds") == 1
    assert "DRY RUN" in capsys.readouterr().out


def test_close_one_failing_league_warns_and_the_next_goes_on(capsys):
    f = Rec(targets={"nhl": {"total"}, "nba": {"total"}}, odds={"nba": [pin_event()]}, fail={"nhl"})
    out = run_close(NOW, f.deps())
    assert "failed" in out["leagues"]["nhl"] and len(f.closes) == 1
    assert "::warning::take-close: nhl close failed" in capsys.readouterr().out


def test_close_missing_tables_warn_and_do_nothing(capsys):
    f = Rec(tables=["take_closes"])
    assert run_close(NOW, f.deps())["status"] == "no_table" and f.order == []
    assert "migration_take_these.sql" in capsys.readouterr().out


def test_main_exit_codes(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "x")
    monkeypatch.setenv("ODDS_API_KEY", "y")
    monkeypatch.setattr(take_close, "_now", lambda: NOW)
    monkeypatch.setattr(take_close, "_db_deps", lambda: Rec(targets={"nhl": {"total"}}, fail={"nhl"}).deps())
    assert take_close.main(["--dry-run"]) == 1                          # the only attempted league failed
    monkeypatch.setattr(take_close, "_db_deps", lambda: Rec(targets={"nhl": {"total"}}, odds={"nhl": [pin_event()]}).deps())
    assert take_close.main(["--dry-run"]) == 0
    monkeypatch.setattr(take_close, "_db_deps", lambda: Rec(tables=["take_closes"]).deps())
    assert take_close.main([]) == 0
