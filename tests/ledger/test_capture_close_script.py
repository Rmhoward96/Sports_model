"""scripts/capture_close.py end to end on injected IO: idle runs make no call, windowing, retry, budget guard,
Pinnacle/regions handling, what is stored. No network, no database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("capture_close", ROOT / "scripts" / "capture_close.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules["capture_close"] = cc            # @dataclass needs its module registered
_spec.loader.exec_module(cc)

NOW = datetime(2026, 10, 18, 16, 40, tzinfo=timezone.utc)


def iso(minutes):
    return (NOW + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def game(pk, minutes, home="Home", away="Away"):
    return {"game_pk": pk, "home_name": home, "away_name": away, "commence_time": iso(minutes)}


def event(eid, minutes, home="Home", away="Away"):
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": iso(minutes)}


def odds_event(eid, minutes, home="Home", away="Away"):
    price = lambda p: {"price": p}
    return {**event(eid, minutes, home, away), "bookmakers": [
        {"key": "pinnacle", "markets": [
            {"key": "h2h", "outcomes": [{"name": home, **price(-150)}, {"name": away, **price(130)}]},
            {"key": "spreads", "outcomes": [{"name": home, "point": -3.5, **price(-110)}, {"name": away, "point": 3.5, **price(-110)}]}]}]}


class Fake:
    """Injected IO that records every call; any call not expected in a scenario can be made to explode."""

    def __init__(self, **over):
        self.calls, self.stored_snaps, self.stored_caps = [], [], []
        self.remaining_value = over.pop("remaining", "20000")
        self.cost = over.pop("cost", "6")
        self.candidates = over.pop("candidates", [])
        self.schedule = over.pop("schedule", {})
        self.captured = over.pop("captured", set())
        self.prop_markets = over.pop("prop_markets", {})
        self.events = over.pop("events", [])
        self.line_events = over.pop("line_events", [])
        self.props_payload = over.pop("props_payload", {"bookmakers": []})
        self.fail = over.pop("fail", set())
        self.fail_event = over.pop("fail_event", {})              # Odds API event id -> exception to raise on props
        self.store_log = []
        self.missing = over.pop("missing", [])
        assert not over, over

    def _log(self, name, *args):
        self.calls.append((name, *args))

    def deps(self):
        def fetch_game_odds(cfg, regions):
            self._log("lines", cfg.key, regions)
            if "lines" in self.fail:
                raise RuntimeError("odds api 500")
            return self.line_events

        def fetch_event_props(eid, markets, cfg, regions):
            self._log("props", eid, tuple(markets), regions)
            if eid in self.fail_event:
                raise self.fail_event[eid]
            if "props" in self.fail:
                raise RuntimeError("odds api 500")
            return self.props_payload

        return cc.Deps(
            missing_tables=lambda: self.missing,
            load_candidates=lambda now, end: self.candidates,
            load_captured=lambda pks: self.captured,
            load_prop_markets=lambda sport, pks: self.prop_markets,
            fetch_schedule=lambda sport: self._log("schedule", sport) or self.schedule.get(sport, []),
            match=cc.ingest_odds._matcher_for,
            fetch_events=lambda cfg: self._log("events", cfg.key) or self.events,
            fetch_game_odds=fetch_game_odds, fetch_event_props=fetch_event_props,
            remaining=lambda: self.remaining_value, last_cost=lambda: self.cost,
            store_snapshots=lambda rows: self.store_log.append("snaps") or self.stored_snaps.extend(rows) or len(rows),
            store_captures=lambda rows: self.store_log.append("caps") or self.stored_caps.extend(rows) or len(rows))

    def network_calls(self):
        return [c for c in self.calls if c[0] in ("events", "lines", "props", "schedule")]


def cand(sport, pk, minutes):
    return {"sport": sport, "game_pk": pk, "commence_time": iso(minutes)}


def run(fake, **kw):
    return cc.run_capture(NOW, fake.deps(), **kw)


# ---- idle runs --------------------------------------------------------------------------------------

def test_idle_run_makes_no_network_call_at_all():
    fake = Fake()                                                  # the DB knows of no kickoff
    assert run(fake) == {"status": "idle"} and fake.calls == []


def test_a_kickoff_far_away_is_idle_too():
    fake = Fake(candidates=[cand("nfl", 1, 180), cand("cfb", 2, -30)])
    assert run(fake) == {"status": "idle"} and fake.calls == []


def test_nearby_kickoff_but_nothing_in_the_window_makes_no_odds_api_call():
    fake = Fake(candidates=[cand("nfl", 1, 45)], schedule={"nfl": [game(1, 45)]})        # 45 min out: too early
    out = run(fake)
    assert out["status"] == "ok" and out["credits"] == 0
    assert fake.calls == [("schedule", "nfl")]                      # schedule only: no events, lines or props call


def test_missing_tables_warn_and_stop_before_anything_else(capsys):
    fake = Fake(missing=["close_capture"], candidates=[cand("nfl", 1, 12)])
    assert run(fake) == {"status": "no_table"} and fake.calls == []
    assert "::warning::capture-close: table close_capture missing" in capsys.readouterr().out


# ---- a normal capture ----------------------------------------------------------------------------

def _nfl_fake(**kw):
    base = dict(candidates=[cand("nfl", 1, 12), cand("nfl", 2, 40)],
                schedule={"nfl": [game(1, 12), game(2, 40, "H2", "A2")]},
                events=[event("ev1", 12), event("ev2", 40, "H2", "A2")],
                line_events=[odds_event("ev1", 12), odds_event("ev2", 40, "H2", "A2")])
    return Fake(**{**base, **kw})


def test_lines_capture_stores_only_the_target_games_and_one_row_per_event():
    fake = _nfl_fake()
    out = run(fake)
    assert out["credits"] == 6 and out["snapshots"] == len(fake.stored_snaps) > 0
    assert {r["game_pk"] for r in fake.stored_snaps} == {1}                           # game 2 is 40 min out
    assert {r["captured_at"] for r in fake.stored_snaps} == {NOW.isoformat()}
    ((sport, pk, kind, status),) = out["rows"]
    assert (sport, pk, kind, status) == ("nfl", 1, "lines", "ok")
    (cap,) = fake.stored_caps
    assert cap["event_id"] == "ev1" and cap["credits_used"] == 6 and cap["regions"] == "us,eu"
    assert cap["minutes_to_kick"] == pytest.approx(12) and cap["captured_at"] == NOW.isoformat()
    assert ("lines", "nfl", "us,eu") in fake.calls and not [c for c in fake.calls if c[0] == "props"]


def test_two_target_games_split_one_call_cost_exactly():
    fake = _nfl_fake(candidates=[cand("nfl", 1, 12), cand("nfl", 2, 14)],
                     schedule={"nfl": [game(1, 12), game(2, 14, "H2", "A2")]},
                     line_events=[odds_event("ev1", 12), odds_event("ev2", 14, "H2", "A2")], cost="6")
    run(fake)
    assert sorted(c["credits_used"] for c in fake.stored_caps) == [3, 3]
    assert len([c for c in fake.calls if c[0] == "lines"]) == 1                      # ONE call per sport


def test_an_event_without_prices_is_recorded_empty():
    fake = _nfl_fake(line_events=[])
    out = run(fake)
    assert out["rows"] == [("nfl", 1, "lines", "empty")]


def test_already_captured_events_cost_nothing():
    fake = _nfl_fake(captured={("nfl", 1, "lines")})
    out = run(fake)
    assert out["credits"] == 0 and fake.stored_caps == [] and "events" not in [c[0] for c in fake.calls]


def test_a_failed_call_is_recorded_failed_and_retried_next_run():
    fake = _nfl_fake(fail={"lines"})
    out = run(fake)
    assert out["rows"] == [("nfl", 1, "lines", "failed")] and out["credits"] == 0
    assert fake.stored_caps[0]["status"] == "failed" and fake.stored_snaps == []
    # 'failed' is not in the loader's final set, so the next run selects the event again and succeeds
    fake2 = _nfl_fake(captured=set())
    assert run(fake2)["rows"] == [("nfl", 1, "lines", "ok")]


def test_the_credit_cost_falls_back_to_markets_times_regions_without_the_header():
    fake = _nfl_fake(cost=None)
    assert run(fake)["credits"] == 6


# ---- props -------------------------------------------------------------------------------------------

PROP_PAYLOAD = {"id": "ev1", "commence_time": iso(12), "bookmakers": [
    {"key": "draftkings", "markets": [{"key": "player_reception_yds", "outcomes": [
        {"name": "Over", "description": "A. Brown", "point": 64.5, "price": -110},
        {"name": "Under", "description": "A. Brown", "point": 64.5, "price": -110}]}]}]}


def test_props_are_captured_only_for_events_with_ledger_prop_picks_and_only_their_markets():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, props_payload=PROP_PAYLOAD, cost="2")
    out = run(fake)
    assert ("props", "ev1", ("player_reception_yds",), "us,eu") in fake.calls
    kinds = {(c["kind"], c["status"], c["credits_used"]) for c in fake.stored_caps}
    assert kinds == {("lines", "ok", 2), ("props", "ok", 2)}
    props_rows = [r for r in fake.stored_snaps if r["player_name"]]
    assert {(r["market"], r["side"], r["line"]) for r in props_rows} == {("reception_yds", "over", 64.5), ("reception_yds", "under", 64.5)}
    assert {r["captured_at"] for r in fake.stored_snaps} == {NOW.isoformat()}
    assert out["credits"] == 4


def test_no_prop_picks_means_no_props_call():
    fake = _nfl_fake(prop_markets={})
    run(fake)
    assert not [c for c in fake.calls if c[0] == "props"]


def test_prop_regions_override_is_used_and_pinnacle_free_runs_pay_half():
    fake = _nfl_fake(prop_markets={1: {"rec_yds", "rush_yds", "receptions", "pass_tds"}}, props_payload=PROP_PAYLOAD, cost=None)
    run(fake, prop_regions="us")
    (call,) = [c for c in fake.calls if c[0] == "props"]
    assert call[3] == "us" and len(call[2]) == 4
    caps = {c["kind"]: c["credits_used"] for c in fake.stored_caps}
    assert caps["props"] == 4                                          # 4 markets x 1 region (8 with us,eu)
    fake2 = _nfl_fake(prop_markets={1: {"rec_yds", "rush_yds", "receptions", "pass_tds"}}, props_payload=PROP_PAYLOAD, cost=None)
    run(fake2, prop_regions="us,eu")
    assert {c["kind"]: c["credits_used"] for c in fake2.stored_caps}["props"] == 8


def test_props_already_captured_are_not_bought_again_and_a_props_retry_needs_no_lines_call():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, props_payload=PROP_PAYLOAD, captured={("nfl", 1, "lines")})
    run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == ["props"]
    fake2 = _nfl_fake(prop_markets={1: {"rec_yds"}}, captured={("nfl", 1, "lines"), ("nfl", 1, "props")})
    run(fake2)
    assert not [c for c in fake2.calls if c[0] in ("lines", "props")]


def test_a_failed_props_call_is_recorded_failed_without_losing_the_lines_capture():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, fail={"props"})
    out = run(fake)
    assert sorted((k, s) for _, _, k, s in out["rows"]) == [("lines", "ok"), ("props", "failed")]


def test_cfb_has_no_props_and_never_asks_for_the_ledger():
    fake = Fake(candidates=[cand("cfb", 9, 12)], schedule={"cfb": [game(9, 12)]}, events=[event("c1", 12)],
                line_events=[odds_event("c1", 12)], prop_markets={9: {"rec_yds"}})
    run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == ["lines"]


# ---- incremental storage ----------------------------------------------------------------------------

def _three_prop_games(**kw):
    games = [(i, 10 + i, f"H{i}", f"A{i}") for i in (1, 2, 3)]
    base = dict(candidates=[cand("nfl", pk, m) for pk, m, _, _ in games],
                schedule={"nfl": [game(pk, m, h, a) for pk, m, h, a in games]},
                events=[event(f"ev{pk}", m, h, a) for pk, m, h, a in games],
                line_events=[odds_event(f"ev{pk}", m, h, a) for pk, m, h, a in games],
                prop_markets={pk: {"rec_yds"} for pk, *_ in games}, props_payload=PROP_PAYLOAD, cost="2")
    return Fake(**{**base, **kw})


def test_storage_order_is_snapshots_then_captures_after_the_lines_call_and_after_each_props_event():
    fake = _three_prop_games()
    run(fake)
    assert fake.store_log == ["snaps", "caps"] * 4                  # lines, then props for events 1-3


def test_snapshots_are_stored_before_their_capture_rows():
    seen = []
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, props_payload=PROP_PAYLOAD)
    deps = fake.deps()
    deps.store_snapshots = lambda rows: seen.append(("snaps", len(rows))) or len(rows)
    deps.store_captures = lambda rows: seen.append(("caps", [r["status"] for r in rows])) or len(rows)
    cc.run_capture(NOW, deps)
    assert [k for k, _ in seen] == ["snaps", "caps", "snaps", "caps"]


def test_an_error_in_the_third_props_event_keeps_the_lines_and_events_one_and_two():
    fake = _three_prop_games(fail_event={"ev3": RuntimeError("odds api 500")})
    out = run(fake)
    got = {(c["game_pk"], c["kind"]): c["status"] for c in fake.stored_caps}
    assert got == {(1, "lines"): "ok", (2, "lines"): "ok", (3, "lines"): "ok",
                   (1, "props"): "ok", (2, "props"): "ok", (3, "props"): "failed"}
    assert out["credits"] == 2 + 2 + 2                  # lines (header cost 2) + props 1 and 2


def test_an_interrupt_in_the_third_props_event_still_leaves_everything_bought_so_far_stored():
    fake = _three_prop_games(fail_event={"ev3": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run(fake)
    got = {(c["game_pk"], c["kind"]) for c in fake.stored_caps}
    assert got == {(1, "lines"), (2, "lines"), (3, "lines"), (1, "props"), (2, "props")}
    assert {r["game_pk"] for r in fake.stored_snaps if r["player_name"]} == {1, 2}


def test_a_failed_lines_row_is_stored_immediately_even_if_props_then_raise():
    fake = _three_prop_games(fail={"lines"}, fail_event={"ev1": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run(fake)
    assert {(c["game_pk"], c["kind"], c["status"]) for c in fake.stored_caps} == {
        (1, "lines", "failed"), (2, "lines", "failed"), (3, "lines", "failed")}


def test_props_targets_with_only_unmapped_markets_are_recorded_empty_and_never_retried():
    fake = _nfl_fake(prop_markets={1: {"nonsense"}}, captured={("nfl", 1, "lines")})
    out = run(fake)
    assert out["rows"] == [("nfl", 1, "props", "empty")] and not [c for c in fake.calls if c[0] in ("lines", "props")]
    (cap,) = fake.stored_caps
    assert cap["credits_used"] == 0 and cap["markets"] == ""
    # 'empty' is final: the next run selects nothing
    fake2 = _nfl_fake(prop_markets={1: {"nonsense"}}, captured={("nfl", 1, "lines"), ("nfl", 1, "props")})
    assert run(fake2)["rows"] == []


def test_mapped_and_unmapped_props_targets_mix():
    fake = _three_prop_games(prop_markets={1: {"rec_yds"}, 2: {"nonsense"}}, captured={("nfl", 3, "lines")})
    out = run(fake)
    got = {(pk, k): s for _, pk, k, s in out["rows"]}
    assert got[(1, "props")] == "ok" and got[(2, "props")] == "empty"
    assert [c[1] for c in fake.calls if c[0] == "props"] == ["ev1"]


# ---- the as-of time ----------------------------------------------------------------------------------

def test_naive_now_is_read_as_utc():
    assert cc._parse_now("2026-10-18T16:40:00") == NOW
    assert cc._parse_now("2026-10-18T16:40:00Z") == NOW
    assert cc._parse_now("2026-10-18T12:40:00-04:00") == NOW
    assert cc._parse_now("2026-10-18T16:40:00").tzinfo is not None


def test_main_passes_a_utc_aware_now_to_the_run(monkeypatch):
    seen = {}
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setattr(cc, "_db_deps", lambda: object())
    monkeypatch.setattr(cc, "run_capture", lambda now, deps, **kw: seen.update(now=now, **kw) or {"status": "idle"})
    assert cc.main(["--dry-run", "--now", "2026-10-18T16:40:00"]) == 0
    assert seen["now"] == NOW and seen["dry_run"] is True


# ---- budget guard ------------------------------------------------------------------------------------

def test_below_three_thousand_credits_skips_props_but_keeps_lines(capsys):
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, remaining="2500")
    run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == ["lines"]
    assert "skipping prop close captures" in capsys.readouterr().out


def test_below_one_thousand_credits_skips_everything_and_stores_nothing(capsys):
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, remaining="999")
    out = run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == [] and fake.stored_snaps == [] and fake.stored_caps == []
    assert out["credits"] == 0 and "skipping ALL close captures" in capsys.readouterr().out


def test_an_unreadable_credit_header_fails_open_with_a_warning(capsys):
    fake = _nfl_fake(remaining=None)
    run(fake)
    assert "lines" in [c[0] for c in fake.calls] and "unavailable" in capsys.readouterr().out


# ---- dry run, sports, main -------------------------------------------------------------------------

def test_dry_run_plans_but_spends_and_writes_nothing():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}})
    out = run(fake, dry_run=True)
    assert out["planned"] == [{"sport": "nfl", "lines": [1], "props": [1]}]
    assert not [c for c in fake.calls if c[0] in ("events", "lines", "props")] and fake.stored_caps == []


def test_one_sport_failing_does_not_stop_the_next():
    fake = Fake(candidates=[cand("nfl", 1, 12), cand("cfb", 2, 12)],
                schedule={"cfb": [game(2, 12)]}, events=[event("c1", 12)], line_events=[odds_event("c1", 12)])
    deps = fake.deps()

    def schedule(sport):
        if sport == "nfl":
            raise RuntimeError("espn down")
        return fake.schedule[sport]

    deps.fetch_schedule = schedule
    out = cc.run_capture(NOW, deps)
    assert out["rows"] == [("cfb", 2, "lines", "ok")]


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert cc.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_default_db_queries(monkeypatch):
    from sportsmodel import db
    seen = []

    class Cur:
        def execute(self, sql, params):
            seen.append((sql, params))

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn(Cur):
        def cursor(self):
            return Cur()

    monkeypatch.setattr(db, "get_postgres", lambda: Conn())
    deps = cc._db_deps()
    deps.load_candidates(NOW, NOW + timedelta(hours=1))
    deps.load_captured([1, 2])
    deps.load_prop_markets("nfl", [1])
    sqls = [s for s, _ in seen]
    assert "FROM predictions_current" in sqls[0] and "commence_time > %s AND commence_time <= %s" in sqls[0]
    assert "UNION" in sqls[0] and "SELECT DISTINCT sport, game_pk, commence_time FROM pick_ledger" in sqls[0]
    assert seen[0][1] == (NOW, NOW + timedelta(hours=1), NOW, NOW + timedelta(hours=1))
    assert "status IN ('ok', 'empty')" in sqls[1] and seen[1][1] == ([1, 2],)
    assert "FROM pick_ledger" in sqls[2] and "source IN ('ev_prop', 'parlay_leg')" in sqls[2]
    assert "sport = %s" in sqls[2] and seen[2][1] == ("nfl", [1])
