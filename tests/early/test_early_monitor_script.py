"""scripts/early_monitor.py end to end on injected IO: bookmakers fetch, the credit floor, snapshot storage, incremental
ticket writes, failure isolation, dry run and the guards. No network, no database."""
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from .builders import NOW, StubTable, iso, mass_at

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("early_monitor", ROOT / "scripts" / "early_monitor.py")
em = importlib.util.module_from_spec(_spec)
sys.modules["early_monitor"] = em            # @dataclass needs its module registered
_spec.loader.exec_module(em)

KICK = NOW + timedelta(hours=72)
M = 0.0318


def schedule_game(pk, home="Home", away="Away", kick=KICK):
    return {"game_pk": pk, "home_name": home, "away_name": away, "commence_time": iso(kick)}


def outcomes_total(line, over, under):
    return [{"name": "Over", "point": line, "price": over}, {"name": "Under", "point": line, "price": under}]


def total_event(eid, home="Home", away="Away", kick=KICK, pin_line=55.5, dk_line=54.5):
    def book(key, line, over=-110, under=-110):
        return {"key": key, "markets": [{"key": "totals", "outcomes": outcomes_total(line, over, under)}]}
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": iso(kick),
            "bookmakers": [book("pinnacle", pin_line), book("draftkings", dk_line), book("bovada", 50.5)]}


def spread_event(eid, home="NFL Home", away="NFL Away", kick=KICK):
    def book(key, home_line):
        return {"key": key, "markets": [{"key": "spreads", "outcomes": [
            {"name": home, "point": home_line, "price": -110}, {"name": away, "point": -home_line, "price": -110}]}]}
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": iso(kick),
            "bookmakers": [book("pinnacle", -3.5), book("betmgm", -2.5)]}


TABLES = {("cfb", "total"): StubTable(mass_at((55, 56), M)), ("cfb", "spread"): StubTable(mass_at(3, M)),
          ("nfl", "spread"): StubTable(mass_at(3, M))}


class Fake:
    """Injected IO that records every call and every store, in order."""

    def __init__(self, **over):
        self.log, self.snaps, self.tickets = [], [], []
        self.remaining_value = over.pop("remaining", "20000")
        self.cost = over.pop("cost", {"cfb": "2", "nfl": "1"})
        self.schedule = over.pop("schedule", {"cfb": [schedule_game(1)], "nfl": [schedule_game(2, "NFL Home", "NFL Away")]})
        self.events = over.pop("events", {"cfb": [total_event("c1")], "nfl": [spread_event("n1")]})
        self.existing = over.pop("existing", set())
        self.fail = over.pop("fail", {})                    # sport -> exception raised by the odds call
        self.fail_store = over.pop("fail_store", {})        # "snaps" / "tickets" -> exception
        self.fail_prime = over.pop("fail_prime", None)
        self.prime_remaining = over.pop("prime_remaining", None)
        self.missing = over.pop("missing", [])
        assert not over, over

    def deps(self):
        state = {"remaining": self.prime_remaining if self.prime_remaining is not None else None, "cost": None}

        def prime(cfg):
            self.log.append(("prime", cfg.key))
            if self.fail_prime:
                raise self.fail_prime
            if self.prime_remaining is not None:
                state["remaining"] = self.prime_remaining

        def fetch_odds(cfg, bookmakers, markets):
            self.log.append(("odds", cfg.key, tuple(bookmakers), tuple(markets)))
            if cfg.key in self.fail:
                raise self.fail[cfg.key]
            state["remaining"], state["cost"] = self.remaining_value, self.cost[cfg.key]
            return self.events[cfg.key]

        def store_snaps(rows):
            self.log.append(("snaps", len(rows)))
            if "snaps" in self.fail_store:
                raise self.fail_store["snaps"]
            self.snaps.extend(rows)
            return len(rows)

        def store_tickets(rows):
            self.log.append(("tickets", len(rows)))
            if "tickets" in self.fail_store:
                raise self.fail_store["tickets"]
            self.tickets.extend(rows)
            self.existing = self.existing | {(t["sport"], t["game_pk"], t["market"], t["side"]) for t in rows}
            return len(rows)

        return em.Deps(
            missing_tables=lambda: self.missing,
            load_existing=lambda sport, now: self.existing,
            load_tables=lambda: TABLES,
            fetch_schedule=lambda sport: self.log.append(("schedule", sport)) or self.schedule[sport],
            match=em.ingest_odds._matcher_for, prime=prime, fetch_odds=fetch_odds,
            remaining=lambda: state["remaining"], last_cost=lambda: state["cost"],
            store_snapshots=store_snaps, store_tickets=store_tickets)

    def calls(self, name):
        return [c for c in self.log if c[0] == name]


def run(fake, **kw):
    return em.run_poll(NOW, fake.deps(), run_id="run-7", **kw)


# ---- the poll itself --------------------------------------------------------------------------------------

def test_one_odds_call_per_sport_with_the_ten_bookmakers_and_the_right_markets():
    fake = Fake()
    out = run(fake)
    cfb, nfl = fake.calls("odds")
    assert cfb[1] == "cfb" and cfb[2] == tuple(em.polling.bookmakers_param().split(",")) and len(cfb[2]) == 10
    assert cfb[3] == ("spreads", "totals")
    assert nfl[1] == "nfl" and nfl[3] == ("spreads",)
    assert out["credits"] == 3 and out["status"] == "ok"


def test_the_free_events_call_primes_the_credit_header_once_before_any_paid_call():
    fake = Fake()
    run(fake)
    names = [c[0] for c in fake.log]
    assert names.index("prime") < names.index("odds") and names.count("prime") == 1


def test_snapshots_use_the_ingest_row_shape_for_the_tested_markets_and_listed_books_only():
    fake = Fake()
    run(fake)
    cfb = [r for r in fake.snaps if r["game_pk"] == 1]
    assert {r["book"] for r in cfb} == {"pinnacle", "draftkings"}                       # bovada was not asked for
    assert {r["market"] for r in cfb} == {"total"} and {r["captured_at"] for r in fake.snaps} == {NOW.isoformat()}
    row = next(r for r in cfb if r["book"] == "draftkings" and r["side"] == "over")
    assert set(row) == {"game_pk", "market", "side", "player_name", "book", "line", "price", "captured_at", "commence_time"}
    assert row["line"] == 54.5 and row["price"] == -110 and row["player_name"] == ""
    assert {r["market"] for r in fake.snaps if r["game_pk"] == 2} == {"spread"}


def test_a_ticket_is_logged_with_this_polls_run_id_and_the_matched_game_pk():
    fake = Fake()
    out = run(fake)
    (cfb,) = [t for t in fake.tickets if t["sport"] == "cfb"]
    assert (cfb["game_pk"], cfb["side"], cfb["book"], cfb["book_line"]) == (1, "over", "draftkings", 54.5)
    assert cfb["poll_run_id"] == "run-7" and cfb["observed_at"] == NOW
    (nfl,) = [t for t in fake.tickets if t["sport"] == "nfl"]
    assert (nfl["market"], nfl["side"], nfl["book"], nfl["book_line"]) == ("spread", "home", "betmgm", -2.5)
    assert out["tickets"] == 2 and out["snapshots"] == len(fake.snaps)


def test_events_that_match_no_game_are_dropped():
    fake = Fake(events={"cfb": [total_event("x", home="Nobody", away="Else")], "nfl": []})
    out = run(fake)
    assert fake.snaps == [] and fake.tickets == [] and out["sports"]["cfb"]["matched"] == 0


def test_a_second_run_in_the_same_hour_logs_no_new_ticket():
    fake = Fake()
    run(fake)
    first = list(fake.tickets)
    assert len(first) == 2
    run(fake)                                                         # a duplicated cron firing: same prices, same book
    assert fake.tickets == first


def test_tickets_that_already_exist_are_not_logged_again():
    fake = Fake(existing={("cfb", 1, "total", "over")})
    run(fake)
    assert [t for t in fake.tickets if t["sport"] == "cfb"] == []


def test_existing_keys_with_a_string_game_pk_still_block_a_second_ticket():
    fake = Fake(existing={("cfb", "1", "total", "over"), ("nfl", "2", "spread", "home")})
    run(fake)
    assert fake.tickets == []                                         # int game_pk in the rows, str in the keys: no mismatch


def test_flag_poll_gets_only_this_polls_rows_never_stored_snapshots(monkeypatch):
    seen = {}
    real = em.flags.flag_poll
    monkeypatch.setattr(em.flags, "flag_poll", lambda rows, **kw: seen.setdefault(kw["sport"], list(rows)) and real(rows, **kw))
    fake = Fake()
    run(fake)
    assert {r["captured_at"] for rows in seen.values() for r in rows} == {NOW.isoformat()}
    assert all(type(r["game_pk"]) is int for rows in seen.values() for r in rows)


# ---- credit floor ----------------------------------------------------------------------------------------

def test_at_1500_remaining_the_next_call_still_runs_at_1499_it_does_not(capsys):
    ok = Fake(remaining="1500")
    run(ok)
    assert [c[1] for c in ok.calls("odds")] == ["cfb", "nfl"]
    low = Fake(remaining="1499")
    out = run(low)
    assert [c[1] for c in low.calls("odds")] == ["cfb"] and out["skipped"] == ["nfl"]
    printed = capsys.readouterr().out
    assert "::warning::early-monitor: 1499 Odds API credits left (< 1500)" in printed


def test_an_unknown_first_value_runs_the_first_call_and_the_value_it_returns_guards_the_second():
    fake = Fake(remaining="900")                       # nothing known until the first paid response says 900
    out = run(fake)
    assert [c[1] for c in fake.calls("odds")] == ["cfb"] and out["credits"] == 2


def test_a_low_balance_read_by_the_free_call_stops_even_the_first_paid_call(capsys):
    fake = Fake(prime_remaining="1200")
    out = run(fake)
    assert fake.calls("odds") == [] and out["credits"] == 0 and out["skipped"] == ["cfb", "nfl"]
    assert "skipping its remaining calls" in capsys.readouterr().out


def test_if_the_free_call_fails_the_first_call_still_runs(capsys):
    fake = Fake(fail_prime=RuntimeError("events 500"))
    run(fake)
    assert [c[1] for c in fake.calls("odds")] == ["cfb", "nfl"]
    assert "credit check call failed" in capsys.readouterr().out


def test_credits_fall_back_to_markets_times_one_region_without_the_cost_header():
    fake = Fake(cost={"cfb": None, "nfl": None})
    assert run(fake)["credits"] == 3


# ---- incremental writes and failures -------------------------------------------------------------------------

def test_each_sports_snapshots_then_tickets_are_stored_before_the_next_sport_is_fetched():
    fake = Fake()
    run(fake)
    stores = [c[0] for c in fake.log if c[0] in ("odds", "snaps", "tickets")]
    assert stores == ["odds", "snaps", "tickets", "odds", "snaps", "tickets"]


def test_a_failure_in_the_second_sport_keeps_the_first_sports_tickets(capsys):
    fake = Fake(fail={"nfl": RuntimeError("odds api 500")})
    out = run(fake)
    assert [t["sport"] for t in fake.tickets] == ["cfb"] and out["sports"]["nfl"]["failed"].startswith("RuntimeError")
    assert "nfl poll failed" in capsys.readouterr().out


def test_a_failed_poll_is_not_retried_and_the_other_sport_goes_on():
    fake = Fake(fail={"cfb": RuntimeError("odds api 500")})
    out = run(fake)
    assert len(fake.calls("odds")) == 2 and [t["sport"] for t in fake.tickets] == ["nfl"]
    assert out["credits"] == 1


def test_a_snapshot_write_failure_does_not_stop_the_tickets():
    fake = Fake(fail_store={"snaps": RuntimeError("db hiccup")})
    out = run(fake)
    assert out["tickets"] == 2 and out["snapshots"] == 0


def test_an_interrupt_in_the_second_sport_still_leaves_the_first_sports_tickets_stored():
    fake = Fake(fail={"nfl": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run(fake)
    assert [t["sport"] for t in fake.tickets] == ["cfb"]


def test_the_schedule_and_existing_ticket_reads_happen_before_credits_are_spent():
    fake = Fake()
    deps = fake.deps()
    deps.load_existing = lambda sport, now: (_ for _ in ()).throw(RuntimeError("db down"))
    out = em.run_poll(NOW, deps)
    assert fake.calls("odds") == [] and out["credits"] == 0


# ---- guards and dry run ------------------------------------------------------------------------------------

def test_missing_tables_warn_and_stop_before_any_network_call(capsys):
    fake = Fake(missing=["early_tickets"])
    assert run(fake) == {"status": "no_table"} and fake.log == []
    assert "::warning::early-monitor: table early_tickets missing" in capsys.readouterr().out


def test_dry_run_polls_but_writes_nothing_and_reports_what_it_would_log():
    fake = Fake()
    out = run(fake, dry_run=True)
    assert len(fake.calls("odds")) == 2 and fake.snaps == [] and fake.tickets == []
    assert fake.calls("snaps") == [] and fake.calls("tickets") == []
    assert [t["sport"] for t in out["would_write"]] == ["cfb", "nfl"] and out["tickets"] == 0


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert em.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_a_dry_run_is_also_refused_under_the_test_ledger_guard(monkeypatch, capsys):
    monkeypatch.setattr(em, "_db_deps", lambda: pytest.fail("must not build DB deps under the guard"))
    monkeypatch.setattr(em, "run_poll", lambda *a, **k: pytest.fail("must not poll under the guard"))
    assert em.main(["--dry-run"]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_main_passes_run_id_and_dry_run_through(monkeypatch, capsys):
    seen = {}
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)             # the root conftest sets the guard; this test exercises the run
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setenv("ODDS_API_KEY", "unused")
    monkeypatch.setenv("GITHUB_RUN_ID", "4242")
    monkeypatch.setattr(em, "_db_deps", lambda: object())
    monkeypatch.setattr(em, "run_poll", lambda now, deps, **kw: seen.update(now=now, **kw) or {"status": "ok", "would_write": [
        {"sport": "cfb", "game_pk": 1, "market": "total", "side": "over", "book": "draftkings", "book_line": 54.5,
         "price": -110, "ev": 0.0152, "hours_to_kick": 72.0}]})
    assert em.main(["--dry-run"]) == 0
    assert seen["run_id"] == "4242" and seen["dry_run"] is True and seen["now"].tzinfo is not None
    assert "DRY RUN would log: cfb 1 total over draftkings +54.5 -110 ev=+1.52%" in capsys.readouterr().out


def test_main_needs_both_secrets(monkeypatch, capsys):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    assert em.main([]) == 1 and "ODDS_API_KEY is not set" in capsys.readouterr().out
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    assert em.main([]) == 1


def test_production_wiring_uses_the_db_helpers_with_timeouts(monkeypatch):
    from sportsmodel import db
    seen = {}
    monkeypatch.setattr(db, "missing_early_tables", lambda **kw: seen.update(missing=kw) or [])
    monkeypatch.setattr(db, "early_ticket_keys", lambda sport, now, **kw: seen.update(keys=(sport, now, kw)) or set())
    monkeypatch.setattr(db, "insert_early_tickets", lambda rows, **kw: seen.update(insert=kw) or len(rows))
    deps = em._db_deps()
    deps.missing_tables()
    deps.load_existing("cfb", NOW)
    deps.store_tickets([{}])
    assert seen["missing"] == {"connect_timeout": 10, "statement_timeout_ms": 15_000}
    assert seen["keys"] == ("cfb", NOW, {"connect_timeout": 10, "statement_timeout_ms": 15_000})
    assert seen["insert"] == {"connect_timeout": 10, "statement_timeout_ms": 15_000}
    assert deps.store_snapshots is db.upsert_odds_snapshot


# ---- a dead monitor is visible -------------------------------------------------------------------------------

def _main_with(monkeypatch, fake, argv=()):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setenv("ODDS_API_KEY", "unused")
    monkeypatch.setattr(em, "_db_deps", fake.deps)
    return em.main(list(argv))


def test_main_exits_1_when_every_attempted_sport_failed(monkeypatch):
    fake = Fake(fail={"cfb": RuntimeError("401"), "nfl": RuntimeError("401")})
    assert _main_with(monkeypatch, fake) == 1


def test_main_exits_0_when_one_sport_fails_and_one_succeeds(monkeypatch):
    fake = Fake(fail={"nfl": RuntimeError("500")})
    assert _main_with(monkeypatch, fake) == 0


def test_main_exits_0_when_the_credit_floor_skips_everything(monkeypatch):
    fake = Fake(prime_remaining="1200")
    assert _main_with(monkeypatch, fake) == 0 and fake.calls("odds") == []


def test_main_exits_0_when_the_first_sport_fails_and_the_floor_skips_the_second(monkeypatch):
    fake = Fake(fail={"cfb": RuntimeError("500")}, remaining="1200")
    assert _main_with(monkeypatch, fake) == 0                          # one attempted sport failed, but the floor ended the run


def test_main_exits_0_when_the_tables_are_missing(monkeypatch):
    assert _main_with(monkeypatch, Fake(missing=["early_tickets"])) == 0


def test_the_books_seen_and_missing_line_is_printed_after_each_successful_poll(capsys):
    fake = Fake()                                                     # the stub events carry pinnacle + draftkings + bovada (cfb)
    out = run(fake)
    printed = capsys.readouterr().out
    assert "[early-monitor] cfb books seen: ['draftkings', 'pinnacle']; missing: " in printed
    assert "'espnbet'" in printed.split("cfb books seen")[1].split("\n")[0]
    assert "[early-monitor] nfl books seen: ['betmgm', 'pinnacle']; missing: " in printed
    assert out["sports"]["cfb"]["books_missing"] == [b for b in em.BOOKMAKERS if b not in ("draftkings", "pinnacle")]


def test_the_dry_run_prints_the_same_books_line(capsys):
    run(Fake(), dry_run=True)
    assert "cfb books seen: ['draftkings', 'pinnacle']; missing:" in capsys.readouterr().out


def test_a_failed_poll_prints_no_books_line(capsys):
    run(Fake(fail={"cfb": RuntimeError("500")}))
    assert "cfb books seen" not in capsys.readouterr().out
