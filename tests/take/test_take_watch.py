"""scripts/take_watch.py: the anchored tick loop, the close-then-pull tick, pull_due, and failure isolation.
Fake clock, sleep and Deps only (no network, no DB, no real waiting)."""
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sportsmodel.take import constants

_SPEC = importlib.util.spec_from_file_location(
    "take_watch", Path(__file__).resolve().parents[2] / "scripts" / "take_watch.py")
take_watch = importlib.util.module_from_spec(_SPEC)
sys.modules["take_watch"] = take_watch          # @dataclass resolves its module through sys.modules
_SPEC.loader.exec_module(take_watch)
Deps, pull_due, run_watch, every_hours_from = take_watch.Deps, take_watch.pull_due, take_watch.run_watch, take_watch.every_hours_from

T0 = datetime(2026, 11, 2, 14, 7, tzinfo=UTC)               # a pull hour
MIN = timedelta(minutes=1)


class Clock:
    """now() / sleep() on a fake timeline; `work` advances time during a tick (an overrun)."""
    def __init__(self, start=T0):
        self.t, self.slept = start, []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += timedelta(seconds=seconds)


class Rig:
    """Replaces run_close / run_pull and builds Deps. Records (name, now, dry_run, run_id) per call."""
    def __init__(self, monkeypatch, clock, *, last_pull=None, close_boom=(), pull_boom=(), pull_credits=3,
                 last_pull_boom=False, work=None):
        self.clock, self.calls, self.last_pull, self.last_pull_calls = clock, [], last_pull, 0
        self.close_boom, self.pull_boom, self.pull_credits = set(close_boom), set(pull_boom), pull_credits
        self.last_pull_boom, self.work = last_pull_boom, work or {}
        monkeypatch.setattr(take_watch.take_close, "run_close", self._close)
        monkeypatch.setattr(take_watch.take_pull, "run_pull", self._pull)

    def _tick_no(self):
        return sum(1 for c in self.calls if c[0] == "close")

    def _close(self, now, deps, *, run_id="local", dry_run=False):
        self.calls.append(("close", now, dry_run, run_id, deps))
        if self._tick_no() in self.close_boom:
            raise RuntimeError("close boom")
        self.clock.t += timedelta(seconds=self.work.get(self._tick_no(), 0))
        return {"status": "ok", "graded": 0}

    def _pull(self, now, deps, *, run_id="local", dry_run=False, leagues=()):
        self.calls.append(("pull", now, dry_run, run_id, deps))
        if self._tick_no() in self.pull_boom:
            raise RuntimeError("pull boom")
        return {"status": "ok", "credits": self.pull_credits, "would_write": []}

    def _last_pull(self):
        self.last_pull_calls += 1
        if self.last_pull_boom:
            raise RuntimeError("db down")
        return self.last_pull

    def deps(self):
        return Deps(close_deps="CLOSE", pull_deps="PULL", last_pull_at=self._last_pull,
                    now=self.clock.now, sleep=self.clock.sleep)

    def names(self):
        return [c[0] for c in self.calls]

    def times(self, name):
        return [c[1] for c in self.calls if c[0] == name]


def watch(monkeypatch, *, minutes=30, tick=300, start=T0, every_hours=2, dry_run=False, **rig_kw):
    clock = Clock(start)
    rig = Rig(monkeypatch, clock, **rig_kw)
    ticks = run_watch(rig.deps(), minutes=minutes, tick_seconds=tick, every_hours=every_hours,
                      run_id="r1", dry_run=dry_run)
    return rig, ticks


# ---- constants --------------------------------------------------------------------------------

def test_constants():
    assert constants.WATCH_MINUTES == 340 and constants.WATCH_TICK_SEC == 300
    assert constants.PULL_HOURS_UTC == frozenset(list(range(12, 24)) + list(range(0, 5)))
    assert (constants.CLOSE_LO_MIN, constants.CLOSE_HI_MIN) == (5, 12)
    assert constants.CLOSE_HI_MIN - constants.CLOSE_LO_MIN > constants.WATCH_TICK_SEC / 60   # every game gets a capture


# ---- the loop ---------------------------------------------------------------------------------

def test_ticks_are_anchored_to_the_start_and_the_deadline_is_respected(monkeypatch):
    rig, ticks = watch(monkeypatch, minutes=30, tick=300)
    assert rig.times("close") == [T0 + 5 * MIN * k for k in range(6)]        # 0, 5, ..., 25; the slot at 30 is the deadline
    assert ticks == 6 and rig.clock.slept == [300.0] * 5
    assert rig.clock.t == T0 + 25 * MIN                                       # it never sleeps past the last tick


def test_an_overrunning_tick_skips_missed_slots_instead_of_bunching(monkeypatch):
    rig, ticks = watch(monkeypatch, minutes=30, tick=300, work={1: 12 * 60})   # tick 1 (at 0) overruns by 12 minutes
    assert rig.times("close") == [T0, T0 + 15 * MIN, T0 + 20 * MIN, T0 + 25 * MIN]
    assert rig.clock.slept[0] == 180.0                                         # slept from +12 to the +15 slot


def test_work_that_ends_exactly_on_a_slot_moves_to_the_next_slot(monkeypatch):
    rig, _ = watch(monkeypatch, minutes=20, tick=300, work={1: 300})
    assert rig.times("close") == [T0, T0 + 10 * MIN, T0 + 15 * MIN]


def test_an_overrun_past_the_deadline_stops_without_sleeping(monkeypatch):
    rig, ticks = watch(monkeypatch, minutes=10, tick=300, work={1: 11 * 60})
    assert ticks == 1 and rig.clock.slept == []


def test_the_first_tick_runs_at_once_even_for_a_tiny_deadline(monkeypatch):
    rig, ticks = watch(monkeypatch, minutes=1, tick=300)
    assert ticks == 1 and rig.times("close") == [T0] and rig.clock.slept == []


def test_the_default_run_is_68_ticks(monkeypatch):
    clock = Clock()
    rig = Rig(monkeypatch, clock)
    assert run_watch(rig.deps(), run_id="r1") == 68                           # 340 min / 5 min, slot 68 is the deadline
    assert clock.t == T0 + 335 * MIN


def test_the_run_id_dry_run_and_deps_are_passed_through(monkeypatch):
    rig, _ = watch(monkeypatch, minutes=1, dry_run=True)
    (_, _, dry, run_id, deps), (_, _, dry_p, run_id_p, deps_p) = rig.calls
    assert (dry, run_id, deps) == (True, "r1", "CLOSE") and (dry_p, run_id_p, deps_p) == (True, "r1", "PULL")


# ---- the tick ---------------------------------------------------------------------------------

def test_close_runs_every_tick_even_outside_pull_hours(monkeypatch):
    rig, _ = watch(monkeypatch, minutes=20, start=datetime(2026, 11, 2, 7, 0, tzinfo=UTC))
    assert rig.names() == ["close"] * 4 and rig.last_pull_calls == 0          # no pull hour: no database read either


def test_close_runs_before_the_pull_within_a_tick(monkeypatch):
    rig, _ = watch(monkeypatch, minutes=1)
    assert rig.names() == ["close", "pull"]


def test_pull_runs_once_per_cadence_when_the_pull_is_remembered(monkeypatch):
    # the database never learns of the pull (a dry run logs nothing), so the in-memory time must stop a repeat
    rig, _ = watch(monkeypatch, minutes=60, dry_run=True, last_pull=None)
    assert rig.times("pull") == [T0]


def test_pull_that_spent_nothing_is_rechecked_next_tick(monkeypatch):
    rig, _ = watch(monkeypatch, minutes=20, pull_credits=0)
    assert rig.times("pull") == [T0 + 5 * MIN * k for k in range(4)]


def test_pull_waits_for_the_logged_cadence(monkeypatch):
    rig, _ = watch(monkeypatch, minutes=30, last_pull=T0 - 100 * MIN, pull_credits=0)
    assert rig.times("pull") == [T0 + 10 * MIN + 5 * MIN * k for k in range(4)]   # 110 minutes since the last pull


@pytest.mark.parametrize("hour,expected", [(4, True), (5, False), (11, False), (12, True), (23, True), (0, True)])
def test_pull_hours_boundaries(hour, expected):
    now = datetime(2026, 11, 2, hour, 30, tzinfo=UTC)
    assert pull_due(now, None, 2) is expected


@pytest.mark.parametrize("ago,expected", [(100, False), (109, False), (110, True), (115, True)])
def test_pull_due_cadence_with_a_ten_minute_slack(ago, expected):
    now = datetime(2026, 11, 2, 14, 7, tzinfo=UTC)
    assert pull_due(now, now - ago * MIN, 2) is expected


def test_pull_due_with_a_longer_cadence_and_no_last_pull():
    now = datetime(2026, 11, 2, 14, 7, tzinfo=UTC)
    assert pull_due(now, now - 160 * MIN, 3) is False and pull_due(now, now - 160 * MIN, 2) is True
    assert pull_due(now, None, 3) is True


def test_a_pull_older_than_the_cadence_is_not_pulled_outside_pull_hours():
    now = datetime(2026, 11, 2, 8, 0, tzinfo=UTC)
    assert pull_due(now, now - 600 * MIN, 2) is False


@pytest.mark.parametrize("raw,expected", [("2", 2), ("3", 3), ("", 2), (None, 2), ("x", 2), ("0", 2), ("-1", 2), ("1", 1)])
def test_every_hours_falls_back_like_take_pull_due(raw, expected):
    assert every_hours_from(raw) == expected
    assert take_watch.take_pull.due(datetime(2026, 11, 2, expected, tzinfo=UTC), raw) is True    # the cadence is the same rule


def test_last_pull_failure_warns_and_counts_as_none(monkeypatch, capsys):
    rig, _ = watch(monkeypatch, minutes=1, last_pull_boom=True)
    assert rig.names() == ["close", "pull"]
    assert "::warning::take-watch: could not read the last pull time: RuntimeError: db down" in capsys.readouterr().out


# ---- failure isolation ------------------------------------------------------------------------

def test_a_close_exception_does_not_stop_the_pull_or_the_loop(monkeypatch, capsys):
    rig, ticks = watch(monkeypatch, minutes=15, close_boom={1}, pull_credits=0)
    assert ticks == 3 and rig.names() == ["close", "pull", "close", "pull", "close", "pull"]
    assert "::warning::take-watch: close failed: RuntimeError: close boom" in capsys.readouterr().out


def test_a_pull_exception_warns_and_the_loop_goes_on(monkeypatch, capsys):
    rig, ticks = watch(monkeypatch, minutes=15, pull_boom={1})
    assert ticks == 3 and rig.names().count("close") == 3
    assert "::warning::take-watch: pull failed: RuntimeError: pull boom" in capsys.readouterr().out
    assert len(rig.times("pull")) == 2                                           # retried next tick, then remembered


def test_a_failed_pull_does_not_count_as_paid(monkeypatch):
    rig, _ = watch(monkeypatch, minutes=15, pull_boom={1, 2})
    assert rig.times("pull") == [T0 + 5 * MIN * k for k in range(3)]             # nothing remembered, so it tries again


def test_result_lines_are_printed(monkeypatch, capsys):
    watch(monkeypatch, minutes=1)
    out = capsys.readouterr().out
    assert "[take-close] {'status': 'ok', 'graded': 0}" in out and "[take-pull]" in out
    assert "[take-watch] done after 1 ticks" in out


# ---- main -------------------------------------------------------------------------------------

def test_main_exits_1_at_once_when_the_environment_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(take_watch, "_db_deps", lambda: pytest.fail("must not wire up without the environment"))
    monkeypatch.setattr(take_watch, "run_watch", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ODDS_API_KEY", "k")
    assert take_watch.main([]) == 1 and "DATABASE_URL is not set" in capsys.readouterr().out
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.delenv("ODDS_API_KEY")
    assert take_watch.main([]) == 1 and "ODDS_API_KEY is not set" in capsys.readouterr().out


def test_main_passes_the_options_through_and_exits_0(monkeypatch):
    seen = {}
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setenv("TAKE_PULL_EVERY_HOURS", "4")
    monkeypatch.setenv("GITHUB_RUN_ID", "99")
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.setattr(take_watch, "_db_deps", lambda: "DEPS")
    monkeypatch.setattr(take_watch, "run_watch", lambda deps, **kw: seen.update(deps=deps, **kw) or 1)
    assert take_watch.main(["--dry-run", "--minutes", "12", "--tick-seconds", "60"]) == 0
    assert seen == {"deps": "DEPS", "minutes": 12.0, "tick_seconds": 60.0, "every_hours": 4, "run_id": "99", "dry_run": True}
    seen.clear()
    assert take_watch.main([]) == 0
    assert seen["minutes"] == constants.WATCH_MINUTES and seen["tick_seconds"] == constants.WATCH_TICK_SEC
    assert seen["dry_run"] is False


def test_main_rejects_a_non_positive_tick():
    with pytest.raises(SystemExit):
        take_watch.main(["--tick-seconds", "0"])
