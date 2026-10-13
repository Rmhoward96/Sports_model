"""Best-effort recording: a ledger failure logs a warning and never fails the writer."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel import db
from sportsmodel.ledger import core, record

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
KICK = NOW + timedelta(hours=5)


def _obs(**kw):
    return core.observation(**{**dict(source="ev_game", sport="cfb", game_pk=9, market="spread", side="home",
                                      commence_time=KICK, line=-3.5, best_price=-110), **kw})


@pytest.fixture(autouse=True)
def _fresh_warnings():
    record.reset_warnings()
    yield
    record.reset_warnings()


class FakeStore:
    def __init__(self, missing=(), last=None, fail=None):
        self.missing, self.last, self.fail, self.inserted, self.asked = list(missing), last or {}, fail, [], []

    def missing_tables(self):
        if self.fail == "missing":
            raise ConnectionError("db down")
        return self.missing

    def latest(self, keys):
        self.asked.append(keys)
        if self.fail == "latest":
            raise RuntimeError("boom")
        return self.last

    def insert(self, rows):
        if self.fail == "insert":
            raise RuntimeError("insert refused")
        self.inserted += rows
        return len(rows)


def test_new_picks_are_inserted_with_the_run_id(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    store = FakeStore()
    out = record.record_observations([_obs()], writer="build_ev_board", now=NOW, store=store)
    assert out["status"] == "ok" and out["appended"] == 1
    (row,) = store.inserted
    assert row["writer"] == "build_ev_board" and row["writer_run"] == "12345" and row["observed_at"] == NOW


def test_local_runs_are_labelled_local(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    store = FakeStore()
    record.record_observations([_obs()], writer="w", now=NOW, store=store)
    assert store.inserted[0]["writer_run"] == "local"


def test_unchanged_picks_insert_nothing():
    o = _obs()
    store = FakeStore(last={o["pick_key"]: {"best_price": -110, "line": -3.5, "model_prob": None, "ref_prob": None}})
    out = record.record_observations([o], writer="w", now=NOW, store=store)
    assert out["appended"] == 0 and store.inserted == []


def test_missing_table_warns_and_skips(capsys):
    store = FakeStore(missing=["pick_ledger"])
    out = record.record_observations([_obs()], writer="w", now=NOW, store=store)
    assert out == {"status": "no_table"} and store.inserted == []
    assert "::warning::pick-ledger: table pick_ledger missing" in capsys.readouterr().out


def test_every_failure_mode_is_swallowed_with_a_warning(capsys):
    for mode in ("missing", "latest", "insert"):
        record.reset_warnings()               # warnings are once-per-(writer, status); reset between modes
        out = record.record_observations([_obs()], writer="w", now=NOW, store=FakeStore(fail=mode))
        assert out["status"] == "error", mode
    assert capsys.readouterr().out.count("::warning::pick-ledger") == 3


def test_safe_record_swallows_a_broken_adapter(capsys):
    def build():
        raise KeyError("matchup")

    out = record.safe_record(build, writer="w", store=FakeStore())
    assert out["status"] == "error" and "building observations failed" in capsys.readouterr().out


def test_safe_record_passes_adapter_output_through():
    store = FakeStore()
    out = record.safe_record(lambda: [_obs(), None], writer="w", now=NOW, store=store)
    assert out["appended"] == 1


def test_empty_batches_never_touch_the_store():
    store = FakeStore(fail="missing")
    assert record.record_observations([], writer="w", store=store) == {"status": "empty"}


def test_the_test_suite_cannot_reach_the_production_ledger(monkeypatch):
    """conftest.py sets SM_LEDGER_OFF: with no injected store the default DB store is never built."""
    assert record.writes_disabled()
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("DB touched")))
    assert record.record_observations([_obs()], writer="w", now=NOW) == {"status": "disabled"}
    monkeypatch.setenv("SM_LEDGER_OFF", "0")
    assert not record.writes_disabled()


def test_warnings_are_printed_once_per_writer_and_status(capsys):
    for _ in range(3):
        record.record_observations([_obs()], writer="w", now=NOW, store=FakeStore(fail="latest"))
        record.record_observations([_obs()], writer="w", now=NOW, store=FakeStore(missing=["pick_ledger"]))
    record.record_observations([_obs()], writer="other", now=NOW, store=FakeStore(fail="latest"))
    assert capsys.readouterr().out.count("::warning::pick-ledger") == 3     # w/error, w/no_table, other/error


def test_exception_text_in_the_warning_and_result_is_truncated(capsys):
    class Loud(FakeStore):
        def latest(self, keys):
            raise RuntimeError("x" * 5000)

    out = record.record_observations([_obs()], writer="w", now=NOW, store=Loud())
    assert len(out["error"]) <= 200 and len(capsys.readouterr().out) < 400


def test_the_db_store_refuses_every_call_while_writes_are_disabled(monkeypatch):
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("DB touched")))
    store = record.DbStore()
    for call in (store.missing_tables, lambda: store.latest(["k"]), lambda: store.insert([{}])):
        with pytest.raises(RuntimeError, match="SM_LEDGER_OFF"):
            call()
    out = record.record_observations([_obs()], writer="w", now=NOW, store=store)   # swallowed, not raised
    assert out["status"] == "error"


def test_one_bad_candidate_is_skipped_not_the_batch(capsys):
    store = FakeStore()
    bad_thunk = lambda: _obs(ref_kind="vibes")
    good_thunk = lambda: _obs(game_pk=10)
    out = record.safe_record(lambda: [_obs(), bad_thunk, good_thunk, {"no": "pick_key"}], writer="w", now=NOW,
                             store=store)
    assert out["status"] == "ok" and out["appended"] == 2 and out["skipped"] == 2
    assert len(store.inserted) == 2
