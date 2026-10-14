"""The ledger hooks in each pick writer: they see exactly the picks the writer just wrote, run AFTER the writer's
own database write, and can never fail the writer. Every IO call in the scripts is stubbed; nothing here can reach
a database (SM_LEDGER_OFF is set by the root conftest.py, and the scripts' DB helpers are replaced)."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel import db
from sportsmodel.ledger import record, sources
from sportsmodel.shrink.store import ShrinkTable

ROOT = Path(__file__).resolve().parents[2]
KICK = datetime.now(timezone.utc) + timedelta(hours=30)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"ledger_hook_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _fresh_warnings():
    record.reset_warnings()
    yield
    record.reset_warnings()


class Recorder:
    """Stands in for `safe_record`: builds the candidates the way the real one does (callables are called, a
    candidate that raises is dropped) and keeps what it built, in call order with the writer's own events."""

    def __init__(self, events=None):
        self.events, self.calls = events if events is not None else [], []

    def __call__(self, build, *, writer, **kw):
        self.events.append(("ledger", writer))
        self.calls.append((writer, sources.build_all(list(build()))))
        return {"status": "ok"}


# ---- stubs for each writer: (setup, run) ----------------------------------------------------------------

def _stub_board(mod, monkeypatch, events):
    game = {"sport": "nfl", "game_pk": 101, "matchup": "A @ B", "commence_time": KICK}
    odds = []
    for market, side, line, price in [
            ("moneyline", "home", None, -150), ("moneyline", "away", None, 130),
            ("spread", "home", -3.5, -110), ("spread", "away", 3.5, -110),
            ("total", "over", 47.5, -110), ("total", "under", 47.5, -110)]:
        odds.append({"game_pk": 101, "market": market, "side": side, "book": "pinnacle", "line": line, "price": price})
    odds.append({"game_pk": 101, "market": "spread", "side": "home", "book": "draftkings", "line": -3.5, "price": 105})
    monkeypatch.setattr(mod, "load_upcoming_games", lambda sport: [game])
    monkeypatch.setattr(mod, "load_latest_odds", lambda pks: odds)
    monkeypatch.setattr(mod, "upsert_ev_picks", lambda rows: events.append(("upsert", len(rows))) or len(rows))
    monkeypatch.setattr(mod, "clear_other_side_picks", lambda rows: events.append(("clear", 0)) or 0)
    monkeypatch.setattr(mod, "build_parlay", lambda cands: {"parlay_id": "p1", "n_legs": 2, "parlay_price": 260, "ev": 0.1})
    monkeypatch.setattr(mod, "upsert_ev_parlays", lambda rows: events.append(("parlay_upsert", len(rows))) or 0)
    monkeypatch.setattr(mod, "demote_stale_parlays", lambda *a: events.append(("demote", 0)) or 0)
    monkeypatch.setattr(sys, "argv", ["build_ev_board.py", "--sport", "nfl"])


def _prop_row(**kw):
    return {"sport": "nfl", "game_pk": 1, "player_id": "00-1", "player_name": "A. Brown", "market": "rec_yds",
            "side": "over", "line": 64.5, "model_version": "props-sim-v1", "commence_time": KICK,
            "model_prob": 0.58, "market_prob": 0.5, "ev_best": 0.05, "best_book": "dk", "best_price": -105, **kw}


def _stub_props(mod, monkeypatch, events, rows=None):
    rows = rows if rows is not None else [_prop_row(is_pick=True), _prop_row(player_id="00-2", is_pick=False)]
    monkeypatch.setattr(mod, "load_sim_rows", lambda sport: [{"game_pk": 1}])
    monkeypatch.setattr(mod, "load_latest_prop_odds", lambda sport, pks: [])
    monkeypatch.setattr(mod, "assemble_prop_line_rows", lambda a, b: [])
    monkeypatch.setattr(mod, "upsert_nfl_prop_lines", lambda r: events.append(("lines", len(r))) or 0)
    monkeypatch.setattr(mod, "assemble_prop_rows", lambda *a, **k: rows)
    monkeypatch.setattr(mod, "upsert_ev_prop_picks", lambda r: events.append(("upsert", len(r))))
    monkeypatch.setattr(mod, "clear_stale_prop_line_picks", lambda picks: events.append(("clear", len(picks))) or 0)


def _stub_parlays(mod, monkeypatch, events):
    leg = lambda pk, market: {"key": f"g:{pk}", "kind": "game", "sport": "nfl", "game_pk": pk, "market": market,
                              "side": "home", "line": -3.5, "prob": 0.6, "price": -110, "commence_time": KICK,
                              "player_id": None, "player_name": None, "label": f"Team {pk}"}
    ticket = {"parlay_id": "dk|g:1|g:2", "book": "dk", "legs": [leg(1, "spread"), leg(2, "spread")],
              "first_commence": None, "ev": 0.05, "parlay_price": 260}
    monkeypatch.setattr(mod, "load_game_picks", lambda: [])
    monkeypatch.setattr(mod, "load_prop_picks", lambda: [])
    monkeypatch.setattr(mod, "assemble_game_legs", lambda a, b: [])
    monkeypatch.setattr(mod, "assemble_prop_legs", lambda a, b: [])
    monkeypatch.setattr(mod, "build_best_parlays", lambda legs, excluded_keys: [ticket])
    monkeypatch.setattr(mod, "locked_leg_keys", lambda: frozenset())
    monkeypatch.setattr(mod, "replace_unlocked_best_parlays", lambda t: events.append(("replace", len(t))) or len(t))


def _pred_row(**kw):
    return {"sport": "nfl", "game_pk": 5, "model_version": "sim-nfl-ml-v2", "commence_time": KICK,
            "home_win_prob": 0.62, "pred_home_score": 27.0, "pred_away_score": 20.0, "market_spread": -3.0,
            "market_total": 44.5, "margin_dist": None, "total_dist": None, **kw}


def _stub_cfb(mod, monkeypatch, events):
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: events.append(("upsert", len(rows))))


def _stub_sim_nfl(mod, monkeypatch, events):
    monkeypatch.setattr(mod, "game_prediction_row",
                        lambda g, sims, cfg, mv: {**_pred_row(), "margin_dist": {}, "total_dist": {}})
    monkeypatch.setattr(mod.nfl_config, "load_gameline", lambda: None)
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: events.append(("upsert", len(rows))))


def _stub_sim_mlb(mod, monkeypatch, events):
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: events.append(("upsert", len(rows))) or len(rows))
    monkeypatch.setattr(mod, "upsert_prop_predictions", lambda rows: events.append(("props", len(rows))) or len(rows))
    monkeypatch.setattr(mod.config, "DATABASE_URL", "postgresql://x")


def _mlb_pred():
    return {k: v for k, v in _pred_row().items() if k != "sport"} | {"model_version": "sim-mlb-v1"}


# (module, stub, run) -- run() drives the writer exactly as production does
WRITERS = {
    "build_ev_board": ("build_ev_board", _stub_board, lambda mod: mod.main()),
    "build_ev_props_board": ("build_ev_props_board", _stub_props, lambda mod: mod.main(["--sport", "nfl"])),
    "build_best_parlays": ("build_best_parlays", _stub_parlays, lambda mod: mod.main()),
    "generate_cfb": ("generate_cfb", _stub_cfb, lambda mod: mod.write_game_rows([_pred_row(sport="cfb")])),
    "generate_sim_nfl": ("generate_sim_nfl", _stub_sim_nfl,
                         lambda mod: mod._write_game_lines([{"game_pk": 5}], {5: object()}, {5})),
    "generate_sim": ("generate_sim", _stub_sim_mlb, lambda mod: mod.write_predictions([_mlb_pred()], [{"game_pk": 5}])),
}


# ---- build_ev_board ---------------------------------------------------------------------------------

def test_build_ev_board_records_its_surfaced_picks_after_its_own_writes(monkeypatch):
    mod, events = load_script("build_ev_board"), []
    _stub_board(mod, monkeypatch, events)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main()
    assert [e[0] for e in events] == ["upsert", "clear", "parlay_upsert", "demote", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "build_ev_board"
    (o,) = obs
    assert (o["source"], o["market"], o["side"], o["line"], o["best_price"], o["best_book"]) == \
        ("ev_game", "spread", "home", -3.5, 105, "draftkings")
    assert o["ref_kind"] == "pinnacle_novig" and o["ref_prob"] == pytest.approx(0.5) and o["model_version"] == "ev-pilot-v1"
    assert o["model_prob"] is None


def test_build_ev_board_survives_a_ledger_failure(monkeypatch, capsys):
    mod, events = load_script("build_ev_board"), []
    _stub_board(mod, monkeypatch, events)
    monkeypatch.setattr(sources, "ev_game_candidates", lambda *a: (_ for _ in ()).throw(KeyError("boom")))
    mod.main()                                                    # real safe_record: must not raise
    assert "::warning::pick-ledger: build_ev_board: building observations failed" in capsys.readouterr().out
    assert ("upsert", 3) in events and ("clear", 0) in events   # the board itself was written


# ---- build_ev_props_board ---------------------------------------------------------------------------

def test_build_ev_props_board_records_only_is_pick_rows_after_the_upsert(monkeypatch):
    mod, events = load_script("build_ev_props_board"), []
    _stub_props(mod, monkeypatch, events)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main(["--sport", "nfl"])
    assert [e[0] for e in events] == ["lines", "upsert", "clear", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "build_ev_props_board" and [o["player_id"] for o in obs] == ["00-1"]
    assert obs[0]["line"] == 64.5 and obs[0]["ref_prob"] == 0.5 and obs[0]["model_prob"] == 0.58


def test_build_ev_props_board_records_the_main_line_not_the_alternates(monkeypatch):
    """Real `assemble_prop_rows` over odds that hold a main line (3 books) plus alternate lines (1 book each):
    the board picks the main line, and the ledger sees exactly ONE observation for that pick key, at that line."""
    mod, events = load_script("build_ev_props_board"), []
    sim = [{"game_pk": 1, "player_id": "00-1", "name": "A. Brown", "market": "rec_yds", "mean": 66.0,
            "dist": {"kind": "normal", "mean": 66.0, "sd": 20.0}, "commence_time": KICK, "matchup": "A @ B"}]

    def odds(line, book, side, price):
        return {"game_pk": 1, "market": "reception_yds", "side": side, "player_name": "A. Brown", "book": book,
                "line": line, "price": price, "captured_at": KICK}

    odds_rows = [odds(64.5, b, s, -110) for b in ("draftkings", "fanduel", "caesars") for s in ("over", "under")]
    odds_rows += [odds(54.5, "draftkings", "over", -200), odds(54.5, "draftkings", "under", 150),
                  odds(79.5, "draftkings", "over", 160), odds(79.5, "draftkings", "under", -220)]
    monkeypatch.setattr(mod, "load_sim_rows", lambda sport: sim)
    monkeypatch.setattr(mod, "load_latest_prop_odds", lambda sport, pks: odds_rows)
    monkeypatch.setattr(mod, "upsert_nfl_prop_lines", lambda r: 0)
    monkeypatch.setattr(mod, "upsert_ev_prop_picks", lambda r: events.append(("upsert", len(r))))
    monkeypatch.setattr(mod, "clear_stale_prop_line_picks", lambda picks: 0)
    # k = 1: the raw model decides, as before the shrink; the test is about WHICH LINE is recorded, not the probability
    monkeypatch.setattr(mod, "load_table", lambda: ShrinkTable({("nfl", "reception_yds"): 1.0}))
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main(["--sport", "nfl"])
    ((_, obs),) = rec.calls
    assert [(o["player_id"], o["market"], o["line"]) for o in obs] == [("00-1", "rec_yds", 64.5)]
    assert len({o["pick_key"] for o in obs}) == len(obs)


# ---- build_best_parlays -----------------------------------------------------------------------------

def test_build_best_parlays_records_every_leg_after_the_replace(monkeypatch):
    mod, events = load_script("build_best_parlays"), []
    _stub_parlays(mod, monkeypatch, events)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main()
    assert [e[0] for e in events] == ["replace", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "build_best_parlays" and len(obs) == 2
    assert {o["parlay_id"] for o in obs} == {"dk|g:1|g:2"} and obs[0]["model_version"] == "ev-parlays-v1"


# ---- the three model-game writers --------------------------------------------------------------------

def test_generate_cfb_write_game_rows_upserts_then_records(monkeypatch):
    mod, events = load_script("generate_cfb"), []
    _stub_cfb(mod, monkeypatch, events)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.write_game_rows([_pred_row(sport="cfb")])
    assert [e[0] for e in events] == ["upsert", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "generate_cfb" and {o["market"] for o in obs} == {"moneyline", "spread", "total"}


def test_generate_sim_nfl_game_lines_record_the_rows_they_wrote(monkeypatch):
    mod, events = load_script("generate_sim_nfl"), []
    _stub_sim_nfl(mod, monkeypatch, events)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod._write_game_lines([{"game_pk": 5}], {5: object()}, {5})
    assert [e[0] for e in events] == ["upsert", "ledger"]
    assert rec.calls[0][0] == "generate_sim_nfl" and len(rec.calls[0][1]) >= 1


def test_generate_sim_records_after_the_prop_predictions_too(monkeypatch):
    mod, events = load_script("generate_sim"), []
    _stub_sim_mlb(mod, monkeypatch, events)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.write_predictions([_mlb_pred()], [{"game_pk": 5}])
    assert [e[0] for e in events] == ["upsert", "props", "ledger"]
    (writer, obs) = rec.calls[0]
    assert writer == "generate_sim" and {o["sport"] for o in obs} == {"mlb"}


def test_generate_sim_without_a_database_never_records(monkeypatch):
    mod = load_script("generate_sim")
    rec = Recorder()
    monkeypatch.setattr(mod, "safe_record", rec)
    monkeypatch.setattr(mod.config, "DATABASE_URL", "")
    monkeypatch.setattr(mod, "get_duckdb", lambda: (_ for _ in ()).throw(RuntimeError("stop at the duckdb branch")))
    with pytest.raises(RuntimeError, match="duckdb branch"):
        mod.write_predictions([_mlb_pred()], [{"game_pk": 5}])
    assert rec.calls == []


def test_a_ledger_failure_never_fails_a_model_game_writer(monkeypatch, capsys):
    mod = load_script("generate_cfb")
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: None)
    monkeypatch.setattr(sources, "model_game_candidates", lambda rows: (_ for _ in ()).throw(RuntimeError("x")))
    mod.write_game_rows([_pred_row()])                            # real safe_record swallows the adapter error
    assert "::warning::pick-ledger: generate_cfb" in capsys.readouterr().out


def test_the_hooks_never_reach_the_database_under_test(monkeypatch):
    """SM_LEDGER_OFF (conftest.py) means even the real safe_record cannot touch the DB from a script main()."""
    assert record.writes_disabled()
    mod = load_script("generate_cfb")
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: None)
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("DB touched")))
    mod.write_game_rows([_pred_row()])


# ---- every writer behaves identically whatever state the ledger is in ---------------------------------------

class _TablesMissing:
    def missing_tables(self):
        return ["pick_ledger"]

    def latest(self, keys):
        raise AssertionError("must not be reached")

    def insert(self, rows):
        raise AssertionError("must not be reached")


class _DbDown:
    def missing_tables(self):
        raise ConnectionError("db down")

    latest = insert = missing_tables


def _ledger_mode(mode, monkeypatch):
    """Put the ledger in a failure state without ever touching a database."""
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("DB touched")))
    if mode == "off":
        monkeypatch.setenv("SM_LEDGER_OFF", "1")
        return
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)             # the ledger is armed, the store is broken
    monkeypatch.setattr(record, "DbStore", _TablesMissing if mode == "tables_missing" else _DbDown)


def _run_writer(name, monkeypatch):
    modname, stub, run = WRITERS[name]
    mod, events = load_script(modname), []
    stub(mod, monkeypatch, events)
    return mod, events, run


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_the_ledger_is_the_last_database_step_of_every_writer(name, monkeypatch):
    """A slow or dead ledger must never delay or skip one of the writer's own writes: every one comes first."""
    modname, stub, run = WRITERS[name]
    mod, events = load_script(modname), []
    stub(mod, monkeypatch, events)
    monkeypatch.setattr(mod, "safe_record", Recorder(events))
    run(mod)
    assert [e for e in events if e[0] == "ledger"] == [("ledger", name)]
    assert events[-1] == ("ledger", name) and len(events) >= 2


def test_generate_sim_ledger_is_last_even_without_prop_rows(monkeypatch):
    mod, events = load_script("generate_sim"), []
    _stub_sim_mlb(mod, monkeypatch, events)
    monkeypatch.setattr(mod, "safe_record", Recorder(events))
    mod.write_predictions([_mlb_pred()], [])
    assert [e[0] for e in events] == ["upsert", "ledger"]


@pytest.mark.parametrize("name", sorted(WRITERS))
def test_a_broken_sources_import_cannot_crash_a_writer(name, monkeypatch, capsys):
    """sportsmodel.ledger.sources is imported inside the hook, so even an ImportError there is just a warning."""
    modname, stub, run = WRITERS[name]
    mod, events = load_script(modname), []
    stub(mod, monkeypatch, events)
    monkeypatch.setitem(sys.modules, "sportsmodel.ledger.sources", None)       # `import` now raises ImportError
    run(mod)
    assert [e[0] for e in events]                                                # the writer still wrote
    assert f"::warning::pick-ledger: {name}: building observations failed" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(WRITERS))
@pytest.mark.parametrize("mode", ["tables_missing", "db_down", "off"])
def test_a_writer_behaves_identically_whatever_state_the_ledger_is_in(name, mode, monkeypatch, capsys):
    # baseline: the writer with its ledger hook stubbed out entirely
    mod, base_events, run = _run_writer(name, monkeypatch)
    monkeypatch.setattr(mod, "safe_record", lambda build, **kw: {"status": "stubbed"})
    run(mod)
    assert base_events                                              # the writer did write something

    # the real hook against a broken ledger: same writes, no exception, nothing escapes
    mod, events, run = _run_writer(name, monkeypatch)
    _ledger_mode(mode, monkeypatch)
    capsys.readouterr()
    run(mod)
    assert events == base_events
    # ...and the ledger really was attempted: a broken store warns once, an OFF ledger stays silent
    assert ("::warning::pick-ledger" in capsys.readouterr().out) == (mode != "off")
