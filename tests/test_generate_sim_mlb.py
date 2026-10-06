"""MLB producer wiring in scripts/generate_sim.py: first-pitch times + never re-predict a started game.

Loads the script module directly (it isn't a package). No network, no DB.
"""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_sim.py"
_spec = importlib.util.spec_from_file_location("generate_sim_mlb_under_test", _PATH)
generate_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_sim)

NOW = datetime(2026, 10, 6, 20, 0, tzinfo=timezone.utc)


def _g(pk, day="2026-10-06"):
    return {"game_pk": pk, "game_date": day}


def test_attach_commence_times_reads_one_schedule_call_per_date():
    calls = []

    def fake_schedule(day):
        calls.append(day)
        return [{"game_pk": 1, "commence_time": "2026-10-06T22:00:00Z"},
                {"game_pk": 2, "commence_time": "2026-10-07T01:30:00Z"},
                {"game_pk": 3, "commence_time": "2026-10-07T20:00:00Z"}]

    games = [_g(1), _g(2), _g(3, day="2026-10-07"), _g(99)]
    generate_sim.attach_commence_times(games, fetch_schedule=fake_schedule)
    assert calls == ["2026-10-06", "2026-10-07"]
    assert [g["commence_time"] for g in games] == [
        "2026-10-06T22:00:00Z", "2026-10-07T01:30:00Z", "2026-10-07T20:00:00Z", None]


def test_attach_commence_times_survives_a_statsapi_failure():
    def flaky(day):
        if day == "2026-10-07":
            raise RuntimeError("boom")
        return [{"game_pk": 1, "commence_time": "2026-10-06T22:00:00Z"}]

    games = [_g(1), _g(3, day="2026-10-07")]
    generate_sim.attach_commence_times(games, fetch_schedule=flaky)
    assert games[0]["commence_time"] == "2026-10-06T22:00:00Z" and games[1]["commence_time"] is None


def test_started_only_when_first_pitch_is_known_and_past():
    assert generate_sim.started({"commence_time": "2026-10-06T19:59:00Z"}, NOW) is True
    assert generate_sim.started({"commence_time": "2026-10-06T20:00:00Z"}, NOW) is True
    assert generate_sim.started({"commence_time": "2026-10-06T22:00:00Z"}, NOW) is False
    assert generate_sim.started({"commence_time": None}, NOW) is False      # unknown -> still predicted
    assert generate_sim.started({}, NOW) is False


def test_markets_stay_the_pre_pause_published_set():
    # hits / hrr / home_run were dropped (206941b); reactivation must not republish them.
    assert generate_sim.SIM_BATTER_MARKETS == ["total_bases"]
    assert generate_sim.ANALYTIC_PITCHER_MARKETS == ["pitcher_ks", "hits_allowed"]


def test_market_lines_from_rows_newest_per_market_and_ignores_other_markets():
    rows = [(1, "spread", -1.5), (1, "spread", -1.0), (1, "total", 8.5), (1, "moneyline", None),
            (2, "total", 7.0), (2, "total", None), (3, "spread", None)]
    assert generate_sim.market_lines_from_rows(rows) == {
        1: {"market_spread": -1.5, "market_total": 8.5},
        2: {"market_spread": None, "market_total": 7.0},
    }


def test_load_market_lines_is_empty_without_a_database_or_games(monkeypatch):
    monkeypatch.setattr(generate_sim.config, "DATABASE_URL", None)
    assert generate_sim.load_market_lines([1, 2]) == {}
    monkeypatch.setattr(generate_sim.config, "DATABASE_URL", "postgresql://x")
    assert generate_sim.load_market_lines([]) == {}


def test_load_market_lines_swallows_a_failed_read(monkeypatch, capsys):
    import sportsmodel.db as db
    monkeypatch.setattr(generate_sim.config, "DATABASE_URL", "postgresql://x")
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert generate_sim.load_market_lines([1]) == {}
    assert "market line lookup failed" in capsys.readouterr().out
