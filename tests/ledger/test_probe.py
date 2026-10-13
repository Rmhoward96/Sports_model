"""The Pinnacle-props probe: verdict logic, event choice, the odds client's credit headers and the workflow."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel.ingest import odds
from sportsmodel.ledger.probe import pinnacle_props_verdict
from tests.test_workflows_props_ml import WF, load, runs, steps_of

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("probe_pinnacle_props", ROOT / "scripts" / "probe_pinnacle_props.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)


def _book(key, *market_outcomes):
    return {"key": key, "markets": [{"key": f"player_m{i}", "outcomes": [{}] * n}
                                    for i, n in enumerate(market_outcomes)]}


def test_verdict_pinnacle_has_props():
    v = pinnacle_props_verdict({"bookmakers": [_book("pinnacle", 4, 0, 2), _book("betfair_ex_eu", 6)]})
    assert v["verdict"] == "pinnacle_has_props" and v["pinnacle_outcomes"] == 6
    assert v["pinnacle_markets"] == ["player_m0", "player_m2"] and v["bookmaker_outcomes"]["betfair_ex_eu"] == 6


def test_verdict_pinnacle_absent_when_other_books_have_props():
    v = pinnacle_props_verdict({"bookmakers": [_book("betfair_ex_eu", 3), _book("pinnacle", 0, 0)]})
    assert v["verdict"] == "pinnacle_absent" and v["pinnacle_outcomes"] == 0


@pytest.mark.parametrize("payload", [None, {}, {"bookmakers": []}, {"bookmakers": [_book("pinnacle", 0)]}])
def test_verdict_inconclusive_when_no_book_returned_any_prop(payload):
    assert pinnacle_props_verdict(payload)["verdict"] == "inconclusive"


def test_next_event_is_the_soonest_unstarted_one():
    ev = lambda i, h: {"id": i, "commence_time": (NOW + timedelta(hours=h)).isoformat()}
    assert probe.next_event([ev("late", 80), ev("past", -2), ev("soon", 30), {"id": "x"}], NOW)["id"] == "soon"
    assert probe.next_event([ev("past", -2)], NOW) is None


def test_run_reports_the_verdict_and_credits():
    asked = []
    events = [{"id": "e1", "home_team": "B", "away_team": "A", "commence_time": (NOW + timedelta(hours=30)).isoformat()}]
    rep = probe.run(NOW, lambda: events, lambda eid: asked.append(eid) or {"bookmakers": [_book("pinnacle", 5)]},
                    lambda: "19000", lambda: "8")
    assert asked == ["e1"] and rep["verdict"] == "pinnacle_has_props" and rep["event"] == "A @ B"
    assert (rep["credits_last"], rep["credits_remaining"]) == ("8", "19000")


def test_run_with_no_upcoming_event_spends_nothing():
    def boom(eid):
        raise AssertionError("paid call")
    rep = probe.run(NOW, lambda: [], boom, lambda: "19000", lambda: None)
    assert rep["verdict"] == "inconclusive" and rep["reason"] == "no upcoming NFL event"


def test_the_probe_costs_about_eight_credits_one_event_eu_only():
    from sportsmodel import sports
    assert len(sports.get("nfl").prop_market_map) == 8          # 8 markets x 1 region x 1 event = 8 credits


def test_main_refuses_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert probe.main() == 1 and "ODDS_API_KEY is not set" in capsys.readouterr().out


# ---- odds client: credit headers --------------------------------------------------------------------

class _Resp:
    headers = {"x-requests-remaining": "18990", "x-requests-last": "8"}

    def raise_for_status(self):
        pass

    def json(self):
        return {"ok": True}


class _Client:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None):
        return _Resp()


def test_get_records_remaining_and_last_call_cost(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(odds.httpx, "Client", _Client)
    assert odds._get("/sports/x/events", {}) == {"ok": True}
    assert odds.last_requests_remaining == "18990" and odds.last_requests_last == "8"


# ---- workflow --------------------------------------------------------------------------------------

def test_probe_workflow_is_manual_read_only_and_runs_the_script_with_the_key():
    wf = load("probe-pinnacle-props.yml")
    assert list(wf["on"]) == ["workflow_dispatch"] and wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    (step,) = [s for s in steps_of(job) if runs("scripts/probe_pinnacle_props.py")(s)]
    assert step["env"]["ODDS_API_KEY"] == "${{ secrets.ODDS_API_KEY }}" and "DATABASE_URL" not in step["env"]
    text = (WF / "probe-pinnacle-props.yml").read_text()
    assert "git push" not in text and "DATABASE_URL" not in text
