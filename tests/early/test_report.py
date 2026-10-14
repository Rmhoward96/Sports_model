"""report.py: summary numbers, the markdown for both statuses and the decision block round trip. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel.early import passtest, report

from .builders import res, spreads, totals

AS_OF = datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc)          # a Tuesday


def tick(i, sport="cfb", market="total", days_ago=2, kick_hours_ago=60):
    return {"id": i, "sport": sport, "market": market, "observed_at": AS_OF - timedelta(days=days_ago),
            "commence_time": AS_OF - timedelta(hours=kick_hours_ago)}


def result(ticket_id, **kw):
    return res(**kw) | {"ticket_id": ticket_id}


def test_summary_counts_tickets_by_market_and_week_and_ungraded():
    tickets = [tick(1), tick(2, days_ago=10), tick(3, sport="nfl", market="spread"), tick(4, kick_hours_ago=3),
               tick(5, kick_hours_ago=100)]
    s = report.summarize(tickets, [result(1)], AS_OF)
    assert s["tickets"]["cfb_total"] == {"all": 4, "week": 3} and s["tickets"]["nfl_spread"] == {"all": 1, "week": 1}
    assert s["tickets"]["cfb_spread"] == {"all": 0, "week": 0} and s["tickets_total"] == 5
    assert s["ungraded"] == 3          # 2, 3 and 5 started long ago without a result; 4 kicked off 3 hours ago (not yet due); 1 graded


def test_summary_close_coverage_and_missing_share():
    results = [result(1), result(2, kind="missing", ev=None), result(3), result(4, kind="missing", ev=None)]
    s = report.summarize([], results, AS_OF)
    assert (s["graded"], s["with_close"], s["missing_close"], s["missing_share"]) == (4, 2, 2, 0.5)
    assert report.summarize([], [], AS_OF)["missing_share"] is None


def test_running_numbers_use_only_tickets_with_a_pinnacle_close():
    results = [result(1, ev=0.04, avail=True), result(2, ev=-0.02, avail=False, game_pk=2),
               result(3, kind="missing", ev=None, game_pk=3)]
    m = report.summarize([], results, AS_OF)["markets"]["cfb_total"]
    assert m["n"] == 2 and m["mean_close_ev"] == pytest.approx(0.01) and m["beat_rate"] == 0.5 and m["avail_share"] == 0.5


def test_a_collecting_report_states_progress_and_never_says_pass_or_fail():
    results = [result(i, game_pk=i) for i in range(10)]
    decision = passtest.decide(results, None, "2026-11-03")
    text = report.render_report(as_of=AS_OF, summary=report.summarize([], results, AS_OF), decision=decision)
    assert "COLLECTING" in text and "10 of 150" in text and "10 of 60 CFB totals" in text
    assert "PASS" not in text and "FAIL" not in text and "INSUFFICIENT" not in text
    assert "bowl season" in text and "Running numbers" in text


def test_a_judged_report_lists_each_markets_verdict_and_criteria():
    results = [result(i, game_pk=i) for i in range(100)] + [
        result(1000 + i, sport="cfb", market="spread", game_pk=1000 + i) for i in range(50)] + [
        result(2000 + i, sport="nfl", market="spread", game_pk=2000 + i) for i in range(39)]
    decision = passtest.decide(results, None, "2026-11-03")
    text = report.render_report(as_of=AS_OF, summary=report.summarize([], results, AS_OF), decision=decision)
    assert "JUDGED on 2026-11-03" in text and "| CFB totals | PASS |" in text and "| CFB spreads | PASS |" in text
    assert "| NFL spreads | INSUFFICIENT |" in text and "39 graded tickets" in text
    assert "separate short design for live alerts" in text and "never gated" in text
    assert "informational only" not in text                       # judged this very week


def test_a_later_report_carries_the_frozen_verdict_and_says_so():
    frozen = passtest.decide(totals(100) + spreads(45), None, "2026-11-03")
    later = datetime(2026, 11, 17, 15, 52, tzinfo=timezone.utc)
    text = report.render_report(as_of=later, summary=report.summarize([], [], later), decision=frozen)
    assert "JUDGED on 2026-11-03" in text and "Later weeks below are informational only" in text


def test_the_decision_block_round_trips_and_the_latest_earlier_one_is_chosen():
    d1 = passtest.decide([], None, "2026-10-20")
    d2 = passtest.decide(totals(100) + spreads(45), None, "2026-11-03")
    r1 = report.render_report(as_of=datetime(2026, 10, 20, 15, 52, tzinfo=timezone.utc), summary=report.summarize([], [], AS_OF), decision=d1)
    r2 = report.render_report(as_of=datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc), summary=report.summarize([], [], AS_OF), decision=d2)
    assert report.parse_report(r2)["decision"]["status"] == "judged"
    assert report.latest_decision([r1, r2], datetime(2026, 11, 10, 15, 52, tzinfo=timezone.utc))["status"] == "judged"
    assert report.latest_decision([r1, r2], datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc))["status"] == "collecting"
    assert report.latest_decision([r1, r2], datetime(2026, 10, 20, 15, 52, tzinfo=timezone.utc)) is None
    assert report.latest_decision(["not a report", ""], AS_OF) is None


def test_parse_ignores_a_corrupt_block():
    assert report.parse_report("<!-- early-monitor: {not json} -->") is None and report.parse_report(None) is None


def test_ungraded_separates_not_finished_yet_from_stale_beyond_48_hours():
    tickets = [tick(1, kick_hours_ago=3),        # kicked off 3 h ago: not due yet, in neither count
               tick(2, kick_hours_ago=10),       # past the settle window, within 48 h: waiting on a final
               tick(3, kick_hours_ago=48),       # exactly 48 h: not yet stale
               tick(4, kick_hours_ago=49),       # stale
               tick(5, kick_hours_ago=200),      # stale
               tick(6, kick_hours_ago=200)]      # stale but graded: counts nowhere
    s = report.summarize(tickets, [result(6)], AS_OF)
    assert s["ungraded"] == 4 and s["ungraded_stale"] == 2
    text = report.render_report(as_of=AS_OF, summary=s, decision=passtest.decide([], None, "2026-11-03"))
    assert "4 tickets whose game started more than 4 hours ago are not graded yet" in text
    assert "2 of them started more than 48 hours ago and are still ungraded" in text and "prediction_accuracy" in text


def test_the_stale_ungraded_line_is_shown_even_when_zero():
    s = report.summarize([tick(1)], [result(1)], AS_OF)
    text = report.render_report(as_of=AS_OF, summary=s, decision=passtest.decide([], None, "2026-11-03"))
    assert s["ungraded_stale"] == 0 and "0 of them started more than 48 hours ago" in text


def test_an_earlier_judged_block_wins_over_any_later_block():
    judged = report.render_report(as_of=datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc), summary=report.summarize([], [], AS_OF),
                                  decision=passtest.decide(totals(100) + spreads(45), None, "2026-11-03"))
    drifted = report.render_report(as_of=datetime(2026, 11, 10, 15, 52, tzinfo=timezone.utc), summary=report.summarize([], [], AS_OF),
                                   decision=passtest.decide([], None, "2026-11-10"))        # a hand-edited / re-collecting report
    assert report.latest_decision([judged, drifted], datetime(2026, 11, 17, 15, 52, tzinfo=timezone.utc))["judged_on"] == "2026-11-03"
