"""The weekly shrink report text and its machine-readable decision block. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel.shrink import fit, passtest, report

AS_OF = datetime(2026, 11, 3, 15, 30, tzinfo=timezone.utc)


def doc(n=400):
    legs = [{"season": 2026, "week": 4, "game_pk": i, "market": "rush_yds", "p": 0.6, "m": 0.5, "y": 0.54, "version": "v2"}
            for i in range(n)]
    return fit.build_document(legs, "v2", "2026-11-03T15:30:00+00:00")


COLLECTING = {"status": "collecting", "scored": 640, "need_scored": 2000,
              "running": {"n": 640, "games": 320, "brier_market": 0.2501, "brier_shrunk": 0.2499, "gain": 0.0002,
                          "lower": -0.0005, "slope": 0.91}}
JUDGED = {"status": "judged", "judged_on": "2026-12-08", "scored": 2010, "need_scored": 2000, "running": COLLECTING["running"],
          "verdict": "FAIL", "criteria": [{"name": n, "passed": ok, "detail": "x"} for n, ok in zip(passtest.CRITERIA, (True, False, True))]}


def test_a_collecting_report_shows_k_per_market_progress_and_the_running_numbers():
    text = report.render_report(as_of=AS_OF, doc=doc(), decision=COLLECTING)
    assert text.startswith("# Shrink prop probabilities: report of 2026-11-03")
    assert "COLLECTING" in text and "640 of 2000" in text
    assert "| pooled | 0.400 | 400 | 0.400 |" in text and "| rush_yds | 0.400 | 400 | 0.400 |" in text
    assert "| pass_tds | 0.000 | 0 | n/a |" in text
    assert "gain +0.00020" in text and "slope 0.910" in text and "lower bound -0.00050" in text
    assert "model version `v2`" in text and "season 2026 week 4 to season 2026 week 4" in text


def test_a_judged_report_names_the_verdict_and_says_it_is_frozen():
    text = report.render_report(as_of=AS_OF, doc=doc(), decision=JUDGED)
    assert "JUDGED on 2026-12-08: FAIL" in text and "informational only" in text
    assert "MISSED" in text and "the user and Claude decide together" in text


def test_an_empty_fit_still_renders():
    text = report.render_report(as_of=AS_OF, doc=fit.build_document([], None, "t"), decision=passtest.decide([], None, "2026-11-03"))
    assert "no legs yet" in text and "0 of 2000" in text


def test_the_decision_block_round_trips():
    block = report.parse_report(report.render_report(as_of=AS_OF, doc=doc(), decision=JUDGED))
    assert block["decision"] == JUDGED and block["as_of"] == AS_OF.isoformat()


@pytest.mark.parametrize("text", ["", None, "no block here", "<!-- shrink-gate: {not json} -->"])
def test_garbage_has_no_block(text):
    assert report.parse_report(text) is None


def _rep(days_from_now, decision):
    return report.render_report(as_of=AS_OF + timedelta(days=days_from_now), doc=doc(), decision=decision)


def test_latest_decision_prefers_the_earliest_judged_report_before_the_date():
    early, later = dict(JUDGED, judged_on="2026-12-08"), dict(JUDGED, judged_on="2026-12-15", verdict="PASS")
    reports = [_rep(35, early), _rep(42, later), _rep(7, COLLECTING)]
    assert report.latest_decision(reports, AS_OF + timedelta(days=60)) == early


def test_latest_decision_ignores_reports_on_or_after_the_date_and_falls_back_to_the_latest_collecting_one():
    reports = [_rep(7, COLLECTING), _rep(14, dict(COLLECTING, scored=900))]
    assert report.latest_decision(reports, AS_OF + timedelta(days=14))["scored"] == 640        # the day-14 report is not "before"
    assert report.latest_decision(reports, AS_OF + timedelta(days=30))["scored"] == 900
    assert report.latest_decision([], AS_OF) is None and report.latest_decision(["junk"], AS_OF) is None


def test_a_corrupt_newer_report_does_not_unfreeze_an_older_judged_verdict():
    reports = [_rep(35, JUDGED), "<!-- shrink-gate: {broken -->"]
    assert report.latest_decision(reports, AS_OF + timedelta(days=90)) == JUDGED


def test_the_report_states_the_pinnacle_limit_of_the_fit_prices_in_one_line():
    text = report.render_report(as_of=AS_OF, doc=doc(), decision=COLLECTING)
    (line,) = [l for l in text.splitlines() if "Pinnacle" in l]
    assert "all books" in line and "major books" in line and "board" in line and "excludes Pinnacle" in line
