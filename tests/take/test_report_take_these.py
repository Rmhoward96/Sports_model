"""scripts/report_take_these.py: the weekly markdown report and the one-time verdict write. Fake data only."""
import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "report_take_these", Path(__file__).resolve().parents[2] / "scripts" / "report_take_these.py")
rep = importlib.util.module_from_spec(_SPEC)
sys.modules["report_take_these"] = rep
_SPEC.loader.exec_module(rep)

AS_OF = datetime(2026, 12, 7, 13, 0, tzinfo=UTC)               # Monday 08:00 ET
FIRST_SEEN = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)
GRADED0 = datetime(2026, 11, 4, 3, 0, tzinfo=UTC)
OCT = datetime(2026, 10, 20, 18, 0, tzinfo=UTC)


def row(i, *, league="nfl", market="spread", beat=True, clv=0.01, graded_by="clv", first_seen=FIRST_SEEN,
        better_line=False, outcome="win", units=0.9):
    return {"id": i, "event_id": f"e{i}", "league": league, "market": market, "side": "home", "first_seen": first_seen,
            "graded_by": graded_by, "clv": clv, "beat": beat, "better_line": better_line, "outcome": outcome,
            "units": units, "graded_at": None if graded_by is None else GRADED0 + timedelta(minutes=i)}


def rows(n, **kw):
    return [row(i, **kw) for i in range(1, n + 1)]


def many(n_beat, n=300):
    return [row(i, beat=i <= n_beat) for i in range(1, n + 1)]


def call(job, league, credits, missing=None, days_ago=1):
    return {"job": job, "league": league, "credits_used": credits, "books_missing": missing,
            "called_at": AS_OF - timedelta(days=days_ago)}


def section(md, heading):
    """The text of one '## heading' section."""
    body = md.split(f"## {heading}", 1)[1]
    return body.split("\n## ", 1)[0]


def test_report_contains_progress_and_caveat():
    md, new = rep.build_report(AS_OF, rows(10), None, [])
    assert "10 of 300 graded picks since Nov 1" in md
    assert "Your own Pikkit beat-close rate over the same dates is on the Scorecard page " \
           "(Pikkit's close is not necessarily Pinnacle's)." in md
    assert new is None and "not judged yet" in md


def test_progress_shows_the_three_criteria_live():
    md, _ = rep.build_report(AS_OF, many(150, n=200), None, [])                  # 75% beat, +0.01 CLV
    prog = section(md, "Progress")
    assert "200 of 300 graded picks" in prog
    assert "| beat_rate>=0.55 | 75.0% | yes |" in prog
    assert "| mean_clv>0 | +0.0100 | yes |" in prog
    assert "| lower>0 | +0.0100 | yes |" in prog
    md, _ = rep.build_report(AS_OF, many(100, n=200), None, [])                  # 50% beat
    assert "| beat_rate>=0.55 | 50.0% | no |" in section(md, "Progress")


def test_october_picks_do_not_count_toward_progress():
    md, new = rep.build_report(AS_OF, rows(5) + rows(40, first_seen=OCT), None, [])
    assert "5 of 300 graded picks" in md and new is None


def test_newly_judged_returns_the_row_and_states_the_verdict():
    md, new = rep.build_report(AS_OF, many(168), None, [])
    assert new is not None and new["passed"] is True and new["n"] == 300 and len(new["pick_ids"]) == 300
    assert new["judged_at"] == AS_OF and set(new["criteria"]) == {"beat_rate>=0.55", "mean_clv>0", "lower>0"}
    assert "**Verdict: PASSED**" in md and "judged 2026-12-07" in md
    md, new = rep.build_report(AS_OF, many(150), None, [])
    assert new["passed"] is False and "**Verdict: FAILED**" in md


def test_existing_verdict_is_carried_and_never_rewritten():
    old = {"judged_at": datetime(2026, 12, 1, 3, 0, tzinfo=UTC), "n": 300, "beat_rate": 0.5, "mean_clv": -0.002,
           "lower": -0.01, "passed": False, "pick_ids": [1, 2, 3],
           "criteria": {"beat_rate>=0.55": False, "mean_clv>0": False, "lower>0": False}}
    md, new = rep.build_report(AS_OF, many(299, n=400), old, [])                 # data that would now pass
    assert new is None and "**Verdict: FAILED**" in md and "judged 2026-11-30" in md      # 03:00Z = 22:00 ET the day before
    assert "| beat_rate>=0.55 | 50.0% | no |" in section(md, "Progress")


def test_dates_are_us_eastern():
    late = datetime(2026, 12, 8, 3, 0, tzinfo=UTC)                                # 22:00 ET on the 7th
    md, _ = rep.build_report(late, rows(3), None, [])
    assert "2026-12-07" in md and "2026-12-08" not in md
    assert rep.report_path(Path("r"), late) == Path("r/2026-12-08-take-these.md")     # the filename is the as-of date


def test_per_league_and_market_tables():
    rs = [row(1, league="nfl", market="spread", beat=True, clv=0.02), row(2, league="nfl", market="spread", beat=False, clv=-0.01),
          row(3, league="nfl", market="total", beat=True, clv=0.03), row(4, league="epl", market="ml", beat=False, clv=-0.02),
          row(5, league="epl", market="ml", graded_by="direction", clv=None, beat=True)]
    md, _ = rep.build_report(AS_OF, rs, None, [])
    leagues = section(md, "Per league")
    assert "| nfl | 3 | 66.7% | +0.0133 |" in leagues
    assert "| epl | 2 | 50.0% | -0.0200 |" in leagues                              # the direction pick has no CLV
    assert leagues.index("| nfl |") < leagues.index("| epl |")                      # league order of the model
    markets = section(md, "Per market")
    assert "| spread | 2 | 50.0% | +0.0050 |" in markets and "| total | 1 | 100.0% | +0.0300 |" in markets
    assert "| ml | 2 | 50.0% | -0.0200 |" in markets


def test_better_line_picks_are_reported_separately():
    rs = [row(1, better_line=True, beat=True, clv=0.04), row(2, better_line=True, beat=False, clv=-0.02),
          row(3), row(4)]
    md, _ = rep.build_report(AS_OF, rs, None, [])
    assert "2 better-line picks: beat-close 50.0%, mean CLV +0.0100" in section(md, "Better-line picks")
    md, _ = rep.build_report(AS_OF, rows(3), None, [])
    assert "No better-line picks graded yet" in section(md, "Better-line picks")


def test_direction_no_close_and_ungraded_counts():
    rs = [row(1), row(2, graded_by="direction", clv=None), row(3, graded_by="direction", clv=None),
          row(4, graded_by="no_close", clv=None, beat=None), row(5, graded_by=None, clv=None, beat=None, outcome=None, units=None),
          row(6, graded_by=None, clv=None, beat=None, outcome=None, units=None), row(7, first_seen=OCT, graded_by="no_close", clv=None, beat=None)]
    md, _ = rep.build_report(AS_OF, rs, None, [])
    grading = section(md, "Grading")
    assert "Graded by line direction (no CLV): 2" in grading
    assert "No valid close (not counted in the test): 1" in grading
    assert "Awaiting grade: 2" in grading                                        # the October no_close pick is not counted


def test_results_record_and_units():
    rs = [row(1, outcome="win", units=0.91), row(2, outcome="win", units=1.2), row(3, outcome="loss", units=-1.0),
          row(4, outcome="push", units=0.0), row(5, outcome=None, units=None), row(6, first_seen=OCT, outcome="win", units=5.0)]
    md, _ = rep.build_report(AS_OF, rs, None, [])
    results = section(md, "Results")
    assert "W-L-P 2-1-1" in results and "+1.110 units" in results and "4 settled" in results


def test_october_shakedown_is_labelled_not_judged():
    rs = (rows(2) + [row(10, first_seen=OCT, beat=True, clv=0.02, outcome="win", units=1.0),
                     row(11, first_seen=OCT, beat=False, clv=-0.01, outcome="loss", units=-1.0),
                     row(12, first_seen=OCT, graded_by=None, clv=None, beat=None, outcome=None, units=None)])
    md, _ = rep.build_report(AS_OF, rs, None, [])
    assert "## October shakedown (not judged)" in md
    shake = section(md, "October shakedown (not judged)")
    assert "3 picks first seen before Nov 1" in shake and "2 graded" in shake
    assert "beat-close 50.0%" in shake and "mean CLV +0.0050" in shake and "W-L-P 1-1-0" in shake


def test_no_october_picks_says_so():
    md, _ = rep.build_report(AS_OF, rows(3), None, [])
    assert "No picks were logged before Nov 1" in section(md, "October shakedown (not judged)")


def test_credits_per_job_and_books_missing_from_the_call_log():
    calls = [call("pull", "nfl", 3, ["fanatics"]), call("pull", "nfl", 3, ["fanatics", "ballybet"]), call("pull", "mlb", 3, []),
             call("close", "nfl", 1, []), call("close", "nfl", None, []), call("scores", "nfl", 2, [])]
    md, _ = rep.build_report(AS_OF, rows(3), None, calls)
    credits = section(md, "Credits used")
    assert "| pull | 9 |" in credits and "| close | 1 |" in credits and "| scores | 2 |" in credits
    assert "| total | 12 |" in credits
    books = section(md, "Books missing")
    assert "nfl: fanatics (2 of 5 calls), ballybet (1 of 5 calls)" in books
    assert "mlb" not in books


def test_empty_call_log_reads_zero_and_none():
    md, _ = rep.build_report(AS_OF, rows(3), None, [])
    assert "| total | 0 |" in section(md, "Credits used")
    assert "None in the last 7 days" in section(md, "Books missing")


# ---- main ---------------------------------------------------------------------------------------

def _wire(monkeypatch, *, rows_=(), verdict=None, calls=(), tables=(), inserted=None):
    monkeypatch.setenv("DATABASE_URL", "x")
    monkeypatch.setattr(rep, "writes_disabled", lambda: False)
    monkeypatch.setattr(rep, "_missing_tables", lambda: list(tables))
    monkeypatch.setattr(rep, "_collect", lambda since: (list(rows_), verdict, list(calls)))
    log = [] if inserted is None else inserted
    monkeypatch.setattr(rep, "_insert_verdict", lambda r: log.append(r) or True)
    return log


def test_main_writes_the_report_file(monkeypatch, tmp_path):
    _wire(monkeypatch, rows_=rows(10))
    assert rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path)]) == 0
    text = (tmp_path / "2026-12-07-take-these.md").read_text()
    assert "10 of 300 graded picks since Nov 1" in text


def test_main_with_zero_picks_writes_nothing(monkeypatch, tmp_path, capsys):
    _wire(monkeypatch, rows_=[])
    assert rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == [] and "no picks" in capsys.readouterr().out


def test_main_inserts_the_verdict_once_when_newly_judged(monkeypatch, tmp_path):
    log = _wire(monkeypatch, rows_=many(168))
    assert rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path)]) == 0
    (new,) = log
    assert new["passed"] is True and new["n"] == 300 and new["judged_at"] == AS_OF
    assert "**Verdict: PASSED**" in (tmp_path / "2026-12-07-take-these.md").read_text()


def test_main_does_not_insert_when_a_verdict_exists(monkeypatch, tmp_path):
    old = {"judged_at": AS_OF, "n": 300, "beat_rate": 0.5, "mean_clv": 0.0, "lower": 0.0, "passed": False,
           "pick_ids": [], "criteria": {"beat_rate>=0.55": False, "mean_clv>0": False, "lower>0": False}}
    log = _wire(monkeypatch, rows_=many(168), verdict=old)
    assert rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path)]) == 0
    assert log == []


def test_main_failed_verdict_insert_writes_no_report(monkeypatch, tmp_path):
    _wire(monkeypatch, rows_=many(168))

    def boom(r):
        raise RuntimeError("db down")
    monkeypatch.setattr(rep, "_insert_verdict", boom)
    with pytest.raises(RuntimeError):
        rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_main_collects_the_last_seven_days_of_calls(monkeypatch, tmp_path):
    seen = []
    _wire(monkeypatch, rows_=rows(3))
    monkeypatch.setattr(rep, "_collect", lambda since: seen.append(since) or (rows(3), None, []))
    rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path)])
    assert seen == [AS_OF - timedelta(days=7)]


def test_main_missing_tables_or_env(monkeypatch, tmp_path, capsys):
    _wire(monkeypatch, rows_=rows(3), tables=["take_verdict"])
    assert rep.main(["--reports-dir", str(tmp_path)]) == 0 and list(tmp_path.iterdir()) == []
    assert "migration_take_these.sql" in capsys.readouterr().out
    monkeypatch.delenv("DATABASE_URL")
    assert rep.main(["--reports-dir", str(tmp_path)]) == 1


def test_main_dry_run_and_ledger_off_print_instead_of_writing(monkeypatch, tmp_path, capsys):
    log = _wire(monkeypatch, rows_=many(168))
    assert rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == [] and log == [] and "**Verdict: PASSED**" in capsys.readouterr().out
    monkeypatch.setattr(rep, "writes_disabled", lambda: True)
    assert rep.main(["--as-of", "2026-12-07T13:00:00Z", "--reports-dir", str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == [] and log == []
