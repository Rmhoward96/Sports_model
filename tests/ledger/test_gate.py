"""The weekly gate: coverage maths, the row-audit, two-consecutive-weekend logic, the report round trip, and
scripts/report_ledger_gate.py on a fake query function. Pure / no database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel.ledger import gate

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("report_ledger_gate", ROOT / "scripts" / "report_ledger_gate.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["report_ledger_gate"] = rg
_spec.loader.exec_module(rg)

END = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)          # a Tuesday
START = END - timedelta(days=7)


# ---- window ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 10, 20, 15, 45, tzinfo=timezone.utc), END),                        # Tuesday afternoon
    (datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc), END),
    (datetime(2026, 10, 20, 11, 59, tzinfo=timezone.utc), END - timedelta(days=7)),    # before noon: last week's
    (datetime(2026, 10, 23, 3, 0, tzinfo=timezone.utc), END),                          # Friday
    (datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc), END)])                         # Monday
def test_window_end_is_the_latest_tuesday_noon(now, expected):
    assert gate.window_end_for(now) == expected


# ---- coverage ----------------------------------------------------------------------------------------

def test_lines_coverage_counts_only_football_closes_within_thirty_minutes():
    games = [("nfl", 1), ("nfl", 2), ("cfb", 3), ("cfb", 4), ("mlb", 5), ("nfl", 1)]
    mins = {("nfl", 1): 12.0, ("nfl", 2): 31.0, ("cfb", 3): 0.0, ("mlb", 5): 5.0}          # cfb 4: no capture at all
    cov = gate.lines_coverage(games, mins)
    assert cov == {"covered": 2, "total": 4, "pct": 50.0}


def test_lines_coverage_boundaries_and_an_empty_window():
    assert gate.lines_coverage([("nfl", 1)], {("nfl", 1): 30.0})["covered"] == 1
    assert gate.lines_coverage([("nfl", 1)], {("nfl", 1): -0.5})["covered"] == 0            # captured after kickoff
    assert gate.lines_coverage([], {}) == {"covered": 0, "total": 0, "pct": None}


def test_props_coverage_needs_a_kind_and_the_exact_line():
    rows = [{"close_kind": "pinnacle_novig", "close_same_line": True},
            {"close_kind": "us_consensus_novig", "close_same_line": True},
            {"close_kind": "pinnacle_novig", "close_same_line": False},          # Pinnacle on another number
            {"close_kind": "missing", "close_same_line": None}]
    assert gate.props_coverage(rows) == {"covered": 2, "total": 4, "pct": 50.0}
    assert gate.props_coverage([])["pct"] is None


def test_is_prop_pick_counts_parlay_legs_that_have_a_player():
    assert gate.is_prop_pick("ev_prop", None) is True
    assert gate.is_prop_pick("parlay_leg", "p123") is True
    assert gate.is_prop_pick("parlay_leg", None) is False                                  # a game-line leg
    assert gate.is_prop_pick("parlay_leg", "") is False
    assert gate.is_prop_pick("ev_game", "p1") is False and gate.is_prop_pick("model_game", None) is False


# ---- audit -------------------------------------------------------------------------------------------

PREV = {"n": 100, "md5": "abc", "max_id": 100, "as_of": "2026-10-13T12:00:00+00:00"}


def test_audit_baseline_unchanged_and_tampered():
    assert gate.audit_check(None, {"n": 5, "md5": "x"})["passed"] is True
    assert gate.audit_check(PREV, {"n": 100, "md5": "abc"})["passed"] is True
    assert gate.audit_check(PREV, {"n": 99, "md5": "abc"})["passed"] is False              # a row vanished
    bad = gate.audit_check(PREV, {"n": 100, "md5": "zzz"})                                 # a row was edited
    assert bad["passed"] is False and "was 100 / abc, now 100 / zzz" in bad["detail"]
    assert gate.audit_check(PREV, None)["passed"] is False


# ---- evaluation and the two-weekend rule ---------------------------------------------------------------

GOOD_LINES = {"covered": 19, "total": 20, "pct": 95.0}
GOOD_PROPS = {"covered": 45, "total": 50, "pct": 90.0}
GOOD_AUDIT = {"passed": True, "detail": "ok"}


def test_every_check_must_pass_and_the_thresholds_are_inclusive():
    assert gate.evaluate_week(GOOD_LINES, GOOD_PROPS, GOOD_AUDIT)["week_pass"] is True
    for lines, props, audit in [({"covered": 18, "total": 20, "pct": 90.0}, GOOD_PROPS, GOOD_AUDIT),
                                (GOOD_LINES, {"covered": 44, "total": 50, "pct": 88.0}, GOOD_AUDIT),
                                (GOOD_LINES, GOOD_PROPS, {"passed": False, "detail": "tampered"})]:
        assert gate.evaluate_week(lines, props, audit)["week_pass"] is False


def test_nothing_measured_is_a_fail_not_a_vacuous_pass():
    ev = gate.evaluate_week({"covered": 0, "total": 0, "pct": None}, GOOD_PROPS, GOOD_AUDIT)
    assert ev["week_pass"] is False and "nothing measured" in ev["checks"][0]["detail"]


def test_step_one_needs_two_consecutive_passing_weekends():
    assert gate.step_status(True, START, None) == {"streak": 1, "step1_pass": False}
    prev = {"week_pass": True, "window_end": START.isoformat(), "streak": 1}
    assert gate.step_status(True, START, prev) == {"streak": 2, "step1_pass": True}
    assert gate.step_status(False, START, prev) == {"streak": 0, "step1_pass": False}
    assert gate.step_status(True, START, {**prev, "week_pass": False}) == {"streak": 1, "step1_pass": False}
    gap = {**prev, "window_end": (START - timedelta(days=7)).isoformat()}                  # a skipped week breaks the streak
    assert gate.step_status(True, START, gap) == {"streak": 1, "step1_pass": False}


# ---- report round trip -----------------------------------------------------------------------------------

def _report(pass_week=True, streak=1):
    ev = gate.evaluate_week(GOOD_LINES if pass_week else {"covered": 1, "total": 20, "pct": 5.0}, GOOD_PROPS, GOOD_AUDIT)
    st = {"streak": streak, "step1_pass": streak >= 2}
    audit = {**GOOD_AUDIT, "state": {"n": 7, "md5": "m", "max_id": 7, "as_of": END.isoformat()}}
    return gate.render_report(window_start=START, window_end=END, evaluation=ev, status=st, audit=audit,
                              lines=GOOD_LINES, props=GOOD_PROPS,
                              summary=[{"source": "ev_prop", "n": 50, "mean_clv_prob": -0.021, "beat_close_pct": 0.31, "units": -2.5}])


def test_report_round_trips_through_its_machine_block():
    text = _report()
    assert text.startswith("# Pick ledger gate: weekend ending 2026-10-20")
    assert "| ev_prop | 50 | -2.10 pp | 31.0% | -2.50 |" in text
    p = gate.parse_report(text)
    assert p["week_pass"] is True and p["streak"] == 1 and p["window_end"] == END.isoformat()
    assert p["audit"] == {"n": 7, "md5": "m", "max_id": 7, "as_of": END.isoformat()}


def test_report_states_whether_pinnacle_was_requested_for_props():
    base = dict(window_start=START, window_end=END, evaluation=gate.evaluate_week(GOOD_LINES, GOOD_PROPS, GOOD_AUDIT),
                status={"streak": 1, "step1_pass": False}, audit={**GOOD_AUDIT, "state": {}}, lines=GOOD_LINES,
                props=GOOD_PROPS, summary=[])
    both = gate.render_report(**base, prop_regions={"us,eu": 12, "us": 3})
    assert "`us` x 3, `us,eu` x 12" in both and "Pinnacle (EU) prop prices were requested." in both
    us_only = gate.render_report(**base, prop_regions={"us": 15})
    assert "Pinnacle was NOT requested for props" in us_only
    assert "Prop closes captured this weekend: none." in gate.render_report(**base)


def test_report_verdict_text():
    assert "STEP 1 PASSED" in _report(streak=2)
    assert "one more consecutive passing weekend" in _report(streak=1)
    assert "steps 2 and 3 stay blocked" in _report(pass_week=False, streak=0)


def test_latest_previous_picks_the_nearest_earlier_report_and_ignores_junk():
    older = _report().replace(END.isoformat(), (END - timedelta(days=14)).isoformat())
    prior = _report()
    assert gate.latest_previous([older, "no block here", prior], END + timedelta(days=7))["window_end"] == END.isoformat()
    assert gate.latest_previous([prior], END) is None                                   # not strictly earlier
    assert gate.parse_report("<!-- ledger-gate: {not json} -->") is None


# ---- the script on a fake query function --------------------------------------------------------------

def fake_q(games=(), lines=(), props=(), summary=(), audit=(7, "m"), regions=()):
    """props rows are (close_kind, close_same_line) for ev_prop picks, or the full (source, player_id, kind, same)."""
    seen = []

    def q(sql, params=()):
        seen.append(sql)
        if "max(id) FROM pick_ledger" in sql:
            return [(audit[0],)]
        if "FROM pick_ledger t" in sql:
            return [audit]
        if "kind = 'props'" in sql:
            return list(regions)
        if "DISTINCT sport, game_pk" in sql:
            return list(games)
        if "FROM close_capture" in sql:
            return list(lines)
        if "close_same_line FROM" in sql:
            return [("ev_prop", None) + tuple(r) if len(r) == 2 else tuple(r) for r in props]
        return list(summary)

    q.seen = seen
    return q


def test_collect_and_build_report_pass_then_fail_on_a_tampered_ledger():
    games = [("nfl", 1), ("cfb", 2)]
    q = fake_q(games=games, lines=[("nfl", 1, 14.0), ("cfb", 2, 9.0)],
               props=[("pinnacle_novig", True)] * 9 + [("missing", None)], audit=(7, "m"))
    data = rg.collect(q, START, END, None)
    text, res = rg.build_report(END, data, None)
    assert res["evaluation"]["week_pass"] is True and res["status"] == {"streak": 1, "step1_pass": False}
    assert "Prop closes captured this weekend: none." in text
    prev = gate.parse_report(text)
    q2 = fake_q(games=games, lines=[("nfl", 1, 14.0), ("cfb", 2, 9.0)], props=[("pinnacle_novig", True)] * 10, audit=(7, "CHANGED"))
    data2 = rg.collect(q2, END, END + timedelta(days=7), prev)
    _, res2 = rg.build_report(END + timedelta(days=7), data2, prev)
    assert res2["evaluation"]["week_pass"] is False
    assert [c["passed"] for c in res2["evaluation"]["checks"]] == [True, True, False]
    assert all("INSERT" not in s and "UPDATE" not in s and "DELETE" not in s for s in q.seen + q2.seen)


def test_second_clean_weekend_passes_step_one():
    q = fake_q(games=[("nfl", 1)], lines=[("nfl", 1, 14.0)], props=[("us_consensus_novig", True)] * 10)
    data = rg.collect(q, START, END, None)
    text, _ = rg.build_report(END, data, None)
    prev = gate.parse_report(text)
    nxt = END + timedelta(days=7)
    data2 = rg.collect(q, END, nxt, prev)
    _, res = rg.build_report(nxt, data2, prev)
    assert res["status"] == {"streak": 2, "step1_pass": True}


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert rg.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


# ---- the audit by insertion order (id), on a tiny in-memory ledger -------------------------------------------

def ledger_q(rows):
    """Answers the two audit queries from `rows` (list of dicts with an `id`), like Postgres would: max(id), and
    count + md5 over every column of the rows with id <= the bound, ordered by id."""
    import hashlib

    def q(sql, params=()):
        if "max(id) FROM pick_ledger" in sql:
            return [(max((r["id"] for r in rows), default=0),)]
        if "FROM pick_ledger t" in sql:
            assert "t.id <= %s" in sql and "ORDER BY t.id" in sql and "observed_at" not in sql
            sel = sorted((r for r in rows if r["id"] <= params[0]), key=lambda r: r["id"])
            md5 = hashlib.md5("".join(hashlib.md5(repr(sorted(r.items())).encode()).hexdigest() for r in sel).encode()).hexdigest() if sel else None
            return [(len(sel), md5)]
        return []
    return q


def _ledger():
    return [{"id": i, "pick_key": f"k{i}", "observed_at": "2026-10-15T00:00:00+00:00", "best_price": -110} for i in (1, 2, 3)]


def _week(rows, end, previous):
    data = rg.collect(ledger_q(rows), end - timedelta(days=7), end, previous)
    text, res = rg.build_report(end, data, previous)
    return gate.parse_report(text), res


def test_audit_records_the_max_id_and_ignores_a_late_insert_with_an_old_observed_at():
    rows = _ledger()
    prev, _ = _week(rows, END, None)
    assert prev["audit"]["max_id"] == 3 and prev["audit"]["n"] == 3
    rows.append({"id": 4, "pick_key": "late", "observed_at": "2026-10-10T00:00:00+00:00", "best_price": 120})   # old observed_at
    _, res = _week(rows, END + timedelta(days=7), prev)
    assert res["evaluation"]["checks"][2]["passed"] is True


def test_audit_fails_on_an_updated_row_in_any_column():
    rows = _ledger()
    prev, _ = _week(rows, END, None)
    rows[1]["best_price"] = -105
    _, res = _week(rows, END + timedelta(days=7), prev)
    assert res["evaluation"]["checks"][2]["passed"] is False


def test_audit_fails_on_a_deleted_row():
    rows = _ledger()
    prev, _ = _week(rows, END, None)
    del rows[0]
    _, res = _week(rows, END + timedelta(days=7), prev)
    chk = res["evaluation"]["checks"][2]
    assert chk["passed"] is False and "was 3 /" in chk["detail"] and "now 2 /" in chk["detail"]


def test_parlay_prop_legs_count_toward_prop_coverage():
    props = [("ev_prop", None, "pinnacle_novig", True)] * 4 + [
        ("parlay_leg", "p1", "us_consensus_novig", True), ("parlay_leg", "p2", "missing", None),
        ("parlay_leg", None, "missing", None)]                                              # game-line leg: not a prop
    data = rg.collect(fake_q(props=props), START, END, None)
    assert gate.props_coverage(data["props"]) == {"covered": 5, "total": 6, "pct": pytest.approx(100 * 5 / 6)}
