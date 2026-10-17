"""The pre-registered CLV verdict: first 300 graded since Nov 1, judged once. Pure functions, no IO."""
from datetime import UTC, datetime, timedelta

import pytest

from sportsmodel.early import passtest
from sportsmodel.take import constants, verdict
from sportsmodel.take.verdict import cluster_lower, criteria, decide, stats, test_rows as pick_test_rows, verdict_row

AS_OF = datetime(2026, 12, 7, 13, 0, tzinfo=UTC)
FIRST_SEEN = datetime(2026, 11, 3, 18, 0, tzinfo=UTC)
GRADED0 = datetime(2026, 11, 4, 3, 0, tzinfo=UTC)


def row(i, *, beat=True, clv=0.01, graded_by="clv", first_seen=FIRST_SEEN, event=None, better_line=False,
        graded_at=None):
    return {"id": i, "event_id": event or f"e{i}", "league": "nfl", "market": "spread", "side": "home",
            "first_seen": first_seen, "graded_by": graded_by, "clv": clv, "beat": beat, "better_line": better_line,
            "graded_at": graded_at or GRADED0 + timedelta(minutes=i)}


def rows(n, **kw):
    return [row(i, **kw) for i in range(1, n + 1)]


def with_beats(n_beat, *, n=300, clv_beat=0.01, clv_miss=0.01):
    return [row(i, beat=i <= n_beat, clv=clv_beat if i <= n_beat else clv_miss) for i in range(1, n + 1)]


def test_not_judged_below_300():
    d = decide(rows(299), None, AS_OF)
    assert d["judged"] is False and d["n"] == 299 and "passed" not in d


def test_judges_first_300_by_graded_at():
    # ids ascend, graded_at descends: the first 300 graded are the HIGHEST ids, not the first-seen order
    rs = [row(i, graded_at=GRADED0 + timedelta(minutes=1000 - i)) for i in range(1, 321)]
    d = decide(rs, None, AS_OF)
    expected = [r["id"] for r in sorted(rs, key=lambda r: (r["graded_at"], r["id"]))[:300]]
    assert d["judged"] is True and len(d["pick_ids"]) == 300 and d["pick_ids"] == expected
    assert d["pick_ids"][0] == 320 and 1 not in d["pick_ids"] and d["n"] == 300


def test_graded_at_ties_break_on_id():
    same = GRADED0
    rs = [row(i, graded_at=same) for i in (5, 3, 4)]
    assert [r["id"] for r in pick_test_rows(rs)] == [3, 4, 5]


def test_later_grades_do_not_change_the_judged_set():
    base = rows(300)
    later = base + [row(301, graded_at=GRADED0 + timedelta(days=30))]
    assert decide(base, None, AS_OF)["pick_ids"] == decide(later, None, AS_OF)["pick_ids"]


def test_october_picks_excluded():
    october = datetime(2026, 10, 31, 23, 0, tzinfo=UTC)
    assert decide(rows(300, first_seen=october), None, AS_OF)["judged"] is False
    # exactly TEST_START counts
    assert len(pick_test_rows(rows(3, first_seen=constants.TEST_START))) == 3


def test_only_clv_and_direction_grades_count():
    mixed = [row(1), row(2, graded_by="direction", clv=None), row(3, graded_by="no_close", clv=None, beat=None),
             {**row(4), "graded_by": None}]
    assert [r["id"] for r in pick_test_rows(mixed)] == [1, 2]


def test_stats_counts_and_rates():
    rs = [row(1, beat=True, clv=0.02, better_line=True), row(2, beat=False, clv=-0.01),
          row(3, graded_by="direction", clv=None, beat=True), row(4, graded_by="direction", clv=None, beat=False)]
    s = stats(rs)
    assert s["n"] == 4 and s["n_clv"] == 2 and s["n_direction"] == 2 and s["n_better_line"] == 1
    assert s["beat_rate"] == pytest.approx(0.5)                     # direction rows count in the beat rate
    assert s["mean_clv"] == pytest.approx(0.005)                    # but not in the mean CLV
    assert s["lower"] is not None


def test_stats_of_nothing_is_none():
    s = stats([])
    assert s["n"] == 0 and s["beat_rate"] is None and s["mean_clv"] is None and s["lower"] is None
    d = stats([row(1, graded_by="direction", clv=None)])
    assert d["mean_clv"] is None and d["lower"] is None and d["beat_rate"] == 1.0 and d["n_clv"] == 0


def test_pass_at_56_percent():
    d = decide(with_beats(168), None, AS_OF)                        # 168/300 = 56%, every CLV +0.01
    assert d["judged"] is True and d["passed"] is True
    assert d["criteria"] == {"beat_rate>=0.55": True, "mean_clv>0": True, "lower>0": True}
    assert d["beat_rate"] == pytest.approx(0.56) and d["mean_clv"] == pytest.approx(0.01)


def test_fail_at_54_percent():
    d = decide(with_beats(162), None, AS_OF)                        # 162/300 = 54%
    assert d["passed"] is False
    assert d["criteria"] == {"beat_rate>=0.55": False, "mean_clv>0": True, "lower>0": True}


def test_beat_rate_boundary_is_inclusive():
    assert decide(with_beats(165), None, AS_OF)["criteria"]["beat_rate>=0.55"] is True      # exactly 55%
    assert decide(with_beats(164), None, AS_OF)["criteria"]["beat_rate>=0.55"] is False


def test_fail_on_negative_mean_clv():
    d = decide(with_beats(168, clv_beat=0.01, clv_miss=-0.05), None, AS_OF)
    assert d["criteria"] == {"beat_rate>=0.55": True, "mean_clv>0": False, "lower>0": False}
    assert d["passed"] is False


def test_fail_on_lower_bound_alone():
    # beat 56%, mean CLV just above 0, but noisy enough that the clustered 95% lower bound is below 0
    d = decide(with_beats(168, clv_beat=0.1, clv_miss=-0.12), None, AS_OF)
    assert d["mean_clv"] > 0 and d["lower"] < 0
    assert d["criteria"] == {"beat_rate>=0.55": True, "mean_clv>0": True, "lower>0": False}
    assert d["passed"] is False


def test_zero_mean_clv_fails_and_no_clv_at_all_fails():
    d = decide(with_beats(168, clv_beat=0.0, clv_miss=0.0), None, AS_OF)
    assert d["criteria"]["mean_clv>0"] is False and d["passed"] is False
    direction = [row(i, graded_by="direction", clv=None) for i in range(1, 301)]
    d = decide(direction, None, AS_OF)
    assert d["judged"] and d["mean_clv"] is None and d["lower"] is None and d["passed"] is False
    assert criteria(d) == {"beat_rate>=0.55": True, "mean_clv>0": False, "lower>0": False}


def test_existing_verdict_never_recomputed():
    old = {"judged": True, "passed": False}
    assert decide(rows(400), old, AS_OF) == {**old, "carried": True}


def test_verdict_row_maps_the_table_columns():
    d = decide(with_beats(168), None, AS_OF)
    r = verdict_row(d, AS_OF)
    assert set(r) == {"judged_at", "n", "beat_rate", "mean_clv", "lower", "passed", "criteria", "pick_ids"}
    assert r["judged_at"] == AS_OF and r["n"] == 300 and r["passed"] is True and len(r["pick_ids"]) == 300
    none = verdict_row({**d, "mean_clv": None, "lower": None, "passed": False}, AS_OF)
    assert none["mean_clv"] == 0.0 and none["lower"] == 0.0         # the table columns are NOT NULL


def test_cluster_lower_deterministic():
    rs = [row(i, clv=(i % 7 - 3) / 100, event=f"e{i // 3}") for i in range(1, 91)]
    a, b = cluster_lower(rs), cluster_lower(rs)
    assert a == b and a is not None
    assert cluster_lower(rs, seed=1) != a
    assert cluster_lower([]) is None and cluster_lower([row(1, clv=None)]) is None


def test_cluster_lower_is_clustered_by_event():
    values = [(i % 9 - 3) / 100 for i in range(60)]
    one_event = [row(i, clv=v, event="same") for i, v in enumerate(values)]
    many = [row(i, clv=v, event=f"e{i}") for i, v in enumerate(values)]
    mean = sum(values) / len(values)
    assert cluster_lower(one_event) == pytest.approx(mean)          # one cluster: every resample is the full mean
    assert cluster_lower(many) < mean - 0.001                       # independent events: a real spread below the mean
    # pairing rows into shared events widens the interval versus treating them as independent
    paired = [row(i, clv=v, event=f"e{i // 2}") for i, v in enumerate(sorted(values))]
    unpaired = [row(i, clv=v, event=f"e{i}") for i, v in enumerate(sorted(values))]
    assert cluster_lower(paired) < cluster_lower(unpaired)


def test_cluster_lower_matches_the_early_monitor_algorithm():
    vals = [(i % 11 - 4) / 50 for i in range(80)]
    ours = [{"event_id": f"g{i // 2}", "clv": v} for i, v in enumerate(vals)]
    theirs = [{"sport": "nfl", "game_pk": f"g{i // 2}", "close_ev": v} for i, v in enumerate(vals)]
    assert cluster_lower(ours, resamples=2500, seed=passtest.BOOTSTRAP_SEED) == pytest.approx(
        passtest.cluster_bootstrap_lower(theirs, resamples=2500, seed=passtest.BOOTSTRAP_SEED))


def test_defaults_are_the_pre_registered_values():
    import inspect
    p = inspect.signature(cluster_lower).parameters
    assert (p["resamples"].default, p["seed"].default, p["percentile"].default) == (10_000, 20261017, 2.5)
    assert verdict.CRITERIA == ("beat_rate>=0.55", "mean_clv>0", "lower>0")
