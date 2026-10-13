"""Structural checks of .github/workflows/capture-close.yml: cadence, jitter-safe minutes, game-day-only hours,
serialised runs, secrets, never commits. Also hosts the tiny cron evaluator the other workflow tests reuse."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

CAPTURE = "capture-close.yml"


def _expand(field: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            a, b = map(int, part.split("-"))
        else:
            a = b = int(part)
            if step != 1:
                b = hi
        out |= set(range(a, b + 1, step))
    return out


def cron_sets(expr: str):
    m, h, dom, mon, dow = expr.split()
    assert dom == "*", expr
    return _expand(m, 0, 59), _expand(h, 0, 23), _expand(mon, 1, 12), _expand(dow, 0, 6)


def fires(expr: str, dt: datetime) -> bool:
    m, h, mon, dow = cron_sets(expr)
    return dt.minute in m and dt.hour in h and dt.month in mon and (dt.isoweekday() % 7) in dow


def any_fire_in(crons: list[str], start: datetime, end: datetime) -> bool:
    t = start.replace(second=0, microsecond=0)
    while t <= end:
        if any(fires(c, t) for c in crons):
            return True
        t += timedelta(minutes=1)
    return False


def utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def crons():
    return [c["cron"] for c in load(CAPTURE)["on"]["schedule"]]


def test_four_schedules_with_ten_minute_cadence_off_the_peak_minutes(crons):
    assert len(crons) == 4
    football, baseball = crons[:2], crons[2:]
    for c in football:
        assert cron_sets(c)[0] == {3, 13, 23, 33, 43, 53}
    for c in baseball:
        assert cron_sets(c)[0] == {8, 18, 28, 38, 48, 58}
    assert not (cron_sets(football[0])[0] & cron_sets(baseball[0])[0])          # the schedules never fire together
    assert all(0 not in cron_sets(c)[0] and 30 not in cron_sets(c)[0] for c in crons)


def test_seasons_football_aug_to_feb_baseball_mar_to_nov(crons):
    assert cron_sets(crons[0])[2] == {1, 2, 8, 9, 10, 11, 12} == cron_sets(crons[1])[2]
    assert cron_sets(crons[2])[2] == set(range(3, 12)) == cron_sets(crons[3])[2]


@pytest.mark.parametrize("label, kick, months_ok", [
    ("Thu-night NFL 8:15pm ET", utc(2026, 10, 16, 0, 15), True),           # Fri 00:15 UTC
    ("Fri-night CFB 7pm ET", utc(2026, 10, 16, 23, 0), True),
    ("Sat noon CFB", utc(2026, 10, 17, 16, 0), True),
    ("Sat late CFB 10:30pm ET", utc(2026, 10, 18, 2, 30), True),            # Sun 02:30 UTC
    ("Sat Hawaii CFB 11:59pm ET", utc(2026, 10, 18, 4, 0), True),
    ("Sun London NFL 9:30am ET", utc(2026, 10, 18, 13, 30), True),
    ("Sun 1pm ET NFL", utc(2026, 10, 18, 17, 0), True),
    ("Sun 4:25pm ET NFL", utc(2026, 10, 18, 20, 25), True),
    ("SNF", utc(2026, 10, 19, 0, 20), True),                                # Mon 00:20 UTC
    ("MNF", utc(2026, 10, 20, 0, 15), True),                                # Tue 00:15 UTC
    ("Thanksgiving 4:30pm ET", utc(2026, 11, 26, 21, 30), True),
    ("Saturday NFL in December", utc(2026, 12, 19, 21, 30), True),
    ("Bowl game in January", utc(2027, 1, 2, 1, 0), True),
    ("Super Bowl Sunday", utc(2027, 2, 14, 23, 30), True),
    ("Wednesday bowl-week kickoff 8pm ET", utc(2026, 12, 24, 1, 0), True),            # Thu 01:00 UTC
    ("Tuesday MACtion 7pm ET", utc(2026, 11, 4, 0, 0), True),                         # Wed 00:00 UTC
    ("Thursday MACtion noon ET", utc(2026, 11, 5, 16, 0), True),
    ("MLB day game", utc(2026, 10, 21, 17, 8), True),
    ("MLB west-coast night game", utc(2026, 10, 22, 2, 10), True),
    ("MLB 1:05pm ET Sunday in April", utc(2027, 4, 11, 17, 5), True),
])
def test_every_kickoff_has_a_run_inside_its_capture_window(crons, label, kick, months_ok):
    lo, hi = kick - timedelta(minutes=25), kick - timedelta(minutes=5)
    assert any_fire_in(crons, lo, hi), label


def test_football_covers_every_weekday_in_season_so_midweek_bowls_and_maction_are_caught(crons):
    football = crons[:2]
    for day in range(19, 26):                                    # a full Mon-Sun week of October
        assert any_fire_in(football, utc(2026, 10, day, 13, 0), utc(2026, 10, day, 23, 59)), day
        assert any_fire_in(football, utc(2026, 10, day, 0, 0), utc(2026, 10, day, 4, 59)), day


def test_the_offseason_never_fires(crons):
    football, baseball = crons[:2], crons[2:]
    assert not any_fire_in(football, utc(2026, 7, 5, 0, 0), utc(2026, 7, 12, 0, 0))        # football in July
    assert not any_fire_in(baseball, utc(2027, 1, 5, 0, 0), utc(2027, 1, 12, 0, 0))        # baseball in January
    assert not any_fire_in(baseball, utc(2026, 10, 21, 3, 0), utc(2026, 10, 21, 14, 59))   # baseball overnight / morning


def test_runs_per_week_are_bounded(crons):
    t, n = utc(2026, 10, 12, 0, 0), 0
    while t < utc(2026, 10, 19, 0, 0):
        n += any(fires(c, t) for c in crons)
        t += timedelta(minutes=1)
    assert 700 <= n <= 1250          # idle-by-design runs (October overlap is the peak); each exits after one DB read


def test_dispatch_has_a_dry_run_input():
    inp = load(CAPTURE)["on"]["workflow_dispatch"]["inputs"]["dry_run"]
    assert inp["type"] == "boolean" and str(inp["default"]).lower() == "false"


def test_job_runs_the_script_with_secrets_regions_and_serialises_runs():
    wf = load(CAPTURE)
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["group"] == "capture-close" and str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
    (job,) = wf["jobs"].values()
    assert str(job["timeout-minutes"]) == "8"
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/capture_close.py"))]
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert step["env"]["ODDS_API_KEY"] == "${{ secrets.ODDS_API_KEY }}"
    assert step["env"]["CLOSE_PROP_REGIONS"] == "${{ vars.CLOSE_PROP_REGIONS }}"
    assert "'--dry-run'" in step["run"]
    uv = steps[step_index(steps, lambda st: "setup-uv" in str(st.get("uses", "")))]
    assert str(uv["with"]["enable-cache"]).lower() == "true"
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("capture_close.py"))


def test_capture_workflow_never_commits_and_never_sets_the_test_guard():
    text = (WF / CAPTURE).read_text()
    assert "git push" not in text and "git commit" not in text and "SM_LEDGER_OFF" not in text
