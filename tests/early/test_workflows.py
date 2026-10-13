"""Structural checks of the three early-monitor workflows: the hourly window, the env-passed dry run, secrets, serialised
runs, and that only the report workflow commits. Reuses the tiny cron evaluator of the capture-close tests."""
from datetime import timedelta

import pytest

from tests.ledger.test_workflows_capture import cron_sets, fires, utc
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

MONITOR, GRADE, REPORT = "early-monitor.yml", "grade-early-tickets.yml", "report-early-tickets.yml"
CRONS = ["0 16-23 * 1,2,8-12 0", "0 * * 1,2,8-12 1,2", "0 0-3 * 1,2,8-12 3"]


@pytest.fixture(scope="module")
def crons():
    return [c["cron"] for c in load(MONITOR)["on"]["schedule"]]


def fired(crons, start, end):
    out, t = [], start
    while t < end:
        if any(fires(c, t) for c in crons):
            out.append(t)
        t += timedelta(minutes=1)
    return out


# ---- early-monitor.yml ---------------------------------------------------------------------------------------

def test_the_three_crons_are_the_specs():
    assert [c["cron"] for c in load(MONITOR)["on"]["schedule"]] == CRONS


def test_hourly_at_minute_zero_60_polls_a_week(crons):
    week = fired(crons, utc(2026, 10, 12, 0, 0), utc(2026, 10, 19, 0, 0))
    assert len(week) == 60 and {t.minute for t in week} == {0}
    assert len(fired(crons, utc(2026, 11, 2, 0, 0), utc(2026, 11, 9, 0, 0))) == 60


def test_the_window_runs_from_sunday_1600_to_wednesday_0300_utc(crons):
    week = fired(crons, utc(2026, 10, 11, 0, 0), utc(2026, 10, 18, 0, 0))
    assert week[0] == utc(2026, 10, 11, 16, 0) and week[-1] == utc(2026, 10, 14, 3, 0)         # Sun 16:00 ... Wed 03:00
    assert not fired(crons, utc(2026, 10, 14, 4, 0), utc(2026, 10, 18, 16, 0))                  # Wed 04:00 until Sun 16:00
    assert not fired(crons, utc(2026, 10, 11, 0, 0), utc(2026, 10, 11, 16, 0))                  # Sunday before 16:00


def test_football_months_only_aug_to_feb(crons):
    assert cron_sets(crons[0])[2] == {1, 2, 8, 9, 10, 11, 12} == cron_sets(crons[1])[2] == cron_sets(crons[2])[2]
    for month in (3, 4, 5, 6, 7):
        assert not fired(crons, utc(2027, month, 1, 0, 0), utc(2027, month, 8, 0, 0))
    assert fired(crons, utc(2026, 8, 30, 0, 0), utc(2026, 9, 6, 0, 0))                          # the first week of the season


def test_dispatch_has_an_env_passed_dry_run_input():
    wf = load(MONITOR)
    inp = wf["on"]["workflow_dispatch"]["inputs"]["dry_run"]
    assert inp["type"] == "boolean" and str(inp["default"]).lower() == "false"
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/early_monitor.py"))]
    assert step["env"]["DRY_RUN"] == "${{ inputs.dry_run }}"
    assert '[ "$DRY_RUN" = "true" ]' in step["run"] and "--dry-run" in step["run"]
    assert "${{" not in step["run"]                                   # no expression interpolated into the shell script


def test_job_has_the_secrets_serialises_runs_and_caches_uv():
    wf = load(MONITOR)
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["group"] == "early-monitor" and str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
    (job,) = wf["jobs"].values()
    assert str(job["timeout-minutes"]) == "6"
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/early_monitor.py"))]
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}" and step["env"]["ODDS_API_KEY"] == "${{ secrets.ODDS_API_KEY }}"
    uv = steps[step_index(steps, lambda st: "setup-uv" in str(st.get("uses", "")))]
    assert str(uv["with"]["enable-cache"]).lower() == "true"
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("early_monitor.py"))


# ---- grade-early-tickets.yml ---------------------------------------------------------------------------------

def test_grading_runs_daily_at_1420_after_the_other_graders_and_never_spends_credits():
    wf = load(GRADE)
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["20 14 * * *"]
    assert wf["on"]["workflow_dispatch"]["inputs"]["days"]["default"] == "14"
    assert wf["permissions"] == {"contents": "read"} and wf["concurrency"]["group"] == "grade-early-tickets"
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/grade_early_tickets.py"))]
    assert step["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}", "DAYS": "${{ github.event.inputs.days || '14' }}"}
    assert step["run"] == 'uv run python scripts/grade_early_tickets.py --days "$DAYS"'
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("grade_early_tickets.py"))
    text = (WF / GRADE).read_text()
    assert "git push" not in text and "ODDS_API_KEY" not in text.split("jobs:")[1]
    assert "10 14 * * *" in (WF / "grade-ledger.yml").read_text()                                # after grade-ledger
