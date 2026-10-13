"""Structural checks of .github/workflows/report-ledger-gate.yml (weekly, commits only the report) and the rule that
no ledger workflow carries the test-suite guard."""
from tests.ledger.test_workflows_capture import cron_sets
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of


def test_gate_workflow_runs_weekly_on_tuesday_after_grading_and_commits_only_the_report():
    wf = load("report-ledger-gate.yml")
    (cron,) = [c["cron"] for c in wf["on"]["schedule"]]
    m, h, mon, dow = cron_sets(cron)
    assert dow == {2} and h == {15} and m == {47}                                       # Tuesday, after grade-ledger 14:10
    assert wf["permissions"] == {"contents": "write"} and "workflow_dispatch" in wf["on"]
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    gen = steps[step_index(steps, runs("scripts/report_ledger_gate.py"))]
    assert gen["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"}
    commit = steps[step_index(steps, runs("git commit"))]
    assert "git add docs/superpowers/reports/\n" in commit["run"] and "[skip ci]" in commit["run"]
    assert step_index(steps, runs("scripts/report_ledger_gate.py")) < step_index(steps, runs("git commit"))
    assert "git add ." not in commit["run"] and "git add -A" not in commit["run"]
    assert wf["concurrency"]["group"] == "report-ledger-gate"


def test_no_ledger_workflow_sets_the_test_guard():
    for name in ("capture-close.yml", "grade-ledger.yml", "report-ledger-gate.yml", "probe-pinnacle-props.yml"):
        assert "SM_LEDGER_OFF" not in (WF / name).read_text(), name
