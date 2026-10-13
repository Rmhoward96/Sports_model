"""Structural checks of .github/workflows/grade-ledger.yml: nightly, after the other graders, free of Odds API credits."""
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of


def test_grade_ledger_runs_nightly_after_the_other_graders_and_never_spends_credits():
    wf = load("grade-ledger.yml")
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["10 14 * * *"]
    assert wf["on"]["workflow_dispatch"]["inputs"]["days"]["default"] == "14"
    assert wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/grade_ledger.py"))]
    assert step["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"}          # no ODDS_API_KEY: grading is free
    assert step["run"] == 'uv run python scripts/grade_ledger.py --days "${{ github.event.inputs.days || \'14\' }}"'
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("grade_ledger.py"))
    text = (WF / "grade-ledger.yml").read_text()
    assert "git push" not in text and "ODDS_API_KEY" not in text.split("jobs:")[1]
    # later than grade-ev (13:20) and grade-ev-props (13:30) so the night's finals and actuals are in
    assert "20 13 * * *" in (WF / "grade-ev.yml").read_text() and "30 13 * * *" in (WF / "grade-ev-props.yml").read_text()
