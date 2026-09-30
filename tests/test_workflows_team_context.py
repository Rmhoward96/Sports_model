"""Structural checks of .github/workflows/build-team-context.yml (matchup /
history / rankings Task 6): daily 13:00 UTC + dispatch, runs the team-context
job for the chosen sport with DATABASE_URL only (the CFBD key belongs to
build-cfb-advanced.yml), and never commits. Reuses the block-YAML reader."""
from __future__ import annotations

from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

NAME = "build-team-context.yml"


def test_triggers_daily_1300_utc_and_dispatch():
    wf = load(NAME)
    crons = [c["cron"] for c in wf["on"]["schedule"]]
    assert crons == ["0 13 * * *"]
    inp = wf["on"]["workflow_dispatch"]["inputs"]["sport"]
    assert inp["default"] == "all"
    assert inp["options"] == ["all", "nfl", "cfb"]


def test_job_runs_script_with_database_url_only():
    wf = load(NAME)
    assert wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/build_team_context.py"))]
    assert step["run"] == 'uv run python scripts/build_team_context.py --sport "$SPORT"'
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert step["env"]["SPORT"] == "${{ inputs.sport || 'all' }}"
    assert "CFBD_API_KEY" not in (WF / NAME).read_text()
    assert int(job["timeout-minutes"]) <= 30
    assert job["concurrency"]["group"] == "build-team-context"
    # uv sync before the job
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("build_team_context.py"))


def test_workflow_never_commits():
    text = (WF / NAME).read_text()
    assert "git push" not in text and "git commit" not in text
