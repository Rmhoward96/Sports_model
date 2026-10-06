"""Structural checks of .github/workflows/build-game-info.yml: daily 14:00 UTC plus 16:00 / 22:00 UTC on
football days, runs the game-info job with DATABASE_URL + CFBD_API_KEY, never commits."""
from __future__ import annotations

from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

NAME = "build-game-info.yml"


def test_schedule_daily_plus_football_days_and_dispatch():
    wf = load(NAME)
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["0 14 * * *", "0 16 * * 0,1,4,5,6", "0 22 * * 0,1,4,5,6"]
    inp = wf["on"]["workflow_dispatch"]["inputs"]["sport"]
    assert inp["default"] == "all" and inp["options"] == ["all", "nfl", "cfb"]
    days = wf["on"]["schedule"][1]["cron"].split()[4].split(",")
    assert sorted(days) == ["0", "1", "4", "5", "6"]          # Sun Mon Thu Fri Sat


def test_job_runs_the_script_with_both_secrets_and_serializes_runs():
    wf = load(NAME)
    assert wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/build_game_info.py"))]
    assert step["run"] == 'uv run python scripts/build_game_info.py --sport "$SPORT"'
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert step["env"]["CFBD_API_KEY"] == "${{ secrets.CFBD_API_KEY }}"
    assert step["env"]["SPORT"] == "${{ inputs.sport || 'all' }}"
    assert job["concurrency"]["group"] == "build-game-info"
    assert str(job["concurrency"]["cancel-in-progress"]).lower() == "false"
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("build_game_info.py"))


def test_workflow_never_commits():
    text = (WF / NAME).read_text()
    assert "git push" not in text and "git commit" not in text
